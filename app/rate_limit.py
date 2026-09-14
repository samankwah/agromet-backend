"""A quota for the routes that spend money.

`/api/chat` takes an unauthenticated POST and turns it into a paid model call.
Before this module there was nothing between the open internet and that bill:
no login, no key, no counter. One loop left running overnight is the whole
budget.

**What this is honest about.** The counter lives in process memory. On a
long-running host that is exactly right, and it is the cheapest thing that
works. On a serverless host each cold instance starts with an empty counter, so
a distributed burst spread across instances can slip through. That is a real
limitation, not one to paper over. It still catches what actually happens --
one client hammering a warm instance -- and the ceilings that always hold are
elsewhere: the message length cap in `schemas.py` and `max_output_tokens` on the
model call bound what a single request can ever cost.

`Limiter` keeps no global state of its own, so swapping the tally for a shared
store (Vercel KV, Upstash, Redis) means reimplementing one class against the
same three methods, not unpicking the call sites.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """The answer to "may this request run?", with what to tell the caller."""

    allowed: bool
    #: Seconds until the caller may try again. 0 when allowed.
    retry_after: int = 0
    #: Which ceiling was hit: "burst" or "daily". Empty when allowed.
    reason: str = ""

    @property
    def message(self) -> str:
        """Plain words for the farmer, not for the operator.

        No numbers and no quota vocabulary: someone who has asked a lot of
        questions in a row does not need to be told about windows and limits,
        they need to know it will work again and roughly when.
        """
        if self.allowed:
            return ""
        if self.reason == "daily":
            return "You have asked AgroMet AI a lot of questions today. Please try again tomorrow."
        minutes = max(1, round(self.retry_after / 60))
        if minutes == 1:
            return "You are asking faster than AgroMet AI can keep up. Please try again in a minute."
        return f"You are asking faster than AgroMet AI can keep up. Please try again in about {minutes} minutes."


class Limiter:
    """A sliding window and a daily ceiling over the same timestamps.

    One deque per key holds the last day of request times. The burst check reads
    its tail, the daily check reads its length, and pruning happens on the way
    in, so nothing has to be swept on a timer -- which matters on a host where
    there may be no process alive between two requests to run a timer.
    """

    #: How much slack a key class gets over the base ceiling.
    #:
    #: An address is a far coarser identity than a device, and in rural Ghana it
    #: is routinely shared: one tower, one NAT, a whole village behind a single
    #: public address. Holding an IP to one farmer's ceiling would cut off
    #: everybody in that village the moment one of them asked a lot of
    #: questions. So the address gets room for a crowd and the device carries
    #: the real per-farmer limit, which is the identity a caller cannot dodge
    #: without also dropping the id the app sends.
    DEFAULT_MULTIPLIERS = {"ip": 10}

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int,
        daily_limit: int,
        max_keys: int = 5000,
        multipliers: dict[str, int] | None = None,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.daily_limit = daily_limit
        #: Bounds memory on a long-running host. Ordered by last use, so the
        #: entry evicted is the one idle longest -- never the busy caller the
        #: limit exists for.
        self.max_keys = max_keys
        self.multipliers = dict(self.DEFAULT_MULTIPLIERS if multipliers is None else multipliers)
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def _scale(self, key: str) -> int:
        return self.multipliers.get(key.split(":", 1)[0], 1)

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits.get(key)
        if hits is None:
            hits = deque()
            self._hits[key] = hits
        self._hits.move_to_end(key)

        day_ago = now - 86400
        while hits and hits[0] < day_ago:
            hits.popleft()

        while len(self._hits) > self.max_keys:
            self._hits.popitem(last=False)

        return hits

    def check(self, keys: list[str], now: float | None = None) -> Decision:
        """Whether every key is under both ceilings. Records nothing.

        Several keys because one request is both a device and an address, and a
        caller who rotates one still has the other. The strictest verdict wins.
        """
        now = time.time() if now is None else now

        for key in keys:
            hits = self._prune(key, now)
            scale = self._scale(key)

            daily_limit = self.daily_limit * scale
            if daily_limit and len(hits) >= daily_limit:
                oldest = hits[0]
                return Decision(False, max(1, int(oldest + 86400 - now)), "daily")

            window_start = now - self.window_seconds
            recent = [hit for hit in hits if hit >= window_start]
            burst_limit = self.limit * scale
            if burst_limit and len(recent) >= burst_limit:
                return Decision(False, max(1, int(recent[0] + self.window_seconds - now)), "burst")

        return Decision(True)

    def record(self, keys: list[str], now: float | None = None) -> None:
        """Count one request against every key."""
        now = time.time() if now is None else now
        for key in keys:
            self._prune(key, now).append(now)

    def reset(self) -> None:
        """Test hook. Nothing in the app calls this."""
        self._hits.clear()


def client_keys(device_id: str | None, ip: str | None) -> list[str]:
    """The identities one request is counted against.

    The device id is the app's own guest id, which survives a changing mobile IP
    -- the common case in rural Ghana, where one tower's subscribers can share
    an address and a farmer's address changes as they move. The IP is what
    remains when a caller declines to send an id, which is itself a signal: a
    scripted caller is the one that omits it.
    """
    keys = []
    if device_id:
        keys.append(f"device:{device_id.strip()[:64]}")
    if ip:
        keys.append(f"ip:{ip}")
    return keys or ["anonymous"]


def client_ip(headers, fallback: str | None = None) -> str | None:
    """The caller's address as the platform reports it.

    `X-Forwarded-For` is trusted because on a hosted deployment the platform
    writes it and the app is not reachable except through the platform. Behind
    nothing at all it is spoofable, which is why it is only ever one of the keys
    and never the only one.
    """
    forwarded = headers.get("x-forwarded-for") if headers else None
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return fallback
