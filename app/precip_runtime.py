"""Hourly precipitation over Ghana, fetched once and served to everyone.

Mirrors ``s2s_runtime``: a module-level dict with a TTL and an ``asyncio`` lock,
refreshed off the read path, serving whatever it already has.

**Why this one has to be proxied at all.** The rain map is the first screen that
wants a field rather than a point, and Open-Meteo weights a request by its
location count. Fetched from each device, several hundred points per phone
consumes the free tier in proportion to the user base rather than to time --
which is to say it works in testing and fails on launch day. Cached here it is
one upstream call per TTL for everyone.

**The arithmetic that sets the TTL.** The grid is 341 land points (see
``precip_grid.json``). At one refresh an hour that is 341 x 24 = 8,184 weighted
calls a day, inside Open-Meteo's 10,000. Halving the TTL to thirty minutes puts
it at 16,368 and over the limit, so the interval is not a free knob. The data is
hourly in any case, so a shorter TTL would mostly re-fetch the same numbers.

**The grid is a shipped asset, not computed here.** It is generated from the
same ``ghanaBoundaries.json`` the app draws with, by snapping the 865 tagged
display cells onto a quarter-degree lattice and deduplicating. That gives a land
mask for free and, more to the point, means the server and the client cannot
disagree about which cells exist.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
GRID_PATH = DATA_DIR / "precip_grid.json"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# One day back and two forward. The past day matters as much as the forecast:
# the satellite half of the map runs about six hours behind, so without the
# model's own recent hours the timeline has a hole between the last measurement
# and now.
PAST_DAYS = 1
FORECAST_DAYS = 2

# See the module header: this is bounded by the free tier, not by taste.
CACHE_TTL_SECONDS = 3600

UPSTREAM_TIMEOUT = 30.0

DATA_SOURCES = ["Open-Meteo (CC BY 4.0)"]


def _load_grid() -> tuple[list[tuple[float, float]], float]:
    try:
        payload = json.loads(GRID_PATH.read_text(encoding="utf-8"))
        points = [(float(lat), float(lng)) for lat, lng in payload["points"]]
        return points, float(payload.get("stepDeg", 0.25))
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("precipitation grid asset is unreadable")
        return [], 0.25


GRID, STEP_DEG = _load_grid()

_CACHE: dict = {}
_CACHE_STAMP: float = 0.0
_LAST_ERROR: str | None = None
_REFRESH_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------

async def _fetch_field() -> dict:
    """One request for the whole field.

    Open-Meteo accepts a comma-separated coordinate list and answers with an
    array in the same order, which is the same trick ``s2s_runtime`` uses for its
    165 ensemble points and ``hazards`` for its sixteen regions.
    """
    if not GRID:
        raise RuntimeError("precipitation grid asset is empty")

    params = {
        "latitude": ",".join(f"{lat:.2f}" for lat, _ in GRID),
        "longitude": ",".join(f"{lng:.2f}" for _, lng in GRID),
        "hourly": "precipitation",
        "timezone": "UTC",
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "cell_selection": "land",
    }

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        response = await client.get(FORECAST_URL, params=params)
    response.raise_for_status()

    payload = response.json()
    # A single-coordinate request returns an object, a multi-coordinate one an
    # array. We always send hundreds; normalise defensively anyway.
    entries = [payload] if isinstance(payload, dict) else payload

    times = ((entries[0] or {}).get("hourly") or {}).get("time") or []
    if not times:
        raise RuntimeError("upstream returned no hours")

    # Rows are hours and columns are cells, because a frame of the animation is
    # one hour across every cell. Transposing here means the client never has to.
    values: list[list[float]] = []
    for hour in range(len(times)):
        row: list[float] = []
        for cell in range(len(GRID)):
            series = ((entries[cell] or {}).get("hourly") or {}).get("precipitation") or []
            reading = series[hour] if hour < len(series) else None
            # A missing hour is genuinely no recorded rain for that step, which
            # is the one place a zero-fill is honest.
            row.append(0.0 if reading is None else round(float(reading), 2))
        values.append(row)

    # Open-Meteo returns zone-less stamps and we asked for UTC, so mark them as
    # such rather than leaving the client to assume it.
    return {"times": [f"{time}Z" if not time.endswith("Z") else time for time in times], "values": values}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_age_seconds() -> float | None:
    return None if not _CACHE else time.time() - _CACHE_STAMP


def is_stale() -> bool:
    age = cache_age_seconds()
    return age is None or age > CACHE_TTL_SECONDS


def cached_snapshot() -> tuple[dict, float]:
    return _CACHE, _CACHE_STAMP


def last_error() -> str | None:
    return _LAST_ERROR


async def refresh(force: bool = False) -> bool:
    """Refresh the cache. Returns True when new data was stored.

    The lock stops a burst of concurrent page loads on a cold container from each
    firing their own upstream request.
    """
    global _CACHE, _CACHE_STAMP, _LAST_ERROR

    if not force and not is_stale():
        return False

    async with _REFRESH_LOCK:
        if not force and not is_stale():
            return False
        try:
            snapshot = await _fetch_field()
        except Exception as exc:
            _LAST_ERROR = f"{type(exc).__name__}: {exc}"
            logger.warning("precipitation refresh failed: %s", _LAST_ERROR)
            return False

        _CACHE = snapshot
        _CACHE_STAMP = time.time()
        _LAST_ERROR = None
        return True


async def ensure_fresh() -> None:
    """Refresh only when there is nothing to serve at all.

    An expired snapshot is still served and revalidated behind the response, so a
    page load never waits on the upstream timeout.
    """
    if not _CACHE:
        await refresh(force=True)


def metadata() -> dict:
    """Provenance block attached to every response."""
    stamp = _CACHE_STAMP and datetime.fromtimestamp(_CACHE_STAMP, tz=timezone.utc).isoformat()
    return {
        "issuedAt": stamp or None,
        "stale": is_stale(),
        "stepDeg": STEP_DEG,
        "sources": DATA_SOURCES,
        "error": _LAST_ERROR,
    }


def reset_cache() -> None:
    """Drop everything. For tests only."""
    global _CACHE, _CACHE_STAMP, _LAST_ERROR
    _CACHE = {}
    _CACHE_STAMP = 0.0
    _LAST_ERROR = None
