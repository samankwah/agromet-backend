"""Open-Meteo weather, fetched and cached per location.

Structured like `hazard_runtime`, for the same reasons: an in-process TTL cache
with a refresh lock and stale-while-revalidate reads. See that module's header
for the full rationale; the short version is that this backend runs serverless
on Vercel, where `/tmp` is per-instance and empty on every cold start, so a
SQLite cache table would buy nothing a dict does not.

What is deliberately *not* here: normalisation. The route hands Open-Meteo's own
response back inside the house envelope, and the mobile app maps it to its
domain types. That is not laziness — the app also calls Open-Meteo directly when
this backend is unreachable, and one mapping written once in TypeScript is
better than two that have to be kept in step across languages. The cost is that
swapping to a different provider later means changing this shape, not just this
file.

Ambee is the road not taken. `main.py` still carries two Ambee routes, but
`AMBEE_API_KEY` has never been set, so every call short-circuits to 503. Ghana's
own hazard index already runs on Open-Meteo (`hazards.py`), which needs no key.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo publishes hourly. Half that keeps "current conditions" honest
# without hammering a free service — the hazards module's six hours is right for
# a daily index and much too slow for a temperature reading.
CACHE_TTL_SECONDS = 30 * 60
UPSTREAM_TIMEOUT = 20.0

# Ghana spans about 5°N-11°N and 3°W-1°E. Rounding the cache key to 2dp is
# roughly a kilometre, so the ten home towns each get their own entry while a
# jittery GPS fix cannot fragment the cache.
CACHE_PRECISION = 2

FORECAST_DAYS = 7

CURRENT_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
)

DAILY_FIELDS = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "precipitation_sum",
    "precipitation_probability_max",
    "sunrise",
    "sunset",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
)

# The day-detail screen charts the first eight. `apparent_temperature_max`,
# `wind_gusts_10m_max` (daily) and `cape` (hourly) are for the severe-weather
# alert in `mobile/src/shared/domain/weatherHazards.ts` — gusts and feels-like
# heat are what a farmer is warned about, and mean wind / air temp understate
# both. `uv_index` and `cape` are dimensionless; everything else arrives in
# °C / % / mm / km/h, which is what the app assumes.
HOURLY_FIELDS = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "weather_code",
    "wind_speed_10m",
    "uv_index",
    "cape",
)

_CACHE: dict[str, dict] = {}
_CACHE_STAMPS: dict[str, float] = {}
_REFRESH_LOCK = asyncio.Lock()
_LAST_ERROR: str | None = None


def cache_key(lat: float, lng: float) -> str:
    return f"{round(lat, CACHE_PRECISION)},{round(lng, CACHE_PRECISION)}"


def is_stale(key: str) -> bool:
    stamp = _CACHE_STAMPS.get(key)
    if not stamp:
        return True
    return (time.time() - stamp) > CACHE_TTL_SECONDS


def cached_bundle(key: str) -> dict | None:
    return _CACHE.get(key)


async def _fetch(lat: float, lng: float) -> dict:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lng:.4f}",
        "current": ",".join(CURRENT_FIELDS),
        "daily": ",".join(DAILY_FIELDS),
        "hourly": ",".join(HOURLY_FIELDS),
        # Ghana is UTC+0 year-round with no DST, so asking for local time costs
        # nothing and means the date boundaries in `daily` line up with the
        # farmer's day rather than with UTC midnight somewhere else.
        "timezone": "Africa/Accra",
        "forecast_days": FORECAST_DAYS,
    }

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        response = await client.get(FORECAST_URL, params=params)
        response.raise_for_status()
        return response.json()


async def refresh(lat: float, lng: float, force: bool = False) -> bool:
    """Fetch and store one location. True when new data was written.

    The lock stops a cold container serving ten carousel towns at once from
    firing ten simultaneous upstream requests for the same coordinates.
    """
    global _LAST_ERROR

    key = cache_key(lat, lng)
    if not force and not is_stale(key):
        return False

    async with _REFRESH_LOCK:
        # Another coroutine may have refreshed this key while we waited.
        if not force and not is_stale(key):
            return False
        try:
            bundle = await _fetch(lat, lng)
        except Exception as exc:
            _LAST_ERROR = f"{type(exc).__name__}: {exc}"
            logger.warning("weather refresh failed for %s: %s", key, _LAST_ERROR)
            return False

        _CACHE[key] = bundle
        _CACHE_STAMPS[key] = time.time()
        _LAST_ERROR = None
        return True


async def ensure_fresh(lat: float, lng: float) -> None:
    """Block only when there is nothing at all to serve for this location.

    Once a bundle exists the read path returns it immediately and leaves
    revalidation to a background task, so a page load is never held open behind
    a twenty-second upstream timeout.
    """
    if cache_key(lat, lng) not in _CACHE:
        await refresh(lat, lng, force=True)


def metadata(key: str) -> dict:
    """Provenance block attached to every response."""
    stamp = _CACHE_STAMPS.get(key)
    return {
        "fetchedAt": datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat() if stamp else None,
        "stale": is_stale(key),
        "source": "Open-Meteo",
        "error": _LAST_ERROR,
    }


def reset_cache() -> None:
    """Test hook. Nothing in the app calls this."""
    global _LAST_ERROR
    _CACHE.clear()
    _CACHE_STAMPS.clear()
    _LAST_ERROR = None
