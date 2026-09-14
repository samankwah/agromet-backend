"""Fetching, caching and assembly for the flood/drought endpoints.

Separated from ``hazards.py`` (pure scoring) and from ``main.py`` (routing) so
that the I/O and cache behaviour can be reasoned about on its own.

Cache design is dictated by the deployment. ``resolve_database_path`` in
``main.py`` puts SQLite under ``tempfile.gettempdir()`` on serverless, so the
database is empty on every cold start and cannot be the primary store. The
primary store is therefore a module-level dict with a TTL; SQLite is a
write-through mirror used only when the file is actually writable, which is what
gives local development persistence across restarts.

The read path never blocks on the network when it has anything to serve, and it
never returns a 500 because upstream is down -- a monitoring page showing a
clearly-labelled stale reading is far more useful than an error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .hazard_math import spi
from .hazards import (
    ARCHIVE_URL,
    CLIMATOLOGY_LABEL,
    DATA_SOURCES,
    DISCHARGE_CLIMATOLOGY_LABEL,
    FLOOD_DAILY,
    FLOOD_URL,
    FORECAST_DAILY,
    FORECAST_DAYS,
    FORECAST_HOURLY,
    FORECAST_URL,
    GHANA_REGIONS,
    PAST_DAYS,
    band_for,
    dry_spell_length,
    score_drought,
    score_flood,
    spi_class,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
CLIMATOLOGY_PATH = DATA_DIR / "hazard_climatology.json"

CACHE_TTL_SECONDS = 6 * 3600
UPSTREAM_TIMEOUT = 25.0
GHANA_TZ = timezone(timedelta(0))  # Africa/Accra is UTC year-round, no DST

# Number of trailing days of series data returned to the charts. The full 92-day
# past window is fetched for the statistics but sending all of it for sixteen
# regions makes the summary payload needlessly large.
SERIES_DAYS = 90

# ERA5 publishes with roughly a five-day lag; step back further so a partially
# filled final day never truncates the accumulation.
ARCHIVE_LAG_DAYS = 7
ACCUMULATION_DAYS = 90

_CACHE: dict[str, dict] = {}
_CACHE_STAMP: float = 0.0
_REFRESH_LOCK = asyncio.Lock()
_LAST_ERROR: str | None = None


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------

def _load_climatology() -> dict:
    if not CLIMATOLOGY_PATH.exists():
        logger.warning(
            "hazard climatology asset missing at %s -- indices that need a baseline "
            "(SPI, discharge percentile) will report as unavailable until "
            "'python -m backend.scripts.build_hazard_climatology' has been run.",
            CLIMATOLOGY_PATH,
        )
        return {"regions": {}, "baseline": None}
    try:
        return json.loads(CLIMATOLOGY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("hazard climatology asset is unreadable")
        return {"regions": {}, "baseline": None}


CLIMATOLOGY = _load_climatology()


def has_climatology() -> bool:
    return bool(CLIMATOLOGY.get("regions"))


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------

async def _fetch_all() -> tuple[list[dict], list[dict], list[dict]]:
    """Three multi-coordinate requests covering all sixteen regions.

    The third one -- the ERA5 archive -- exists because of a measured bias, and
    removing it would silently break every drought reading:

    Over an identical 84-day window in August 2026, the forecast API's
    ``past_days`` rainfall came to 45% of the ERA5 archive total for Western,
    41% for Greater Accra and 71% for Northern. The forecast API serves a
    best-match model, not ERA5. Feeding those totals into a gamma fitted on
    ERA5 put Western at SPI -3.09 ("extreme drought") during a season that was
    in fact wetter than normal.

    So both halves of the SPI ratio must come from the same model. The
    accumulation is taken from ERA5, which lags about five days -- irrelevant
    for a 90-day sum. The forecast API is still the right source for everything
    forward-looking and for anything needing today's value.
    """
    regions = list(GHANA_REGIONS.values())

    forecast_params = {
        "latitude": ",".join(f"{r.lat:.4f}" for r in regions),
        "longitude": ",".join(f"{r.lon:.4f}" for r in regions),
        "daily": ",".join(FORECAST_DAILY),
        "hourly": ",".join(FORECAST_HOURLY),
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "Africa/Accra",
    }
    flood_params = {
        "latitude": ",".join(f"{r.river_lat:.4f}" for r in regions),
        "longitude": ",".join(f"{r.river_lon:.4f}" for r in regions),
        "daily": ",".join(FLOOD_DAILY),
        "past_days": 30,
        "forecast_days": FORECAST_DAYS,
    }

    # ERA5 lags a few days, so ask for a window that ends before the lag and is
    # long enough to still contain a full 90-day accumulation.
    archive_end = datetime.now(GHANA_TZ).date() - timedelta(days=ARCHIVE_LAG_DAYS)
    archive_params = {
        "latitude": forecast_params["latitude"],
        "longitude": forecast_params["longitude"],
        "start_date": (archive_end - timedelta(days=ACCUMULATION_DAYS - 1)).isoformat(),
        "end_date": archive_end.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "Africa/Accra",
    }

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        weather_response, flood_response, archive_response = await asyncio.gather(
            client.get(FORECAST_URL, params=forecast_params),
            client.get(FLOOD_URL, params=flood_params),
            client.get(ARCHIVE_URL, params=archive_params),
            return_exceptions=True,
        )

    def unpack(response, label: str, required: bool):
        if isinstance(response, Exception):
            if required:
                raise response
            logger.warning("hazard upstream %s unavailable: %s", label, response)
            return []
        try:
            response.raise_for_status()
        except Exception:
            if required:
                raise
            logger.warning("hazard upstream %s returned %s", label, response.status_code)
            return []
        payload = response.json()
        # A single-coordinate request returns an object; a multi-coordinate one
        # returns an array. We always send sixteen, but normalise defensively.
        return [payload] if isinstance(payload, dict) else payload

    # The forecast call is the only hard requirement. Losing the flood or
    # archive call degrades individual indices to "unavailable" rather than
    # taking the whole page down.
    weather = unpack(weather_response, "forecast", required=True)
    flood = unpack(flood_response, "flood", required=False)
    archive = unpack(archive_response, "archive", required=False)
    return weather, flood, archive


def _daily_mean_from_hourly(hourly: dict, keys: tuple[str, ...], day_count: int) -> list[float | None]:
    """Collapse hourly soil-moisture layers into one daily mean series."""
    series: list[list[float]] = [[] for _ in range(day_count)]
    times = hourly.get("time") or []
    layers = [hourly.get(key) or [] for key in keys]
    for index in range(len(times)):
        values = [layer[index] for layer in layers if index < len(layer) and layer[index] is not None]
        if not values:
            continue
        day = index // 24
        if day < day_count:
            series[day].append(sum(values) / len(values))
    return [sum(v) / len(v) if v else None for v in series]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _build_region(name: str, weather: dict, flood: dict, archive: dict, today: str) -> dict:
    region = GHANA_REGIONS[name]
    climate = CLIMATOLOGY.get("regions", {}).get(name)

    daily = weather.get("daily") or {}
    dates: list[str] = daily.get("time") or []
    rain: list[float | None] = daily.get("precipitation_sum") or []
    et0: list[float | None] = daily.get("et0_fao_evapotranspiration") or []
    soil = _daily_mean_from_hourly(weather.get("hourly") or {}, FORECAST_HOURLY, len(dates))

    # The response is past days followed by forecast days; the boundary is today.
    split = dates.index(today) if today in dates else max(0, len(dates) - FORECAST_DAYS)

    past_rain = [v for v in rain[:split] if v is not None]
    future_rain = [v for v in rain[split:] if v is not None]
    past_soil = [v for v in soil[:split] if v is not None]

    # --- drought ------------------------------------------------------------
    # The accumulation MUST come from the same model as the climatology it is
    # compared against. See the note on _fetch_all: using the forecast API here
    # understated rainfall by up to 59% and manufactured extreme droughts.
    archive_rain = [v for v in ((archive.get("daily") or {}).get("precipitation_sum") or []) if v is not None]
    spi_value = None
    accumulation = sum(archive_rain[-ACCUMULATION_DAYS:]) if len(archive_rain) >= 60 else None
    month_key = today[5:7]
    month_stats = (climate or {}).get("months", {}).get(month_key)
    if accumulation is not None and month_stats:
        spi_value = spi(
            accumulation,
            month_stats["shape"],
            month_stats["scale"],
            month_stats["zeroProbability"],
        )

    forward_et0 = sum(v for v in et0[split:split + 7] if v is not None)
    forward_rain = sum(future_rain[:7])
    drought_score, drought_drivers = score_drought(
        spi_value=spi_value,
        soil_moisture_now=past_soil[-1] if past_soil else None,
        soil_moisture_window=past_soil[-90:] or None,
        dry_spell_days=dry_spell_length(rain[:split]) if split else None,
        water_deficit_7d=max(0.0, forward_et0 - forward_rain),
    )

    # --- flood --------------------------------------------------------------
    flood_daily = flood.get("daily") or {}
    discharge_dates: list[str] = flood_daily.get("time") or []
    discharge: list[float | None] = flood_daily.get("river_discharge") or []
    discharge_split = (
        discharge_dates.index(today) if today in discharge_dates
        else max(0, len(discharge_dates) - FORECAST_DAYS)
    )
    forward_discharge = [v for v in discharge[discharge_split:] if v is not None]
    peak_discharge = max(forward_discharge) if forward_discharge else None

    ladder = ((climate or {}).get("discharge") or {}).get("ladder") or None
    flood_score, flood_drivers, riverine = score_flood(
        discharge_now=peak_discharge,
        discharge_reference=ladder,
        rain_next_7d=sum(future_rain[:7]) if future_rain else None,
        rain_7d_p95=(month_stats or {}).get("rain7dP95Mm"),
        rain_max_1d=max(future_rain[:7]) if future_rain else None,
        rain_day_p95=(month_stats or {}).get("rainDayP95Mm"),
        soil_moisture_now=past_soil[-1] if past_soil else None,
        soil_moisture_90d_max=max(past_soil[-90:]) if past_soil else None,
    )

    def trim(values: list, count: int = SERIES_DAYS) -> list:
        return values[max(0, split - count):split + FORECAST_DAYS]

    return {
        "region": name,
        "agroZone": region.agro_zone,
        "centroid": [region.lat, region.lon],
        "riverPoint": [region.river_lat, region.river_lon],
        "riverine": riverine,
        "flood": {
            "score": round(flood_score, 1),
            "band": band_for(flood_score),
            "drivers": [d.as_dict() for d in flood_drivers],
        },
        "drought": {
            "score": round(drought_score, 1),
            "band": band_for(drought_score),
            "spi": None if spi_value is None else round(spi_value, 2),
            "spiClass": spi_class(spi_value),
            "precip90dMm": None if accumulation is None else round(accumulation, 1),
            "precip90dNormalMm": (month_stats or {}).get("normalMm"),
            "drivers": [d.as_dict() for d in drought_drivers],
        },
        "dominant": "flood" if flood_score >= drought_score else "drought",
        # Two rainfall series, kept separate on purpose rather than spliced
        # into one line. `observed` is ERA5 and is the series the 90-day total
        # and the SPI are actually computed from, so the chart and the headline
        # figure can never disagree. `forecast` is the forward week from the
        # forecast model. They come from different models and are drawn as
        # different marks; pretending otherwise is how the SPI bug happened.
        "series": {
            "observed": {
                "source": "ERA5",
                "dates": (archive.get("daily") or {}).get("time") or [],
                "precipitation": (archive.get("daily") or {}).get("precipitation_sum") or [],
                "normalMm": (month_stats or {}).get("normalMm"),
            },
            "forecast": {
                "source": "Open-Meteo forecast",
                "dates": dates[split:],
                "precipitation": rain[split:],
                "et0": et0[split:],
            },
            "soilMoisture": {"dates": trim(dates), "values": trim(soil)},
        },
        "discharge": {
            "dates": discharge_dates,
            "values": discharge,
            "forecastFrom": (
                discharge_dates[discharge_split] if discharge_split < len(discharge_dates) else None
            ),
            "median": ((climate or {}).get("discharge") or {}).get("median"),
            "p90": ((climate or {}).get("discharge") or {}).get("p90"),
            "p95": ((climate or {}).get("discharge") or {}).get("p95"),
        },
    }


async def compute_snapshot() -> dict[str, dict]:
    """Fetch upstream and score every region. Raises if upstream is unusable."""
    weather, flood, archive = await _fetch_all()
    today = datetime.now(GHANA_TZ).date().isoformat()

    snapshot: dict[str, dict] = {}
    for index, name in enumerate(GHANA_REGIONS):
        if index >= len(weather):
            break
        try:
            snapshot[name] = _build_region(
                name,
                weather[index],
                flood[index] if index < len(flood) else {},
                archive[index] if index < len(archive) else {},
                today,
            )
        except Exception:
            # One malformed region must not lose the other fifteen.
            logger.exception("failed to score region %s", name)
    if not snapshot:
        raise RuntimeError("upstream returned no usable regions")
    return snapshot


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_age_seconds() -> float | None:
    return None if not _CACHE else time.time() - _CACHE_STAMP


def is_stale() -> bool:
    age = cache_age_seconds()
    return age is None or age > CACHE_TTL_SECONDS


def cached_snapshot() -> tuple[dict[str, dict], float]:
    return _CACHE, _CACHE_STAMP


def last_error() -> str | None:
    return _LAST_ERROR


async def refresh(force: bool = False) -> bool:
    """Refresh the cache. Returns True when new data was stored.

    The lock stops a burst of concurrent page loads on a cold container from
    each firing their own pair of upstream requests.
    """
    global _CACHE, _CACHE_STAMP, _LAST_ERROR

    if not force and not is_stale():
        return False

    async with _REFRESH_LOCK:
        # Another coroutine may have refreshed while we waited for the lock.
        if not force and not is_stale():
            return False
        try:
            snapshot = await compute_snapshot()
        except Exception as exc:
            _LAST_ERROR = f"{type(exc).__name__}: {exc}"
            logger.warning("hazard refresh failed: %s", _LAST_ERROR)
            return False

        _CACHE = snapshot
        _CACHE_STAMP = time.time()
        _LAST_ERROR = None
        return True


async def ensure_fresh() -> None:
    """Refresh only when there is nothing to serve, or the data has expired.

    Called from the read path. When a usable cache already exists this returns
    immediately and the refresh is left to the background task, so a page load
    is never blocked behind a 25-second upstream timeout.
    """
    if not _CACHE:
        await refresh(force=True)


def metadata() -> dict:
    """Provenance block attached to every response."""
    stamp = _CACHE_STAMP and datetime.fromtimestamp(_CACHE_STAMP, tz=timezone.utc).isoformat()
    return {
        "computedAt": stamp or None,
        "stale": is_stale(),
        "baseline": CLIMATOLOGY.get("baseline") or CLIMATOLOGY_LABEL,
        "dischargeBaseline": CLIMATOLOGY.get("dischargeBaseline") or DISCHARGE_CLIMATOLOGY_LABEL,
        "hasClimatology": has_climatology(),
        "sources": DATA_SOURCES,
        "error": _LAST_ERROR,
    }
