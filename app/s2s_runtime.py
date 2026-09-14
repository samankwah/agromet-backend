"""The weeks 2-to-4 outlook, fetched, binned and cached.

Mirrors ``hazard_runtime``: a module-level dict with a TTL and an ``asyncio``
lock, refreshed off the read path, serving whatever it already has. See that
module's header for why the cache lives in the process rather than in SQLite --
the same serverless cold-start reasoning applies.

**Why GEFS and not ECMWF.** The product this is modelled on is ECMWF's
subseasonal ensemble, but that data is licensed and its free archive lags about
three weeks, which cannot drive a forward outlook. Open-Meteo does expose
``ecmwf_ifs025``, and it returns 51 members -- but every value past day 14 is
null, because that is where the operational ensemble stops. NOAA's GEFS 0.5°
extended run carries 31 members to about day 33, which covers days 14 to 28 with
room to spare. So the method here is the ECMWF one (an ensemble binned into
climatological terciles) with GEFS underneath.

**What the resolution costs.** GEFS is 0.5 degrees beyond day 10, roughly 55 km.
This used to sample sixteen region centroids, which is what made a district view
dishonest -- a district map drawn from sixteen points invents everything between
them. It now fetches the model's own lattice over Ghana (165 points, see
``ghana_grid_points``), so the field is real everywhere it is drawn. The map is
blocky at 55 km and neighbouring districts inside one cell read identically; that
is the truth about the resolution, and the client says so rather than implying
otherwise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from .s2s import (
    LEAD_DAYS,
    WINDOW_DAYS,
    agreement_confidence,
    cell_id,
    dominant_category,
    ensemble_percentile,
    ghana_grid_points,
    nearest_cell,
    tercile_probabilities,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
CLIMATOLOGY_PATH = DATA_DIR / "s2s_climatology.json"

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_MODEL = "gfs05"
ENSEMBLE_DAILY = ("precipitation_sum", "temperature_2m_max")

# One day past the end of the window, so a short run is detected as missing data
# rather than silently truncating the sum.
FORECAST_DAYS = LEAD_DAYS + WINDOW_DAYS + 1

# GEFS's extended run publishes once a day. Six hours matches `hazard_runtime`
# and keeps a cold container from re-fetching on every request.
CACHE_TTL_SECONDS = 6 * 3600
UPSTREAM_TIMEOUT = 30.0
GHANA_TZ = timezone(timedelta(0))  # Africa/Accra is UTC year-round, no DST

DATA_SOURCES = [
    {
        "id": "gefs",
        "label": "NOAA GEFS",
        "detail": "Global Ensemble Forecast System, 0.5 degree, 31 members",
        "url": "https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs.php",
    },
    {
        "id": "era5",
        "label": "ERA5",
        "detail": "ECMWF reanalysis, 1995-2024 tercile baseline",
        "url": "https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5",
    },
]

# The spread band on the detail chart. Deliberately not the full envelope: one
# stray member would set the band's width and make every cell look equally
# uncertain.
SPREAD_LOW_Q = 0.10
SPREAD_HIGH_Q = 0.90

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
            "s2s climatology asset missing at %s -- the subseasonal outlook will "
            "report as unavailable until "
            "'python -m backend.scripts.build_s2s_climatology' has been run.",
            CLIMATOLOGY_PATH,
        )
        return {"cells": {}, "baseline": None}
    try:
        return json.loads(CLIMATOLOGY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("s2s climatology asset is unreadable")
        return {"cells": {}, "baseline": None}


CLIMATOLOGY = _load_climatology()


def has_climatology() -> bool:
    return bool(CLIMATOLOGY.get("cells"))


def _week_baseline(key: str, issued: date) -> dict | None:
    """The tercile boundaries for this grid cell and this issue week."""
    weeks = ((CLIMATOLOGY.get("cells") or {}).get(key) or {}).get("weeks") or {}
    return weeks.get(f"{issued.isocalendar()[1]:02d}")


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------

async def _fetch_ensemble() -> list[dict]:
    """One request for the model's whole field over Ghana.

    Open-Meteo accepts a comma-separated coordinate list on the ensemble
    endpoint, so 165 points with 31 members each is a single call of about eight
    seconds -- the same trick `hazard_runtime` uses for sixteen.
    """
    points = ghana_grid_points()
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
        "longitude": ",".join(f"{lng:.4f}" for _, lng in points),
        "daily": ",".join(ENSEMBLE_DAILY),
        "models": ENSEMBLE_MODEL,
        "forecast_days": FORECAST_DAYS,
        "timezone": "Africa/Accra",
    }

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        response = await client.get(ENSEMBLE_URL, params=params)
    response.raise_for_status()

    payload = response.json()
    # A single-coordinate request returns an object, a multi-coordinate one an
    # array. We always send 165; normalise defensively.
    return [payload] if isinstance(payload, dict) else payload


def _member_series(daily: dict, variable: str) -> list[list[float | None]]:
    """Every ensemble member's series for one variable.

    Open-Meteo names the control run ``precipitation_sum`` and the perturbed
    members ``precipitation_sum_member01`` upward, so the control is picked up by
    the same prefix match. It is a legitimate member of the distribution and is
    counted as one.
    """
    keys = sorted(key for key in daily if key == variable or key.startswith(f"{variable}_member"))
    return [daily[key] for key in keys]


def _reduce_window(series: list[float | None], *, mean: bool) -> float | None:
    """One member's weeks 2-to-4 value, or None if the run is short."""
    window = series[LEAD_DAYS : LEAD_DAYS + WINDOW_DAYS]
    if len(window) < WINDOW_DAYS:
        return None

    if mean:
        # A gap in temperature cannot be zero-filled without inventing a cold
        # day, so an incomplete member is dropped from the distribution instead.
        if any(value is None for value in window):
            return None
        return sum(float(value) for value in window) / len(window)  # type: ignore[arg-type]

    # Rainfall: a missing day is genuinely no recorded rain for that step.
    return sum(0.0 if value is None else float(value) for value in window)


def _daily_spread(members: list[list[float | None]], *, mean: bool) -> dict | None:
    """The window day by day: ensemble mean inside a 10th-to-90th band.

    Computed here rather than on request because the members are in hand exactly
    once, and refetching one cell's ensemble to draw a chart would be an upstream
    call per tap. Ninety numbers per cell is nothing to keep; the raw members
    would be a hundred times that.
    """
    columns: list[list[float]] = []
    for offset in range(WINDOW_DAYS):
        day: list[float] = []
        for series in members:
            index = LEAD_DAYS + offset
            if index < len(series) and series[index] is not None:
                day.append(float(series[index]))
        if not day:
            return None
        columns.append(day)

    return {
        "mean": [round(sum(day) / len(day), 2) for day in columns],
        "low": [round(ensemble_percentile(day, SPREAD_LOW_Q), 2) for day in columns],
        "high": [round(ensemble_percentile(day, SPREAD_HIGH_Q), 2) for day in columns],
    }


def _build_variable(
    daily: dict,
    variable: str,
    p33: float | None,
    p67: float | None,
    normal: float | None,
    *,
    mean: bool,
) -> dict | None:
    members = _member_series(daily, variable)
    reduced = [_reduce_window(series, mean=mean) for series in members]
    usable = [value for value in reduced if value is not None]
    if not usable:
        return None

    # The deterministic view is the ensemble mean of the window: a rainfall total
    # in mm, or a mean daily maximum in degrees. It needs no baseline, so it
    # still works for a cell whose climatology failed to bake.
    built: dict = {
        "value": round(sum(usable) / len(usable), 2),
        "members": len(usable),
        "normal": normal,
        "series": _daily_spread(members, mean=mean),
    }

    probabilities = None if p33 is None or p67 is None else tercile_probabilities(reduced, p33, p67)
    if probabilities is not None:
        built.update({
            "probabilities": {
                "below": round(probabilities["below"], 3),
                "normal": round(probabilities["normal"], 3),
                "above": round(probabilities["above"], 3),
            },
            "category": dominant_category(probabilities),
            "confidence": agreement_confidence(probabilities),
            # True when the baseline's two boundaries coincide, which happens in
            # a dry season that is mostly zeros. The split is then arithmetically
            # fine and means nothing; the client shows it as "no signal".
            "noSignal": probabilities["degenerate"],
        })
    return built


def _build_cell(lat: float, lng: float, entry: dict, issued: date) -> dict | None:
    """One grid cell: both variables, probabilistic and deterministic.

    A missing baseline no longer loses the cell. The deterministic field is an
    ensemble mean and needs no climatology, so a cell whose bake failed still
    draws on the deterministic map and simply carries no probabilities.
    """
    baseline = _week_baseline(cell_id(lat, lng), issued) or {}
    daily = entry.get("daily") or {}

    rainfall = _build_variable(
        daily, "precipitation_sum",
        baseline.get("rainP33Mm"), baseline.get("rainP67Mm"), baseline.get("rainNormalMm"),
        mean=False,
    )
    temperature = _build_variable(
        daily, "temperature_2m_max",
        baseline.get("tempP33C"), baseline.get("tempP67C"), baseline.get("tempNormalC"),
        mean=True,
    )
    if rainfall is None and temperature is None:
        return None

    return {
        "id": cell_id(lat, lng),
        "lat": lat,
        "lng": lng,
        "rainfall": rainfall,
        "temperature": temperature,
    }


async def compute_snapshot() -> dict[str, dict]:
    """Fetch the field and reduce every cell. Raises if upstream is unusable."""
    entries = await _fetch_ensemble()
    issued = datetime.now(GHANA_TZ).date()
    points = ghana_grid_points()

    snapshot: dict[str, dict] = {}
    for index, (lat, lng) in enumerate(points):
        if index >= len(entries):
            break
        try:
            cell = _build_cell(lat, lng, entries[index], issued)
            if cell:
                snapshot[cell["id"]] = cell
        except Exception:
            # One malformed cell must not lose the other hundred and sixty-four.
            logger.exception("failed to reduce cell %s", cell_id(lat, lng))

    if not snapshot:
        raise RuntimeError("upstream returned no usable cells")
    return snapshot


def cell_at(lat: float, lng: float) -> dict | None:
    """The cell covering a place, for the detail chart and a reader's own card."""
    snapshot, _ = cached_snapshot()
    if not snapshot:
        return None
    point = nearest_cell(lat, lng, ghana_grid_points())
    return None if point is None else snapshot.get(cell_id(*point))


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
            snapshot = await compute_snapshot()
        except Exception as exc:
            _LAST_ERROR = f"{type(exc).__name__}: {exc}"
            logger.warning("s2s refresh failed: %s", _LAST_ERROR)
            return False

        _CACHE = snapshot
        _CACHE_STAMP = time.time()
        _LAST_ERROR = None
        return True


async def ensure_fresh() -> None:
    """Refresh only when there is nothing to serve at all.

    An expired cache is still served, and revalidated behind the response, so a
    page load never waits on a 30-second upstream timeout.
    """
    if not _CACHE:
        await refresh(force=True)


def window_dates(issued: date | None = None) -> tuple[str, str]:
    """The calendar days the outlook actually covers."""
    start = (issued or datetime.now(GHANA_TZ).date()) + timedelta(days=LEAD_DAYS)
    return start.isoformat(), (start + timedelta(days=WINDOW_DAYS - 1)).isoformat()


def metadata() -> dict:
    """Provenance block attached to every response."""
    stamp = _CACHE_STAMP and datetime.fromtimestamp(_CACHE_STAMP, tz=timezone.utc).isoformat()
    window_start, window_end = window_dates()
    return {
        "issuedAt": stamp or None,
        "stale": is_stale(),
        "windowStart": window_start,
        "windowEnd": window_end,
        "leadDays": LEAD_DAYS,
        "windowDays": WINDOW_DAYS,
        "model": "NOAA GEFS 0.5 degree",
        "baseline": CLIMATOLOGY.get("baseline"),
        "hasClimatology": has_climatology(),
        "sources": DATA_SOURCES,
        "error": _LAST_ERROR,
    }


def reset_cache() -> None:
    """Drop everything. For tests only."""
    global _CACHE, _CACHE_STAMP, _LAST_ERROR
    _CACHE = {}
    _CACHE_STAMP = 0.0
    _LAST_ERROR = None
