"""Bake the weeks 2-to-4 tercile baseline into a committed JSON asset.

Run this manually, not on a request path:

    python -m backend.scripts.build_s2s_climatology
    python -m backend.scripts.build_s2s_climatology --rebake   # ignore cache

Why a committed file rather than a table we populate at runtime: the same reason
``build_hazard_climatology`` gives. ``resolve_database_path`` in
``backend/app/main.py`` puts the SQLite file under ``tempfile.gettempdir()`` when
running serverless, so the database is empty on every cold start. Thirty years of
ERA5 do not change between deploys, so baking once means production never calls
the archive API at all.

**What a tercile baseline is for.** A subseasonal forecast is not "35 mm of rain".
It is "more likely than not to be in the wettest third of what this place
normally gets at this time of year". That requires knowing what the thirds *are*,
which is what this file computes: for every region and every ISO week, the 33rd
and 67th percentiles of the weeks 2-to-4 rainfall total across 1995-2024. The
runtime then counts how many ensemble members land in each third.

Bucketing by ISO week rather than pooling the year is not optional. Ghana's
rainfall is strongly seasonal -- comparing an August total against a January one
would make every August look catastrophically wet.

**Per grid cell, not per region.** The baseline has to be computed wherever the
forecast is, and the forecast is now the model's own 0.5 degree field -- 165
points over Ghana, from ``ghana_grid_points``. A region-level baseline applied to
a gridded forecast would put region-shaped steps into the thresholds while the
forecast varied smoothly across them.

That is 165 multi-decade downloads if done one at a time. The archive endpoint
takes a comma-separated coordinate list, so they go in chunks instead: five
coordinates and two variables per request, about thirty requests, checkpointed
after each.

Output: ``backend/app/data/s2s_climatology.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT.parent))

from backend.app.hazards import (  # noqa: E402
    ARCHIVE_URL,
    CLIMATOLOGY_END,
    CLIMATOLOGY_LABEL,
    CLIMATOLOGY_START,
)
from backend.app.s2s import (  # noqa: E402
    LEAD_DAYS,
    WINDOW_DAYS,
    cell_id,
    ghana_grid_points,
    quantile,
)

OUTPUT_PATH = BACKEND_ROOT / "app" / "data" / "s2s_climatology.json"

# The hazard bake's cache is keyed by region, so it cannot serve a per-cell
# baseline; this bake downloads its own and keeps it in the same shape, keyed by
# `cell_id`. Checkpointed after every chunk so an interrupted run resumes rather
# than restarting thirty requests.
CACHE_PATH = BACKEND_ROOT / "app" / "data" / ".s2s_grid_cache.json"

# Five coordinates and two variables over thirty years is about 18 seconds and
# roughly 110k numbers. Larger chunks start timing out; smaller ones waste the
# per-request overhead.
CHUNK_SIZE = 5


def request_json(url: str, params: dict, *, attempts: int = 14) -> dict | list:
    """GET with exponential backoff.

    Lifted from ``build_hazard_climatology`` for the same reason it exists there:
    Open-Meteo enforces hourly quotas and a 30-year daily series is an expensive
    call, so a 429 means waiting minutes rather than seconds. The caller
    checkpoints after every region, so giving up costs only a re-run.
    """
    query = urllib.parse.urlencode(params)
    delay = 15.0
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{url}?{query}", timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            print(f"      429, backing off {delay:.0f}s", flush=True)
        except Exception:
            if attempt == attempts - 1:
                raise
            print(f"      transient error, retrying in {delay:.0f}s", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 900.0)
    raise RuntimeError("unreachable")


def fetch_chunk(points: list[tuple[float, float]]) -> list[dict]:
    """Thirty years of daily rainfall and maximum temperature for a few cells."""
    payload = request_json(ARCHIVE_URL, {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
        "longitude": ",".join(f"{lng:.4f}" for _, lng in points),
        "start_date": CLIMATOLOGY_START,
        "end_date": CLIMATOLOGY_END,
        "daily": "precipitation_sum,temperature_2m_max",
        "timezone": "Africa/Accra",
    })
    # A single-coordinate request returns an object, a multi-coordinate one an
    # array. Normalise so a one-cell final chunk behaves like the others.
    return [payload] if isinstance(payload, dict) else payload


def windows_by_iso_week(
    values: list[float | None],
    *,
    reduce_mean: bool,
) -> dict[int, list[float]]:
    """Every weeks 2-to-4 window in the record, bucketed by the ISO week it was issued in.

    Indexed from the issue date, not the window: at runtime we know today's date
    and need the distribution for a forecast made *today*, so the bucket key has
    to be the issue date's ISO week. Roughly seven issue days times thirty years
    gives about 210 samples per bucket, which is a sound base for two
    percentiles.

    Missing days are treated as zero rainfall, matching ``rolling_totals`` in the
    hazard bake. For temperature that would be a lie, so a window containing a
    gap is dropped instead.
    """
    start = date.fromisoformat(CLIMATOLOGY_START)
    span = LEAD_DAYS + WINDOW_DAYS
    buckets: dict[int, list[float]] = {week: [] for week in range(1, 54)}

    for index in range(len(values) - span + 1):
        window = values[index + LEAD_DAYS : index + span]

        if reduce_mean:
            if any(value is None for value in window):
                continue
            reduced = sum(float(value) for value in window) / len(window)  # type: ignore[arg-type]
        else:
            reduced = sum(0.0 if value is None else float(value) for value in window)

        issued = start + timedelta(days=index)
        buckets[issued.isocalendar()[1]].append(reduced)

    return buckets


def build_cell(lat: float, lng: float, rainfall: list[float | None], temperature: list[float | None]) -> dict:
    rain_by_week = windows_by_iso_week(rainfall, reduce_mean=False)
    temp_by_week = windows_by_iso_week(temperature, reduce_mean=True)

    weeks: dict[str, dict] = {}
    for week in range(1, 54):
        rain = sorted(rain_by_week[week])
        temp = sorted(temp_by_week[week])
        if not rain:
            continue

        weeks[f"{week:02d}"] = {
            "n": len(rain),
            "rainP33Mm": round(quantile(rain, 1 / 3), 2),
            "rainP67Mm": round(quantile(rain, 2 / 3), 2),
            "rainNormalMm": round(sum(rain) / len(rain), 2),
            "tempP33C": round(quantile(temp, 1 / 3), 2) if temp else None,
            "tempP67C": round(quantile(temp, 2 / 3), 2) if temp else None,
            "tempNormalC": round(sum(temp) / len(temp), 2) if temp else None,
        }

    return {"lat": lat, "lng": lng, "weeks": weeks}


def publish(cache: dict, points: list[tuple[float, float]]) -> dict:
    """Bin everything downloaded so far and write it out.

    Called after every chunk rather than once at the end. The download is
    rate-limited to roughly a day of wall clock, and an all-or-nothing write
    meant a run that was 91% done served exactly nothing, while any interruption
    threw the published result away entirely. A partial baseline is genuinely
    useful: a cell without one carries no tercile split, and the map already
    draws only the cells that have one.
    """
    cells = {}
    for lat, lng in points:
        raw = cache.get(cell_id(lat, lng))
        if not raw or not raw.get("precipitation"):
            continue
        cells[cell_id(lat, lng)] = build_cell(lat, lng, raw["precipitation"], raw["temperature"])

    OUTPUT_PATH.write_text(json.dumps({
        "baseline": CLIMATOLOGY_LABEL,
        "start": CLIMATOLOGY_START,
        "end": CLIMATOLOGY_END,
        "leadDays": LEAD_DAYS,
        "windowDays": WINDOW_DAYS,
        "gridStepDeg": 0.5,
        "builtAt": date.today().isoformat(),
        "cells": cells,
    }, indent=1), encoding="utf-8")
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebake", action="store_true", help="ignore the download cache")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    points = ghana_grid_points()

    cache: dict = {}
    if CACHE_PATH.exists() and not args.rebake:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        print(f"resuming with {len(cache)}/{len(points)} cell(s) already downloaded")

    pending = [point for point in points if cell_id(*point) not in cache]
    chunks = [pending[i : i + CHUNK_SIZE] for i in range(0, len(pending), CHUNK_SIZE)]

    for index, chunk in enumerate(chunks, start=1):
        print(f"[{index:3d}/{len(chunks)}] {len(chunk)} cell(s)...", flush=True)
        for point, entry in zip(chunk, fetch_chunk(chunk)):
            daily = entry.get("daily") or {}
            cache[cell_id(*point)] = {
                "precipitation": daily.get("precipitation_sum") or [],
                "temperature": daily.get("temperature_2m_max") or [],
            }
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
        publish(cache, points)
        time.sleep(2.0)

    cells = publish(cache, points)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nwrote {OUTPUT_PATH} ({size_kb:.0f} KB) -- {len(cells)}/{len(points)} cells")

    # A north-south transect: the wet forest belt should dwarf the Sahel edge.
    this_week = f"{date.today().isocalendar()[1]:02d}"
    print(f"\nsanity check -- weeks 2-to-4 rainfall terciles at week {this_week}, along 1.0W:")
    for lat, lng in points:
        if lng != -1.0:
            continue
        week = (cells.get(cell_id(lat, lng)) or {}).get("weeks", {}).get(this_week)
        if week:
            print(f"  {lat:5.1f}N  {week['rainP33Mm']:6.1f} / {week['rainP67Mm']:6.1f} mm   (n={week['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
