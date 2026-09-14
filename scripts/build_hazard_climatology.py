"""Bake the 30-year flood/drought baseline into a committed JSON asset.

Run this manually, not on a request path:

    python -m backend.scripts.build_hazard_climatology
    python -m backend.scripts.build_hazard_climatology --rebake   # ignore cache

Why a committed file rather than a table we populate at runtime:
``resolve_database_path`` in ``backend/app/main.py`` puts the SQLite file under
``tempfile.gettempdir()`` when running serverless, so the database is empty on
every cold start. A climatology is static by definition -- thirty years of ERA5
do not change between deploys -- so baking it once and shipping the result means
production never calls the archive API at all.

Output: ``backend/app/data/hazard_climatology.json``.
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

from backend.app.hazard_math import fit_gamma  # noqa: E402
from backend.app.hazards import (  # noqa: E402
    ARCHIVE_URL,
    CLIMATOLOGY_END,
    CLIMATOLOGY_LABEL,
    CLIMATOLOGY_START,
    DISCHARGE_CLIMATOLOGY_LABEL,
    FLOOD_URL,
    GHANA_REGIONS,
)

OUTPUT_PATH = BACKEND_ROOT / "app" / "data" / "hazard_climatology.json"
CACHE_PATH = BACKEND_ROOT / "app" / "data" / ".climatology_cache.json"

ACCUMULATION_DAYS = 90
DISCHARGE_LADDER_STEPS = 101  # 0th..100th percentile inclusive


def request_json(url: str, params: dict, *, attempts: int = 14) -> dict | list:
    """GET with exponential backoff.

    Open-Meteo enforces hourly and daily quotas, and a 30-year daily series is
    an expensive call. A 429 here usually means an hourly bucket needs to
    refill, so the backoff has to be patient in minutes, not seconds -- hence
    the 15-minute ceiling. The caller checkpoints after every region, so an
    eventual give-up costs nothing but a re-run.
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


def rolling_totals(values: list[float | None], window: int) -> list[tuple[date, float]]:
    """Every complete ``window``-day total, tagged with its end date."""
    start = date.fromisoformat(CLIMATOLOGY_START)
    cleaned = [0.0 if v is None else float(v) for v in values]
    if len(cleaned) < window:
        return []

    totals: list[tuple[date, float]] = []
    running = sum(cleaned[:window])
    totals.append((start + timedelta(days=window - 1), running))
    for i in range(window, len(cleaned)):
        running += cleaned[i] - cleaned[i - window]
        totals.append((start + timedelta(days=i), running))
    return totals


def quantile_ladder(values: list[float], steps: int = DISCHARGE_LADDER_STEPS) -> list[float]:
    """Downsample a long series to an ascending ladder of quantiles.

    Storing 11,000 daily discharges per region would bloat the asset for no
    gain -- percentile lookups against a 101-point ladder are accurate to one
    percentile, which is all the UI ever displays.
    """
    ordered = sorted(values)
    if not ordered:
        return []
    last = len(ordered) - 1
    return [ordered[round(i / (steps - 1) * last)] for i in range(steps)]


def fetch_precipitation(region) -> list[float | None]:
    payload = request_json(ARCHIVE_URL, {
        "latitude": f"{region.lat:.4f}",
        "longitude": f"{region.lon:.4f}",
        "start_date": CLIMATOLOGY_START,
        "end_date": CLIMATOLOGY_END,
        "daily": "precipitation_sum",
        "timezone": "Africa/Accra",
    })
    return payload["daily"]["precipitation_sum"]


def fetch_discharge(region) -> list[float | None]:
    payload = request_json(FLOOD_URL, {
        "latitude": f"{region.river_lat:.4f}",
        "longitude": f"{region.river_lon:.4f}",
        "start_date": CLIMATOLOGY_START,
        "end_date": CLIMATOLOGY_END,
        "daily": "river_discharge",
    })
    return payload["daily"]["river_discharge"]


def build_region(name: str, region, raw: dict) -> dict:
    precipitation = raw["precipitation"]
    discharge = [v for v in raw["discharge"] if v is not None]

    # --- drought baseline: a gamma fit per calendar month -------------------
    # SPI compares a 90-day total against the distribution of the SAME 90-day
    # window in other years, because Ghana's rainfall is strongly seasonal --
    # comparing a January total against an August one would be meaningless.
    by_month: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for end_date, total in rolling_totals(precipitation, ACCUMULATION_DAYS):
        by_month[end_date.month].append(total)

    months: dict[str, dict] = {}
    for month in range(1, 13):
        totals = by_month[month]
        shape, scale, zero_probability = fit_gamma(totals)
        ordered = sorted(totals)
        months[f"{month:02d}"] = {
            "n": len(totals),
            "shape": round(shape, 6),
            "scale": round(scale, 6),
            "zeroProbability": round(zero_probability, 6),
            "normalMm": round(sum(totals) / len(totals), 2) if totals else 0.0,
            "medianMm": round(ordered[len(ordered) // 2], 2) if ordered else 0.0,
        }

    # --- flood baseline: 7-day rainfall p95 per calendar month --------------
    weekly_by_month: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for end_date, total in rolling_totals(precipitation, 7):
        weekly_by_month[end_date.month].append(total)
    for month in range(1, 13):
        weekly = sorted(weekly_by_month[month])
        p95 = weekly[int(len(weekly) * 0.95)] if weekly else 0.0
        months[f"{month:02d}"]["rain7dP95Mm"] = round(p95, 2)

    # --- flood baseline: what counts as a heavy DAY here ---------------------
    # Pluvial flooding is driven by intensity, not by monthly totals, and what
    # counts as intense is local: Greater Accra's 90-day normal is 270 mm
    # against Western's 660 mm, so the same 50 mm day is a far rarer event on
    # the coast than in the rainforest. Scoring the heaviest forecast day
    # against fixed national anchors flattened exactly that difference.
    #
    # Taken over WET days only. Including dry days would drag the percentile
    # toward zero in the dry season and make any rain at all look extreme.
    start = date.fromisoformat(CLIMATOLOGY_START)
    wet_by_month: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for index, value in enumerate(precipitation):
        if value is None or value < 1.0:
            continue
        wet_by_month[(start + timedelta(days=index)).month].append(float(value))
    for month in range(1, 13):
        wet = sorted(wet_by_month[month])
        months[f"{month:02d}"]["wetDayCount"] = len(wet)
        months[f"{month:02d}"]["rainDayP95Mm"] = round(wet[int(len(wet) * 0.95)], 2) if wet else 0.0

    ladder = quantile_ladder(discharge)
    return {
        "region": name,
        "agroZone": region.agro_zone,
        "centroid": [region.lat, region.lon],
        "riverPoint": [region.river_lat, region.river_lon],
        "months": months,
        "discharge": {
            "ladder": [round(v, 3) for v in ladder],
            "median": round(ladder[50], 3) if ladder else 0.0,
            "p90": round(ladder[90], 3) if ladder else 0.0,
            "p95": round(ladder[95], 3) if ladder else 0.0,
            "n": len(discharge),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebake", action="store_true", help="ignore the raw-download cache")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # The raw download is the expensive part (32 multi-decade requests), so it
    # is cached separately from the fit. Re-running to tweak the statistics
    # then costs nothing and hits no rate limit.
    cache: dict = {}
    if CACHE_PATH.exists() and not args.rebake:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        print(f"resuming with {len(cache)} region(s) already downloaded")

    for index, (name, region) in enumerate(GHANA_REGIONS.items(), start=1):
        if name in cache:
            continue
        print(f"[{index:2d}/16] {name}", flush=True)
        print("        precipitation...", flush=True)
        precipitation = fetch_precipitation(region)
        time.sleep(2.0)
        print("        discharge...", flush=True)
        discharge = fetch_discharge(region)
        cache[name] = {"precipitation": precipitation, "discharge": discharge}
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
        time.sleep(2.0)

    print("\nfitting...")
    regions = {name: build_region(name, region, cache[name]) for name, region in GHANA_REGIONS.items()}

    OUTPUT_PATH.write_text(json.dumps({
        "baseline": CLIMATOLOGY_LABEL,
        "dischargeBaseline": DISCHARGE_CLIMATOLOGY_LABEL,
        "start": CLIMATOLOGY_START,
        "end": CLIMATOLOGY_END,
        "accumulationDays": ACCUMULATION_DAYS,
        "builtAt": date.today().isoformat(),
        "regions": regions,
    }, indent=1), encoding="utf-8")

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nwrote {OUTPUT_PATH} ({size_kb:.0f} KB)")

    print("\nsanity check -- median discharge by region (expect the Volta basin high):")
    for name in sorted(regions, key=lambda r: -regions[r]["discharge"]["median"]):
        d = regions[name]["discharge"]
        aug = regions[name]["months"]["08"]
        riverine = "riverine" if d["p95"] >= 100 else "pluvial "
        print(f"  {name:15s} median {d['median']:9.2f} m3/s  p95 {d['p95']:9.2f}  {riverine}"
              f"  Aug 90d normal {aug['normalMm']:7.1f} mm  shape {aug['shape']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
