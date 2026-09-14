"""The live figures the assistant answers from.

The app already serves everything a farmer's question needs -- the forecast for
their region, the flood and drought bands, the market table -- and until now the
assistant could see none of it. It was told the farmer's region only inside the
canned reply served when no provider key was configured. So a question like "is
the rain coming this week?" got a paragraph of seasonal generality, from a model
whose knowledge of Ghanaian rainfall is a year out of date at best.

This module builds the block that fixes that: a short, labelled, plain-text
summary of what is true for this farmer right now, handed to the model beside
their question. Three rules shaped it.

**Only what was asked for.** Every source costs latency, and on a serverless host
latency is the whole budget. A price question does not need the river discharge,
so the message is scanned for what it is about and only those sources are
touched. Nothing matches, and the farmer still gets their region and crops,
which costs nothing at all.

**Never block on a source.** Each lookup gets a couple of seconds and the whole
block gets four. Anything slower is dropped from the block, not waited for and
not raised: an answer without the market row beats no answer, and both beat the
platform killing the request.

**Figures with their provenance.** Each section says where it came from and when
it was fetched, because the prompt tells the model to quote the day a figure
belongs to, and it cannot do that from bare numbers.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from . import hazard_runtime, weather_runtime
from .database import get_connection
from .hazards import GHANA_REGIONS, resolve_region

logger = logging.getLogger(__name__)

#: Per-source budget. Open-Meteo answers in well under a second warm; this is
#: the ceiling on how long a cold cache may hold up a farmer's answer.
SOURCE_TIMEOUT_SECONDS = 2.5

#: Ceiling on the whole block, since the sources run concurrently.
BLOCK_TIMEOUT_SECONDS = 4.0

#: Days of forecast rendered. Seven is what the app shows, and what "this week"
#: means to the person asking.
FORECAST_DAYS = 7

#: Market rows rendered. Four keeps the block short when a farmer grows several
#: things and asks a broad price question.
MAX_MARKET_ROWS = 4


# --- intent -----------------------------------------------------------------
#
# Substring matching on a lowercased message, deliberately. It is crude, it is
# free, and it fails in the safe direction: a false positive costs one cached
# lookup, while the alternative -- a model call to classify the question before
# the model call that answers it -- doubles both the latency and the bill.

WEATHER_WORDS = (
    "rain", "rains", "raining", "weather", "forecast", "outlook", "temperature",
    "hot", "cold", "humid", "wind", "storm", "sun", "cloud", "dry", "wet",
    "plant", "planting", "sow", "sowing", "harvest", "spray", "spraying",
    "irrigate", "irrigation", "season", "week", "today", "tomorrow", "weekend",
)

HAZARD_WORDS = (
    "flood", "flooding", "flooded", "drought", "dry spell", "river", "overflow",
    "waterlog", "warning", "alert", "disaster", "erosion", "washed away",
)

MARKET_WORDS = (
    "price", "prices", "pricing", "sell", "selling", "sale", "market", "buyer",
    "buyers", "cedi", "cedis", "ghs", "ghc", "cost", "worth", "profit", "margin",
    "trader", "traders",
)


def detect_intents(message: str) -> set[str]:
    """Which sources this question needs."""
    text = (message or "").lower()
    intents = set()
    if any(word in text for word in WEATHER_WORDS):
        intents.add("weather")
    if any(word in text for word in HAZARD_WORDS):
        intents.add("hazard")
    if any(word in text for word in MARKET_WORDS):
        intents.add("market")
    return intents


# --- rendering --------------------------------------------------------------

#: WMO weather codes, in the words a person would use. Open-Meteo returns the
#: number, and a number tells the model nothing.
WEATHER_CODES = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}

#: What the rains do in each agro-ecological zone. The prompt tells the model to
#: think in Ghanaian seasons; this tells it which season the farmer is in, since
#: the north has one and the south has two.
ZONE_SEASONS = {
    "Coastal Savannah": "two rainy seasons, major rains around March to July and minor rains around September to November",
    "Rainforest": "two rainy seasons, major rains around March to July and minor rains around September to November",
    "Deciduous Forest": "two rainy seasons, major rains around March to July and minor rains around September to November",
    "Forest-Savannah Transition": "two rainy seasons, major rains around April to July and minor rains around September to October",
    "Guinea Savannah": "one rainy season, roughly May to October",
    "Sudan Savannah": "one rainy season, roughly June to September",
}


def _describe_code(code) -> str:
    try:
        return WEATHER_CODES.get(int(code), "")
    except (TypeError, ValueError):
        return ""


def _day_label(iso_date: str) -> str:
    try:
        return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%a %d %b")
    except (TypeError, ValueError):
        return iso_date


def _fetched_label(iso_timestamp: str | None) -> str | None:
    """"09:12 today", not "2026-08-31T09:12:03.481920+00:00".

    The model is asked to say how current a figure is, and it will repeat what
    it is given. A machine timestamp read back to a farmer is noise; the hour is
    the part that tells them whether this is fresh.
    """
    if not iso_timestamp:
        return None
    try:
        stamp = datetime.fromisoformat(iso_timestamp)
    except (TypeError, ValueError):
        return None

    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)

    now = datetime.now(timezone.utc)
    if stamp.date() == now.date():
        return f"{stamp:%H:%M} today"
    return f"{stamp:%H:%M} on {stamp:%a %d %b}"


def _region_point(region: str | None):
    if not region:
        return None
    resolved = resolve_region(region)
    return GHANA_REGIONS.get(resolved) if resolved else None


def render_farmer(
    region: str | None,
    district: str | None,
    crops: list[str] | None,
    town: str | None = None,
) -> str:
    """Who is asking, and when. Always present, even with no region set."""
    # Ghana keeps UTC+0 all year, so UTC is the farmer's wall clock.
    now = datetime.now(timezone.utc)
    lines = [f"Today: {now.strftime('%A %d %B %Y')}, {now.strftime('%H:%M')} local time."]

    where = town or district or region
    if where:
        # Named in the order a person would say it, and never promoted: a town
        # is written as a town, because the mobile app asks for a town and an
        # answer that says "your district" about it has invented a fact.
        if town and region:
            line = f"Farmer's area: {town}, {region} region"
        elif district and region:
            line = f"Farmer's area: {district} district, {region} region"
        elif region:
            line = f"Farmer's area: {region} region"
        else:
            line = f"Farmer's area: {where}"
        point = _region_point(region)
        if point:
            line += f". Agro-ecological zone: {point.agro_zone}"
            season = ZONE_SEASONS.get(point.agro_zone)
            if season:
                line += f" ({season})"
        lines.append(line + ".")
    else:
        lines.append(
            "Farmer's area: not set in the app. Ask which district they farm in before giving figures."
        )

    if crops:
        lines.append(f"Crops they grow: {', '.join(crops)}.")

    return "\n".join(lines)


def render_weather(bundle: dict | None, region: str, fetched_at: str | None) -> str | None:
    """Current conditions and the week, as lines a model can quote."""
    if not bundle:
        return None

    current = bundle.get("current") or {}
    daily = bundle.get("daily") or {}
    dates = daily.get("time") or []
    if not current and not dates:
        return None

    stamp = f", fetched {fetched_at}" if fetched_at else ""
    lines = [f"WEATHER for {region} region (Open-Meteo{stamp}):"]

    if current:
        parts = []
        description = _describe_code(current.get("weather_code"))
        if description:
            parts.append(description)
        if current.get("temperature_2m") is not None:
            parts.append(f"{current['temperature_2m']}C")
        if current.get("apparent_temperature") is not None:
            parts.append(f"feels {current['apparent_temperature']}C")
        if current.get("relative_humidity_2m") is not None:
            parts.append(f"humidity {current['relative_humidity_2m']}%")
        if current.get("wind_speed_10m") is not None:
            parts.append(f"wind {current['wind_speed_10m']} km/h")
        if current.get("precipitation") is not None:
            parts.append(f"rain in the last hour {current['precipitation']} mm")
        if parts:
            lines.append("  Now: " + ", ".join(parts) + ".")

    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    rain = daily.get("precipitation_sum") or []
    chance = daily.get("precipitation_probability_max") or []
    codes = daily.get("weather_code") or []

    for index, date in enumerate(dates[:FORECAST_DAYS]):
        bits = []
        low = lows[index] if index < len(lows) else None
        high = highs[index] if index < len(highs) else None
        if low is not None and high is not None:
            bits.append(f"{low}-{high}C")
        millimetres = rain[index] if index < len(rain) else None
        if millimetres is not None:
            bits.append(f"rain {millimetres} mm")
        probability = chance[index] if index < len(chance) else None
        if probability is not None:
            bits.append(f"{probability}% chance of rain")
        description = _describe_code(codes[index] if index < len(codes) else None)
        if description:
            bits.append(description)
        if bits:
            lines.append(f"  {_day_label(date)}: " + ", ".join(bits) + ".")

    return "\n".join(lines) if len(lines) > 1 else None


def render_hazard(payload: dict | None) -> str | None:
    """The flood and drought bands for one region, with their advisories.

    Advisories hang off each hazard block rather than off the region, and a
    block carrying `source: gmet-bulletin` is a forecaster's words rather than
    this app's arithmetic. That distinction is passed through verbatim: the
    prompt forbids softening a GMet advisory, and it can only obey that if it is
    told which lines came from GMet.
    """
    if not payload:
        return None

    flood = payload.get("flood") or {}
    drought = payload.get("drought") or {}
    if not flood.get("band") and not drought.get("band"):
        return None

    lines = [f"FLOOD AND DROUGHT for {payload.get('region', 'this region')} (AgroMet monitor):"]

    if flood.get("band"):
        lines.append(f"  Flood risk: {flood['band']}.")
    if drought.get("band"):
        detail = f"  Drought risk: {drought['band']}"
        if drought.get("precip90dMm") is not None and drought.get("precip90dNormalMm") is not None:
            detail += (
                f". Rain over the last 90 days {drought['precip90dMm']} mm"
                f" against a normal of {drought['precip90dNormalMm']} mm"
            )
        lines.append(detail + ".")

    for hazard, block in (("Flood", flood), ("Drought", drought)):
        official = block.get("source") == "gmet-bulletin"
        if official and block.get("headline"):
            lines.append(f"  {hazard}, GMet bulletin in force: {block['headline']}")
        for advisory in (block.get("advisories") or [])[:2]:
            if isinstance(advisory, str) and advisory.strip():
                origin = "GMet advisory" if official else "advice"
                lines.append(f"  {hazard} {origin}: {advisory.strip()}")

    return "\n".join(lines) if len(lines) > 1 else None


def render_market(rows: list[dict]) -> str | None:
    """Indicative prices, named as indicative.

    They are a reference table, not a live exchange feed, and an answer that
    implies otherwise is a figure a farmer could travel a day on.
    """
    if not rows:
        return None

    lines = ["MARKET PRICES (AgroMet reference table, indicative, not a live market feed):"]
    for row in rows[:MAX_MARKET_ROWS]:
        parts = []
        if row.get("price") is not None:
            parts.append(f"GHS {row['price']} per {row.get('unit', 'unit')}")
        if row.get("trend"):
            parts.append(f"trend {row['trend']}")
        if row.get("demand"):
            parts.append(f"demand {row['demand']}")
        if parts:
            lines.append(f"  {row.get('name') or row.get('slug')}: " + ", ".join(parts) + ".")

    return "\n".join(lines) if len(lines) > 1 else None


# --- sources ----------------------------------------------------------------


async def _weather_section(region: str | None) -> str | None:
    point = _region_point(region)
    if not point:
        return None

    key = weather_runtime.cache_key(point.lat, point.lon)
    try:
        # Bounded: a cold cache gets one short chance to fill, and whatever is
        # already there is used either way. `ensure_fresh` holds its lock through
        # an async context manager, so a cancelled wait releases it cleanly.
        await asyncio.wait_for(weather_runtime.ensure_fresh(point.lat, point.lon), SOURCE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.info("Chat context: weather lookup for %s exceeded its budget.", point.name)
    except Exception:
        logger.warning("Chat context: weather lookup for %s failed.", point.name, exc_info=True)

    bundle = weather_runtime.cached_bundle(key)
    fetched_at = _fetched_label((weather_runtime.metadata(key) or {}).get("fetchedAt"))
    return render_weather(bundle, point.name, fetched_at)


async def _hazard_section(region: str | None) -> str | None:
    name = resolve_region(region) if region else None
    if not name:
        return None

    try:
        await asyncio.wait_for(hazard_runtime.ensure_fresh(), SOURCE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.info("Chat context: hazard snapshot exceeded its budget.")
    except Exception:
        logger.warning("Chat context: hazard snapshot failed.", exc_info=True)

    snapshot, _ = hazard_runtime.cached_snapshot()
    payload = (snapshot or {}).get(name)
    if not payload:
        return None

    # Imported here rather than at module scope: `routers/chat.py` imports this
    # module, so a top-level import back from here would close a loop. Worth
    # the lazy import to reuse the one implementation of override handling --
    # the advisories a farmer is shown in the app and the ones the assistant
    # quotes have to be the same lines, and a second copy of this logic here
    # is how they would drift apart.
    try:
        from .routers.hazards import _active_overrides, _apply_overrides

        with get_connection() as connection:
            payload = _apply_overrides(payload, _active_overrides(connection))
    except Exception:
        logger.warning("Chat context: could not apply hazard overrides.", exc_info=True)

    return render_hazard(payload)


def market_rows(message: str, crops: list[str] | None) -> list[dict]:
    """Rows for what the farmer named, else for what they grow.

    A synchronous SQLite read of a table with a few dozen rows: fast enough to
    run inline, and there is no upstream here to time out.
    """
    text = (message or "").lower()
    try:
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT slug, name, category, price, unit, trend, demand FROM commodities ORDER BY name"
            ).fetchall()
    except Exception:
        logger.warning("Chat context: could not read the commodities table.", exc_info=True)
        return []

    entries = [dict(row) for row in rows]
    named = [
        row for row in entries
        if (row.get("name") or "").lower() in text or (row.get("slug") or "").replace("-", " ") in text
    ]
    if named:
        return named

    wanted = {crop.lower() for crop in (crops or [])}
    return [row for row in entries if (row.get("name") or "").lower() in wanted or row.get("slug") in wanted]


async def _market_section(message: str, crops: list[str] | None) -> str | None:
    return render_market(market_rows(message, crops))


# --- assembly ---------------------------------------------------------------


async def build_context_block(
    message: str,
    *,
    region: str | None = None,
    district: str | None = None,
    town: str | None = None,
    crops: list[str] | None = None,
) -> str:
    """The DATA block for one question. Never raises, and never comes back empty.

    The farmer section always renders, so even an off-topic question is answered
    by something that knows what day it is and where the person is standing.
    """
    intents = detect_intents(message)
    sections: list[str] = [render_farmer(region, district, crops, town)]

    jobs = []
    if "weather" in intents:
        jobs.append(_weather_section(region))
    if "hazard" in intents:
        jobs.append(_hazard_section(region))
    if "market" in intents:
        jobs.append(_market_section(message, crops))

    if jobs:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*jobs, return_exceptions=True), BLOCK_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.warning("Chat context timed out; answering without live figures.")
            results = []

        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Chat context source failed: %s", result)
                continue
            if result:
                sections.append(result)

    return (
        "DATA. Live figures for this farmer, from the AgroMet app itself. "
        "Answer from these, and quote the day a figure belongs to. "
        "Do not invent any figure that is not here.\n" + "\n".join(sections)
    )
