"""Flood and drought index definitions for Ghana's sixteen regions.

Scope of this module: constants, the upstream request shapes, and the scoring
that turns raw meteorology into a band. It holds no database code and no FastAPI
handlers -- those live in ``main.py`` -- so the scoring is testable in isolation.

What the numbers actually are
-----------------------------
Flood and drought are scored on one shared 0-100 scale with one shared set of
bands, so a reader can compare a flood severity against a drought severity. Both
are built only from measurable quantities: river discharge, rainfall,
soil moisture and reference evapotranspiration, each ranked against a 30-year
baseline where a baseline exists.

There is deliberately **no vulnerability or exposure multiplier**. A per-region
fudge factor for "this district floods a lot" cannot be sourced or audited, and
inventing one would quietly corrupt every score on the page. Where local
knowledge needs to override the model -- Bagre spillage, a failed drain, a
NADMO evacuation -- that is what a published GMet bulletin is for, and the
override is shown to the reader as an override.

Known limits, restated verbatim in the UI methodology panel:
  * GloFAS is a ~5 km global hydrological model, not a gauge network. It
    indicates river-system stress, not street-level inundation.
  * Several southern regions have no major monitored reach; their flooding is
    pluvial. Those regions drop the discharge term entirely rather than pretend
    a headwater stream represents them -- see ``RIVERINE_MIN_MEDIAN_CMS``.
  * ERA5 is a reanalysis. It under-represents localised convective rainfall,
    which is the dominant flood-producing mechanism in northern Ghana.
  * SPI needs 30 years to be meaningful; we have exactly 30, which is the floor.
  * Nothing here models dam operations (Bagre, Kompienga, Akosombo), which
    drive much of the White Volta flood risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hazard_math import clamp, linear_score, percentile_of, piecewise

# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionPoint:
    """One region's sampling geometry.

    ``lat``/``lon`` is the mean of the region's district centroids taken from
    ``frontend/src/assets/ghana-regions.json`` -- used for the weather sample.

    ``river_lat``/``river_lon`` is a separate point on the region's dominant
    watercourse. It has to be separate: GloFAS sampled at an administrative
    centroid returns a trickle. These were found by ranking a 0.15-degree
    national grid of GloFAS cells by peak-season discharge and taking each
    region's maximum, rather than by hand-picking coordinates.
    """

    name: str
    lat: float
    lon: float
    agro_zone: str
    river_lat: float
    river_lon: float
    districts: int


GHANA_REGIONS: dict[str, RegionPoint] = {
    "Greater Accra": RegionPoint(
        name="Greater Accra", lat=5.6483, lon=-0.1106,
        agro_zone="Coastal Savannah", river_lat=5.7833, river_lon=0.6667, districts=28,
    ),
    "Central": RegionPoint(
        name="Central", lat=5.5194, lon=-1.1164,
        agro_zone="Coastal Savannah", river_lat=5.6333, river_lon=-1.5833, districts=23,
    ),
    "Western": RegionPoint(
        name="Western", lat=5.2300, lon=-2.0424,
        agro_zone="Rainforest", river_lat=5.1833, river_lon=-2.7833, districts=14,
    ),
    "Western North": RegionPoint(
        name="Western North", lat=6.2767, lon=-2.7854,
        agro_zone="Rainforest", river_lat=5.4833, river_lon=-2.6333, districts=9,
    ),
    "Ashanti": RegionPoint(
        name="Ashanti", lat=6.6924, lon=-1.5829,
        agro_zone="Deciduous Forest", river_lat=6.3833, river_lon=-1.7333, districts=41,
    ),
    "Eastern": RegionPoint(
        name="Eastern", lat=6.1993, lon=-0.4811,
        agro_zone="Deciduous Forest", river_lat=6.3833, river_lon=0.0667, districts=33,
    ),
    "Volta": RegionPoint(
        name="Volta", lat=6.3635, lon=0.6400,
        agro_zone="Coastal Savannah", river_lat=6.6833, river_lon=0.0667, districts=17,
    ),
    "Oti": RegionPoint(
        name="Oti", lat=7.7982, lon=0.2426,
        agro_zone="Guinea Savannah", river_lat=7.4333, river_lon=0.0667, districts=9,
    ),
    "Bono": RegionPoint(
        name="Bono", lat=7.5560, lon=-2.5870,
        agro_zone="Forest-Savannah Transition", river_lat=8.1833, river_lon=-2.1833, districts=12,
    ),
    "Bono East": RegionPoint(
        name="Bono East", lat=7.8133, lon=-1.3219,
        agro_zone="Forest-Savannah Transition", river_lat=8.1833, river_lon=-0.5333, districts=11,
    ),
    "Ahafo": RegionPoint(
        name="Ahafo", lat=6.9165, lon=-2.6037,
        agro_zone="Deciduous Forest", river_lat=6.8333, river_lon=-2.4833, districts=6,
    ),
    "Northern": RegionPoint(
        name="Northern", lat=9.3815, lon=-0.3704,
        agro_zone="Guinea Savannah", river_lat=9.8333, river_lon=-0.9833, districts=16,
    ),
    "Savannah": RegionPoint(
        name="Savannah", lat=9.2357, lon=-1.6141,
        agro_zone="Guinea Savannah", river_lat=8.6333, river_lon=-0.9833, districts=7,
    ),
    "North East": RegionPoint(
        name="North East", lat=10.3599, lon=-0.3950,
        agro_zone="Guinea Savannah", river_lat=10.1333, river_lon=-1.1333, districts=6,
    ),
    "Upper East": RegionPoint(
        name="Upper East", lat=10.8704, lon=-0.6832,
        agro_zone="Sudan Savannah", river_lat=10.4333, river_lon=-1.1333, districts=15,
    ),
    "Upper West": RegionPoint(
        name="Upper West", lat=10.4421, lon=-2.4896,
        agro_zone="Guinea Savannah", river_lat=9.8333, river_lon=-2.7833, districts=11,
    ),
}

REGION_NAMES: list[str] = list(GHANA_REGIONS)

# ``Brong-Ahafo`` was split into Bono, Bono East and Ahafo in 2019. Older parts
# of the frontend still carry the legacy key, so accept it on input and map it
# to the region that kept the former capital (Sunyani).
LEGACY_REGION_ALIASES: dict[str, str] = {
    "Brong-Ahafo": "Bono",
    "Brong Ahafo": "Bono",
}


def resolve_region(name: str | None) -> str | None:
    """Normalise a caller-supplied region name, tolerating case and the legacy key."""
    if not name:
        return None
    candidate = " ".join(str(name).split())
    if candidate in GHANA_REGIONS:
        return candidate
    if candidate in LEGACY_REGION_ALIASES:
        return LEGACY_REGION_ALIASES[candidate]
    folded = candidate.casefold()
    for region in GHANA_REGIONS:
        if region.casefold() == folded:
            return region
    for legacy, target in LEGACY_REGION_ALIASES.items():
        if legacy.casefold() == folded:
            return target
    return None


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo accepts comma-separated coordinate lists and returns an array, so a
# full refresh across all sixteen regions is two requests, not thirty-two.
FORECAST_DAILY = ("precipitation_sum", "et0_fao_evapotranspiration", "temperature_2m_max")
FORECAST_HOURLY = ("soil_moisture_0_to_1cm", "soil_moisture_3_to_9cm")
FLOOD_DAILY = ("river_discharge",)

PAST_DAYS = 92
FORECAST_DAYS = 7
CLIMATOLOGY_START = "1995-01-01"
CLIMATOLOGY_END = "2024-12-31"
CLIMATOLOGY_LABEL = "ERA5 1995-2024"
DISCHARGE_CLIMATOLOGY_LABEL = "GloFAS reanalysis 1995-2024"

DATA_SOURCES = [
    {
        "id": "open-meteo",
        "label": "Open-Meteo",
        "detail": "Forecast and archive APIs",
        "url": "https://open-meteo.com/",
    },
    {
        "id": "glofas",
        "label": "GloFAS v4",
        "detail": "Copernicus Emergency Management Service river discharge",
        "url": "https://global-flood.emergency.copernicus.eu/",
    },
    {
        "id": "era5",
        "label": "ERA5",
        "detail": "ECMWF reanalysis, 1995-2024 baseline",
        "url": "https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5",
    },
]

# The limits stated in this module's docstring, in the form the UI shows them.
# They are served from /api/hazards/methodology rather than hardcoded in the
# frontend so the caveats can never drift away from the maths that needs them.
HAZARD_LIMITS: list[str] = [
    "River discharge comes from GloFAS, a roughly 5 km global model, not from "
    "gauges on Ghanaian rivers. It indicates river-system stress, not street-level flooding.",
    "Regions without a major monitored river reach are scored on rainfall alone. "
    "Their flooding is usually drainage-related, which no rainfall model can predict at street level.",
    "ERA5 is a reanalysis and under-represents localised convective storms, which "
    "cause much of the flash flooding in northern Ghana.",
    "SPI needs at least 30 years of record to be meaningful. This baseline is "
    "exactly 30 years, which is the accepted minimum rather than a comfortable margin.",
    "Dam operations at Bagre, Kompienga and Akosombo are not modelled, yet they "
    "drive a large share of White Volta flooding. Published GMet bulletins cover this gap.",
    "Scores describe conditions across a whole region. Any individual farm can "
    "differ sharply from its regional figure.",
]

# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

# One scale for both hazards. Five levels, lowest first. ``min_score`` is
# inclusive; a band runs up to the next band's minimum.
SEVERITY_BANDS: tuple[tuple[str, int], ...] = (
    ("normal", 0),
    ("watch", 25),
    ("moderate", 45),
    ("severe", 65),
    ("extreme", 85),
)

BAND_ORDER: dict[str, int] = {name: index for index, (name, _) in enumerate(SEVERITY_BANDS)}


def band_for(score: float | None) -> str:
    """Map a 0-100 score onto a severity band."""
    if score is None:
        return "unavailable"
    result = SEVERITY_BANDS[0][0]
    for name, minimum in SEVERITY_BANDS:
        if score >= minimum:
            result = name
    return result


# The WMO/McKee SPI classification, kept separate from the 0-100 band scale
# because SPI has its own published thresholds and we should not silently
# reinterpret them.
def spi_class(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value <= -2.0:
        return "extreme drought"
    if value <= -1.5:
        return "severe drought"
    if value <= -1.0:
        return "moderate drought"
    if value < 1.0:
        return "near normal"
    if value < 1.5:
        return "moderately wet"
    if value < 2.0:
        return "very wet"
    return "extremely wet"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Below this climatological 95th-percentile discharge, the GloFAS reach carrying
# a region is a headwater stream rather than a river worth monitoring, and its
# percentile says nothing useful about regional flooding. Those regions are
# scored on rainfall alone and the response flags ``riverine: false`` so the UI
# can say so out loud instead of showing a meaningless river figure.
#
# The test is on the 95th percentile, not the median, because Ghana's rivers are
# strongly seasonal. The Black Volta at Upper West has a median of only 17 m3/s
# but reaches 467 m3/s in flood season -- judging it on the median would discard
# a genuine riverine flood signal for exactly the months that matter.
RIVERINE_MIN_P95_CMS = 100.0

FLOOD_WEIGHTS = {
    "discharge": 0.40,
    "rain_7d": 0.30,
    "rain_max_1d": 0.20,
    "saturation": 0.10,
}

DROUGHT_WEIGHTS = {
    "spi": 0.45,
    "soil_moisture": 0.25,
    "dry_spell": 0.20,
    "water_deficit": 0.10,
}

# Rainfall depths, in mm, that anchor the single-day flood term. Roughly: a
# 20 mm day is unremarkable, 50 mm strains urban drainage, 100 mm+ is a
# damaging event anywhere in Ghana.
RAIN_MAX_1D_ANCHORS = [(0.0, 0.0), (20.0, 20.0), (50.0, 60.0), (100.0, 95.0), (150.0, 100.0)]

# Where there is no river to measure, flooding is drainage-limited and driven by
# how hard it rains rather than by the weekly total, so the single-day term
# carries more of the score.
PLUVIAL_RAIN_MAX_1D_WEIGHT = 0.45

# Consecutive rainless days that anchor the dry-spell term. Three weeks without
# rain in the growing season is a crop-loss event.
DRY_SPELL_FULL_SCORE_DAYS = 21.0

# Seven-day cumulative (ET0 - rainfall) that anchors the forward-deficit term.
WATER_DEFICIT_FULL_SCORE_MM = 40.0

# Above this SPI the season is running wetter than normal, and the drought score
# is held inside the normal band whatever the short-term terms say.
WET_SEASON_SPI_FLOOR = 0.5


@dataclass
class Driver:
    """One contributing measurement, carried to the UI so a score can be audited."""

    key: str
    label: str
    value: float | None
    unit: str
    score: float
    weight: float
    percentile: float | None = None
    gloss: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "value": None if self.value is None else round(self.value, 2),
            "unit": self.unit,
            "score": round(self.score, 1),
            "weight": round(self.weight, 3),
            "percentile": None if self.percentile is None else round(self.percentile, 1),
            "gloss": self.gloss,
        }


def _combine(drivers: list[Driver]) -> float:
    """Weighted mean over the drivers that actually have data.

    Renormalising over available weight -- rather than treating a missing term
    as zero -- is what lets a non-riverine region drop the discharge term
    without its flood score being artificially suppressed.
    """
    usable = [d for d in drivers if d.value is not None and d.weight > 0.0]
    total_weight = sum(d.weight for d in usable)
    if total_weight <= 0.0:
        return 0.0
    return clamp(sum(d.score * d.weight for d in usable) / total_weight)


def score_flood(
    *,
    discharge_now: float | None,
    discharge_reference: list[float] | None,
    rain_next_7d: float | None,
    rain_7d_p95: float | None,
    rain_max_1d: float | None,
    rain_day_p95: float | None,
    soil_moisture_now: float | None,
    soil_moisture_90d_max: float | None,
) -> tuple[float, list[Driver], bool]:
    """Score flood hazard 0-100.

    ``discharge_reference`` is the ascending 30-year daily discharge series for
    this reach. Returns ``(score, drivers, riverine)``.
    """
    drivers: list[Driver] = []

    # discharge_reference is an ascending 101-point quantile ladder, so index 95
    # is the 95th percentile.
    riverine = bool(
        discharge_reference
        and discharge_now is not None
        and len(discharge_reference) > 95
        and discharge_reference[95] >= RIVERINE_MIN_P95_CMS
    )

    if riverine:
        pct = percentile_of(discharge_now, discharge_reference)
        drivers.append(Driver(
            key="discharge",
            label="River discharge",
            value=discharge_now,
            unit="m3/s",
            # Discharge only starts to mean anything near the top of its own
            # record, so the bottom half of the distribution is flat at zero.
            score=linear_score(pct, 50.0, 99.0),
            weight=FLOOD_WEIGHTS["discharge"],
            percentile=pct,
            gloss=f"Higher than {pct:.0f}% of daily flows on this reach since 1995",
        ))

    if rain_next_7d is not None:
        pct = None
        if rain_7d_p95 and rain_7d_p95 > 0.0:
            score = linear_score(rain_next_7d, rain_7d_p95 * 0.4, rain_7d_p95 * 1.3)
            pct = min(100.0, rain_next_7d / rain_7d_p95 * 95.0)
            gloss = f"{rain_next_7d:.0f} mm forecast against a seasonal 95th percentile of {rain_7d_p95:.0f} mm"
        else:
            score = linear_score(rain_next_7d, 20.0, 200.0)
            gloss = f"{rain_next_7d:.0f} mm forecast over the next 7 days"
        drivers.append(Driver(
            key="rain7d", label="7-day rainfall", value=rain_next_7d, unit="mm",
            score=score, weight=FLOOD_WEIGHTS["rain_7d"], percentile=pct, gloss=gloss,
        ))

    if rain_max_1d is not None:
        # Intensity is judged against what a heavy day means HERE, when the
        # climatology provides it. A 50 mm day is an ordinary rainforest
        # afternoon and a rare event on the coastal plain, and the fixed
        # national anchors could not tell those apart -- which mattered most for
        # exactly the pluvial-flooding regions that have no river term to lean
        # on. The anchors remain as the fallback.
        if rain_day_p95 and rain_day_p95 > 0.0:
            score = piecewise(rain_max_1d, [
                (0.0, 0.0),
                (rain_day_p95 * 0.5, 20.0),
                (rain_day_p95, 60.0),
                (rain_day_p95 * 1.8, 100.0),
            ])
            gloss = (
                f"{rain_max_1d:.0f} mm in a day, against {rain_day_p95:.0f} mm "
                "for a heavy day here at this time of year"
            )
        else:
            score = piecewise(rain_max_1d, RAIN_MAX_1D_ANCHORS)
            gloss = f"{rain_max_1d:.0f} mm expected in a single day"

        drivers.append(Driver(
            key="rainMax1d", label="Heaviest forecast day", value=rain_max_1d, unit="mm",
            score=score,
            weight=FLOOD_WEIGHTS["rain_max_1d"],
            gloss=gloss,
        ))

    if soil_moisture_now is not None and soil_moisture_90d_max:
        saturation = soil_moisture_now / soil_moisture_90d_max if soil_moisture_90d_max > 0 else 0.0
        drivers.append(Driver(
            key="saturation", label="Soil saturation", value=saturation, unit="fraction",
            score=linear_score(saturation, 0.5, 1.0),
            weight=FLOOD_WEIGHTS["saturation"],
            gloss=f"Topsoil at {saturation * 100:.0f}% of its wettest level in the last 90 days",
        ))

    if not riverine:
        # With no river term, the remaining weights are renormalised anyway --
        # but a drainage-limited region floods because of how hard it rained,
        # not how much fell over a week. Promote intensity accordingly.
        for driver in drivers:
            if driver.key == "rainMax1d":
                driver.weight = PLUVIAL_RAIN_MAX_1D_WEIGHT

    return _combine(drivers), drivers, riverine


def score_drought(
    *,
    spi_value: float | None,
    soil_moisture_now: float | None,
    soil_moisture_window: list[float] | None,
    dry_spell_days: int | None,
    water_deficit_7d: float | None,
) -> tuple[float, list[Driver]]:
    """Score drought hazard 0-100."""
    drivers: list[Driver] = []

    if spi_value is not None:
        # SPI -1.0 is a moderate drought, -2.0 an extreme one. Anchor the scale
        # so those land inside the matching severity bands.
        drivers.append(Driver(
            key="spi", label="Rainfall anomaly (SPI-90)", value=spi_value, unit="sigma",
            score=clamp(50.0 - 25.0 * spi_value),
            weight=DROUGHT_WEIGHTS["spi"],
            gloss=f"90-day rainfall is {spi_class(spi_value)} against the {CLIMATOLOGY_LABEL} baseline",
        ))

    if soil_moisture_now is not None and soil_moisture_window:
        ordered = sorted(soil_moisture_window)
        pct = percentile_of(soil_moisture_now, ordered)
        drivers.append(Driver(
            key="soilMoisture", label="Soil moisture", value=soil_moisture_now, unit="m3/m3",
            score=clamp(100.0 - pct),
            weight=DROUGHT_WEIGHTS["soil_moisture"],
            percentile=pct,
            # Deliberately NOT called a climatological percentile: this ranks
            # against the region's own recent 90 days, not against 30 years.
            gloss=f"Wetter than {pct:.0f}% of the last 90 days in this region",
        ))

    if dry_spell_days is not None:
        drivers.append(Driver(
            key="drySpell", label="Days since rain", value=float(dry_spell_days), unit="days",
            score=clamp(dry_spell_days / DRY_SPELL_FULL_SCORE_DAYS * 100.0),
            weight=DROUGHT_WEIGHTS["dry_spell"],
            gloss=f"{dry_spell_days} consecutive days below 1 mm",
        ))

    if water_deficit_7d is not None:
        drivers.append(Driver(
            key="waterDeficit", label="Forward water deficit", value=water_deficit_7d, unit="mm",
            score=clamp(water_deficit_7d / WATER_DEFICIT_FULL_SCORE_MM * 100.0),
            weight=DROUGHT_WEIGHTS["water_deficit"],
            gloss=f"Evapotranspiration exceeds forecast rainfall by {water_deficit_7d:.0f} mm over 7 days",
        ))

    score = _combine(drivers)

    # A region cannot be in drought while it has had meaningfully more rain than
    # normal. Without this guard the corroborating terms outvote the index
    # itself: in August 2026 Greater Accra topped the drought table on dry-spell
    # and soil-moisture readings while its 90-day rainfall was 57% ABOVE normal
    # (SPI +1.61). Short-term dryness inside a wetter-than-normal season is worth
    # watching, but it is not agricultural drought, and publishing it as the
    # country's worst case discredits every other number on the page.
    if spi_value is not None and spi_value >= WET_SEASON_SPI_FLOOR:
        score = min(score, SEVERITY_BANDS[1][1] - 1)

    return score, drivers


# ---------------------------------------------------------------------------
# Advisories
# ---------------------------------------------------------------------------

# A deliberately CLOSED set of sentences, not generated prose.
#
# The frontend translates UI text at runtime through a batching cache keyed on
# the literal string (see frontend/src/components/common/T.jsx). Per-region
# generated wording would miss that cache on every load in Twi, Ewe, Ga or
# Dagbani -- sixteen regions times four languages of avoidable translation
# traffic. Holding the wording constant and varying only which entries apply
# keeps the cache hit rate high and keeps the agronomy reviewable by GMet.
ADVISORY_TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("flood", "watch"): [
        "Clear field drains, culverts and farm roads before the next rainfall.",
        "Lift stored grain, seed and fertiliser off the floor onto pallets.",
    ],
    ("flood", "moderate"): [
        "Open drainage channels on low-lying plots and check that outlets run freely.",
        "Harvest any crop that is mature enough to lift early.",
        "Move livestock, seed stock and agrochemicals off the floodplain.",
    ],
    ("flood", "severe"): [
        "Move livestock, planting material and agrochemicals to higher ground now.",
        "Do not attempt to cross flooded roads, culverts or spillways.",
        "Record affected plots and report them to your district NADMO office.",
        "Delay any fertiliser application until floodwater has drained.",
    ],
    ("flood", "extreme"): [
        "Follow NADMO instructions for your district, including any evacuation order.",
        "Move people and livestock to higher ground before water levels rise further.",
        "Do not enter flooded fields; submerged wells and channels are not visible.",
        "Keep drinking water separate from floodwater to avoid contamination.",
    ],
    ("drought", "watch"): [
        "Wait for at least 20 mm of rain across two days before planting.",
        "Mulch between rows to slow soil-water loss.",
    ],
    ("drought", "moderate"): [
        "Give priority to irrigating crops at flowering and grain filling.",
        "Switch any remaining planting to a short-duration variety.",
        "Weed early; weeds compete directly for the little soil water left.",
    ],
    ("drought", "severe"): [
        "Irrigate only the highest-value blocks; spreading water thinly loses the whole field.",
        "Arrange supplementary feed and check that livestock watering points still yield.",
        "Hold back from replanting until rainfall re-establishes.",
        "Harvest fodder and crop residues now rather than leaving them to dry out.",
    ],
    ("drought", "extreme"): [
        "Reduce stocking rates or move livestock to areas with reliable water.",
        "Protect remaining seed stock for the next season rather than planting it now.",
        "Contact your district agriculture office about drought relief and feed support.",
        "Prioritise household and livestock drinking water over irrigation.",
    ],
}

# Additions that only make sense in particular agroecological zones.
ZONE_ADVISORIES: dict[tuple[str, str], list[str]] = {
    ("flood", "Coastal Savannah"): [
        "Check that lagoon and estuary outlets are open; backed-up tides worsen inland flooding.",
    ],
    ("flood", "Guinea Savannah"): [
        "Watch for upstream dam spillage announcements; White Volta levels can rise with no local rain.",
    ],
    ("flood", "Sudan Savannah"): [
        "Watch for upstream dam spillage announcements; White Volta levels can rise with no local rain.",
    ],
    ("drought", "Sudan Savannah"): [
        "Space plantings across the short rains rather than committing all seed at once.",
    ],
    ("drought", "Guinea Savannah"): [
        "Favour early-maturing sorghum, millet or cowpea over long-duration maize.",
    ],
}


def advisories_for(hazard: str, band: str, agro_zone: str) -> list[str]:
    """Guidance for one region, most urgent band first.

    ``normal`` and ``unavailable`` deliberately return nothing: a page that
    always has advice to give trains readers to ignore the advice.
    """
    entries = list(ADVISORY_TEMPLATES.get((hazard, band), []))
    if entries:
        entries.extend(ZONE_ADVISORIES.get((hazard, agro_zone), []))
    return entries


def dry_spell_length(daily_rain: list[float | None], threshold_mm: float = 1.0) -> int:
    """Consecutive days ending today with rainfall below ``threshold_mm``."""
    count = 0
    for value in reversed(daily_rain):
        if value is None or value >= threshold_mm:
            break
        count += 1
    return count
