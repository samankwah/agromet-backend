"""The maths behind the weeks 2-to-4 outlook.

Pure functions only -- no I/O, no HTTP, no FastAPI. The bake script
(``backend/scripts/build_s2s_climatology.py``) and the request-path runtime
(``backend/app/s2s_runtime.py``) both import from here, which is the point: if
the window the baseline is computed over ever drifted from the window the
forecast is summed over, every probability in the app would be quietly wrong and
nothing would look broken. Sharing the constants makes that impossible.

Same split as ``hazard_math.py`` versus ``hazards.py``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

# "Weeks 2 to 4" means days 14 to 28 after the forecast is issued: fifteen days,
# starting a fortnight out. Days 0-13 are deliberately excluded -- that range is
# already covered, far better, by the Daily and 7-Day segments, and a
# subseasonal product that overlapped them would invite a farmer to compare two
# forecasts of the same days and find them disagreeing.
LEAD_DAYS = 14
WINDOW_DAYS = 15

# NOAA's GEFS extended run carries real data to about day 33, so day 28 sits
# comfortably inside it. ECMWF's operational ensemble stops at day 14 and cannot
# serve this window at all, which is why the model underneath is GEFS -- see the
# plan and `s2s_runtime`.
MAX_USABLE_LEAD_DAYS = 33


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


def quantile(sorted_values: list[float], q: float) -> float:
    """The value at quantile ``q`` of an ascending list, linearly interpolated.

    Written here rather than pulled from numpy because the backend has no numpy
    dependency and this is nine lines. Interpolating rather than picking the
    nearest rank matters at the sample sizes involved (~210 per ISO week): with
    nearest-rank the 33rd and 67th percentiles can collide on a coarse sample and
    silently produce an empty middle tercile.
    """
    if not sorted_values:
        raise ValueError("quantile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low]) * (1 - weight) + float(sorted_values[high]) * weight


# ---------------------------------------------------------------------------
# Terciles
# ---------------------------------------------------------------------------

CATEGORIES = ("below", "normal", "above")


def tercile_probabilities(
    members: list[float | None],
    p33: float,
    p67: float,
) -> dict | None:
    """How the ensemble splits across the three climatological thirds.

    Returns ``None`` when no member carried a value -- an absent forecast, which
    the caller must report as unavailable rather than as a flat 33/33/33. Those
    two things look identical on a chart and mean opposite things.

    Boundary values count as normal (``< p33`` below, ``> p67`` above). That is
    not a rounding convenience: in Ghana's dry season a third or more of the
    historical record is exactly zero, so ``p33`` is 0.0 and a member forecasting
    no rain is not *below* normal -- no rain is precisely what normal looks like
    there. Treating it as below-normal would paint the Sahel drought-red every
    January.

    ``degenerate`` marks the case where the two boundaries coincide, which
    happens when most of the record is zero. The split is then arithmetically
    valid but carries no information, and the caller should say so rather than
    publish "100% normal" as though it were a confident finding.
    """
    usable = [float(value) for value in members if value is not None]
    if not usable:
        return None

    below = sum(1 for value in usable if value < p33)
    above = sum(1 for value in usable if value > p67)
    normal = len(usable) - below - above
    count = len(usable)

    return {
        "below": below / count,
        "normal": normal / count,
        "above": above / count,
        "members": count,
        "degenerate": p33 == p67,
    }


def dominant_category(probabilities: dict) -> str:
    """Which third the ensemble favours, ties going to ``normal``.

    A tie between below and above is not a forecast of either; it is a forecast
    of nothing in particular, and ``normal`` is the honest label for that.
    """
    ranked = sorted(CATEGORIES, key=lambda name: probabilities[name], reverse=True)
    top = ranked[0]
    if probabilities[top] == probabilities["normal"]:
        return "normal"
    return top


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

# A three-way split has a floor of about 33% per category: that *is* climatology,
# and an ensemble landing there has told us nothing. These thresholds are how far
# above that floor the winning share has to sit before the outlook is worth
# leaning on.
#
# Judgement calls, deliberately conservative, and the first thing to revisit if
# forecasters say the app overclaims. Weeks 3-4 rainfall skill over West Africa
# is modest, so "low" being the common answer is the expected outcome, not a bug.
CONFIDENCE_MODERATE_SHARE = 0.40
CONFIDENCE_HIGH_SHARE = 0.55


def agreement_confidence(probabilities: dict) -> str:
    """How much the ensemble agrees, as the UI's three-step confidence."""
    if probabilities.get("degenerate"):
        return "low"

    share = max(probabilities[name] for name in CATEGORIES)
    if share >= CONFIDENCE_HIGH_SHARE:
        return "high"
    if share >= CONFIDENCE_MODERATE_SHARE:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# The model's own grid
# ---------------------------------------------------------------------------

# Ghana's extent, matching `mobile/src/shared/data/ghanaBoundaries.json`'s
# `bounds`. Restated here rather than imported because the backend must not
# depend on a mobile asset, and these are fixed geography.
GHANA_BOUNDS = {"minLat": 4.74540, "maxLat": 11.17445, "minLng": -3.25491, "maxLng": 1.19948}

# GEFS is half a degree beyond day 10, which is the whole of this window.
GRID_STEP_DEG = 0.5


def ghana_grid_points() -> list[tuple[float, float]]:
    """The model's native 0.5 degree lattice over Ghana: 165 points.

    Sampling sixteen region centroids was what made district granularity
    dishonest — a district map drawn from sixteen points invents everything
    between them. Fetching the field the model actually produces removes that
    objection: at 0.5 degrees the map is blocky, but every block is real.

    Snapped to the half-degree lattice and padded by one step, so the country's
    edges sit inside the grid rather than on its boundary. Open-Meteo snaps
    coordinates to the same lattice, so asking for these exact points means the
    response needs no interpolation.

    Shared with the bake for the reason `LEAD_DAYS` is: if the two ever disagreed
    about which cells exist, every probability would be computed against the
    wrong baseline and nothing would look broken.
    """
    def snap(value: float) -> float:
        return round(value / GRID_STEP_DEG) * GRID_STEP_DEG

    points: list[tuple[float, float]] = []
    lat = snap(GHANA_BOUNDS["minLat"])
    while lat <= GHANA_BOUNDS["maxLat"] + GRID_STEP_DEG:
        lng = snap(GHANA_BOUNDS["minLng"])
        while lng <= GHANA_BOUNDS["maxLng"] + GRID_STEP_DEG:
            points.append((round(lat, 2), round(lng, 2)))
            lng += GRID_STEP_DEG
        lat += GRID_STEP_DEG
    return points


def cell_id(lat: float, lng: float) -> str:
    """A stable key for one grid cell, used by the asset, the cache and the
    series endpoint. Fixed to two decimals so 4.5 and 4.500001 cannot become two
    different cells."""
    return f"{lat:.2f},{lng:.2f}"


def nearest_cell(lat: float, lng: float, points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """The grid cell a place falls in, or None if the grid does not reach it.

    Squared distance rather than a great-circle metric: over a country five
    degrees across, on a lattice this coarse, the two never disagree about which
    cell is nearest, and this needs no trigonometry.

    The distance guard is not defensive padding. Without it "nearest" always
    succeeds, so a request from Kenya would come back with a Ghanaian forecast
    wearing Kenyan coordinates -- wrong in the way that looks completely
    plausible. One grid step of tolerance covers a point anywhere inside a cell
    whose centre is on the lattice; beyond that the grid simply does not cover
    the place, and saying so is the only honest answer.
    """
    if not points:
        return None

    best = min(points, key=lambda p: (p[0] - lat) ** 2 + (p[1] - lng) ** 2)
    if abs(best[0] - lat) > GRID_STEP_DEG or abs(best[1] - lng) > GRID_STEP_DEG:
        return None
    return best


def ensemble_percentile(values: list[float], q: float) -> float:
    """One ensemble's percentile at a single lead day, for the spread band.

    Thin wrapper over `quantile` so the chart's p10/p90 and the baseline's
    p33/p67 cannot end up using different definitions of a percentile.
    """
    return quantile(sorted(values), q)
