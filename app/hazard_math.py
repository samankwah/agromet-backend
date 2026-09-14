"""Pure statistics for the flood and drought indices.

Deliberately dependency-free. ``backend/requirements.txt`` is six packages and
the app ships as a Vercel serverless function, so pulling in numpy/scipy to get
one gamma fit and one inverse normal CDF is a bad trade. Everything here is
stdlib ``math``, and does no I/O, so it is unit-testable on its own.

References:
  Thom, H.C.S. (1958) "A note on the gamma distribution", Monthly Weather Review.
  McKee, Doesken & Kleist (1993) "The relationship of drought frequency and
    duration to time scales" -- the original SPI definition.
  Acklam, P.J. "An algorithm for computing the inverse normal cumulative
    distribution function" (rational approximation, absolute error < 1.15e-9).
"""

from __future__ import annotations

import math

__all__ = [
    "SPI_LIMIT",
    "clamp",
    "fit_gamma",
    "gamma_cdf",
    "inverse_normal_cdf",
    "linear_score",
    "percentile_of",
    "piecewise",
    "spi",
]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Gamma distribution
# ---------------------------------------------------------------------------

def fit_gamma(values: list[float]) -> tuple[float, float, float]:
    """Fit a gamma distribution to precipitation totals via Thom's approximation.

    Precipitation series contain exact zeros, which a gamma cannot represent
    (its support is x > 0). The standard SPI treatment is a mixed distribution:
    fit the gamma to the positive values only, and carry the probability of a
    zero separately so it can be mixed back in at the CDF stage.

    Returns ``(shape, scale, zero_probability)``. Shape and scale are ``0.0``
    when there is too little positive data to fit; callers must treat that as
    "no valid fit" rather than as a degenerate distribution.
    """
    if not values:
        return 0.0, 0.0, 1.0

    positives = [v for v in values if v > 0.0]
    zero_probability = (len(values) - len(positives)) / len(values)

    # Two positives can technically be fitted, but the result is noise.
    if len(positives) < 4:
        return 0.0, 0.0, zero_probability

    mean = sum(positives) / len(positives)
    mean_log = sum(math.log(v) for v in positives) / len(positives)

    # Thom's estimator. A is non-negative by Jensen's inequality, and collapses
    # to zero only when every positive value is identical -- which would send
    # the shape parameter to infinity.
    a = math.log(mean) - mean_log
    if a <= 1e-12:
        return 0.0, 0.0, zero_probability

    shape = (1.0 + math.sqrt(1.0 + (4.0 * a) / 3.0)) / (4.0 * a)
    if shape <= 0.0 or not math.isfinite(shape):
        return 0.0, 0.0, zero_probability

    return shape, mean / shape, zero_probability


def _lower_gamma_series(shape: float, x: float) -> float:
    """Regularized lower incomplete gamma P(shape, x) by series expansion.

    Converges quickly for x < shape + 1.
    """
    term = 1.0 / shape
    total = term
    n = shape
    for _ in range(500):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + shape * math.log(x) - math.lgamma(shape))


def _upper_gamma_continued_fraction(shape: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(shape, x) by modified Lentz.

    Converges quickly for x >= shape + 1.
    """
    tiny = 1e-300
    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b if b != 0.0 else 1.0 / tiny
    h = d
    for i in range(1, 500):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + shape * math.log(x) - math.lgamma(shape))


def gamma_cdf(x: float, shape: float, scale: float) -> float:
    """Regularized lower incomplete gamma -- i.e. the gamma CDF."""
    if shape <= 0.0 or scale <= 0.0 or x <= 0.0:
        return 0.0
    t = x / scale
    if t < shape + 1.0:
        return min(1.0, max(0.0, _lower_gamma_series(shape, t)))
    return min(1.0, max(0.0, 1.0 - _upper_gamma_continued_fraction(shape, t)))


# ---------------------------------------------------------------------------
# Inverse normal CDF (Acklam)
# ---------------------------------------------------------------------------

_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def inverse_normal_cdf(p: float) -> float:
    """Standard normal quantile."""
    if p <= 0.0 or p >= 1.0:
        # Callers clamp before this point. Returning the SPI display bound is
        # safer than raising inside a refresh loop over 16 regions.
        return -3.09 if p <= 0.0 else 3.09

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) /
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))

    if p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        return ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q /
                (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0))

    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) /
             ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))


# ---------------------------------------------------------------------------
# SPI
# ---------------------------------------------------------------------------

SPI_LIMIT = 3.09  # about the 0.1 / 99.9 percentiles; past this the fit is noise


def spi(total_mm: float, shape: float, scale: float, zero_probability: float) -> float | None:
    """Standardised Precipitation Index for one accumulation total.

    Returns ``None`` when the climatological fit is unusable. During the
    northern dry season a 90-day window is legitimately near-zero rainfall and
    the gamma degenerates; reporting ``None`` so the UI can say "outside SPI
    validity" is honest, whereas emitting a fabricated -3.0 is not.
    """
    if shape <= 0.0 or scale <= 0.0:
        return None

    if total_mm > 0.0:
        cumulative = zero_probability + (1.0 - zero_probability) * gamma_cdf(total_mm, shape, scale)
    else:
        # The textbook mixed-distribution form sets H(0) = q, which puts a
        # zero-rainfall period at the TOP of the tied dry mass: with q = 0.60 a
        # totally dry window would score SPI +0.25, i.e. "wetter than normal".
        # That is indefensible on a drought page. Use the mid-rank plotting
        # position for the tie instead, so a dry window reads dry.
        cumulative = zero_probability / 2.0

    # Guard the tails so inverse_normal_cdf never sees exactly 0 or 1.
    epsilon = 1e-6
    cumulative = min(1.0 - epsilon, max(epsilon, cumulative))

    return max(-SPI_LIMIT, min(SPI_LIMIT, inverse_normal_cdf(cumulative)))


# ---------------------------------------------------------------------------
# Helpers used by the scoring layer
# ---------------------------------------------------------------------------

def percentile_of(value: float, sorted_reference: list[float]) -> float:
    """Percentile rank of ``value`` within an ascending reference series, 0-100."""
    n = len(sorted_reference)
    if n == 0:
        return 0.0
    low, high = 0, n
    while low < high:
        mid = (low + high) // 2
        if sorted_reference[mid] <= value:
            low = mid + 1
        else:
            high = mid
    return (low / n) * 100.0


def linear_score(value: float, at_zero: float, at_hundred: float) -> float:
    """Map a value onto 0-100 by interpolating between two anchors.

    Works in either direction, so an inverted scale needs no special case.
    """
    if at_hundred == at_zero:
        return 0.0
    return clamp((value - at_zero) / (at_hundred - at_zero) * 100.0)


def piecewise(value: float, points: list[tuple[float, float]]) -> float:
    """Interpolate along an ascending list of ``(input, score)`` anchors."""
    if not points:
        return 0.0
    if value <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if value <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return points[-1][1]
