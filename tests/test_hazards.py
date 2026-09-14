from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.database import init_db, set_database_path
from backend.app.hazard_math import (
    fit_gamma,
    gamma_cdf,
    inverse_normal_cdf,
    percentile_of,
    piecewise,
    spi,
)
from backend.app.hazards import (
    BAND_ORDER,
    GHANA_REGIONS,
    advisories_for,
    band_for,
    dry_spell_length,
    resolve_region,
    score_drought,
    score_flood,
    spi_class,
)


class GammaMathTests(unittest.TestCase):
    """Pin the statistics against closed forms, not against themselves."""

    def test_gamma_cdf_matches_exponential(self):
        # shape = 1 is Exponential(scale): CDF = 1 - exp(-x / scale)
        for x in (0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0):
            self.assertAlmostEqual(
                gamma_cdf(x, 1.0, 2.0), 1 - math.exp(-x / 2.0), places=12
            )

    def test_gamma_cdf_matches_erlang_two(self):
        # shape = 2, scale = 1: CDF = 1 - (1 + x) e^-x
        for x in (0.1, 1.0, 3.0, 7.0, 15.0, 30.0):
            self.assertAlmostEqual(
                gamma_cdf(x, 2.0, 1.0), 1 - (1 + x) * math.exp(-x), places=12
            )

    def test_gamma_cdf_matches_erlang_three(self):
        for x in (0.5, 2.0, 6.0, 12.0, 25.0):
            t = x / 2.0
            expected = 1 - math.exp(-t) * (1 + t + t * t / 2)
            self.assertAlmostEqual(gamma_cdf(x, 3.0, 2.0), expected, places=12)

    def test_inverse_normal_cdf_matches_published_quantiles(self):
        known = {
            0.5: 0.0,
            0.975: 1.9599639845,
            0.025: -1.9599639845,
            0.99: 2.3263478740,
            0.001: -3.0902323062,
        }
        for probability, quantile in known.items():
            self.assertAlmostEqual(inverse_normal_cdf(probability), quantile, places=8)

    def test_fit_gamma_recovers_known_parameters(self):
        import random

        random.seed(7)
        sample = [random.gammavariate(2.0, 30.0) for _ in range(20000)]
        shape, scale, zero_probability = fit_gamma(sample)
        self.assertAlmostEqual(shape, 2.0, delta=0.15)
        self.assertAlmostEqual(scale, 30.0, delta=2.5)
        self.assertEqual(zero_probability, 0.0)

    def test_fit_gamma_refuses_insufficient_data(self):
        shape, scale, zero_probability = fit_gamma([0.0, 0.0, 1.0])
        self.assertEqual((shape, scale), (0.0, 0.0))
        self.assertAlmostEqual(zero_probability, 2 / 3)

    def test_spi_is_zero_at_the_median_and_monotonic(self):
        import random

        random.seed(11)
        sample = [random.gammavariate(2.0, 30.0) for _ in range(20000)]
        shape, scale, zero_probability = fit_gamma(sample)

        median = sorted(sample)[len(sample) // 2]
        self.assertAlmostEqual(spi(median, shape, scale, zero_probability), 0.0, delta=0.03)

        values = [spi(v, shape, scale, zero_probability) for v in (5, 20, 60, 120, 300)]
        self.assertEqual(values, sorted(values))

    def test_spi_reaches_minus_one_at_the_15th_percentile(self):
        import random

        random.seed(13)
        sample = [random.gammavariate(2.0, 30.0) for _ in range(20000)]
        shape, scale, zero_probability = fit_gamma(sample)
        p16 = sorted(sample)[int(len(sample) * 0.1587)]
        self.assertAlmostEqual(spi(p16, shape, scale, zero_probability), -1.0, delta=0.06)

    def test_spi_returns_none_without_a_usable_fit(self):
        self.assertIsNone(spi(0.0, 0.0, 0.0, 1.0))

    def test_zero_rainfall_never_reads_as_wetter_than_normal(self):
        """The textbook mixed-distribution form puts a dry period at the top of
        the tied dry mass, so with a 60% zero probability a totally dry window
        scores SPI +0.25. That is indefensible on a drought page."""
        import random

        random.seed(17)
        values = [0.0] * 600 + [random.gammavariate(1.5, 10.0) for _ in range(400)]
        shape, scale, zero_probability = fit_gamma(values)
        self.assertAlmostEqual(zero_probability, 0.6)
        self.assertLess(spi(0.0, shape, scale, zero_probability), -0.3)

    def test_percentile_and_piecewise(self):
        self.assertEqual(percentile_of(5, list(range(1, 11))), 50.0)
        self.assertEqual(percentile_of(0, list(range(1, 11))), 0.0)
        self.assertEqual(percentile_of(99, list(range(1, 11))), 100.0)
        self.assertEqual(piecewise(50, [(20, 25), (50, 60), (100, 100)]), 60.0)
        self.assertEqual(piecewise(35, [(20, 25), (50, 60), (100, 100)]), 42.5)


class BandTests(unittest.TestCase):
    def test_band_boundaries_are_inclusive_at_the_minimum(self):
        cases = {
            0: "normal", 24: "normal",
            25: "watch", 44: "watch",
            45: "moderate", 64: "moderate",
            65: "severe", 84: "severe",
            85: "extreme", 100: "extreme",
        }
        for score, band in cases.items():
            self.assertEqual(band_for(score), band, f"score {score}")
        self.assertEqual(band_for(None), "unavailable")

    def test_spi_classification_follows_the_wmo_thresholds(self):
        self.assertEqual(spi_class(-2.5), "extreme drought")
        self.assertEqual(spi_class(-1.6), "severe drought")
        self.assertEqual(spi_class(-1.2), "moderate drought")
        self.assertEqual(spi_class(-0.5), "near normal")
        self.assertEqual(spi_class(0.9), "near normal")
        self.assertEqual(spi_class(2.5), "extremely wet")
        self.assertEqual(spi_class(None), "unavailable")


class RegionTests(unittest.TestCase):
    def test_sixteen_current_regions(self):
        self.assertEqual(len(GHANA_REGIONS), 16)
        self.assertNotIn("Brong-Ahafo", GHANA_REGIONS)

    def test_legacy_region_name_resolves(self):
        self.assertEqual(resolve_region("Brong-Ahafo"), "Bono")
        self.assertEqual(resolve_region("upper east"), "Upper East")
        self.assertEqual(resolve_region("  Volta  "), "Volta")
        self.assertIsNone(resolve_region("Atlantis"))

    def test_river_points_differ_from_centroids(self):
        """A GloFAS sample at an administrative centroid returns a trickle, so
        every region must carry a distinct river-reach point."""
        for name, region in GHANA_REGIONS.items():
            self.assertNotEqual(
                (region.lat, region.lon), (region.river_lat, region.river_lon), name
            )


class ScoringTests(unittest.TestCase):
    def setUp(self):
        # Ascending 101-point ladder, p95 = 950 -> comfortably riverine.
        self.riverine_ladder = [float(i) * 10 for i in range(101)]
        # p95 = 9.5 -> below the threshold, so the discharge term must drop.
        self.headwater_ladder = [float(i) / 10 for i in range(101)]

    def test_riverine_region_uses_the_discharge_term(self):
        score, drivers, riverine = score_flood(
            discharge_now=990.0,
            discharge_reference=self.riverine_ladder,
            rain_next_7d=100.0,
            rain_7d_p95=120.0,
            rain_max_1d=40.0,
            rain_day_p95=None,
            soil_moisture_now=0.3,
            soil_moisture_90d_max=0.4,
        )
        self.assertTrue(riverine)
        self.assertIn("discharge", {d.key for d in drivers})
        self.assertGreater(score, 0)

    def test_headwater_region_drops_the_discharge_term(self):
        score, drivers, riverine = score_flood(
            discharge_now=9.0,
            discharge_reference=self.headwater_ladder,
            rain_next_7d=100.0,
            rain_7d_p95=120.0,
            rain_max_1d=40.0,
            rain_day_p95=None,
            soil_moisture_now=0.3,
            soil_moisture_90d_max=0.4,
        )
        self.assertFalse(riverine)
        self.assertNotIn("discharge", {d.key for d in drivers})
        # The remaining weights are renormalised, so dropping a term must not
        # artificially suppress the score.
        self.assertGreater(score, 0)

    def test_missing_terms_do_not_drag_the_score_to_zero(self):
        only_spi, drivers = score_drought(
            spi_value=-2.0,
            soil_moisture_now=None,
            soil_moisture_window=None,
            dry_spell_days=None,
            water_deficit_7d=None,
        )
        self.assertEqual(len(drivers), 1)
        self.assertEqual(only_spi, 100.0)

    def test_score_is_bounded(self):
        score, _, _ = score_flood(
            discharge_now=10_000.0,
            discharge_reference=self.riverine_ladder,
            rain_next_7d=9_999.0,
            rain_7d_p95=10.0,
            rain_max_1d=900.0,
            rain_day_p95=None,
            soil_moisture_now=1.0,
            soil_moisture_90d_max=1.0,
        )
        self.assertLessEqual(score, 100.0)

    def test_dry_spell_counts_trailing_days_only(self):
        self.assertEqual(dry_spell_length([5.0, 0.2, 0.0, 0.5, 0.0]), 4)
        self.assertEqual(dry_spell_length([0.0, 0.0, 3.0]), 0)
        self.assertEqual(dry_spell_length([]), 0)

    def test_advisories_stay_silent_when_nothing_is_wrong(self):
        self.assertEqual(advisories_for("flood", "normal", "Rainforest"), [])
        self.assertEqual(advisories_for("drought", "unavailable", "Rainforest"), [])

    def test_advisories_add_zone_specific_guidance(self):
        guidance = advisories_for("flood", "severe", "Guinea Savannah")
        self.assertTrue(any("spillage" in line for line in guidance))
        self.assertNotIn(
            "spillage",
            " ".join(advisories_for("flood", "severe", "Rainforest")),
        )


SNAPSHOT = {
    "Northern": {
        "region": "Northern",
        "agroZone": "Guinea Savannah",
        "centroid": [9.38, -0.37],
        "riverPoint": [9.83, -0.98],
        "riverine": True,
        "flood": {"score": 61.0, "band": "moderate", "drivers": []},
        "drought": {"score": 12.0, "band": "normal", "spi": 0.4, "spiClass": "near normal",
                    "precip90dMm": 400.0, "precip90dNormalMm": 401.0, "drivers": []},
        "dominant": "flood",
        "series": {},
        "discharge": {},
    }
}


class EndpointTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        set_database_path(self.db_path)
        init_db()

        from backend.app.main import app

        self.client = TestClient(app)

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def _with_snapshot(self):
        from backend.app import hazard_runtime

        return patch.multiple(
            hazard_runtime,
            _CACHE=dict(SNAPSHOT),
            _CACHE_STAMP=9_999_999_999.0,  # far future, so nothing looks stale
        )

    def test_methodology_is_served_without_upstream(self):
        response = self.client.get("/api/hazards/methodology")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(
            [entry["band"] for entry in data["bands"]],
            ["normal", "watch", "moderate", "severe", "extreme"],
        )
        self.assertTrue(data["limits"])

    def test_summary_degrades_to_200_when_upstream_fails(self):
        """A monitoring page must not show an error screen because a third
        party is down; it reports what it has and says how old it is."""
        from backend.app import hazard_runtime

        with patch.object(
            hazard_runtime, "compute_snapshot", side_effect=RuntimeError("upstream down")
        ), patch.multiple(hazard_runtime, _CACHE={}, _CACHE_STAMP=0.0):
            response = self.client.get("/api/hazards/summary")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertTrue(data["unavailable"])
        self.assertEqual(data["regions"], [])

    def test_summary_reports_computed_values_without_an_override(self):
        with self._with_snapshot():
            response = self.client.get("/api/hazards/summary")

        data = response.json()["data"]
        region = data["regions"][0]
        self.assertFalse(region["flood"]["overridden"])
        self.assertEqual(region["flood"]["source"], "open-meteo")
        self.assertEqual(region["flood"]["band"], "moderate")
        # Advisories are attached from the closed template set.
        self.assertTrue(region["flood"]["advisories"])

    def test_override_supersedes_the_model_but_keeps_it_visible(self):
        token = self._register_and_login()

        create = self.client.post(
            "/api/hazards/overrides",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "region": "Northern",
                "hazard": "flood",
                "band": "extreme",
                "headline": "Bagre spillage under way",
                "advisories": ["Move to higher ground."],
            },
        )
        self.assertEqual(create.status_code, 200, create.text)

        with self._with_snapshot():
            data = self.client.get("/api/hazards/summary").json()["data"]

        flood = data["regions"][0]["flood"]
        self.assertTrue(flood["overridden"])
        self.assertEqual(flood["band"], "extreme")
        self.assertEqual(flood["source"], "gmet-bulletin")
        self.assertEqual(flood["headline"], "Bagre spillage under way")
        # The computed reading survives, so the override is never silent.
        self.assertEqual(flood["computed"]["band"], "moderate")
        self.assertEqual(data["national"]["overriddenCount"], 1)

    def test_expired_override_reverts_without_a_cleanup_job(self):
        token = self._register_and_login()
        self.client.post(
            "/api/hazards/overrides",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "region": "Northern",
                "hazard": "flood",
                "band": "extreme",
                "effectiveFrom": "2020-01-01 00:00:00",
                "effectiveTo": "2020-01-02 00:00:00",
            },
        )

        with self._with_snapshot():
            data = self.client.get("/api/hazards/summary").json()["data"]

        flood = data["regions"][0]["flood"]
        self.assertFalse(flood["overridden"])
        self.assertEqual(flood["band"], "moderate")

    def test_override_requires_authentication(self):
        response = self.client.post(
            "/api/hazards/overrides",
            json={"region": "Northern", "hazard": "flood", "band": "extreme"},
        )
        self.assertIn(response.status_code, (401, 403))

    def test_override_rejects_an_unknown_region(self):
        token = self._register_and_login()
        response = self.client.post(
            "/api/hazards/overrides",
            headers={"Authorization": f"Bearer {token}"},
            json={"region": "Atlantis", "hazard": "flood", "band": "extreme"},
        )
        self.assertEqual(response.status_code, 400)

    def test_unknown_region_detail_is_404(self):
        with self._with_snapshot():
            response = self.client.get("/api/hazards/regions/Atlantis")
        self.assertEqual(response.status_code, 404)

    def _register_and_login(self) -> str:
        self.client.post(
            "/api/v1/auth/register",
            json={"email": "gmet@example.com", "password": "secret123", "name": "GMet"},
        )
        response = self.client.post(
            "/api/v1/auth/login",
            data={"username": "gmet@example.com", "password": "secret123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["access_token"]


if __name__ == "__main__":
    unittest.main()


class ModelConsistencyTests(unittest.TestCase):
    """Two defects found by reading the published output against what a Ghanaian
    reader knows about the country."""

    def test_a_wetter_than_normal_season_cannot_read_as_drought(self):
        """Greater Accra topped the national drought table in August 2026 on
        dry-spell and soil-moisture readings, while its 90-day rainfall was 57%
        above normal. The index has to outrank its own corroborating terms."""
        score, _ = score_drought(
            spi_value=1.61,           # markedly wetter than normal
            soil_moisture_now=0.05,   # but dry right now
            soil_moisture_window=[0.05 + i * 0.004 for i in range(90)],
            dry_spell_days=20,
            water_deficit_7d=39.0,
        )
        self.assertEqual(band_for(score), "normal")

    def test_the_guard_does_not_suppress_a_real_drought(self):
        score, _ = score_drought(
            spi_value=-1.8,
            soil_moisture_now=0.05,
            soil_moisture_window=[0.05 + i * 0.004 for i in range(90)],
            dry_spell_days=20,
            water_deficit_7d=39.0,
        )
        self.assertGreaterEqual(BAND_ORDER[band_for(score)], BAND_ORDER["severe"])

    def test_heavy_rain_is_judged_against_local_normals(self):
        """The same 45 mm day is unremarkable in the rainforest and a rare event
        on the coastal plain. Fixed national anchors could not tell the two
        apart -- which mattered most in exactly the drainage-limited regions
        that have no river term to fall back on."""
        headwater = [float(i) / 10 for i in range(101)]  # p95 well below the riverine floor

        coastal, _, _ = score_flood(
            discharge_now=1.0, discharge_reference=headwater,
            rain_next_7d=50.0, rain_7d_p95=60.0,
            rain_max_1d=45.0, rain_day_p95=25.0,      # a heavy day here is 25 mm
            soil_moisture_now=0.3, soil_moisture_90d_max=0.4,
        )
        rainforest, _, _ = score_flood(
            discharge_now=1.0, discharge_reference=headwater,
            rain_next_7d=50.0, rain_7d_p95=60.0,
            rain_max_1d=45.0, rain_day_p95=60.0,      # a heavy day here is 60 mm
            soil_moisture_now=0.3, soil_moisture_90d_max=0.4,
        )
        self.assertGreater(coastal, rainforest)

    def test_intensity_carries_more_weight_where_there_is_no_river(self):
        riverine_ladder = [float(i) * 10 for i in range(101)]
        headwater = [float(i) / 10 for i in range(101)]

        def weight_of(reference):
            _, drivers, _ = score_flood(
                discharge_now=500.0, discharge_reference=reference,
                rain_next_7d=50.0, rain_7d_p95=60.0,
                rain_max_1d=45.0, rain_day_p95=25.0,
                soil_moisture_now=0.3, soil_moisture_90d_max=0.4,
            )
            return next(d.weight for d in drivers if d.key == "rainMax1d")

        self.assertGreater(weight_of(headwater), weight_of(riverine_ladder))
