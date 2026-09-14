"""The weeks 2-to-4 outlook: the maths, the window, and the degradation."""

from __future__ import annotations

import unittest
from datetime import date, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import s2s_runtime
from backend.app.main import app
from backend.app.s2s import (
    GHANA_BOUNDS,
    LEAD_DAYS,
    WINDOW_DAYS,
    agreement_confidence,
    cell_id,
    dominant_category,
    ensemble_percentile,
    ghana_grid_points,
    nearest_cell,
    quantile,
    tercile_probabilities,
)


class QuantileTests(unittest.TestCase):
    def test_interpolates_between_ranks(self):
        # Nearest-rank would return 3.0 for both here and collapse the middle
        # tercile on a coarse sample, which is what the interpolation avoids.
        self.assertAlmostEqual(quantile([1.0, 2.0, 3.0, 4.0], 1 / 3), 2.0)
        self.assertAlmostEqual(quantile([1.0, 2.0, 3.0, 4.0], 2 / 3), 3.0)

    def test_handles_a_single_sample(self):
        self.assertEqual(quantile([7.5], 1 / 3), 7.5)

    def test_refuses_an_empty_sample(self):
        # Returning 0.0 would read as "no rain normally falls here", a very
        # different claim from "we have no baseline".
        with self.assertRaises(ValueError):
            quantile([], 0.5)


class TercileTests(unittest.TestCase):
    def test_splits_members_across_the_three_thirds(self):
        members = [1.0, 5.0, 9.0, 11.0, 12.0, 15.0, 19.0, 21.0, 30.0, 40.0]
        result = tercile_probabilities(members, 10.0, 20.0)

        self.assertAlmostEqual(result["below"], 0.3)
        self.assertAlmostEqual(result["normal"], 0.4)
        self.assertAlmostEqual(result["above"], 0.3)
        self.assertEqual(result["members"], 10)

    def test_probabilities_always_sum_to_one(self):
        result = tercile_probabilities([float(v) for v in range(31)], 10.0, 20.0)
        self.assertAlmostEqual(result["below"] + result["normal"] + result["above"], 1.0)

    def test_ignores_members_with_no_value(self):
        result = tercile_probabilities([5.0, None, 25.0, None], 10.0, 20.0)
        self.assertEqual(result["members"], 2)
        self.assertAlmostEqual(result["below"], 0.5)
        self.assertAlmostEqual(result["above"], 0.5)

    def test_returns_nothing_when_no_member_has_a_value(self):
        # None and 33/33/33 look identical on a map and mean opposite things:
        # "we could not forecast" versus "the forecast is climatology".
        self.assertIsNone(tercile_probabilities([None, None], 10.0, 20.0))

    def test_a_dry_forecast_is_normal_where_dry_is_normal(self):
        """Zero rain against a zero p33 is not below-normal.

        In the dry season a third or more of the record is exactly zero, so p33
        is 0.0. Counting `<= p33` as below would paint the north drought-red
        every January for entirely typical weather.
        """
        result = tercile_probabilities([0.0] * 31, 0.0, 4.0)
        self.assertEqual(result["below"], 0.0)
        self.assertAlmostEqual(result["normal"], 1.0)

    def test_flags_a_baseline_with_no_spread(self):
        flat = tercile_probabilities([0.0] * 31, 0.0, 0.0)
        self.assertTrue(flat["degenerate"])
        # And it must not be dressed up as a confident finding.
        self.assertEqual(agreement_confidence(flat), "low")

        real = tercile_probabilities([0.0] * 31, 0.0, 4.0)
        self.assertFalse(real["degenerate"])


class CategoryTests(unittest.TestCase):
    def test_picks_the_winner(self):
        self.assertEqual(dominant_category({"below": 0.6, "normal": 0.3, "above": 0.1}), "below")
        self.assertEqual(dominant_category({"below": 0.1, "normal": 0.2, "above": 0.7}), "above")

    def test_a_tie_is_reported_as_normal(self):
        # A dead heat between drier and wetter is not a forecast of either.
        self.assertEqual(dominant_category({"below": 0.4, "normal": 0.4, "above": 0.2}), "normal")

    def test_confidence_tracks_agreement(self):
        climatology = {"below": 0.34, "normal": 0.33, "above": 0.33, "degenerate": False}
        leaning = {"below": 0.45, "normal": 0.30, "above": 0.25, "degenerate": False}
        decided = {"below": 0.70, "normal": 0.20, "above": 0.10, "degenerate": False}

        # A three-way split floors at about a third. An ensemble sitting there
        # has said nothing and must not be reported as moderate.
        self.assertEqual(agreement_confidence(climatology), "low")
        self.assertEqual(agreement_confidence(leaning), "moderate")
        self.assertEqual(agreement_confidence(decided), "high")


class WindowTests(unittest.TestCase):
    def test_the_window_is_days_14_to_28(self):
        self.assertEqual(LEAD_DAYS, 14)
        self.assertEqual(WINDOW_DAYS, 15)
        # Deliberately no overlap with the 7-Day segment: two forecasts of the
        # same day that disagree is worse than one forecast.
        self.assertEqual(LEAD_DAYS + WINDOW_DAYS - 1, 28)

    def test_sums_only_the_window(self):
        self.assertAlmostEqual(s2s_runtime._reduce_window([1.0] * 40, mean=False), 15.0)

    def test_rejects_a_short_run(self):
        """A truncated ensemble must not be summed as though it were complete.

        ECMWF's ensemble stops at day 14, so this is exactly what the wrong model
        looks like: a partial window silently reported as a full one.
        """
        self.assertIsNone(s2s_runtime._reduce_window([1.0] * 20, mean=False))

    def test_temperature_drops_a_member_with_a_gap(self):
        series: list[float | None] = [30.0] * 40
        series[LEAD_DAYS + 3] = None
        # Zero-filling would invent a 0 degree day in Ghana.
        self.assertIsNone(s2s_runtime._reduce_window(series, mean=True))
        # Rainfall treats the same gap as no recorded rain, which is defensible.
        self.assertAlmostEqual(s2s_runtime._reduce_window(series, mean=False), 30.0 * 14)

    def test_member_series_includes_the_control_run(self):
        daily = {
            "time": [],
            "precipitation_sum": [1.0],
            "precipitation_sum_member01": [2.0],
            "precipitation_sum_member02": [3.0],
            "temperature_2m_max": [30.0],
        }
        series = s2s_runtime._member_series(daily, "precipitation_sum")

        # Control plus two perturbed members. The control belongs to the
        # distribution; dropping it would bias the count.
        self.assertEqual(len(series), 3)
        # And the temperature key must not leak into the rainfall members.
        self.assertNotIn([30.0], series)

    def test_window_dates_start_a_fortnight_out(self):
        issued = date(2026, 8, 24)
        start, end = s2s_runtime.window_dates(issued)
        self.assertEqual(start, (issued + timedelta(days=14)).isoformat())
        self.assertEqual(end, (issued + timedelta(days=28)).isoformat())


def ensemble_payload(members: int = 31, days: int = 30, rain: float = 2.0) -> list[dict]:
    """One entry per grid point, in the array shape a multi-coordinate call returns."""
    entry = {"daily": {"time": [f"2026-08-{d:02d}" for d in range(1, days + 1)]}}
    entry["daily"]["precipitation_sum"] = [rain] * days
    entry["daily"]["temperature_2m_max"] = [31.0] * days
    for index in range(1, members):
        entry["daily"][f"precipitation_sum_member{index:02d}"] = [rain] * days
        entry["daily"][f"temperature_2m_max_member{index:02d}"] = [31.0] * days
    return [dict(entry, daily=dict(entry["daily"])) for _ in ghana_grid_points()]


def fake_climatology() -> dict:
    weeks = {
        f"{week:02d}": {
            "n": 210,
            "rainP33Mm": 10.0,
            "rainP67Mm": 40.0,
            "rainNormalMm": 25.0,
            "tempP33C": 29.0,
            "tempP67C": 33.0,
            "tempNormalC": 31.0,
        }
        for week in range(1, 54)
    }
    return {
        "baseline": "ERA5 1995-2024",
        "cells": {
            cell_id(lat, lng): {"lat": lat, "lng": lng, "weeks": weeks}
            for lat, lng in ghana_grid_points()
        },
    }


class GridTests(unittest.TestCase):
    """The lattice the forecast and its baseline are both computed on.

    Sampling sixteen region centroids is what made a district view dishonest.
    Fetching the model's own field is what makes it honest, so the lattice has to
    be the model's, and the bake and the runtime have to agree on it exactly.
    """

    def test_covers_ghana_at_the_model_resolution(self):
        points = ghana_grid_points()
        # 15 latitudes x 11 longitudes, verified against a live Open-Meteo call.
        self.assertEqual(len(points), 165)

        lats = {lat for lat, _ in points}
        lngs = {lng for _, lng in points}
        self.assertLessEqual(min(lats), GHANA_BOUNDS["minLat"])
        self.assertGreaterEqual(max(lats), GHANA_BOUNDS["maxLat"])
        self.assertLessEqual(min(lngs), GHANA_BOUNDS["minLng"])
        self.assertGreaterEqual(max(lngs), GHANA_BOUNDS["maxLng"])

    def test_every_point_sits_on_the_half_degree_lattice(self):
        # Open-Meteo snaps to this lattice. Asking for points that are already on
        # it means the response needs no interpolation.
        for lat, lng in ghana_grid_points():
            self.assertAlmostEqual(lat * 2, round(lat * 2))
            self.assertAlmostEqual(lng * 2, round(lng * 2))

    def test_cell_ids_are_stable_and_unique(self):
        points = ghana_grid_points()
        self.assertEqual(len({cell_id(*p) for p in points}), len(points))
        # Fixed precision, so 4.5 and 4.500001 cannot become two cells.
        self.assertEqual(cell_id(4.5, -3.5), cell_id(4.500001, -3.499999))

    def test_finds_the_cell_covering_a_town(self):
        points = ghana_grid_points()
        # Accra, 5.6037 / -0.187, snaps to the same cell Open-Meteo returned.
        self.assertEqual(nearest_cell(5.6037, -0.187, points), (5.5, 0.0))

    def test_a_place_the_grid_does_not_reach_has_no_cell(self):
        """Without this, "nearest" always succeeds.

        A request from Kenya would come back with a Ghanaian forecast wearing
        Kenyan coordinates: wrong in the way that looks entirely plausible.
        """
        points = ghana_grid_points()
        self.assertIsNone(nearest_cell(-20, 80, points))
        # ...but a point anywhere inside a real cell still resolves.
        self.assertIsNotNone(nearest_cell(6.72, -1.61, points))

    def test_spread_percentiles_share_one_definition(self):
        # The chart's p10/p90 and the baseline's p33/p67 must not drift apart.
        self.assertEqual(ensemble_percentile([1.0, 2.0, 3.0, 4.0], 1 / 3), quantile([1.0, 2.0, 3.0, 4.0], 1 / 3))


class RouteTests(unittest.TestCase):
    def setUp(self):
        s2s_runtime.reset_cache()

    def tearDown(self):
        s2s_runtime.reset_cache()

    def test_serves_the_whole_field_with_a_complete_split(self):
        with patch.object(s2s_runtime, "CLIMATOLOGY", fake_climatology()), \
             patch.object(s2s_runtime, "_fetch_ensemble", return_value=ensemble_payload()):
            with TestClient(app) as client:
                body = client.get("/api/outlook/subseasonal").json()

        data = body["data"]
        self.assertTrue(body["success"])
        self.assertFalse(data["unavailable"])
        self.assertEqual(len(data["cells"]), len(ghana_grid_points()))

        rainfall = data["cells"][0]["rainfall"]
        probabilities = rainfall["probabilities"]
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=2)
        # 15 days at 2 mm is 30 mm, which sits between the 10/40 boundaries, so
        # every member lands in the middle third.
        self.assertAlmostEqual(probabilities["normal"], 1.0)
        self.assertEqual(rainfall["category"], "normal")
        self.assertEqual(rainfall["members"], 31)

    def test_the_map_payload_carries_no_chart_series(self):
        # 90 numbers per cell per variable across 165 cells is about 100 KB the
        # map never draws. The series endpoint serves it one cell at a time.
        with patch.object(s2s_runtime, "CLIMATOLOGY", fake_climatology()), \
             patch.object(s2s_runtime, "_fetch_ensemble", return_value=ensemble_payload()):
            with TestClient(app) as client:
                data = client.get("/api/outlook/subseasonal").json()["data"]

        self.assertNotIn("series", data["cells"][0]["rainfall"])

    def test_deterministic_survives_a_missing_baseline(self):
        """An ensemble mean needs no climatology, so those cells still draw.

        This is why a missing baseline no longer empties the response: the
        deterministic map is unaffected by it, and only the probabilities go.
        """
        with patch.object(s2s_runtime, "CLIMATOLOGY", {"cells": {}, "baseline": None}), \
             patch.object(s2s_runtime, "_fetch_ensemble", return_value=ensemble_payload()):
            self.assertFalse(s2s_runtime.has_climatology())
            with TestClient(app) as client:
                data = client.get("/api/outlook/subseasonal").json()["data"]

        self.assertFalse(data["unavailable"])
        rainfall = data["cells"][0]["rainfall"]
        # 15 days at 2 mm.
        self.assertAlmostEqual(rainfall["value"], 30.0)
        self.assertNotIn("probabilities", rainfall)

    def test_reports_unavailable_when_nothing_could_be_computed(self):
        with patch.object(s2s_runtime, "CLIMATOLOGY", {"cells": {}, "baseline": None}), \
             patch.object(s2s_runtime, "_fetch_ensemble", return_value=[]):
            with TestClient(app) as client:
                body = client.get("/api/outlook/subseasonal").json()

        data = body["data"]
        self.assertTrue(body["success"])
        self.assertTrue(data["unavailable"])
        self.assertEqual(data["cells"], [])

    def test_serves_one_cell_of_spread_for_the_chart(self):
        with patch.object(s2s_runtime, "CLIMATOLOGY", fake_climatology()), \
             patch.object(s2s_runtime, "_fetch_ensemble", return_value=ensemble_payload()):
            with TestClient(app) as client:
                body = client.get("/api/outlook/subseasonal/series?lat=5.6037&lng=-0.187").json()

        data = body["data"]
        self.assertEqual(data["id"], cell_id(5.5, 0.0))

        series = data["rainfall"]
        self.assertEqual(len(series["mean"]), WINDOW_DAYS)
        # The band must contain the line at every step, or the chart lies.
        for low, mean, high in zip(series["low"], series["mean"], series["high"]):
            self.assertLessEqual(low, mean)
            self.assertLessEqual(mean, high)

    def test_a_place_off_the_grid_is_an_error_not_a_flat_chart(self):
        with patch.object(s2s_runtime, "CLIMATOLOGY", fake_climatology()), \
             patch.object(s2s_runtime, "_fetch_ensemble", return_value=ensemble_payload()):
            with TestClient(app) as client:
                # Somewhere in the Indian Ocean.
                response = client.get("/api/outlook/subseasonal/series?lat=-20&lng=80")

        self.assertEqual(response.status_code, 404)

    def test_carries_its_provenance(self):
        with patch.object(s2s_runtime, "CLIMATOLOGY", fake_climatology()), \
             patch.object(s2s_runtime, "_fetch_ensemble", return_value=ensemble_payload()):
            with TestClient(app) as client:
                data = client.get("/api/outlook/subseasonal").json()["data"]

        # The reader is told which model and which baseline produced this, the
        # same obligation the hazard summary carries.
        self.assertIn("GEFS", data["model"])
        self.assertEqual(data["baseline"], "ERA5 1995-2024")
        self.assertEqual(data["leadDays"], 14)
        self.assertTrue(any(source["id"] == "gefs" for source in data["sources"]))


if __name__ == "__main__":
    unittest.main()
