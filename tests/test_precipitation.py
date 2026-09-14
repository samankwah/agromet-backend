"""The rain map's forecast field: the grid, the transposition, and the cache."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.app import precip_runtime
from backend.app.main import app


def _upstream(cells: int, hours: int, fill=lambda cell, hour: float(hour)):
    """An Open-Meteo style response: one entry per coordinate, each with its own
    hourly series."""
    times = [f"2026-08-27T{hour:02d}:00" for hour in range(hours)]
    return [
        {"hourly": {"time": times, "precipitation": [fill(cell, hour) for hour in range(hours)]}}
        for cell in range(cells)
    ]


class GridTests(unittest.TestCase):
    def test_grid_asset_loads(self):
        # The whole design rests on server and client agreeing about which cells
        # exist, and the asset is generated from the same boundaries the app
        # draws with. An empty grid here means that agreement is broken.
        self.assertGreater(len(precip_runtime.GRID), 100)
        self.assertEqual(precip_runtime.STEP_DEG, 0.25)

    def test_grid_stays_over_ghana(self):
        lats = [lat for lat, _ in precip_runtime.GRID]
        lngs = [lng for _, lng in precip_runtime.GRID]
        self.assertGreaterEqual(min(lats), 4.0)
        self.assertLessEqual(max(lats), 11.5)
        self.assertGreaterEqual(min(lngs), -3.5)
        self.assertLessEqual(max(lngs), 1.5)

    def test_grid_has_no_duplicates(self):
        # A duplicated point costs an upstream location for a cell drawn twice,
        # and the budget in the module header assumes the count is honest.
        self.assertEqual(len(set(precip_runtime.GRID)), len(precip_runtime.GRID))

    def test_budget_stays_inside_the_free_tier(self):
        """The TTL is chosen against Open-Meteo's daily limit, so a grid that
        grew without the interval changing would quietly break the budget."""
        daily = len(precip_runtime.GRID) * (86400 / precip_runtime.CACHE_TTL_SECONDS)
        self.assertLess(daily, 10_000)


class FieldShapeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        precip_runtime.reset_cache()

    def tearDown(self):
        precip_runtime.reset_cache()

    async def test_transposes_to_hours_by_cells(self):
        """Rows must be hours and columns cells, because a frame of the
        animation is one hour across the whole country."""
        cells = len(precip_runtime.GRID)
        response = AsyncMock()
        response.json = lambda: _upstream(cells, 5, fill=lambda cell, hour: float(hour))
        response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)):
            field = await precip_runtime._fetch_field()

        self.assertEqual(len(field["values"]), 5)
        self.assertEqual(len(field["values"][0]), cells)
        # Every cell shares the hour's value in this fixture, so a row that came
        # back varying would mean the axes were swapped.
        self.assertEqual(set(field["values"][3]), {3.0})

    async def test_marks_times_as_utc(self):
        # We ask upstream for UTC, so the stamps are UTC; leaving them unmarked
        # invites the client to read them as local.
        response = AsyncMock()
        response.json = lambda: _upstream(len(precip_runtime.GRID), 2)
        response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)):
            field = await precip_runtime._fetch_field()

        self.assertTrue(all(time.endswith("Z") for time in field["times"]))

    async def test_missing_hour_reads_as_no_rain(self):
        # A gap in a precipitation series is genuinely no recorded rain for that
        # step, which is the one place a zero-fill is honest.
        response = AsyncMock()
        response.json = lambda: _upstream(len(precip_runtime.GRID), 3, fill=lambda cell, hour: None)
        response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)):
            field = await precip_runtime._fetch_field()

        self.assertEqual(set(field["values"][0]), {0.0})

    async def test_refuses_a_response_with_no_hours(self):
        response = AsyncMock()
        response.json = lambda: [{"hourly": {"time": [], "precipitation": []}}]
        response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)):
            with self.assertRaises(RuntimeError):
                await precip_runtime._fetch_field()


class EndpointTests(unittest.TestCase):
    def setUp(self):
        precip_runtime.reset_cache()

    def tearDown(self):
        precip_runtime.reset_cache()

    def test_serves_the_field_and_its_grid(self):
        cells = len(precip_runtime.GRID)
        response = AsyncMock()
        response.json = lambda: _upstream(cells, 4)
        response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)) as fetch:
            with TestClient(app) as client:
                body = client.get("/api/precipitation/field").json()["data"]

        self.assertFalse(body["unavailable"])
        # The grid travels with the values because the two are positional.
        self.assertEqual(len(body["grid"]), cells)
        self.assertEqual(len(body["values"][0]), cells)
        self.assertEqual(len(body["times"]), 4)
        self.assertEqual(fetch.await_count, 1)

    def test_second_call_is_served_from_cache(self):
        """The whole reason this endpoint exists: one upstream call serves
        everyone, rather than one per device."""
        response = AsyncMock()
        response.json = lambda: _upstream(len(precip_runtime.GRID), 2)
        response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient.get", AsyncMock(return_value=response)) as fetch:
            with TestClient(app) as client:
                client.get("/api/precipitation/field")
                client.get("/api/precipitation/field")
                client.get("/api/precipitation/field")

        self.assertEqual(fetch.await_count, 1)

    def test_upstream_failure_degrades_rather_than_erroring(self):
        # The map should say the forecast half is missing, not return a 500 that
        # takes the satellite half down with it.
        with patch("httpx.AsyncClient.get", AsyncMock(side_effect=RuntimeError("upstream down"))):
            with TestClient(app) as client:
                payload = client.get("/api/precipitation/field")

        self.assertEqual(payload.status_code, 200)
        body = payload.json()["data"]
        self.assertTrue(body["unavailable"])
        self.assertEqual(body["values"], [])
        self.assertIsNotNone(body["error"])


if __name__ == "__main__":
    unittest.main()
