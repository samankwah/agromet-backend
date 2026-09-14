from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import weather_runtime
from backend.app.main import app


ACCRA = (5.6037, -0.187)


def open_meteo_payload(temperature: float = 25.5) -> dict:
    """The shape Open-Meteo actually returns, trimmed to two hours.

    Times carry no zone suffix — that is Open-Meteo's own format when
    `timezone=Africa/Accra` is requested, and the reason the client has to
    append one before parsing.
    """
    return {
        "latitude": 5.6,
        "longitude": -0.1875,
        "timezone": "Africa/Accra",
        "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
        "current": {
            "time": "2026-08-22T18:30",
            "temperature_2m": temperature,
            "relative_humidity_2m": 83,
            "apparent_temperature": 28.3,
            "precipitation": 0.0,
            "weather_code": 3,
            "wind_speed_10m": 16.0,
        },
        "daily": {
            "time": ["2026-08-22", "2026-08-23"],
            "weather_code": [53, 3],
            "temperature_2m_max": [28.7, 29.1],
            "temperature_2m_min": [24.3, 24.0],
            "precipitation_sum": [1.8, 0.0],
            "precipitation_probability_max": [100, 40],
            "sunrise": ["2026-08-22T05:55", "2026-08-23T05:55"],
            "sunset": ["2026-08-22T18:11", "2026-08-23T18:11"],
            "wind_speed_10m_max": [18.8, 17.2],
        },
        "hourly": {
            "time": ["2026-08-22T00:00", "2026-08-22T01:00"],
            "temperature_2m": [25.5, 25.2],
            "apparent_temperature": [28.9, 28.4],
            "relative_humidity_2m": [88, 89],
            "precipitation": [0.0, 0.2],
            "precipitation_probability": [37, 41],
            "weather_code": [53, 61],
            "wind_speed_10m": [11.2, 10.8],
            "uv_index": [0.0, 0.0],
        },
    }


class WeatherCacheTests(unittest.TestCase):
    """The cache is the whole point of proxying — without it every carousel
    scroll would be ten upstream requests."""

    def setUp(self):
        weather_runtime.reset_cache()

    def test_second_read_is_served_from_cache(self):
        with patch.object(weather_runtime, "_fetch", return_value=open_meteo_payload()) as fetch:
            asyncio.run(weather_runtime.ensure_fresh(*ACCRA))
            asyncio.run(weather_runtime.ensure_fresh(*ACCRA))

        self.assertEqual(fetch.await_count, 1)

    def test_nearby_coordinates_share_one_entry(self):
        # Two GPS fixes metres apart must not fragment the cache.
        self.assertEqual(weather_runtime.cache_key(5.6037, -0.187), weather_runtime.cache_key(5.6041, -0.1874))
        # Two different towns must not collide.
        self.assertNotEqual(weather_runtime.cache_key(*ACCRA), weather_runtime.cache_key(9.4034, -0.8424))

    def test_expiry_forces_a_refetch(self):
        key = weather_runtime.cache_key(*ACCRA)
        with patch.object(weather_runtime, "_fetch", return_value=open_meteo_payload()) as fetch:
            asyncio.run(weather_runtime.ensure_fresh(*ACCRA))
            self.assertFalse(weather_runtime.is_stale(key))

            # Wind the clock past the TTL rather than sleeping through it.
            weather_runtime._CACHE_STAMPS[key] -= weather_runtime.CACHE_TTL_SECONDS + 1
            self.assertTrue(weather_runtime.is_stale(key))

            asyncio.run(weather_runtime.refresh(*ACCRA))
            self.assertEqual(fetch.await_count, 2)

    def test_upstream_failure_degrades_rather_than_raising(self):
        with patch.object(weather_runtime, "_fetch", side_effect=RuntimeError("upstream down")):
            stored = asyncio.run(weather_runtime.refresh(*ACCRA, force=True))

        self.assertFalse(stored)
        self.assertIsNone(weather_runtime.cached_bundle(weather_runtime.cache_key(*ACCRA)))
        self.assertIn("upstream down", weather_runtime.metadata(weather_runtime.cache_key(*ACCRA))["error"])

    def test_a_failed_refresh_keeps_the_last_good_bundle(self):
        # Stale data beats no data: the farmer sees this morning's forecast
        # rather than an error page.
        with patch.object(weather_runtime, "_fetch", return_value=open_meteo_payload(26.0)):
            asyncio.run(weather_runtime.ensure_fresh(*ACCRA))

        key = weather_runtime.cache_key(*ACCRA)
        weather_runtime._CACHE_STAMPS[key] -= weather_runtime.CACHE_TTL_SECONDS + 1

        with patch.object(weather_runtime, "_fetch", side_effect=RuntimeError("upstream down")):
            asyncio.run(weather_runtime.refresh(*ACCRA))

        self.assertEqual(weather_runtime.cached_bundle(key)["current"]["temperature_2m"], 26.0)


class WeatherRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        weather_runtime.reset_cache()

    def test_serves_the_bundle_in_the_house_envelope(self):
        with patch.object(weather_runtime, "_fetch", return_value=open_meteo_payload()):
            response = self.client.get("/api/weather/bundle", params={"lat": ACCRA[0], "lng": ACCRA[1]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertFalse(body["unavailable"])
        self.assertEqual(body["meta"]["source"], "Open-Meteo")
        # Open-Meteo's own shape, passed through — the client owns the mapping.
        self.assertIn("current", body["data"])
        self.assertIn("daily", body["data"])
        self.assertIn("hourly", body["data"])
        self.assertEqual(body["data"]["current"]["temperature_2m"], 25.5)

    def test_reports_unavailable_instead_of_failing(self):
        # The client falls back to calling Open-Meteo itself, so this must be a
        # readable 200 rather than a 502 it has to interpret.
        with patch.object(weather_runtime, "_fetch", side_effect=RuntimeError("upstream down")):
            response = self.client.get("/api/weather/bundle", params={"lat": ACCRA[0], "lng": ACCRA[1]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertTrue(body["unavailable"])
        self.assertIsNone(body["data"])

    def test_requires_coordinates(self):
        self.assertEqual(self.client.get("/api/weather/bundle").status_code, 422)


if __name__ == "__main__":
    unittest.main()
