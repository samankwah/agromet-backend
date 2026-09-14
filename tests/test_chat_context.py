from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from backend.app import chat_context


class IntentTests(unittest.TestCase):
    """Which sources a question reaches for.

    Crude substring matching, so what is worth pinning is the shape of its
    mistakes: it should over-reach into a cached lookup rather than under-reach
    and answer a rainfall question from memory.
    """

    def test_a_rainfall_question_asks_for_the_weather(self):
        self.assertEqual(chat_context.detect_intents("Is the rain coming this week?"), {"weather"})

    def test_a_planting_question_asks_for_the_weather(self):
        """Planting is a weather question wearing different words: the answer
        turns on when the rains arrive."""
        self.assertIn("weather", chat_context.detect_intents("When should I plant maize?"))

    def test_a_price_question_asks_for_the_market(self):
        self.assertEqual(chat_context.detect_intents("What price is maize fetching?"), {"market"})

    def test_a_flood_question_asks_for_the_hazard_monitor(self):
        self.assertEqual(chat_context.detect_intents("Will the river flood my farm?"), {"hazard"})

    def test_one_question_can_need_two_sources(self):
        intents = chat_context.detect_intents("Should I sell my maize before the rains?")

        self.assertEqual(intents, {"weather", "market"})

    def test_an_unrelated_question_reaches_for_nothing(self):
        self.assertEqual(chat_context.detect_intents("Who was the first president of Ghana?"), set())


class FarmerSectionTests(unittest.TestCase):
    def test_names_the_region_and_its_season(self):
        block = chat_context.render_farmer("Northern", None, ["Maize"])

        self.assertIn("Northern region", block)
        # The north has one rainy season and the south has two. Telling the
        # model which it is doing is the difference between "plant in the minor
        # season" being useful advice and being nonsense.
        self.assertIn("Guinea Savannah", block)
        self.assertIn("one rainy season", block)
        self.assertIn("Maize", block)

    def test_prefers_the_district_when_there_is_one(self):
        block = chat_context.render_farmer("Northern", "Tolon", None)

        self.assertIn("Tolon district, Northern region", block)

    def test_says_when_no_area_is_set_rather_than_guessing(self):
        block = chat_context.render_farmer(None, None, None)

        self.assertIn("not set", block)
        self.assertIn("Ask which district", block)


class WeatherSectionTests(unittest.TestCase):
    BUNDLE = {
        "current": {
            "temperature_2m": 29.4,
            "apparent_temperature": 33.1,
            "relative_humidity_2m": 71,
            "wind_speed_10m": 12.3,
            "precipitation": 0.0,
            "weather_code": 3,
        },
        "daily": {
            "time": ["2026-08-31", "2026-09-01"],
            "temperature_2m_max": [31.2, 30.8],
            "temperature_2m_min": [22.4, 22.9],
            "precipitation_sum": [6.4, 12.1],
            "precipitation_probability_max": [70, 85],
            "weather_code": [63, 80],
        },
    }

    def test_renders_figures_with_the_day_they_belong_to(self):
        block = chat_context.render_weather(self.BUNDLE, "Northern", "09:12 today")

        self.assertIn("Mon 31 Aug", block)
        self.assertIn("rain 6.4 mm", block)
        self.assertIn("70% chance of rain", block)
        self.assertIn("09:12 today", block)

    def test_turns_weather_codes_into_words(self):
        """The model is given "moderate rain", not 63. A raw WMO code is not
        something it can quote to a farmer."""
        block = chat_context.render_weather(self.BUNDLE, "Northern", None)

        self.assertIn("overcast", block)
        self.assertIn("moderate rain", block)
        self.assertNotIn("63", block)

    def test_nothing_to_render_is_nothing_rather_than_an_empty_heading(self):
        self.assertIsNone(chat_context.render_weather(None, "Northern", None))
        self.assertIsNone(chat_context.render_weather({}, "Northern", None))


class HazardSectionTests(unittest.TestCase):
    def test_marks_a_gmet_bulletin_as_gmet(self):
        """The prompt forbids softening a forecaster's advisory, which it can
        only obey if it is told which lines came from a forecaster."""
        block = chat_context.render_hazard(
            {
                "region": "Northern",
                "flood": {
                    "band": "high",
                    "source": "gmet-bulletin",
                    "headline": "Heavy rain in the White Volta basin",
                    "advisories": ["Move livestock to higher ground."],
                },
                "drought": {"band": "low", "source": "open-meteo", "advisories": ["No action needed."]},
            }
        )

        self.assertIn("GMet bulletin in force", block)
        self.assertIn("Flood GMet advisory: Move livestock to higher ground.", block)
        self.assertIn("Drought advice: No action needed.", block)


class MarketSectionTests(unittest.TestCase):
    def test_labels_the_prices_as_indicative(self):
        """These come from a reference table, not a live exchange. A farmer can
        travel a day on a price, so the model must not present it as today's."""
        block = chat_context.render_market(
            [{"slug": "maize", "name": "Maize", "price": 320, "unit": "100kg bag", "trend": "rising", "demand": "high"}]
        )

        self.assertIn("indicative", block)
        self.assertIn("GHS 320 per 100kg bag", block)

    def test_no_rows_render_nothing(self):
        self.assertIsNone(chat_context.render_market([]))


class BlockAssemblyTests(unittest.TestCase):
    def run_block(self, message, **kwargs):
        return asyncio.run(chat_context.build_context_block(message, **kwargs))

    def test_an_unrelated_question_still_gets_the_farmer_and_the_date(self):
        with patch.object(chat_context, "_weather_section") as weather:
            block = self.run_block("Who was the first president of Ghana?", region="Volta")

        weather.assert_not_called()
        self.assertIn("Volta region", block)
        self.assertIn("Today:", block)

    def test_a_source_that_times_out_is_dropped_not_raised(self):
        """An answer without the forecast beats no answer, and both beat the
        platform killing the request while a source thinks about it."""

        async def slow(*args, **kwargs):
            await asyncio.sleep(5)
            return "WEATHER: never arrives"

        with patch.object(chat_context, "_weather_section", slow), patch.object(
            chat_context, "BLOCK_TIMEOUT_SECONDS", 0.05
        ):
            block = self.run_block("Is the rain coming?", region="Northern")

        self.assertIn("Northern region", block)
        self.assertNotIn("never arrives", block)

    def test_a_source_that_raises_is_dropped_not_raised(self):
        async def broken(*args, **kwargs):
            raise RuntimeError("upstream on fire")

        with patch.object(chat_context, "_weather_section", broken):
            block = self.run_block("Is the rain coming?", region="Northern")

        self.assertIn("Northern region", block)

    def test_the_block_tells_the_model_not_to_invent_figures(self):
        block = self.run_block("Hello", region="Ashanti")

        self.assertTrue(block.startswith("DATA"))
        self.assertIn("Do not invent", block)


if __name__ == "__main__":
    unittest.main()
