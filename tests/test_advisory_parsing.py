from __future__ import annotations

import unittest

from backend.app.spreadsheet_parser import _build_agromet_preview, _build_poultry_preview


PARAMETERS = [
    "[RAINFALL]",
    "[TEMP]",
    "[HUMIDITY]",
    "[SOIL MOISTURE]",
    "[SOIL TEMP]",
    "[SUNSHINE INTENSITY]",
    "[SUNRISE]",
    "[SUNSET]",
    "[EVAPO-TRANSP.]",
]


def sheet_template_rows(week: str, crop: str = "CT0000000001/Broiler") -> list[list[str]]:
    """One worksheet in the layout every real district bulletin uses.

    Taken from `Jasikan Broiler exceltemplate.xlsx` — crop and poultry bulletins
    are authored on this same template, which is the whole reason the two
    preview builders were merged.
    """
    return [
        ["[ZONE]", "[REGION]", "[DISTRICT]", "[MONTH/YEAR]", "[WEEK]", "[START DATE]", "[END DATE]", "[CROP]"],
        ["Forest", "REG01/Oti", "DS001/Jasikan", "", week, "", "", crop],
        [],
        ["", *PARAMETERS],
        ["[FORECAST]", "60% occurrence", "25/33OC", "0.8", "-", "-", "9 hours", "06:00", "18:10", "3.5-4.0 mm/day"],
        ["[IMPLICATION]", "Low rainfall expected", "Suitable temperature", "Moderate to high effects", "-", "-", "High day light", "Early sunrise", "Late sunset", "Moderate water loss"],
        ["[ADVISORY]", "Ensure access to potable and cool water", "Suitable temperature for brooding", "Ensure minimum ventilation", "-", "-", "Suitable sunlight", "Begin farm activity early", "-", "Monitor drinkers"],
        ["[FORECAST AND ADVISORY TITLE]", "OVERALL SUMMARY WEATHER OUTLOOK & ADVISORY"],
        ["[FORECAST AND ADVISORY BODY]", "During brooding ensure the following, adequate heat and dry litter."],
        ["[SMS/TEXT]", "During brooding, ensure optimal temperature."],
    ]


def generated_template_rows() -> list[list[str]]:
    """The *older* shape, from templateGenerationService's downloadable file.

    No forecast band; management parameters and recommendations as labelled
    sections. Some districts uploaded these, so the parser still has to read them.
    """
    return [
        ["[MANAGEMENT PARAMETERS]"],
        ["Parameter", "Recommended Value"],
        ["Brooding temperature", "32C"],
        ["Stocking density", "10 birds per square metre"],
        [],
        ["[RECOMMENDATION]"],
        ["Check the brooder guard twice daily."],
        ["Keep litter dry and friable."],
    ]


def make_sheet(name: str, rows: list[list[str]]) -> dict:
    """The subset of `parse_xlsx_workbook`'s sheet shape the builders touch."""
    return {
        "name": name,
        "rawRows": rows,
        "rows": rows,
        "headers": rows[0] if rows else [],
        "totalRows": len(rows),
        "filledCells": sum(1 for row in rows for cell in row if cell),
        "colors": [],
        "sections": [],
    }


class PoultryAdvisoryPreviewTests(unittest.TestCase):
    """A poultry bulletin on the district template must parse into activities.

    Before these, poultry uploads went to a builder that only understood the
    generated template, so a real bulletin produced bare sheet names and the app
    had nothing to render.
    """

    def test_parses_each_worksheet_into_an_activity(self):
        sheets = [
            make_sheet("Brooder Management", sheet_template_rows("1-  2")),
            make_sheet("Grower - Finisher Diet", sheet_template_rows("5-  8")),
        ]

        preview = _build_poultry_preview(sheets, {"poultryTypeCode": "PT001", "title": "Jasikan Broiler"})

        self.assertEqual(len(preview["parsedActivities"]), 2)
        self.assertEqual(
            [entry["activity"] for entry in preview["activities"]],
            ["Brooder Management", "Grower - Finisher Diet"],
        )

    def test_keeps_the_nine_parameters_in_spreadsheet_order(self):
        sheets = [make_sheet("Brooder Management", sheet_template_rows("1-  2"))]

        parsed = _build_poultry_preview(sheets, {})["parsedActivities"][0]

        self.assertEqual(
            parsed["weatherParameters"],
            ["RAINFALL", "TEMP", "HUMIDITY", "SOIL MOISTURE", "SOIL TEMP", "SUNSHINE INTENSITY", "SUNRISE", "SUNSET", "EVAPO-TRANSP."],
        )
        self.assertEqual(parsed["forecast"]["RAINFALL"], "60% occurrence")
        self.assertEqual(parsed["implication"]["TEMP"], "Suitable temperature")
        self.assertEqual(parsed["advisory"]["HUMIDITY"], "Ensure minimum ventilation")
        self.assertIn("adequate heat", parsed["summaryBody"])

    def test_week_labels_are_never_coerced_to_numbers(self):
        # Layer bulletins carry prose here — "1 - End", "From point of lay to end
        # of production". Anything that tries to make a number of those loses the
        # only week information the sheet has.
        sheets = [
            make_sheet("Feeding & Watering", sheet_template_rows("1 - End", crop="CT0000000001/Layer")),
            make_sheet("Harvesting and Storage", sheet_template_rows("From point of lay to end of production", crop="CT0000000001/Layer")),
        ]

        preview = _build_poultry_preview(sheets, {})

        self.assertEqual(
            [entry["weekLabel"] for entry in preview["activities"]],
            ["1 - End", "From point of lay to end of production"],
        )

    def test_reads_the_bird_and_place_off_the_sheet_when_metadata_is_thin(self):
        sheets = [make_sheet("Brooder Management", sheet_template_rows("1-  2"))]

        preview = _build_poultry_preview(sheets, {})

        self.assertEqual(preview["crop"], "CT0000000001/Broiler")
        self.assertEqual(preview["regionCode"], "REG01/Oti")
        self.assertEqual(preview["districtCode"], "DS001/Jasikan")

    def test_still_reads_the_older_generated_template(self):
        sheets = [make_sheet("BROODING MANAGEMENT", generated_template_rows())]

        preview = _build_poultry_preview(sheets, {})

        self.assertEqual(preview["managementMetrics"]["Brooding temperature"], "32C")
        self.assertEqual(preview["parsedActivities"], [])
        # With no parsed sheets the metrics ride in the weather-forecast column,
        # which is where the client looks for them.
        self.assertEqual(preview["weatherForecast"], preview["managementMetrics"])
        self.assertTrue(any("Check the brooder guard" in entry.get("text", "") for entry in preview["advisories"]))

    def test_a_parsed_bulletin_puts_real_parameters_in_the_forecast_column(self):
        sheets = [make_sheet("Brooder Management", sheet_template_rows("1-  2"))]

        forecast = _build_poultry_preview(sheets, {})["weatherForecast"]

        self.assertEqual(forecast["source"], "spreadsheet")
        self.assertIn("RAINFALL", forecast["parameters"])

    def test_year_is_carried_through(self):
        sheets = [make_sheet("Brooder Management", sheet_template_rows("1-  2"))]

        # Arrives as a string from multipart form data.
        self.assertEqual(_build_poultry_preview(sheets, {"year": "2026"})["year"], 2026)
        self.assertIsNone(_build_poultry_preview(sheets, {})["year"])


class AgrometAdvisoryPreviewTests(unittest.TestCase):
    """The crop path must be unchanged by the merge."""

    def test_parses_every_sheet_and_keeps_the_commodity_code(self):
        sheets = [make_sheet("Land preparation", sheet_template_rows("1-4", crop="CM001/Maize"))]

        preview = _build_agromet_preview(sheets, {"commodityCode": "CM001"})

        self.assertEqual(len(preview["parsedActivities"]), 1)
        self.assertEqual(preview["commodityCode"], "CM001")
        self.assertEqual(preview["weatherForecast"]["source"], "spreadsheet")
        self.assertNotIn("managementMetrics", preview)
        self.assertNotIn("poultryTypeCode", preview)

    def test_a_sheet_without_a_forecast_band_still_becomes_an_activity(self):
        # Crop has no second shape to fall back to, so a thin sheet must still be
        # recorded rather than dropped.
        sheets = [make_sheet("Notes", generated_template_rows())]

        preview = _build_agromet_preview(sheets, {})

        self.assertEqual(len(preview["parsedActivities"]), 1)
        self.assertEqual(preview["activities"][0]["activity"], "Notes")


if __name__ == "__main__":
    unittest.main()
