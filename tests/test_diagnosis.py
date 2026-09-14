from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.database import init_db, set_database_path
from backend.app.diagnosis import (
    build_unavailable_diagnosis,
    format_image_analysis_response,
    is_inconclusive_diagnosis,
    normalize_kindwise_identification,
    resolve_crop_from_plant_id,
)
from backend.app.main import app


# A real 8x8 JPEG, base64 encoded.
#
# This was five bytes of the ASCII text "hello" until it was noticed that
# nothing in the request path ever decodes an image, so a fixture named
# VALID_IMAGE had never been one. It is a genuine JPEG now so that the name is
# true and so that adding real image validation later does not require rewriting
# every test in this file. See test_non_image_base64_is_currently_accepted for
# the gap itself.
VALID_IMAGE = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8t"
    "MC0oMCUoKSgBBwcHCggKEwoKEygaFhooKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKP/AABEIAAgA"
    "CAMBEQACEQEDEQH/xAGiAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgsQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQy"
    "gZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SV"
    "lpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+gEAAwEBAQEBAQEBAQAAAAAAAAEC"
    "AwQFBgcICQoLEQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3"
    "ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS"
    "09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2gAMAwEAAhEDEQA/AOUr5c+NP//Z"
)

#: Decodable base64 that is not an image at all. Five bytes of "hello".
NON_IMAGE_BASE64 = "aGVsbG8="


def crop_health_payload(probability: float = 0.82, disease_name: str = "Maize streak virus"):
    return {
        "result": {
            "disease": {
                "suggestions": [
                    {
                        "name": disease_name,
                        "probability": probability,
                        "details": {
                            "common_name": disease_name,
                            "description": "Visible streaking on leaf tissue.",
                            "symptoms": ["yellow streaks", "stunted growth"],
                            "severity": "moderate",
                            "treatment": {
                                "prevention": ["Control insect vectors", "Use resistant seed"],
                            },
                        },
                    }
                ]
            }
        }
    }


def plant_id_payload(name: str = "Zea mays", probability: float = 0.9, common_names: list[str] | None = None):
    return {
        "result": {
            "is_plant": {"binary": 0.99, "probability": 0.99},
            "classification": {
                "suggestions": [
                    {
                        "name": name,
                        "probability": probability,
                        "details": {
                            "common_names": common_names or ["maize", "corn"],
                            "taxonomy": {"genus": "Zea", "species": "mays"},
                        },
                    }
                ]
            },
        }
    }


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeAsyncClient:
    payloads = []
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        FakeAsyncClient.calls.append({"url": url, "kwargs": kwargs})
        if not FakeAsyncClient.payloads:
            raise AssertionError("No fake payloads configured for FakeAsyncClient")
        return FakeResponse(FakeAsyncClient.payloads.pop(0))


class DiagnosisNormalizationTests(unittest.TestCase):
    def test_normalize_kindwise_identification_maps_fields(self):
        result = normalize_kindwise_identification(
            crop_health_payload(),
            fallback_crop="maize",
            provider_product="crop.health",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["providerProduct"], "crop.health")
        self.assertEqual(result["plant"], "maize")
        self.assertEqual(result["severity"], "moderate")

    def test_treatment_parts_split_by_when_you_act_on_them(self):
        """Biological and chemical are both "the infection is here now"; prevention
        is for next season. A client showing "do this now" wants only the former."""
        payload = crop_health_payload()
        payload["result"]["disease"]["suggestions"][0]["details"]["treatment"] = {
            "biological": ["Remove and burn infected plants"],
            "chemical": ["Apply a registered foliar spray"],
            "prevention": ["Use resistant seed", "Control insect vectors"],
        }
        result = normalize_kindwise_identification(
            payload, fallback_crop="maize", provider_product="crop.health"
        )

        self.assertEqual(
            result["treatmentParts"]["immediate"],
            ["Remove and burn infected plants", "Apply a registered foliar spray"],
        )
        self.assertEqual(
            result["treatmentParts"]["prevention"],
            ["Use resistant seed", "Control insect vectors"],
        )

    def test_treatment_parts_leave_remedy_untouched(self):
        """The web client reads `remedy`. Splitting the advice for a second client
        must not change the first one's answer."""
        result = normalize_kindwise_identification(
            crop_health_payload(), fallback_crop="maize", provider_product="crop.health"
        )

        self.assertIn("Prevention:", result["remedy"])
        self.assertIn("Control insect vectors", result["remedy"])

    def test_treatment_parts_survive_a_provider_that_sends_prose(self):
        """Kindwise sometimes returns `treatment` as a bare string. There is no
        structure to recover, so the parts are empty rather than guessed at by
        splitting the prose on full stops."""
        payload = crop_health_payload()
        payload["result"]["disease"]["suggestions"][0]["details"]["treatment"] = (
            "Spray weekly. Remove volunteers."
        )
        result = normalize_kindwise_identification(
            payload, fallback_crop="maize", provider_product="crop.health"
        )

        self.assertEqual(result["treatmentParts"], {"immediate": [], "prevention": []})
        self.assertEqual(result["remedy"], "Spray weekly. Remove volunteers.")

    def test_unavailable_diagnosis_still_carries_the_key(self):
        # So a consumer can read treatmentParts without branching on status first.
        result = build_unavailable_diagnosis("No provider configured.")

        self.assertEqual(result["treatmentParts"], {"immediate": [], "prevention": []})

    def test_inconclusive_diagnosis_detects_low_confidence(self):
        result = normalize_kindwise_identification(
            crop_health_payload(probability=0.12, disease_name="Unknown"),
            fallback_crop="maize",
            provider_product="crop.health",
        )
        self.assertTrue(is_inconclusive_diagnosis(result))

    def test_format_image_analysis_response_maps_fields(self):
        formatted = format_image_analysis_response(
            {
                "status": "ok",
                "source": "kindwise",
                "providerProduct": "crop.health",
                "identificationProduct": "plant.id",
                "plant": "Cassava",
                "disease": "Cassava mosaic virus",
                "confidence": 0.61,
                "severity": "moderate",
                "remedy": "Use clean planting material.",
                "evidence": ["mosaic leaf pattern"],
                "notes": ["Confirm with field symptoms"],
                "disclaimer": "Decision support only.",
            }
        )
        self.assertEqual(formatted["analysis"]["identified_disease"], "Cassava mosaic virus")
        self.assertEqual(formatted["providerProduct"], "crop.health")
        self.assertEqual(formatted["identificationProduct"], "plant.id")
        self.assertEqual(formatted["status"], "ok")

    def test_resolve_crop_from_plant_id_maps_supported_alias(self):
        resolved = resolve_crop_from_plant_id(plant_id_payload(common_names=["maize", "corn"]))
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["crop"], "corn")

    def test_resolve_crop_from_plant_id_returns_none_for_unsupported_crop(self):
        resolved = resolve_crop_from_plant_id(
            plant_id_payload(name="Dioscorea alata", common_names=["yam"])
        )
        self.assertIsNotNone(resolved)
        self.assertIsNone(resolved["crop"])


class DiagnosisApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._db_dir = tempfile.TemporaryDirectory()
        cls._db_path = os.path.join(cls._db_dir.name, "test-agromet.db")
        set_database_path(cls._db_path)
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls._db_dir.cleanup()

    def setUp(self):
        FakeAsyncClient.payloads = []
        FakeAsyncClient.calls = []

    def test_crop_diagnosis_returns_unavailable_without_crop_health_key(self):
        with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", ""):
            response = self.client.post("/api/crop-diagnosis", json={"image": VALID_IMAGE, "crop": "maize"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(body["source"], "configuration")

    def test_crop_diagnosis_validates_invalid_image(self):
        with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"):
            response = self.client.post("/api/crop-diagnosis", json={"image": "not-base64"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(body["source"], "validation")

    def test_non_image_base64_is_currently_accepted(self):
        """Documents a gap rather than endorsing it.

        `_is_valid_base64_image` checks only that the string decodes as base64.
        It never checks that the bytes are an image, so five bytes of the text
        "hello" reach the provider as though they were a photograph, and the
        request is spent finding that out.

        Asserting the current behaviour keeps it visible and makes this test
        fail the moment real validation is added, which is the point at which
        somebody should come back and turn this into an assertion that the
        request was rejected.
        """
        FakeAsyncClient.payloads = [crop_health_payload()]
        with patch("backend.app.diagnosis.httpx.AsyncClient", FakeAsyncClient):
            with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"):
                response = self.client.post(
                    "/api/crop-diagnosis",
                    json={"image": NON_IMAGE_BASE64, "crop": "maize"},
                )

        self.assertEqual(response.status_code, 200)
        # It got as far as calling the provider, which is the gap.
        self.assertEqual(len(FakeAsyncClient.calls), 1)

    def test_crop_diagnosis_uses_crop_health_for_supported_crop(self):
        FakeAsyncClient.payloads = [crop_health_payload()]
        with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.diagnosis.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = self.client.post(
                "/api/crop-diagnosis",
                json={"image": VALID_IMAGE, "crop": "maize", "region": "Northern"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["providerProduct"], "crop.health")
        self.assertIsNone(body["identificationProduct"])
        self.assertEqual(len(FakeAsyncClient.calls), 1)
        self.assertIn("crop.kindwise.com", FakeAsyncClient.calls[0]["url"])

    def test_crop_diagnosis_uses_plant_id_when_crop_missing(self):
        FakeAsyncClient.payloads = [
            plant_id_payload(),
            crop_health_payload(),
        ]
        with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.config.KINDWISE_PLANT_ID_API_KEY",
            "plant-id-key",
        ), patch(
            "backend.app.diagnosis.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = self.client.post(
                "/api/crop-diagnosis",
                json={"image": VALID_IMAGE, "region": "Northern"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["providerProduct"], "crop.health")
        self.assertEqual(body["identificationProduct"], "plant.id")
        self.assertEqual(len(FakeAsyncClient.calls), 2)
        self.assertIn("api.plant.id", FakeAsyncClient.calls[0]["url"])
        self.assertIn("crop.kindwise.com", FakeAsyncClient.calls[1]["url"])

    def test_crop_diagnosis_uses_crop_health_when_crop_missing_and_plant_id_unconfigured(self):
        FakeAsyncClient.payloads = [crop_health_payload()]
        with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.config.KINDWISE_PLANT_ID_API_KEY",
            "",
        ), patch(
            "backend.app.diagnosis.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = self.client.post(
                "/api/crop-diagnosis",
                json={"image": VALID_IMAGE},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["providerProduct"], "crop.health")
        self.assertIsNone(body["identificationProduct"])
        self.assertEqual(len(FakeAsyncClient.calls), 1)
        self.assertIn("crop.kindwise.com", FakeAsyncClient.calls[0]["url"])

    def test_crop_diagnosis_uses_crop_health_for_unmappable_plant_id_result(self):
        FakeAsyncClient.payloads = [
            plant_id_payload(name="Dioscorea alata", common_names=["yam"]),
            crop_health_payload(),
        ]
        with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.config.KINDWISE_PLANT_ID_API_KEY",
            "plant-id-key",
        ), patch(
            "backend.app.diagnosis.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = self.client.post("/api/crop-diagnosis", json={"image": VALID_IMAGE})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["providerProduct"], "crop.health")
        self.assertIsNone(body["identificationProduct"])
        self.assertEqual(len(FakeAsyncClient.calls), 2)
        called_urls = [call["url"] for call in FakeAsyncClient.calls]
        self.assertTrue(any("api.plant.id" in url for url in called_urls))
        self.assertTrue(any("crop.kindwise.com" in url for url in called_urls))

    def test_image_analysis_rejects_unsupported_type(self):
        response = self.client.post(
            "/api/image-analysis",
            json={"image": VALID_IMAGE, "analysisType": "soil-analysis", "context": {}},
        )
        self.assertEqual(response.status_code, 400)

    def test_image_analysis_returns_chatbot_shape(self):
        FakeAsyncClient.payloads = [crop_health_payload()]
        with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.diagnosis.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = self.client.post(
                "/api/image-analysis",
                json={"image": VALID_IMAGE, "analysisType": "disease-detection", "context": {"crop": "maize"}},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("analysis", body)
        self.assertEqual(body["providerProduct"], "crop.health")

    def test_authenticated_crop_diagnosis_creates_history(self):
        register = self.client.post(
            "/api/v1/auth/register",
            json={"email": "diag@example.com", "password": "password123", "name": "Diag User"},
        )
        self.assertEqual(register.status_code, 200)
        login = self.client.post(
            "/api/v1/auth/login",
            data={"username": "diag@example.com", "password": "password123"},
        )
        token = login.json()["access_token"]

        FakeAsyncClient.payloads = [crop_health_payload()]
        with patch("backend.app.config.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.diagnosis.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = self.client.post(
                "/api/crop-diagnosis",
                json={"image": VALID_IMAGE, "crop": "maize"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("historyId", body)

        history = self.client.get(
            "/api/diagnosis-history",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(history.status_code, 200)
        history_body = history.json()
        self.assertEqual(history_body["total"], 1)


if __name__ == "__main__":
    unittest.main()
