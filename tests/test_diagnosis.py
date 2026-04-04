from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.database import init_db, set_database_path
from backend.app.diagnosis import (
    format_image_analysis_response,
    is_inconclusive_diagnosis,
    normalize_kindwise_identification,
    resolve_crop_from_plant_id,
)
from backend.app.main import app


VALID_IMAGE = "aGVsbG8="


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
        with patch("backend.app.main.KINDWISE_CROP_HEALTH_API_KEY", ""):
            response = self.client.post("/api/crop-diagnosis", json={"image": VALID_IMAGE, "crop": "maize"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(body["source"], "configuration")

    def test_crop_diagnosis_validates_invalid_image(self):
        with patch("backend.app.main.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"):
            response = self.client.post("/api/crop-diagnosis", json={"image": "not-base64"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(body["source"], "validation")

    def test_crop_diagnosis_uses_crop_health_for_supported_crop(self):
        FakeAsyncClient.payloads = [crop_health_payload()]
        with patch("backend.app.main.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
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
        with patch("backend.app.main.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.main.KINDWISE_PLANT_ID_API_KEY",
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

    def test_crop_diagnosis_returns_unavailable_when_crop_missing_and_plant_id_unconfigured(self):
        with patch("backend.app.main.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.main.KINDWISE_PLANT_ID_API_KEY",
            "",
        ):
            response = self.client.post(
                "/api/crop-diagnosis",
                json={"image": VALID_IMAGE},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(body["source"], "configuration")

    def test_crop_diagnosis_returns_unavailable_for_unmappable_plant_id_result(self):
        FakeAsyncClient.payloads = [
            plant_id_payload(name="Dioscorea alata", common_names=["yam"]),
        ]
        with patch("backend.app.main.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
            "backend.app.main.KINDWISE_PLANT_ID_API_KEY",
            "plant-id-key",
        ), patch(
            "backend.app.diagnosis.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = self.client.post("/api/crop-diagnosis", json={"image": VALID_IMAGE})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(len(FakeAsyncClient.calls), 1)

    def test_image_analysis_rejects_unsupported_type(self):
        response = self.client.post(
            "/api/image-analysis",
            json={"image": VALID_IMAGE, "analysisType": "soil-analysis", "context": {}},
        )
        self.assertEqual(response.status_code, 400)

    def test_image_analysis_returns_chatbot_shape(self):
        FakeAsyncClient.payloads = [crop_health_payload()]
        with patch("backend.app.main.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
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
        with patch("backend.app.main.KINDWISE_CROP_HEALTH_API_KEY", "crop-key"), patch(
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
