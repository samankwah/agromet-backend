from __future__ import annotations

import asyncio
import re
from base64 import b64decode
from binascii import Error as BinasciiError
from typing import Any

import logging

import httpx

logger = logging.getLogger(__name__)


SUPPORTED_IMAGE_ANALYSIS_TYPES = {"disease-detection"}
DEFAULT_DIAGNOSIS_DISCLAIMER = (
    "This result is decision support only. Confirm visible symptoms with a qualified agricultural extension officer before treatment."
)

CROP_HEALTH_SUPPORTED_CROPS = {
    "apple",
    "banana",
    "barley",
    "cassava",
    "citrus",
    "cocoa",
    "coffee",
    "corn",
    "cotton",
    "cucumber",
    "eggplant",
    "garlic",
    "grapevine",
    "oil palm",
    "onion",
    "potato",
    "rice",
    "soybean",
    "sugarcane",
    "tea",
    "tobacco",
    "tomato",
    "wheat",
}

CROP_SYNONYMS = {
    "banana": "banana",
    "capsicum": "eggplant",
    "cassava": "cassava",
    "cocoa": "cocoa",
    "coffee": "coffee",
    "corn": "corn",
    "cowpea": "soybean",
    "eggplant": "eggplant",
    "groundnut": "soybean",
    "groundnuts": "soybean",
    "maize": "corn",
    "okra": None,
    "oil palm": "oil palm",
    "onions": "onion",
    "pepper": "eggplant",
    "plantain": "banana",
    "potatoes": "potato",
    "rice": "rice",
    "soybeans": "soybean",
    "sugar cane": "sugarcane",
    "tomatoes": "tomato",
    "yam": None,
}

KINDWISE_DETAILS = ",".join(
    [
        "common_names",
        "taxonomy",
        "url",
        "description",
        "treatment",
        "symptoms",
        "severity",
        "type",
        "eppo_code",
        "eppo_regulation_status",
    ]
)

PLANT_ID_DETAILS = ",".join(
    [
        "common_names",
        "url",
        "description",
        "taxonomy",
        "synonyms",
    ]
)


def build_unavailable_diagnosis(message: str, source: str = "unavailable") -> dict[str, Any]:
    return {
        "status": "unavailable",
        "isAvailable": False,
        "plant": "Analysis unavailable",
        "disease": "No verified diagnosis",
        "remedy": message,
        "confidence": None,
        "source": source,
        "providerProduct": None,
        "identificationProduct": None,
        "evidence": [],
        "severity": "unknown",
        "notes": [],
        "disclaimer": DEFAULT_DIAGNOSIS_DISCLAIMER,
    }


def _extract_http_error_message(exc: Exception, fallback_message: str) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            payload = exc.response.json()
        except ValueError:
            payload = exc.response.text

        if isinstance(payload, dict):
            for key in ("detail", "message", "error"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        return fallback_message

    return str(exc) or fallback_message


def normalize_crop_name(crop: str | None) -> str | None:
    if not crop:
        return None
    normalized = str(crop).strip().lower()
    return CROP_SYNONYMS.get(normalized, normalized)


def should_use_crop_health(crop: str | None) -> bool:
    normalized = normalize_crop_name(crop)
    return bool(normalized and normalized in CROP_HEALTH_SUPPORTED_CROPS)


def is_inconclusive_diagnosis(result: dict[str, Any]) -> bool:
    confidence = result.get("confidence")
    if confidence is None:
        return True
    if confidence < 0.45:
        return True
    disease = str(result.get("disease") or "").strip().lower()
    return disease in {
        "",
        "healthy",
        "no verified diagnosis",
        "unknown",
        "likely issue not identified",
    }


def _clamp_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1:
        confidence /= 100.0
    return max(0.0, min(confidence, 1.0))


def _normalize_severity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "mild"}:
        return "low"
    if text in {"moderate", "medium"}:
        return "moderate"
    if text in {"high", "severe"}:
        return "high"
    return "unknown"


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _is_valid_base64_image(image: str) -> bool:
    if not image or not isinstance(image, str):
        return False
    payload = image.split(",", 1)[1] if image.startswith("data:") and "," in image else image
    try:
        b64decode(payload, validate=True)
        return True
    except (BinasciiError, ValueError):
        return False


def _extract_first_suggestion(payload: dict[str, Any]) -> dict[str, Any] | None:
    result = payload.get("result") or {}
    suggestions = result.get("disease", {}).get("suggestions")
    if isinstance(suggestions, list) and suggestions:
        return suggestions[0]
    suggestions = result.get("suggestions")
    if isinstance(suggestions, list) and suggestions:
        return suggestions[0]
    return None


def _extract_treatment_text(details: dict[str, Any]) -> str:
    treatment = details.get("treatment")
    if isinstance(treatment, str) and treatment.strip():
        return treatment.strip()
    if isinstance(treatment, dict):
        parts: list[str] = []
        for key in ("prevention", "biological", "chemical"):
            values = _normalize_string_list(treatment.get(key))
            if values:
                parts.append(f"{key.title()}: " + ". ".join(
                    v.rstrip(".") for v in values
                ) + ".")
        if parts:
            return "\n\n".join(parts)
    description = details.get("description") or details.get("wiki_description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return "No treatment recommendation is available."


def _extract_evidence(details: dict[str, Any]) -> list[str]:
    evidence = _normalize_string_list(details.get("symptoms"))
    if evidence:
        return evidence
    description = details.get("description")
    if isinstance(description, str) and description.strip():
        return [description.strip()]
    return []


def _extract_plant_id_suggestion(payload: dict[str, Any]) -> dict[str, Any] | None:
    result = payload.get("result") or {}
    classification = result.get("classification") or {}
    suggestions = classification.get("suggestions")
    if isinstance(suggestions, list) and suggestions:
        return suggestions[0]
    suggestions = result.get("suggestions")
    if isinstance(suggestions, list) and suggestions:
        return suggestions[0]
    return None


def _candidate_crop_names_from_plant_id(suggestion: dict[str, Any]) -> list[str]:
    details = suggestion.get("details") or {}
    candidates: list[str] = []

    for value in (
        suggestion.get("name"),
        details.get("common_name"),
        details.get("local_name"),
    ):
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for value in (
        details.get("common_names"),
        details.get("synonyms"),
    ):
        candidates.extend(_normalize_string_list(value))

    taxonomy = details.get("taxonomy") or {}
    for key in ("genus", "species"):
        value = taxonomy.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    return candidates


def resolve_crop_from_plant_id(payload: dict[str, Any]) -> dict[str, Any] | None:
    suggestion = _extract_plant_id_suggestion(payload)
    if not suggestion:
        return None

    probability = _clamp_confidence(suggestion.get("probability") or suggestion.get("confidence"))
    is_plant = (payload.get("result") or {}).get("is_plant") or {}
    plant_probability = _clamp_confidence(is_plant.get("probability") or is_plant.get("binary"))
    if plant_probability is not None and plant_probability < 0.5:
        return None

    best_supported: str | None = None
    raw_candidates = _candidate_crop_names_from_plant_id(suggestion)
    for candidate in raw_candidates:
        normalized = normalize_crop_name(candidate)
        if normalized in CROP_HEALTH_SUPPORTED_CROPS:
            best_supported = normalized
            break

    return {
        "identifiedPlant": str(suggestion.get("name") or "Unknown plant").strip(),
        "crop": best_supported,
        "confidence": probability,
        "rawCandidates": raw_candidates,
    }


def normalize_kindwise_identification(
    payload: dict[str, Any],
    *,
    fallback_crop: str | None = None,
    provider_product: str,
    identification_product: str | None = None,
) -> dict[str, Any]:
    suggestion = _extract_first_suggestion(payload)
    if not suggestion:
        return build_unavailable_diagnosis(
            "The diagnosis provider did not return any disease suggestions.",
            source="kindwise",
        )

    details = suggestion.get("details") or {}
    disease_raw = (
        details.get("local_name")
        or details.get("common_name")
        or (details.get("common_names") or [None])[0]
        or suggestion.get("name")
        or "Likely issue not identified"
    )
    confidence = _clamp_confidence(suggestion.get("probability") or suggestion.get("confidence"))

    # Extract crop name from crop.health response
    result_block = payload.get("result") or {}
    crop_block = result_block.get("crop") or {}
    crop_name_from_api = crop_block.get("name")
    if not crop_name_from_api:
        crop_suggestions = crop_block.get("suggestions")
        if isinstance(crop_suggestions, list) and crop_suggestions:
            crop_name_from_api = crop_suggestions[0].get("name")

    plant = fallback_crop or crop_name_from_api or "Unknown plant"

    # Clean up disease name: strip mismatched crop suffixes like
    # "Late Blight Of Potato" → "Late Blight" when the actual crop
    # differs, then re-append the correct crop name.
    disease = disease_raw
    of_match = re.search(r"\s+Of\s+\w+$", disease, re.IGNORECASE)
    if of_match:
        base_disease = disease[: of_match.start()]
        disease = f"{base_disease} Of {plant.title()}"
    evidence = _extract_evidence(details)
    notes = []
    if not evidence:
        notes.append("The provider did not return explicit symptoms for this result.")
    if identification_product:
        notes.append(f"Crop context resolved with {identification_product} before disease diagnosis.")

    return {
        "status": "ok",
        "isAvailable": True,
        "plant": str(plant).strip(),
        "disease": str(disease).strip(),
        "remedy": _extract_treatment_text(details),
        "confidence": confidence,
        "source": "kindwise",
        "providerProduct": provider_product,
        "identificationProduct": identification_product,
        "evidence": evidence,
        "severity": _normalize_severity(details.get("severity")),
        "notes": notes,
        "disclaimer": DEFAULT_DIAGNOSIS_DISCLAIMER,
    }


def format_image_analysis_response(diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        "analysis": {
            "identified_disease": diagnosis["disease"],
            "confidence": diagnosis["confidence"],
            "severity": diagnosis.get("severity") or "unknown",
            "treatment": diagnosis["remedy"],
            "prevention_tips": diagnosis.get("notes") or [diagnosis["disclaimer"]],
            "evidence": diagnosis.get("evidence") or [],
            "disclaimer": diagnosis["disclaimer"],
            "plant": diagnosis["plant"],
        },
        "source": diagnosis["source"],
        "providerProduct": diagnosis.get("providerProduct"),
        "identificationProduct": diagnosis.get("identificationProduct"),
        "status": diagnosis["status"],
    }


async def _request_crop_health_identification(
    base_url: str,
    api_key: str,
    image: str,
    *,
    language: str | None = None,
    crop: str | None = None,
    timeout: float = 45.0,
) -> dict[str, Any]:
    normalized_crop = normalize_crop_name(crop)
    payload_image = image if image.startswith("data:") else image
    body = {"images": [payload_image]}
    if normalized_crop:
        body["similar_images"] = False
    params = {
        "details": KINDWISE_DETAILS,
        "language": (language or "en").split(",")[0],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/v1/identification",
            headers={
                "Content-Type": "application/json",
                "Api-Key": api_key,
            },
            params=params,
            json=body,
        )
        response.raise_for_status()
        return response.json()


async def _request_plant_id_identification(
    base_url: str,
    api_key: str,
    image: str,
    *,
    language: str | None = None,
    timeout: float = 45.0,
) -> dict[str, Any]:
    payload_image = image.split(",", 1)[1] if image.startswith("data:") and "," in image else image
    body = {"images": [payload_image], "similar_images": False}
    params = {
        "details": PLANT_ID_DETAILS,
        "language": (language or "en").split(",")[0],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/identification",
            headers={
                "Content-Type": "application/json",
                "Api-Key": api_key,
            },
            params=params,
            json=body,
        )
        response.raise_for_status()
        return response.json()


async def diagnose_crop_image(
    crop_health_api_key: str,
    crop_health_url: str,
    plant_id_api_key: str,
    plant_id_url: str,
    image: str,
    *,
    crop: str | None = None,
    region: str | None = None,
    language: str | None = None,
    context: dict[str, Any] | None = None,
    timeout: float = 45.0,
) -> dict[str, Any]:
    if not _is_valid_base64_image(image):
        return build_unavailable_diagnosis(
            "Disease analysis could not start because the uploaded image payload is invalid.",
            source="validation",
        )

    context = context or {}
    chosen_language = language or str(context.get("language") or "en")
    crop_candidate = normalize_crop_name(crop or context.get("crop"))
    identified_plant_name: str | None = None
    identification_product: str | None = None

    if not crop_health_api_key:
        return build_unavailable_diagnosis(
            "Disease analysis is unavailable because the crop.health API key is not configured.",
            source="configuration",
        )

    plant_id_rejected_as_non_plant = False

    if not should_use_crop_health(crop_candidate):
        # Crop unknown — run plant.id (for crop name) and crop.health
        # (for disease diagnosis) in parallel so that a plant.id failure
        # no longer blocks the diagnosis.
        plant_id_task = None
        if plant_id_api_key:
            plant_id_task = asyncio.create_task(
                _request_plant_id_identification(
                    plant_id_url,
                    plant_id_api_key,
                    image,
                    language=chosen_language,
                    timeout=timeout,
                )
            )

        crop_health_task = asyncio.create_task(
            _request_crop_health_identification(
                crop_health_url,
                crop_health_api_key,
                image,
                language=chosen_language,
                crop=crop_candidate,
                timeout=timeout,
            )
        )

        # Resolve plant.id result (best-effort)
        if plant_id_task:
            try:
                plant_id_payload = await plant_id_task
                is_plant = (plant_id_payload.get("result") or {}).get("is_plant") or {}
                plant_prob = _clamp_confidence(is_plant.get("probability") or is_plant.get("binary"))
                if plant_prob is not None and plant_prob < 0.5:
                    plant_id_rejected_as_non_plant = True
                else:
                    identified_crop = resolve_crop_from_plant_id(plant_id_payload)
                    if identified_crop and identified_crop.get("crop"):
                        crop_candidate = identified_crop["crop"]
                        identified_plant_name = identified_crop.get("identifiedPlant")
                        identification_product = "plant.id"
            except Exception:
                pass  # plant.id failed — crop.health result still used below

        # Resolve crop.health result (required)
        try:
            crop_payload = await crop_health_task
        except Exception as exc:
            return build_unavailable_diagnosis(
                "Disease analysis is temporarily unavailable. Please try another image or try again later."
                if isinstance(exc, httpx.HTTPStatusError)
                else f"Disease analysis is temporarily unavailable: {_extract_http_error_message(exc, 'Crop diagnosis failed.')}",
                source="kindwise",
            )
    else:
        identified_plant_name = crop_candidate
        try:
            crop_payload = await _request_crop_health_identification(
                crop_health_url,
                crop_health_api_key,
                image,
                language=chosen_language,
                crop=crop_candidate,
                timeout=timeout,
            )
        except Exception as exc:
            return build_unavailable_diagnosis(
                "Disease analysis is temporarily unavailable. Please try another image or try again later."
                if isinstance(exc, httpx.HTTPStatusError)
                else f"Disease analysis is temporarily unavailable: {_extract_http_error_message(exc, 'Crop diagnosis failed.')}",
                source="kindwise",
            )

    # Reject non-plant images when plant.id confirms it's not a plant
    if plant_id_rejected_as_non_plant:
        return build_unavailable_diagnosis(
            "The uploaded image does not appear to be a plant. Please upload a clear photo of a crop.",
            source="validation",
        )

    diagnosis = normalize_kindwise_identification(
        crop_payload,
        fallback_crop=identified_plant_name or crop_candidate,
        provider_product="crop.health",
        identification_product=identification_product,
    )

    # Reject low-confidence or inconclusive diagnoses
    if is_inconclusive_diagnosis(diagnosis):
        return build_unavailable_diagnosis(
            "The image could not be confidently identified as a crop disease. "
            "Please upload a clear, well-lit photo of the affected plant.",
            source="validation",
        )

    if region and diagnosis["status"] == "ok":
        diagnosis["notes"] = diagnosis.get("notes", []) + [f"Region context: {region}"]
    return diagnosis
