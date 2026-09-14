"""Crop disease diagnosis: the two entry points (structured request, and the
web app's image-analysis shape) and the signed-in farmer's history of them.

Reads `config.KINDWISE_*` as `config.NAME` for the same reason `chat.py`
does -- `test_diagnosis.py` patches `backend.app.config.KINDWISE_*` and needs
that patch to reach the module that actually calls `diagnose_crop_image` at
request time.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from .. import config
from ..database import decode_payload, encode_payload, get_connection, row_to_dict
from ..deps import get_current_user, get_optional_user
from ..diagnosis import SUPPORTED_IMAGE_ANALYSIS_TYPES, diagnose_crop_image, format_image_analysis_response
from ..schemas import CropDiagnosisRequest, ImageAnalysisRequest

router = APIRouter(tags=["diagnosis"])


def save_diagnosis_record(owner_id: int, diagnosis: dict, crop: str | None, region: str | None) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO diagnosis_records(
                owner_id, provider_product, status, plant, disease, confidence, severity,
                source, crop_context, region_context, result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_id,
                diagnosis.get("providerProduct"),
                diagnosis.get("status"),
                diagnosis.get("plant"),
                diagnosis.get("disease"),
                diagnosis.get("confidence"),
                diagnosis.get("severity"),
                diagnosis.get("source"),
                crop,
                region,
                encode_payload(diagnosis),
            ),
        )
    return cursor.lastrowid


def serialize_diagnosis_record(record: dict) -> dict:
    result = decode_payload(record.get("result_json"))
    return {
        "id": record["id"],
        "providerProduct": record.get("provider_product"),
        "status": record.get("status"),
        "plant": record.get("plant"),
        "disease": record.get("disease"),
        "confidence": record.get("confidence"),
        "severity": record.get("severity"),
        "source": record.get("source"),
        "cropContext": record.get("crop_context"),
        "regionContext": record.get("region_context"),
        "createdAt": record.get("created_at"),
        "result": result,
    }


@router.post("/api/crop-diagnosis")
async def crop_diagnosis(payload: CropDiagnosisRequest, authorization: str | None = Header(default=None)):
    context = dict(payload.context)
    context.setdefault("language", payload.language)
    diagnosis = await diagnose_crop_image(
        config.KINDWISE_CROP_HEALTH_API_KEY,
        config.KINDWISE_CROP_HEALTH_URL,
        config.KINDWISE_PLANT_ID_API_KEY,
        config.KINDWISE_PLANT_ID_URL,
        payload.image,
        crop=payload.crop,
        region=payload.region,
        language=payload.language,
        context=context,
    )
    current_user = get_optional_user(authorization)
    if current_user and diagnosis["status"] == "ok":
        record_id = save_diagnosis_record(current_user["id"], diagnosis, payload.crop, payload.region)
        diagnosis["historyId"] = record_id
    return diagnosis


@router.post("/api/image-analysis")
async def image_analysis(payload: ImageAnalysisRequest, authorization: str | None = Header(default=None)):
    if payload.analysisType not in SUPPORTED_IMAGE_ANALYSIS_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported analysisType '{payload.analysisType}'. Supported types: {', '.join(sorted(SUPPORTED_IMAGE_ANALYSIS_TYPES))}.",
        )

    diagnosis = await diagnose_crop_image(
        config.KINDWISE_CROP_HEALTH_API_KEY,
        config.KINDWISE_CROP_HEALTH_URL,
        config.KINDWISE_PLANT_ID_API_KEY,
        config.KINDWISE_PLANT_ID_URL,
        payload.image,
        crop=payload.context.get("crop"),
        region=payload.context.get("region"),
        language=payload.context.get("language"),
        context=payload.context,
    )
    current_user = get_optional_user(authorization)
    if current_user and diagnosis["status"] == "ok":
        save_diagnosis_record(current_user["id"], diagnosis, payload.context.get("crop"), payload.context.get("region"))
    if diagnosis["status"] != "ok":
        return {
            "analysis": None,
            "source": diagnosis["source"],
            "providerProduct": diagnosis.get("providerProduct"),
            "status": diagnosis["status"],
            "message": diagnosis["remedy"],
        }
    return format_image_analysis_response(diagnosis)


@router.get("/api/diagnosis-history")
def list_diagnosis_history(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    status_filter: str | None = Query(default=None, alias="status"),
    crop: str | None = Query(default=None),
):
    clauses = ["owner_id = ?"]
    params: list[object] = [current_user["id"]]
    if status_filter:
        clauses.append("status = ?")
        params.append(status_filter)
    if crop:
        clauses.append("LOWER(crop_context) = LOWER(?)")
        params.append(crop)
    params.append(limit)
    query = f"""
        SELECT * FROM diagnosis_records
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
    """
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    data = [serialize_diagnosis_record(row_to_dict(row)) for row in rows]
    return {"success": True, "data": data, "total": len(data)}


@router.get("/api/diagnosis-history/{record_id}")
def get_diagnosis_history_item(record_id: int, current_user: dict = Depends(get_current_user)):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM diagnosis_records WHERE id = ? AND owner_id = ?",
            (record_id, current_user["id"]),
        ).fetchone()
    record = row_to_dict(row)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis history item not found.")
    return {"success": True, "data": serialize_diagnosis_record(record)}


@router.delete("/api/diagnosis-history/{record_id}")
def delete_diagnosis_history_item(record_id: int, current_user: dict = Depends(get_current_user)):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM diagnosis_records WHERE id = ? AND owner_id = ?",
            (record_id, current_user["id"]),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis history item not found.")
        connection.execute(
            "DELETE FROM diagnosis_records WHERE id = ? AND owner_id = ?",
            (record_id, current_user["id"]),
        )
    return {"success": True, "message": "Diagnosis history item deleted successfully."}
