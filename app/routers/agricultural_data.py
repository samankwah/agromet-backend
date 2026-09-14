"""The generic upload path: any spreadsheet/JSON a client posts, stored as
an `agricultural_records` row, plus the specific record types (crop and
poultry calendars) that also get a calendar built from them on the way in."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile, status

from ..database import encode_payload, get_connection, row_to_dict
from ..deps import get_current_user
from ..records import insert_calendar_from_record, normalize_record

router = APIRouter(prefix="/api/agricultural-data", tags=["agricultural-data"])


@router.post("/upload")
async def upload_agricultural_data(
    request: Request,
    dataType: str = Form(...),
    file: UploadFile | None = File(default=None),
    authorization: str | None = Header(default=None),
):
    owner_id = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            current_user = get_current_user(token)
            owner_id = current_user["id"]
        except HTTPException:
            owner_id = None

    payload = {}
    if file:
        payload["originalFilename"] = file.filename

    form = await request.form()
    for key, value in form.multi_items():
        if key == "dataType" or key == "file":
            continue
        payload[key] = value

    record_payload = payload
    if file:
        contents = await file.read()
        record_payload["fileSize"] = len(contents)
        record_payload["fileName"] = file.filename
        record_payload["fileContentType"] = file.content_type

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                owner_id,
                dataType,
                encode_payload(record_payload),
                file.filename if file else None,
                file.content_type if file else None,
                record_payload.get("fileSize"),
            ),
        )
        record_id = cursor.lastrowid
        row = connection.execute(
            "SELECT * FROM agricultural_records WHERE id = ?",
            (record_id,),
        ).fetchone()
        calendar = insert_calendar_from_record(connection, record_id, dataType, record_payload)

    response = {"success": True, "data": normalize_record(row_to_dict(row))}
    if calendar:
        response["calendar"] = calendar
        response["calendarType"] = calendar["calendarType"]
        response["commodity"] = calendar["commodity"]
        response["activities"] = len(calendar.get("sampleActivities", []))
        response["message"] = "Agricultural data uploaded and calendar processed successfully."
    return response


@router.get("/{data_type}")
def list_agricultural_data(data_type: str):
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM agricultural_records WHERE data_type = ? ORDER BY created_at DESC, id DESC",
            (data_type,),
        ).fetchall()

    return {"success": True, "data": [normalize_record(row_to_dict(row)) for row in rows]}


@router.delete("/{data_type}/{record_id}")
def delete_agricultural_data(data_type: str, record_id: int):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM agricultural_records WHERE id = ? AND data_type = ?",
            (record_id, data_type),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agricultural record not found.")
        connection.execute(
            "DELETE FROM agricultural_records WHERE id = ? AND data_type = ?",
            (record_id, data_type),
        )

    return {"success": True, "message": "Agricultural record deleted successfully."}
