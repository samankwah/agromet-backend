"""Crop and poultry calendars: the preview-then-commit spreadsheet upload
flow, the "create straight from JSON" shortcut, and the enhanced-calendars
read/search endpoints that back them (including the crop-calendars/district,
/search and /stats routes, which are just enhanced-calendars queries under
an older name)."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status

from ..database import encode_payload, get_connection, row_to_dict
from ..domain import serialize_calendar
from ..records import fetch_calendar_activities, insert_calendar_from_parsed_payload, insert_calendar_from_record
from ..spreadsheet_parser import (
    build_calendar_preview_payload_from_files,
    build_committed_calendar_payload,
    discard_preview_payload,
    get_preview_payload,
)

router = APIRouter(tags=["calendars"])


async def _preview_calendar_upload(
    file: UploadFile,
    metadata: dict,
    calendar_type: str,
    extra_files: list[tuple[str, UploadFile | None]] | None = None,
):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Spreadsheet file is required.")
    file_items = [
        {
            "season": metadata.get("primarySeason") or "Major Season",
            "fileName": file.filename or "",
            "contents": await file.read(),
        }
    ]
    for season, upload in extra_files or []:
        if upload is None:
            continue
        file_items.append(
            {
                "season": season,
                "fileName": upload.filename or "",
                "contents": await upload.read(),
            }
        )
    try:
        preview = build_calendar_preview_payload_from_files(file_items, metadata, calendar_type)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"success": True, "data": preview}


@router.post("/api/crop-calendars/preview")
async def preview_crop_calendar(
    file: UploadFile = File(...),
    minorFile: UploadFile | None = File(default=None),
    region: str = Form(...),
    district: str = Form(...),
    crop: str = Form(...),
    title: str = Form(default=""),
    description: str = Form(default=""),
    year: int | None = Form(default=None),
):
    return await _preview_calendar_upload(
        file,
        {
            "region": region,
            "district": district,
            "crop": crop,
            "title": title or f"{crop} Calendar",
            "description": description,
            "year": year,
            "primarySeason": "Major Season",
        },
        "crop-calendar",
        extra_files=[("Minor Season", minorFile)],
    )


@router.post("/api/poultry-calendars/preview")
async def preview_poultry_calendar(
    file: UploadFile = File(...),
    region: str = Form(...),
    district: str = Form(...),
    poultryType: str = Form(...),
    title: str = Form(default=""),
    description: str = Form(default=""),
    year: int | None = Form(default=None),
):
    return await _preview_calendar_upload(
        file,
        {
            "region": region,
            "district": district,
            "poultryType": poultryType,
            "title": title or f"{poultryType} Calendar",
            "description": description,
            "year": year,
        },
        "poultry-calendar",
    )


@router.post("/api/crop-calendars/commit")
async def commit_crop_calendar(parseToken: str = Form(...)):
    preview_payload = get_preview_payload(parseToken)
    if not preview_payload or preview_payload.get("entityType") != "crop-calendar":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview token not found or expired.")
    payload = build_committed_calendar_payload(preview_payload)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "crop-calendar", encode_payload(payload), None, None, None),
        )
        record_id = cursor.lastrowid
        calendar = insert_calendar_from_parsed_payload(connection, record_id, "crop-calendar", payload)
    discard_preview_payload(parseToken)
    return {"success": True, "data": calendar, "message": "Crop calendar committed successfully."}


@router.post("/api/poultry-calendars/commit")
async def commit_poultry_calendar(parseToken: str = Form(...)):
    preview_payload = get_preview_payload(parseToken)
    if not preview_payload or preview_payload.get("entityType") != "poultry-calendar":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview token not found or expired.")
    payload = build_committed_calendar_payload(preview_payload)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "poultry-calendar", encode_payload(payload), None, None, None),
        )
        record_id = cursor.lastrowid
        calendar = insert_calendar_from_parsed_payload(connection, record_id, "poultry-calendar", payload)
    discard_preview_payload(parseToken)
    return {"success": True, "data": calendar, "message": "Poultry calendar committed successfully."}


@router.get("/api/enhanced-calendars/metadata")
def get_enhanced_calendar_metadata():
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM calendars ORDER BY created_at DESC, id DESC").fetchall()

    calendars = [serialize_calendar(row_to_dict(row)) for row in rows]
    return {
        "commodities": sorted({item["commodity"] for item in calendars if item.get("commodity")}),
        "regions": sorted({item["regionCode"] for item in calendars if item.get("regionCode")}),
        "districts": sorted({item["districtCode"] for item in calendars if item.get("districtCode")}),
        "calendarTypes": sorted({item["calendarType"] for item in calendars if item.get("calendarType")}),
        "totalCalendars": len(calendars),
    }


@router.get("/api/enhanced-calendars")
def list_enhanced_calendars(
    calendarType: str | None = Query(default=None),
    commodity: str | None = Query(default=None),
    regionCode: str | None = Query(default=None),
    districtCode: str | None = Query(default=None),
    year: int | None = Query(default=None),
    search: str | None = Query(default=None),
):
    clauses = ["1=1"]
    params: list[object] = []
    if calendarType:
        clauses.append("calendar_type = ?")
        params.append(calendarType)
    if commodity:
        clauses.append("LOWER(crop) = LOWER(?)")
        params.append(commodity)
    if regionCode:
        clauses.append("region_code = ?")
        params.append(regionCode)
    if districtCode:
        clauses.append("district_code = ?")
        params.append(districtCode)
    if year is not None:
        clauses.append("year = ?")
        params.append(year)
    if search:
        clauses.append("(LOWER(title) LIKE LOWER(?) OR LOWER(crop) LIKE LOWER(?))")
        params.extend([f"%{search}%", f"%{search}%"])

    query = f"SELECT * FROM calendars WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, id DESC"
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()

    data = [serialize_calendar(row_to_dict(row)) for row in rows]
    return {
        "success": True,
        "data": data,
        "total": len(data),
        "filters": {
            "calendarType": calendarType,
            "commodity": commodity,
            "regionCode": regionCode,
            "districtCode": districtCode,
            "year": year,
            "search": search,
        },
        "summary": {
            "calendarTypes": sorted({item["calendarType"] for item in data}),
            "commodities": sorted({item["commodity"] for item in data}),
        },
    }


@router.get("/api/enhanced-calendars/{calendar_id}/activities")
def get_enhanced_calendar_activities(
    calendar_id: int,
    currentWeek: int | None = Query(default=None),
    startWeek: int | None = Query(default=None),
    endWeek: int | None = Query(default=None),
):
    with get_connection() as connection:
        calendar_row = connection.execute("SELECT * FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
        if not calendar_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calendar not found.")
        activities = fetch_calendar_activities(connection, calendar_id, currentWeek, startWeek, endWeek)
    return {
        "success": True,
        "data": {
            "activities": activities,
            "schedule": activities,
        },
        "filters": {"currentWeek": currentWeek, "startWeek": startWeek, "endWeek": endWeek},
    }


@router.get("/api/enhanced-calendars/{calendar_id}")
def get_enhanced_calendar(calendar_id: int):
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calendar not found.")
        activities = fetch_calendar_activities(connection, calendar_id)
    calendar = serialize_calendar(row_to_dict(row))
    calendar["activities"] = activities
    return {"success": True, "data": calendar}


@router.delete("/api/enhanced-calendars/{calendar_id}")
def delete_enhanced_calendar(calendar_id: int):
    with get_connection() as connection:
        row = connection.execute("SELECT source_record_id FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calendar not found.")
        source_record_id = row["source_record_id"]
        connection.execute("DELETE FROM calendars WHERE id = ?", (calendar_id,))
        if source_record_id:
            connection.execute("DELETE FROM agricultural_records WHERE id = ?", (source_record_id,))
    return {"success": True, "message": "Calendar deleted successfully."}


@router.post("/api/crop-calendars/create")
async def create_crop_calendar(request: Request):
    payload = dict(await request.json())
    payload.setdefault("title", payload.get("crop") or "Crop Calendar")
    payload.setdefault("regionCode", payload.get("region"))
    payload.setdefault("districtCode", payload.get("district"))
    payload.setdefault("crop", payload.get("crop") or payload.get("commodity"))
    payload.setdefault("year", payload.get("year"))

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "crop-calendar", encode_payload(payload), None, None, None),
        )
        record_id = cursor.lastrowid
        calendar = insert_calendar_from_record(connection, record_id, "crop-calendar", payload)
    return {"success": True, "data": calendar}


@router.get("/api/crop-calendars/district/{district}")
def get_crop_calendars_by_district(district: str, year: int | None = Query(default=None), crop: str | None = Query(default=None)):
    params: dict[str, object] = {"districtCode": district}
    if year is not None:
        params["year"] = year
    if crop:
        params["commodity"] = crop
    return list_enhanced_calendars(calendarType="seasonal", **params)


@router.get("/api/crop-calendars/search")
def search_crop_calendars(search: str | None = Query(default=None), region: str | None = Query(default=None), district: str | None = Query(default=None)):
    return list_enhanced_calendars(
        calendarType="seasonal",
        commodity=search,
        regionCode=region,
        districtCode=district,
        search=search,
    )


@router.get("/api/crop-calendars/stats")
def get_crop_calendar_stats():
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM calendars WHERE calendar_type = 'seasonal' ORDER BY created_at DESC").fetchall()
    calendars = [serialize_calendar(row_to_dict(row)) for row in rows]
    return {
        "success": True,
        "data": {
            "totalCalendars": len(calendars),
            "regions": sorted({item["regionCode"] for item in calendars if item.get("regionCode")}),
            "commodities": sorted({item["commodity"] for item in calendars if item.get("commodity")}),
        },
    }
