"""Weekly agromet/poultry advisory bulletins: upload, the preview-then-commit
spreadsheet flow, and the read/delete endpoints the dashboards use."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status

from ..database import encode_payload, get_connection, row_to_dict
from ..domain import json_dumps, parse_json_list, serialize_advisory
from ..records import advisory_list_item, insert_weekly_advisory
from ..spreadsheet_parser import build_advisory_preview_payload, build_committed_advisory_payload, discard_preview_payload, get_preview_payload

router = APIRouter(prefix="/api/weekly-advisories", tags=["advisories"])


@router.post("/upload")
async def upload_weekly_advisory(request: Request, file: UploadFile | None = File(default=None)):
    payload = {}
    if file:
        payload["originalFilename"] = file.filename
        contents = await file.read()
        payload["fileSize"] = len(contents)
        payload["fileName"] = file.filename
        payload["fileContentType"] = file.content_type

    form = await request.form()
    for key, value in form.multi_items():
        if key != "file":
            payload[key] = value

    advisory_type = "poultry-advisory" if payload.get("poultryTypeCode") else "agromet-advisory"
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                None,
                advisory_type,
                encode_payload(payload),
                payload.get("fileName"),
                payload.get("fileContentType"),
                payload.get("fileSize"),
            ),
        )
        record_id = cursor.lastrowid
        advisory = insert_weekly_advisory(connection, record_id, advisory_type, payload)

    return {"success": True, "data": advisory, "message": "Weekly advisory uploaded successfully."}


async def _preview_advisory_upload(file: UploadFile, metadata: dict, advisory_type: str):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Spreadsheet file is required.")
    contents = await file.read()
    try:
        preview = build_advisory_preview_payload(contents, metadata, advisory_type)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"success": True, "data": preview}


@router.post("/preview")
async def preview_weekly_advisory(
    file: UploadFile = File(...),
    regionCode: str = Form(...),
    districtCode: str = Form(...),
    commodityCode: str = Form(...),
    title: str = Form(...),
    description: str = Form(default=""),
):
    return await _preview_advisory_upload(
        file,
        {
            "regionCode": regionCode,
            "districtCode": districtCode,
            "commodityCode": commodityCode,
            "title": title,
            "description": description,
        },
        "agromet-advisory",
    )


# Poultry advisory preview/commit share this router (not weekly-advisories'
# prefix -- they're their own top-level paths) because they're the same
# upload flow as the two routes above, just for the poultry data shape.
poultry_router = APIRouter(tags=["advisories"])


@poultry_router.post("/api/poultry-advisories/preview")
async def preview_poultry_advisory(
    file: UploadFile = File(...),
    regionCode: str = Form(...),
    districtCode: str = Form(...),
    poultryTypeCode: str = Form(...),
    breedCode: str = Form(default=""),
    title: str = Form(...),
    description: str = Form(default=""),
):
    return await _preview_advisory_upload(
        file,
        {
            "regionCode": regionCode,
            "districtCode": districtCode,
            "poultryTypeCode": poultryTypeCode,
            "breedCode": breedCode,
            "title": title,
            "description": description,
        },
        "poultry-advisory",
    )


@router.post("/commit")
async def commit_weekly_advisory(
    parseToken: str = Form(...),
    selectedSheets: str = Form(default="[]"),
):
    preview_payload = get_preview_payload(parseToken)
    if not preview_payload or preview_payload.get("entityType") != "agromet-advisory":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview token not found or expired.")

    payload = build_committed_advisory_payload(preview_payload, parse_json_list(selectedSheets))
    payload.update(
        {
            "selectedSheets": json_dumps(payload.get("selectedSheets", [])),
            "weatherForecast": payload.get("weatherForecast", {}),
            "advisories": [item["text"] for item in payload.get("advisories", [])],
        }
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "agromet-advisory", encode_payload(payload), None, None, None),
        )
        record_id = cursor.lastrowid
        advisory = insert_weekly_advisory(connection, record_id, "agromet-advisory", payload)

    discard_preview_payload(parseToken)
    return {"success": True, "data": advisory, "message": "Agromet advisory committed successfully."}


@poultry_router.post("/api/poultry-advisories/commit")
async def commit_poultry_advisory(
    parseToken: str = Form(...),
    selectedSheets: str = Form(default="[]"),
):
    preview_payload = get_preview_payload(parseToken)
    if not preview_payload or preview_payload.get("entityType") != "poultry-advisory":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview token not found or expired.")

    payload = build_committed_advisory_payload(preview_payload, parse_json_list(selectedSheets))
    payload.update(
        {
            "selectedSheets": json_dumps(payload.get("selectedSheets", [])),
            "advisories": [item["text"] for item in payload.get("advisories", [])],
            "weatherForecast": payload.get("managementMetrics", {}),
        }
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "poultry-advisory", encode_payload(payload), None, None, None),
        )
        record_id = cursor.lastrowid
        advisory = insert_weekly_advisory(connection, record_id, "poultry-advisory", payload)

    discard_preview_payload(parseToken)
    return {"success": True, "data": advisory, "message": "Poultry advisory committed successfully."}


@router.get("")
def list_weekly_advisories(
    advisoryType: str | None = Query(default=None),
    regionCode: str | None = Query(default=None),
    districtCode: str | None = Query(default=None),
    commodityCode: str | None = Query(default=None),
    poultryTypeCode: str | None = Query(default=None),
):
    clauses = ["1=1"]
    params: list[object] = []
    if advisoryType:
        clauses.append("advisory_type = ?")
        params.append(advisoryType)
    if regionCode:
        clauses.append("region_code = ?")
        params.append(regionCode)
    if districtCode:
        clauses.append("district_code = ?")
        params.append(districtCode)
    if commodityCode:
        clauses.append("commodity_code = ?")
        params.append(commodityCode)
    if poultryTypeCode:
        clauses.append("poultry_type_code = ?")
        params.append(poultryTypeCode)

    query = f"SELECT * FROM weekly_advisories WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, id DESC"
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    data = [advisory_list_item(row_to_dict(row)) for row in rows]
    return {"success": True, "data": data, "total": len(data)}


@router.delete("/{advisory_id}")
def delete_weekly_advisory(advisory_id: int):
    with get_connection() as connection:
        row = connection.execute("SELECT source_record_id FROM weekly_advisories WHERE id = ?", (advisory_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Weekly advisory not found.")
        source_record_id = row["source_record_id"]
        connection.execute("DELETE FROM weekly_advisories WHERE id = ?", (advisory_id,))
        if source_record_id:
            connection.execute("DELETE FROM agricultural_records WHERE id = ?", (source_record_id,))
    return {"success": True, "message": "Weekly advisory deleted successfully."}


@router.get("/activities")
def list_weekly_advisory_activities(
    region: str | None = Query(default=None),
    district: str | None = Query(default=None),
    crop: str | None = Query(default=None),
    year: int | None = Query(default=None),
):
    clauses = ["1=1"]
    params: list[object] = []
    if region:
        clauses.append("(wa.region_code = ? OR LOWER(wa.region) = LOWER(?) OR LOWER(REPLACE(wa.region, ' Region', '')) = LOWER(?))")
        params.extend([region, region, region])
    if district:
        clauses.append("(wa.district_code = ? OR LOWER(wa.district) = LOWER(?))")
        params.extend([district, district])
    if crop:
        clauses.append("LOWER(wa.crop) = LOWER(?)")
        params.append(crop)
    if year is not None:
        clauses.append("wa.year = ?")
        params.append(year)

    query = f"""
        SELECT waa.id, waa.activity, waa.week_label, wa.id AS advisory_id, wa.region, wa.region_code,
               wa.district, wa.district_code, wa.crop, wa.year
        FROM weekly_advisory_activities waa
        JOIN weekly_advisories wa ON wa.id = waa.advisory_id
        WHERE {' AND '.join(clauses)}
        ORDER BY wa.created_at DESC, waa.id ASC
    """
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()

    data = []
    for row in rows:
        item = row_to_dict(row)
        data.append(
            {
                "id": item["id"],
                "advisory_id": item["advisory_id"],
                "activity": item["activity"],
                "week_label": item["week_label"],
                "region": item.get("region") or item.get("region_code"),
                "district": item.get("district") or item.get("district_code"),
                "crop": item.get("crop"),
                "year": item.get("year"),
            }
        )
    return {"success": True, "data": data}


@router.get("/{advisory_id}")
def get_weekly_advisory(advisory_id: int, activity: str | None = Query(default=None)):
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM weekly_advisories WHERE id = ?", (advisory_id,)).fetchone()
    advisory = row_to_dict(row)
    if not advisory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Weekly advisory not found.")
    result = serialize_advisory(advisory)

    # If an activity name is specified, find its parsed data from advisories
    if activity and isinstance(result.get("advisories"), list):
        for adv in result["advisories"]:
            if isinstance(adv, dict) and adv.get("activity", "").lower() == activity.lower():
                result["activityData"] = adv
                break

    return {"success": True, "data": result}
