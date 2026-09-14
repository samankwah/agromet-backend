"""The calendar/advisory write path, shared by two routers.

`routers/agricultural_data.py`'s generic upload endpoint and
`routers/calendars.py` / `routers/advisories.py`'s dedicated preview+commit
flows both end up inserting the same `calendars` / `calendar_activities` /
`weekly_advisories` / `weekly_advisory_activities` rows from a parsed
payload. That overlap is why these functions live here instead of in
whichever router was written first -- the alternative was two calendars.
"""

from __future__ import annotations

from .database import get_connection, row_to_dict
from .domain import (
    build_activity_rows,
    build_advisory_payload,
    generate_sample_activities,
    infer_calendar_type,
    infer_crop_name,
    infer_weeks,
    json_dumps,
    serialize_advisory,
    serialize_calendar,
    serialize_calendar_activity,
)


def normalize_record(record: dict) -> dict:
    from .database import decode_payload

    payload = decode_payload(record.get("payload_json"))
    normalized = {
        "id": record["id"],
        "dataType": record["data_type"],
        "createdAt": record["created_at"],
        "updatedAt": record["updated_at"],
        "uploadDate": record["created_at"],
        "status": "processed",
        "fileName": record.get("file_name"),
        "fileSize": record.get("file_size"),
        "fileContentType": record.get("file_content_type"),
    }
    normalized.update(payload)
    return normalized


def insert_calendar_from_record(connection, record_id: int, data_type: str, payload: dict) -> dict | None:
    if data_type not in {"crop-calendar", "poultry-calendar", "enhanced-calendar"}:
        return None

    calendar_type = infer_calendar_type(data_type, payload)
    crop = infer_crop_name(data_type, payload)
    title = str(payload.get("title") or payload.get("fileName") or f"{crop.title()} Calendar").strip()
    description = str(payload.get("description") or "")
    region_code = str(payload.get("regionCode") or payload.get("region") or "")
    district_code = str(payload.get("districtCode") or payload.get("district") or "")

    try:
        year = int(payload.get("year")) if payload.get("year") not in (None, "") else None
    except (TypeError, ValueError):
        year = None

    total_weeks = infer_weeks(payload, calendar_type)
    sample_activities = generate_sample_activities(title, crop, calendar_type)
    activity_rows = build_activity_rows(sample_activities, total_weeks)
    metadata = {
        "majorSeason": {"startMonth": "April"} if calendar_type == "seasonal" else {},
        "fileData": {
            "filename": payload.get("fileName") or payload.get("originalFilename"),
            "totalRecords": len(activity_rows),
            "sheets": {},
        },
    }

    cursor = connection.execute(
        """
        INSERT INTO calendars(
            source_record_id, title, description, calendar_type, crop, region_code, region,
            district_code, district, year, total_weeks, cycle_duration, breed_type,
            sample_activities_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            title,
            description,
            calendar_type,
            crop,
            region_code,
            region_code,
            district_code,
            district_code,
            year,
            total_weeks,
            total_weeks if calendar_type == "cycle" else None,
            payload.get("breedCode") or payload.get("breedType"),
            json_dumps(sample_activities),
            json_dumps(metadata),
        ),
    )
    calendar_id = cursor.lastrowid

    for activity in activity_rows:
        connection.execute(
            """
            INSERT INTO calendar_activities(calendar_id, activity_code, activity_name, start_week, end_week, production_week, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                calendar_id,
                activity["activityCode"],
                activity["activityName"],
                activity["startWeek"],
                activity["endWeek"],
                activity["productionWeek"],
                json_dumps(activity),
            ),
        )

    row = connection.execute("SELECT * FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
    return serialize_calendar(row_to_dict(row))


def insert_calendar_from_parsed_payload(connection, record_id: int | None, data_type: str, payload: dict) -> dict:
    calendar_type = payload.get("calendarType") or infer_calendar_type(data_type, payload)
    crop = payload.get("crop") or infer_crop_name(data_type, payload)
    title = str(payload.get("title") or f"{crop} Calendar").strip()
    description = str(payload.get("description") or "")
    region_code = str(payload.get("regionCode") or payload.get("region") or "")
    district_code = str(payload.get("districtCode") or payload.get("district") or "")
    total_weeks = infer_weeks(payload, calendar_type)
    sample_activities = payload.get("sampleActivities") or [item.get("activityName") for item in payload.get("activities", [])[:6]]
    metadata = {
        "fileData": {
            "totalRecords": len(payload.get("activities", [])),
            "sheets": payload.get("sheets", []),
        },
        "previewWarnings": payload.get("warnings", []),
        "seasons": payload.get("seasons", []),
    }

    cursor = connection.execute(
        """
        INSERT INTO calendars(
            source_record_id, title, description, calendar_type, crop, region_code, region,
            district_code, district, year, total_weeks, cycle_duration, breed_type,
            sample_activities_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            title,
            description,
            calendar_type,
            crop,
            region_code,
            region_code,
            district_code,
            district_code,
            payload.get("year"),
            total_weeks,
            total_weeks if calendar_type == "cycle" else None,
            payload.get("breedType"),
            json_dumps(sample_activities),
            json_dumps(metadata),
        ),
    )
    calendar_id = cursor.lastrowid

    for activity in payload.get("activities", []):
        connection.execute(
            """
            INSERT INTO calendar_activities(calendar_id, activity_code, activity_name, start_week, end_week, production_week, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                calendar_id,
                activity.get("activityCode") or activity.get("activityName"),
                activity.get("activityName"),
                activity.get("startWeek"),
                activity.get("endWeek"),
                activity.get("productionWeek"),
                json_dumps(activity.get("metadata") or activity),
            ),
        )

    row = connection.execute("SELECT * FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
    return serialize_calendar(row_to_dict(row))


def insert_weekly_advisory(connection, record_id: int | None, advisory_type: str, payload: dict) -> dict:
    advisory_record, activities = build_advisory_payload(payload, advisory_type)
    cursor = connection.execute(
        """
        INSERT INTO weekly_advisories(
            source_record_id, advisory_type, title, description, region_code, region, district_code,
            district, crop, commodity_code, poultry_type_code, breed_code, year, season,
            summary, weather_forecast_json, advisories_json, sms_advisory
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            advisory_record["advisoryType"],
            advisory_record["title"],
            advisory_record["description"],
            advisory_record["regionCode"],
            advisory_record["region"],
            advisory_record["districtCode"],
            advisory_record["district"],
            advisory_record["crop"],
            advisory_record["commodityCode"],
            advisory_record["poultryTypeCode"],
            advisory_record["breedCode"],
            advisory_record["year"],
            advisory_record["season"],
            advisory_record["summary"],
            json_dumps(advisory_record["weatherForecast"]),
            json_dumps(advisory_record["advisories"]),
            advisory_record["smsAdvisory"],
        ),
    )
    advisory_id = cursor.lastrowid
    for item in activities:
        connection.execute(
            """
            INSERT INTO weekly_advisory_activities(advisory_id, activity, week_label, activity_type)
            VALUES (?, ?, ?, ?)
            """,
            (advisory_id, item["activity"], item["weekLabel"], item["activityType"]),
        )
    row = connection.execute("SELECT * FROM weekly_advisories WHERE id = ?", (advisory_id,)).fetchone()
    return serialize_advisory(row_to_dict(row))


def advisory_list_item(advisory: dict) -> dict:
    item = serialize_advisory(advisory)
    with get_connection() as connection:
        activity_rows = connection.execute(
            "SELECT activity, week_label FROM weekly_advisory_activities WHERE advisory_id = ? ORDER BY id ASC",
            (advisory["id"],),
        ).fetchall()
    activities = [row_to_dict(row) for row in activity_rows]
    item["activityCount"] = len(activities)
    item["activities"] = [row["activity"] for row in activities]
    item["weekLabels"] = [row.get("week_label") for row in activities if row.get("week_label")]
    return item


def fetch_calendar_activities(connection, calendar_id: int, current_week: int | None = None, start_week: int | None = None, end_week: int | None = None) -> list[dict]:
    clauses = ["calendar_id = ?"]
    params: list[object] = [calendar_id]

    if current_week is not None:
        clauses.append("start_week <= ? AND end_week >= ?")
        params.extend([current_week, current_week])
    else:
        if start_week is not None:
            clauses.append("end_week >= ?")
            params.append(start_week)
        if end_week is not None:
            clauses.append("start_week <= ?")
            params.append(end_week)

    query = f"""
        SELECT * FROM calendar_activities
        WHERE {' AND '.join(clauses)}
        ORDER BY start_week, id
    """
    rows = connection.execute(query, params).fetchall()
    return [serialize_calendar_activity(row_to_dict(row)) for row in rows]
