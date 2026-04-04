from __future__ import annotations

import json
from datetime import date, datetime


def parse_json_list(value: str | None) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_json_object(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True)


def slugify_label(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def infer_calendar_type(data_type: str, payload: dict) -> str:
    if data_type == "poultry-calendar":
        return "cycle"
    if data_type == "enhanced-calendar":
        title = str(payload.get("title", "")).lower()
        file_name = str(payload.get("fileName") or payload.get("originalFilename") or "").lower()
        if any(term in f"{title} {file_name}" for term in ("poultry", "broiler", "layer", "bird")):
            return "cycle"
    return "seasonal"


def infer_crop_name(data_type: str, payload: dict) -> str:
    if payload.get("commodity"):
        return str(payload["commodity"])
    if payload.get("commodityName"):
        return str(payload["commodityName"])
    if payload.get("crop"):
        return str(payload["crop"])
    if payload.get("commodityCode"):
        return str(payload["commodityCode"])
    if payload.get("poultryTypeCode"):
        return str(payload["poultryTypeCode"])
    if data_type == "poultry-calendar":
        return "poultry"
    return "general"


def infer_weeks(payload: dict, calendar_type: str) -> int:
    for key in ("totalWeeks", "cycleDuration", "durationWeeks"):
        value = payload.get(key)
        if value not in (None, ""):
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
    return 8 if calendar_type == "cycle" else 16


def generate_sample_activities(title: str, crop: str, calendar_type: str) -> list[str]:
    if calendar_type == "cycle":
        return [
            f"Prepare {crop} housing and equipment",
            "Start feed and water monitoring",
            "Schedule vaccination and health checks",
            "Track weight gain and mortality",
            "Prepare market or next production stage",
        ]
    return [
        f"Land preparation for {crop}",
        "Planting and early establishment",
        "Weeding and nutrient application",
        "Pest and disease monitoring",
        "Harvest planning and post-harvest handling",
    ]


def build_activity_rows(sample_activities: list[str], total_weeks: int) -> list[dict]:
    rows: list[dict] = []
    if not sample_activities:
        return rows

    span = max(1, total_weeks // max(1, len(sample_activities)))
    current_week = 1
    for index, name in enumerate(sample_activities):
        start_week = current_week
        end_week = total_weeks if index == len(sample_activities) - 1 else min(total_weeks, current_week + span - 1)
        rows.append(
            {
                "activityName": name,
                "activityCode": slugify_label(name, f"activity-{index + 1}"),
                "startWeek": start_week,
                "endWeek": end_week,
                "productionWeek": start_week,
            }
        )
        current_week = end_week + 1
    return rows


def _extract_name_from_code_slash(value: str) -> str:
    """Extract the human-readable name from 'CODE/Name' format (e.g., 'REG01/Western Region' -> 'Western Region')."""
    if "/" in value:
        return value.split("/", 1)[1].strip()
    return value


def build_advisory_payload(payload: dict, advisory_type: str) -> tuple[dict, list[dict]]:
    region_code = str(payload.get("regionCode") or payload.get("region") or "")
    district_code = str(payload.get("districtCode") or payload.get("district") or "")
    crop = infer_crop_name(advisory_type, payload)

    # Try to resolve human-readable names from parsedActivities metadata
    parsed_activities_data = payload.get("parsedActivities")
    if isinstance(parsed_activities_data, list) and parsed_activities_data:
        first_meta = parsed_activities_data[0].get("metadata", {})
        region_name = _extract_name_from_code_slash(first_meta.get("region", ""))
        district_name = _extract_name_from_code_slash(first_meta.get("district", ""))
        crop_name = _extract_name_from_code_slash(first_meta.get("crop", ""))
        if region_name:
            region_code = region_code or region_name
        if district_name:
            district_code = district_code or district_name
        if crop_name and crop in (payload.get("commodityCode", ""), ""):
            crop = crop_name
    else:
        region_name = ""
        district_name = ""
    title = str(payload.get("title") or f"{crop.title()} Advisory").strip()
    year_raw = payload.get("year")
    try:
        year = int(year_raw) if year_raw not in (None, "") else datetime.utcnow().year
    except (TypeError, ValueError):
        year = datetime.utcnow().year

    summary = payload.get("summary") or payload.get("description") or f"Weekly advisory for {crop} in {district_code or 'selected district'}."
    advisories = payload.get("advisories") or [
        f"Monitor weather-sensitive operations for {crop}.",
        "Maintain drainage and inspect fields regularly.",
        "Adjust field operations to match rainfall and humidity conditions.",
    ]
    if advisory_type == "poultry-advisory":
        advisories = payload.get("advisories") or [
            "Maintain feed, water, and ventilation checks.",
            "Review vaccination and sanitation routines.",
            "Monitor flock performance and stress indicators.",
        ]

    parsed_activities = payload.get("activities")
    if isinstance(parsed_activities, list) and parsed_activities:
        activities = [
            {
                "activity": str(item.get("activity") or item.get("stage") or f"Activity {index + 1}"),
                "weekLabel": str(item.get("weekLabel") or item.get("timing") or f"Week {index + 1}"),
                "activityType": advisory_type,
            }
            for index, item in enumerate(parsed_activities[:24])
        ]
    else:
        selected_sheets = parse_json_list(payload.get("selectedSheets"))
        activity_names = [f"{sheet} review" for sheet in selected_sheets] or advisories
        activities = [
            {
                "activity": name,
                "weekLabel": f"Week {index + 1}",
                "activityType": advisory_type,
            }
            for index, name in enumerate(activity_names[:8])
        ]

    weather_forecast = payload.get("weatherForecast") or payload.get("weather_forecast") or {
        "rainfall": "Moderate rainfall expected",
        "temperature": "Warm daytime and mild night temperatures",
        "humidity": "Elevated humidity; monitor disease pressure",
    }

    # Use parsedActivities (from new agromet parser) if available
    if isinstance(parsed_activities_data, list) and parsed_activities_data:
        # Store full per-activity data in advisories field
        advisories = parsed_activities_data
        # Use first activity's SMS as default
        sms_advisory = parsed_activities_data[0].get("smsText", "")
    else:
        sms_advisory = payload.get("smsAdvisory") or (advisories[0] if isinstance(advisories, list) and advisories else "")

    record = {
        "advisoryType": advisory_type,
        "title": title,
        "description": str(payload.get("description") or ""),
        "regionCode": region_code,
        "region": region_name or region_code,
        "districtCode": district_code,
        "district": district_name or district_code,
        "crop": crop,
        "commodityCode": str(payload.get("commodityCode") or ""),
        "poultryTypeCode": str(payload.get("poultryTypeCode") or ""),
        "breedCode": str(payload.get("breedCode") or ""),
        "year": year,
        "season": str(payload.get("season") or ""),
        "summary": summary,
        "weatherForecast": weather_forecast,
        "advisories": advisories,
        "smsAdvisory": sms_advisory,
    }
    return record, activities


def serialize_calendar(row: dict) -> dict:
    sample_activities = parse_json_list(row.get("sample_activities_json"))
    metadata = parse_json_object(row.get("metadata_json"))
    return {
        "id": row["id"],
        "title": row.get("title"),
        "description": row.get("description"),
        "calendarType": row.get("calendar_type"),
        "crop": row.get("crop"),
        "commodity": row.get("crop"),
        "region": row.get("region") or row.get("region_code"),
        "regionCode": row.get("region_code"),
        "district": row.get("district") or row.get("district_code"),
        "districtCode": row.get("district_code"),
        "year": row.get("year"),
        "totalWeeks": row.get("total_weeks"),
        "cycleDuration": row.get("cycle_duration"),
        "breedType": row.get("breed_type"),
        "majorSeason": metadata.get("majorSeason", {}),
        "sampleActivities": sample_activities,
        "fileData": metadata.get("fileData", {}),
        "createdAt": row.get("created_at"),
        "updatedAt": row.get("updated_at"),
    }


def serialize_calendar_activity(row: dict) -> dict:
    metadata = parse_json_object(row.get("metadata_json"))
    colors = metadata.get("colors", [])
    primary_color = colors[0] if colors else None
    start_week = row.get("start_week")
    end_week = row.get("end_week")

    # Expand startWeek/endWeek into per-week periods so the frontend can
    # map each week to the correct timeline column.
    periods = []
    if start_week is not None and end_week is not None:
        for week in range(start_week, end_week + 1):
            periods.append({
                "productionWeek": week,
                "startWeek": week,
                "endWeek": week,
                "timelineIndex": week - 1,
                "backgroundColor": primary_color,
            })

    return {
        "id": row["id"],
        "activityId": row.get("activity_code"),
        "activityName": row.get("activity_name"),
        "startWeek": start_week,
        "endWeek": end_week,
        "productionWeek": row.get("production_week"),
        "backgroundColor": primary_color,
        "colors": colors,
        "periods": periods,
    }


def serialize_advisory(row: dict) -> dict:
    weather_forecast = parse_json_object(row.get("weather_forecast_json"))
    advisories = parse_json_list(row.get("advisories_json"))
    return {
        "id": row["id"],
        "title": row.get("title"),
        "description": row.get("description"),
        "advisoryType": row.get("advisory_type"),
        "region": row.get("region") or row.get("region_code"),
        "regionCode": row.get("region_code"),
        "district": row.get("district") or row.get("district_code"),
        "districtCode": row.get("district_code"),
        "crop": row.get("crop"),
        "commodityCode": row.get("commodity_code"),
        "poultryTypeCode": row.get("poultry_type_code"),
        "breedCode": row.get("breed_code"),
        "year": row.get("year"),
        "season": row.get("season"),
        "summary": row.get("summary"),
        "weather_forecast": weather_forecast,
        "weatherForecast": weather_forecast,
        "advisories": advisories,
        "sms_advisory": row.get("sms_advisory"),
        "smsAdvisory": row.get("sms_advisory"),
        "created_at": row.get("created_at"),
    }


def serialize_cycle(row: dict) -> dict:
    today = date.today()
    start_date = date.fromisoformat(row["start_date"])
    total_weeks = max(1, row.get("total_duration_weeks") or 1)
    elapsed_days = max(0, (today - start_date).days)
    current_week = min(total_weeks, elapsed_days // 7 + 1)
    progress_percent = round((current_week / total_weeks) * 100, 2)
    end_date = start_date.fromordinal(start_date.toordinal() + total_weeks * 7)
    return {
        "id": row["id"],
        "calendarId": row.get("calendar_id"),
        "batchName": row.get("batch_name"),
        "commodity": row.get("commodity"),
        "status": row.get("status"),
        "startDate": row.get("start_date"),
        "expectedEndDate": end_date.isoformat(),
        "initialQuantity": row.get("initial_quantity"),
        "currentQuantity": row.get("current_quantity"),
        "notes": row.get("notes"),
        "currentWeek": current_week,
        "totalDurationWeeks": total_weeks,
        "progressPercent": progress_percent,
        "createdAt": row.get("created_at"),
        "updatedAt": row.get("updated_at"),
    }
