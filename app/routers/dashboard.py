"""One aggregate-counts endpoint for the admin dashboard, registered under
both its current and legacy paths."""

from __future__ import annotations

from fastapi import APIRouter

from ..database import get_connection, row_to_dict

router = APIRouter(tags=["dashboard"])


def build_dashboard_stats() -> dict:
    counts = {
        "cropCalendars": 0,
        "agrometAdvisories": 0,
        "poultryCalendars": 0,
        "poultryAdvisories": 0,
    }
    with get_connection() as connection:
        crop_count = connection.execute("SELECT COUNT(*) AS total, MAX(updated_at) AS last_updated FROM calendars WHERE calendar_type = 'seasonal'").fetchone()
        poultry_count = connection.execute("SELECT COUNT(*) AS total, MAX(updated_at) AS last_updated FROM calendars WHERE calendar_type = 'cycle'").fetchone()
        agromet_count = connection.execute("SELECT COUNT(*) AS total, MAX(updated_at) AS last_updated FROM weekly_advisories WHERE advisory_type = 'agromet-advisory'").fetchone()
        poultry_adv_count = connection.execute("SELECT COUNT(*) AS total, MAX(updated_at) AS last_updated FROM weekly_advisories WHERE advisory_type = 'poultry-advisory'").fetchone()
    last_updated = None
    for row, key in (
        (crop_count, "cropCalendars"),
        (agromet_count, "agrometAdvisories"),
        (poultry_count, "poultryCalendars"),
        (poultry_adv_count, "poultryAdvisories"),
    ):
        item = row_to_dict(row)
        counts[key] = item["total"] if item else 0
        if item and item.get("last_updated") and (last_updated is None or item["last_updated"] > last_updated):
            last_updated = item["last_updated"]
    counts["totalRecords"] = sum(counts.values())
    counts["lastUpdated"] = last_updated
    return counts


@router.get("/api/user/dashboard/stats")
@router.get("/user/dashboard/stats")
def get_dashboard_stats():
    return {"success": True, "data": build_dashboard_stats()}
