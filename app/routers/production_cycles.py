"""A production cycle is one batch (a planting, a poultry flock) tracked
against the calendar it was started from."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from ..database import get_connection, row_to_dict
from ..domain import serialize_cycle
from ..records import fetch_calendar_activities
from ..schemas import ProductionCycleCreateRequest, ProductionCycleUpdateRequest

router = APIRouter(prefix="/api/production-cycles", tags=["production-cycles"])


@router.get("")
def list_production_cycles(
    status_filter: str | None = Query(default=None, alias="status"),
    commodity: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    clauses = ["1=1"]
    params: list[object] = []
    if status_filter:
        clauses.append("pc.status = ?")
        params.append(status_filter)
    if commodity:
        clauses.append("LOWER(pc.commodity) = LOWER(?)")
        params.append(commodity)
    params.append(limit)
    query = f"""
        SELECT pc.* FROM production_cycles pc
        WHERE {' AND '.join(clauses)}
        ORDER BY pc.created_at DESC, pc.id DESC
        LIMIT ?
    """
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    data = [serialize_cycle(row_to_dict(row)) for row in rows]
    return {
        "success": True,
        "data": data,
        "total": len(data),
        "summary": {
            "active": sum(1 for item in data if item["status"] == "active"),
            "paused": sum(1 for item in data if item["status"] == "paused"),
            "completed": sum(1 for item in data if item["status"] == "completed"),
        },
    }


@router.post("")
def create_production_cycle(payload: ProductionCycleCreateRequest):
    with get_connection() as connection:
        calendar_row = connection.execute("SELECT * FROM calendars WHERE id = ?", (payload.calendarId,)).fetchone()
        if not calendar_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calendar not found.")
        calendar = row_to_dict(calendar_row)
        # Snapshot the calendar's length onto the cycle. serialize_cycle
        # divides by this to derive currentWeek and progressPercent, so
        # leaving it null made every cycle report "week 1 of 1, 100%".
        # A poultry calendar carries its length in cycle_duration, a crop
        # one in total_weeks.
        duration_weeks = calendar.get("cycle_duration") or calendar.get("total_weeks") or 1

        cursor = connection.execute(
            """
            INSERT INTO production_cycles(
                calendar_id, batch_name, commodity, start_date, status,
                initial_quantity, current_quantity, notes, total_duration_weeks
            )
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)
            """,
            (
                payload.calendarId,
                payload.batchName,
                calendar["crop"],
                payload.startDate,
                payload.initialQuantity,
                payload.initialQuantity,
                payload.notes,
                duration_weeks,
            ),
        )
        cycle_id = cursor.lastrowid
        row = connection.execute("SELECT * FROM production_cycles WHERE id = ?", (cycle_id,)).fetchone()
    return {"success": True, "data": serialize_cycle(row_to_dict(row))}


@router.put("/{cycle_id}")
def update_production_cycle(cycle_id: int, payload: ProductionCycleUpdateRequest):
    updates = []
    params: list[object] = []
    mapping = {
        "status": payload.status,
        "batch_name": payload.batchName,
        "initial_quantity": payload.initialQuantity,
        "current_quantity": payload.currentQuantity,
        "notes": payload.notes,
    }
    for column, value in mapping.items():
        if value is not None:
            updates.append(f"{column} = ?")
            params.append(value)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No production cycle updates provided.")
    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(cycle_id)

    with get_connection() as connection:
        existing = connection.execute("SELECT * FROM production_cycles WHERE id = ?", (cycle_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Production cycle not found.")
        connection.execute(f"UPDATE production_cycles SET {', '.join(updates)} WHERE id = ?", params)
        row = connection.execute("SELECT * FROM production_cycles WHERE id = ?", (cycle_id,)).fetchone()
    return {"success": True, "data": serialize_cycle(row_to_dict(row))}


@router.get("/{cycle_id}/current-activities")
def get_current_cycle_activities(cycle_id: int):
    with get_connection() as connection:
        cycle_row = connection.execute("SELECT * FROM production_cycles WHERE id = ?", (cycle_id,)).fetchone()
        if not cycle_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Production cycle not found.")
        cycle = serialize_cycle(row_to_dict(cycle_row))
        activities = fetch_calendar_activities(connection, cycle["calendarId"], current_week=cycle["currentWeek"])
    return {
        "success": True,
        "data": {
            "currentWeek": cycle["currentWeek"],
            "totalWeeks": cycle["totalDurationWeeks"],
            "progressPercent": cycle["progressPercent"],
            "activities": activities,
            "completedActivities": [],
        },
    }
