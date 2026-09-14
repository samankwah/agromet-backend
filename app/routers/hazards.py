"""Flood and drought monitoring.

Read endpoints never raise on upstream failure. Unlike `weather.ambee_request`
-- which is a proxy, where a 502 is the honest answer -- this is a monitoring
page, and a clearly-labelled stale reading beats an error screen. Failures
surface as `stale`, `degraded` and `error` fields on a 200 response.

`_active_overrides` and `_apply_overrides` are imported by `chat_context.py`
(lazily, inside a function, to sidestep a circular import at load time) so
the assistant quotes the exact same overridden band a farmer sees on the
hazards screen -- one implementation of override handling, not two that could
drift apart.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from .. import hazard_runtime
from ..database import get_connection
from ..deps import get_current_user
from ..domain import json_dumps, parse_json_list
from ..hazards import (
    BAND_ORDER,
    CLIMATOLOGY_LABEL,
    DATA_SOURCES,
    DISCHARGE_CLIMATOLOGY_LABEL,
    DROUGHT_WEIGHTS,
    FLOOD_WEIGHTS,
    HAZARD_LIMITS,
    SEVERITY_BANDS,
    advisories_for,
    resolve_region,
)
from ..schemas import HazardOverrideRequest

router = APIRouter(tags=["hazards"])


def _active_overrides(connection) -> dict[tuple[str, str], dict]:
    """Newest in-force override per (region, hazard).

    Expiry is evaluated in SQL rather than by a cleanup job, so a bulletin
    reverts to the computed value on its own the moment it lapses.
    """
    rows = connection.execute(
        """
        SELECT id, region, hazard, band, headline, advisory_json, issued_by,
               effective_from, effective_to, created_at
        FROM hazard_overrides
        WHERE effective_from <= CURRENT_TIMESTAMP
          AND (effective_to IS NULL OR effective_to >= CURRENT_TIMESTAMP)
        ORDER BY effective_from DESC, id DESC
        """
    ).fetchall()

    active: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["region"], row["hazard"])
        if key in active:
            continue  # the ORDER BY already put the newest first
        active[key] = {
            "id": row["id"],
            "band": row["band"],
            "headline": row["headline"],
            "advisories": parse_json_list(row["advisory_json"]),
            "issuedBy": row["issued_by"],
            "issuedAt": row["effective_from"],
            "effectiveTo": row["effective_to"],
        }
    return active


def _apply_overrides(region_payload: dict, overrides: dict[tuple[str, str], dict]) -> dict:
    """Overlay any in-force bulletin, keeping the computed reading visible.

    Copies the nested hazard blocks rather than mutating them. The snapshot this
    reads from is the long-lived process cache, so writing an override straight
    into it would contaminate every later request -- and the override would
    outlive its own expiry, which is the one thing the effective window exists
    to prevent.
    """
    region_payload = dict(region_payload)
    name = region_payload["region"]
    zone = region_payload["agroZone"]

    for hazard in ("flood", "drought"):
        block = dict(region_payload[hazard])
        region_payload[hazard] = block
        override = overrides.get((name, hazard))
        if override:
            block["computed"] = {"score": block["score"], "band": block["band"]}
            block["band"] = override["band"]
            block["overridden"] = True
            block["source"] = "gmet-bulletin"
            block["headline"] = override["headline"]
            block["issuedBy"] = override["issuedBy"]
            block["issuedAt"] = override["issuedAt"]
            block["effectiveTo"] = override["effectiveTo"]
            block["advisories"] = override["advisories"] or advisories_for(hazard, override["band"], zone)
        else:
            block["overridden"] = False
            block["source"] = "open-meteo"
            block["advisories"] = advisories_for(hazard, block["band"], zone)

    return region_payload


def _summarise(regions: list[dict]) -> dict:
    """National roll-up. Counts, not averages -- an average across sixteen
    regions hides the one region that is actually in trouble."""
    def counts(hazard: str) -> dict:
        tally = {name: 0 for name, _ in SEVERITY_BANDS}
        tally["unavailable"] = 0
        for region in regions:
            tally[region[hazard]["band"]] = tally.get(region[hazard]["band"], 0) + 1
        return tally

    def worst(hazard: str) -> dict | None:
        ranked = [r for r in regions if r[hazard].get("score") is not None]
        if not ranked:
            return None
        top = max(ranked, key=lambda r: (BAND_ORDER.get(r[hazard]["band"], 0), r[hazard]["score"]))
        return {
            "region": top["region"],
            "score": top[hazard]["score"],
            "band": top[hazard]["band"],
            "overridden": top[hazard].get("overridden", False),
        }

    def elevated(hazard: str) -> int:
        return sum(1 for r in regions if BAND_ORDER.get(r[hazard]["band"], 0) >= BAND_ORDER["moderate"])

    return {
        "regionCount": len(regions),
        "floodBands": counts("flood"),
        "droughtBands": counts("drought"),
        "floodElevated": elevated("flood"),
        "droughtElevated": elevated("drought"),
        "highestFlood": worst("flood"),
        "highestDrought": worst("drought"),
        "overriddenCount": sum(
            1 for r in regions for h in ("flood", "drought") if r[h].get("overridden")
        ),
    }


def _strip_series(region_payload: dict) -> dict:
    """Summary rows do not need 97 days of series data per region."""
    trimmed = dict(region_payload)
    trimmed.pop("series", None)
    discharge = dict(trimmed.get("discharge") or {})
    discharge.pop("dates", None)
    discharge.pop("values", None)
    trimmed["discharge"] = discharge
    return trimmed


@router.get("/api/hazards/summary")
async def hazard_summary(background: BackgroundTasks):
    await hazard_runtime.ensure_fresh()
    snapshot, _ = hazard_runtime.cached_snapshot()

    if not snapshot:
        return {
            "success": True,
            "data": {
                "regions": [],
                "national": None,
                "unavailable": True,
                **hazard_runtime.metadata(),
            },
        }

    # Serve what we have immediately and revalidate behind the response, so a
    # page load is never held open by a slow upstream.
    if hazard_runtime.is_stale():
        background.add_task(hazard_runtime.refresh, False)

    with get_connection() as connection:
        overrides = _active_overrides(connection)

    regions = [
        _strip_series(_apply_overrides(payload, overrides))
        for payload in snapshot.values()
    ]

    return {
        "success": True,
        "data": {
            "regions": regions,
            "national": _summarise(regions),
            "unavailable": False,
            **hazard_runtime.metadata(),
        },
    }


@router.get("/api/hazards/regions/{region}")
async def hazard_region(region: str, background: BackgroundTasks):
    resolved = resolve_region(region)
    if not resolved:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown region '{region}'.")

    await hazard_runtime.ensure_fresh()
    snapshot, _ = hazard_runtime.cached_snapshot()
    payload = snapshot.get(resolved)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hazard data is not available yet. Try again shortly.",
        )

    if hazard_runtime.is_stale():
        background.add_task(hazard_runtime.refresh, False)

    with get_connection() as connection:
        overrides = _active_overrides(connection)

    return {
        "success": True,
        "data": {
            **_apply_overrides(payload, overrides),
            **hazard_runtime.metadata(),
        },
    }


@router.get("/api/hazards/methodology")
def hazard_methodology():
    """Everything needed to audit a score, served to the UI disclosure panel."""
    return {
        "success": True,
        "data": {
            "baseline": hazard_runtime.CLIMATOLOGY.get("baseline") or CLIMATOLOGY_LABEL,
            "dischargeBaseline": (
                hazard_runtime.CLIMATOLOGY.get("dischargeBaseline") or DISCHARGE_CLIMATOLOGY_LABEL
            ),
            "bands": [{"band": name, "minScore": minimum} for name, minimum in SEVERITY_BANDS],
            "floodWeights": FLOOD_WEIGHTS,
            "droughtWeights": DROUGHT_WEIGHTS,
            "sources": DATA_SOURCES,
            "limits": HAZARD_LIMITS,
        },
    }


@router.get("/api/hazards/overrides")
def list_hazard_overrides(includeExpired: bool = False):
    clause = "" if includeExpired else (
        "WHERE effective_from <= CURRENT_TIMESTAMP "
        "AND (effective_to IS NULL OR effective_to >= CURRENT_TIMESTAMP)"
    )
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id, region, hazard, band, headline, advisory_json, issued_by,
                   effective_from, effective_to, created_at
            FROM hazard_overrides {clause}
            ORDER BY effective_from DESC, id DESC
            """
        ).fetchall()

    return {
        "success": True,
        "data": [
            {
                "id": row["id"],
                "region": row["region"],
                "hazard": row["hazard"],
                "band": row["band"],
                "headline": row["headline"],
                "advisories": parse_json_list(row["advisory_json"]),
                "issuedBy": row["issued_by"],
                "effectiveFrom": row["effective_from"],
                "effectiveTo": row["effective_to"],
            }
            for row in rows
        ],
    }


@router.post("/api/hazards/overrides")
def create_hazard_override(
    payload: HazardOverrideRequest,
    current_user: dict = Depends(get_current_user),
):
    resolved = resolve_region(payload.region)
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown region '{payload.region}'.",
        )
    if payload.band not in BAND_ORDER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Band must be one of: {', '.join(BAND_ORDER)}.",
        )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO hazard_overrides
                (region, hazard, band, headline, advisory_json, issued_by,
                 effective_from, effective_to, created_by)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?)
            """,
            (
                resolved,
                payload.hazard,
                payload.band,
                payload.headline,
                json_dumps(payload.advisories or []),
                payload.issuedBy,
                payload.effectiveFrom,
                payload.effectiveTo,
                current_user["id"],
            ),
        )
        override_id = cursor.lastrowid

    return {"success": True, "data": {"id": override_id, "region": resolved}}


@router.delete("/api/hazards/overrides/{override_id}")
def delete_hazard_override(override_id: int, current_user: dict = Depends(get_current_user)):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM hazard_overrides WHERE id = ?", (override_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Override not found.")
        connection.execute("DELETE FROM hazard_overrides WHERE id = ?", (override_id,))

    return {"success": True, "data": {"id": override_id}}


@router.post("/api/hazards/refresh")
async def refresh_hazards(current_user: dict = Depends(get_current_user)):
    updated = await hazard_runtime.refresh(force=True)
    return {
        "success": updated,
        "data": hazard_runtime.metadata(),
        "message": "Hazard indices refreshed." if updated else "Refresh failed; cached data retained.",
    }
