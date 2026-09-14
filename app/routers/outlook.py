"""The weeks 2-to-4 subseasonal outlook and the hourly precipitation field
behind the rain map -- two independent GEFS/Open-Meteo-derived products,
each with its own runtime cache, sharing only the serve-then-revalidate
shape `hazards.py`'s endpoints also use."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from .. import precip_runtime, s2s_runtime

router = APIRouter(tags=["outlook"])


def _strip_series(cell: dict) -> dict:
    """The map's copy of a cell, without its 15-day chart series.

    The series is 90 numbers per cell per variable; across 165 cells that is
    roughly 100 KB the map never draws. It is served instead by
    `/api/outlook/subseasonal/series`, one cell at a time, when a district is
    actually selected.
    """
    trimmed = dict(cell)
    for variable in ("rainfall", "temperature"):
        reading = trimmed.get(variable)
        if reading:
            trimmed[variable] = {key: value for key, value in reading.items() if key != "series"}
    return trimmed


@router.get("/api/outlook/subseasonal")
async def subseasonal_outlook(background: BackgroundTasks):
    """The weeks 2-to-4 outlook over the model's own grid.

    Serves the 165-point GEFS field, each cell carrying both the tercile
    probabilities and the deterministic ensemble mean. Admin boundaries are the
    client's business: it already ships Ghana's regions and districts, and
    overlaying them here would put the same geometry in two places.

    Same serve-then-revalidate shape as `hazard_summary`: a stale snapshot is
    returned immediately and refreshed behind the response, so a page load never
    waits on the ensemble fetch.

    `unavailable` means nothing could be computed at all. A *missing baseline*
    no longer empties the response, because the deterministic field needs none --
    those cells simply carry no probabilities, and the client shows the
    deterministic view for them.
    """
    await s2s_runtime.ensure_fresh()
    snapshot, _ = s2s_runtime.cached_snapshot()

    if not snapshot:
        return {
            "success": True,
            "data": {
                "cells": [],
                "unavailable": True,
                **s2s_runtime.metadata(),
            },
        }

    if s2s_runtime.is_stale():
        background.add_task(s2s_runtime.refresh, False)

    return {
        "success": True,
        "data": {
            "cells": [_strip_series(cell) for cell in snapshot.values()],
            "unavailable": False,
            **s2s_runtime.metadata(),
        },
    }


@router.get("/api/precipitation/field")
async def precipitation_field(background: BackgroundTasks):
    """Hourly rainfall over Ghana's land grid, for the rain map's forecast half.

    Proxied rather than fetched by the app, which is the exception the rain map
    forces. `fetchCurrentBatch` in the client goes direct and is right to: it is
    32 points for a display strip. This is several hundred, and Open-Meteo
    weights a request by its location count, so direct it would consume the free
    tier in proportion to how many people open the screen. Cached here it is one
    upstream call an hour for everyone. See `precip_runtime` for the arithmetic
    that fixes the interval.

    Same serve-then-revalidate shape as `subseasonal_outlook`: a stale snapshot
    is returned immediately and refreshed behind the response, so opening the map
    never waits on the fetch.

    The grid travels with the values because the two are positional -- every row
    of `values` is parallel to `grid` -- and a client that inferred the lattice
    itself would silently mis-draw the whole field the day either side changed
    its rounding.
    """
    await precip_runtime.ensure_fresh()
    snapshot, _ = precip_runtime.cached_snapshot()

    if not snapshot:
        return {
            "success": True,
            "data": {
                "grid": [],
                "times": [],
                "values": [],
                "unavailable": True,
                **precip_runtime.metadata(),
            },
        }

    if precip_runtime.is_stale():
        background.add_task(precip_runtime.refresh, False)

    return {
        "success": True,
        "data": {
            "grid": [[lat, lng] for lat, lng in precip_runtime.GRID],
            "times": snapshot["times"],
            "values": snapshot["values"],
            "unavailable": False,
            **precip_runtime.metadata(),
        },
    }


@router.get("/api/outlook/subseasonal/series")
async def subseasonal_series(lat: float, lng: float):
    """One grid cell's day-by-day ensemble spread, for the detail chart.

    Read straight from the cached snapshot -- the members were reduced when the
    field was fetched, so selecting a district costs no upstream call. Returns
    404 rather than an empty series when the place falls outside the grid, so a
    bad coordinate is a visible error rather than a flat chart.
    """
    await s2s_runtime.ensure_fresh()
    cell = s2s_runtime.cell_at(lat, lng)
    if not cell:
        raise HTTPException(status_code=404, detail="No subseasonal outlook covers that location.")

    return {
        "success": True,
        "data": {
            "id": cell["id"],
            "lat": cell["lat"],
            "lng": cell["lng"],
            "rainfall": (cell.get("rainfall") or {}).get("series"),
            "temperature": (cell.get("temperature") or {}).get("series"),
            **s2s_runtime.metadata(),
        },
    }
