"""The Ambee proxy and the Open-Meteo bundle the mobile app's forecast
screens read."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from .. import config, weather_runtime

router = APIRouter(tags=["weather"])


async def ambee_request(path: str, *, params: dict, timeout: float = 20.0) -> dict:
    if not config.AMBEE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ambee integration is unavailable because the API key is not configured.",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{config.AMBEE_BASE_URL.rstrip('/')}/{path.lstrip('/')}",
                headers={
                    "Content-type": "application/json",
                    "x-api-key": config.AMBEE_API_KEY,
                },
                params=params,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text.strip() if exc.response is not None and exc.response.text else ""
        detail = {
            "message": "Ambee upstream request failed.",
            "upstreamStatus": exc.response.status_code if exc.response is not None else None,
            "path": path,
            "params": params,
            "upstreamBody": response_text[:300] if response_text else None,
        }
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc
    except Exception as exc:  # pragma: no cover - defensive proxy handling
        detail = {
            "message": "Ambee upstream request failed.",
            "path": path,
            "params": params,
            "error": str(exc),
        }
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc


@router.get("/api/ambee/weather/latest/by-lat-lng")
async def get_ambee_latest_weather(
    lat: float = Query(...),
    lng: float = Query(...),
):
    payload = await ambee_request(
        "/weather/latest/by-lat-lng",
        params={"lat": lat, "lng": lng},
    )
    return payload


@router.get("/api/ambee/weather/forecast/by-lat-lng")
async def get_ambee_forecast_weather(
    lat: float = Query(...),
    lng: float = Query(...),
):
    payload = await ambee_request(
        "/weather/forecast/by-lat-lng",
        params={"lat": lat, "lng": lng},
    )
    return payload


@router.get("/api/weather/bundle")
async def get_weather_bundle(
    background: BackgroundTasks,
    lat: float = Query(...),
    lng: float = Query(...),
):
    """Current conditions, seven days and the hourly series for one point.

    One route rather than three because Open-Meteo returns all of it in a single
    request, and the app needs all of it: Home reads `current`, the 7-Day
    segment reads `daily`, and day detail reads `hourly`.

    The payload is Open-Meteo's own, wrapped in the house envelope rather than
    normalised here. The mobile app falls back to calling Open-Meteo directly
    when this backend is unreachable, so the mapping has to live somewhere both
    paths share — which means the client, in TypeScript, written once.

    Never raises on upstream failure: a stale bundle beats an error page, and an
    empty one is reported as `unavailable` so the client can fall back.
    """
    key = weather_runtime.cache_key(lat, lng)
    await weather_runtime.ensure_fresh(lat, lng)
    bundle = weather_runtime.cached_bundle(key)

    if not bundle:
        return {
            "success": True,
            "data": None,
            "unavailable": True,
            "meta": weather_runtime.metadata(key),
        }

    # Serve what we have and revalidate behind the response.
    if weather_runtime.is_stale(key):
        background.add_task(weather_runtime.refresh, lat, lng, False)

    return {
        "success": True,
        "data": bundle,
        "unavailable": False,
        "meta": weather_runtime.metadata(key),
    }
