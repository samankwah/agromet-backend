"""Liveness and integration-status endpoints -- what a load balancer and an
operator each check first."""

from __future__ import annotations

from fastapi import APIRouter

from .. import config
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
def api_health():
    return HealthResponse(status="healthy", app=config.APP_NAME)


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", app=config.APP_NAME)


@router.get("/api/integrations/status")
def integrations_status():
    return {
        "success": True,
        "data": {
            "kindwise": {
                "cropHealthConfigured": bool(config.KINDWISE_CROP_HEALTH_API_KEY),
                "plantIdConfigured": bool(config.KINDWISE_PLANT_ID_API_KEY),
            },
            "translation": {
                "provider": "google-translate-fallback",
                "serverTtsEnabled": False,
            },
            "ambee": {
                "configured": bool(config.AMBEE_API_KEY),
                "baseUrl": config.AMBEE_BASE_URL,
            },
            # The assistant's own provider, and the one that was missing here.
            # Without it, "why is every chat answer generic?" could not be
            # answered from the outside, which is exactly when you need to ask.
            "openai": {
                "configured": bool(config.OPENAI_API_KEY),
                "model": config.OPENAI_MODEL,
                "transcribeModel": config.TRANSCRIPTION_MODEL,
            },
        },
    }
