"""Every setting this service reads from its environment, resolved once at
import time -- the single source of truth every router imports from, instead
of each one growing its own `os.getenv` calls with its own defaults that can
drift from each other.

Split out of `main.py`, which used to run this resolution inline before
building the app. Nothing here changed behavior; it only moved.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def is_serverless_runtime() -> bool:
    return os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_local_env() -> None:
    """Load `.env` for local runs, and `.env.example` only when asked.

    `.env.example` used to be loaded automatically whenever `APP_ENV` was not
    the exact string "production". That is a trap on any host that is not
    Vercel: forget to set one variable and the example file supplies
    `SECRET_KEY=change-me-for-production` and `DEBUG=true` to a live server,
    silently, with nothing in the logs to say so.

    It stays available because it is genuinely useful for a first run on a fresh
    clone, but now it has to be asked for by name.
    """
    if is_serverless_runtime():
        return

    load_env_file(BACKEND_ROOT / ".env")
    if os.getenv("USE_EXAMPLE_ENV", "").lower() in ("1", "true", "yes"):
        load_env_file(BACKEND_ROOT / ".env.example")


def resolve_database_path(configured_path: str | None) -> str:
    if is_serverless_runtime():
        if configured_path and Path(configured_path).is_absolute():
            return configured_path
        database_name = Path(configured_path).name if configured_path else "agromet.db"
        return str(Path(tempfile.gettempdir()) / database_name)

    return configured_path or str(BACKEND_ROOT / "agromet.db")


load_local_env()


APP_NAME = os.getenv("APP_NAME", "AgroMet Backend")
APP_ENV = os.getenv("APP_ENV", "production" if is_serverless_runtime() else "development")
DEBUG = os.getenv("DEBUG", "false" if APP_ENV == "production" else "true").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
FRONTEND_ORIGINS = [origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if origin.strip()]
LOCAL_DEV_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$" if APP_ENV != "production" else None
DATABASE_PATH = resolve_database_path(os.getenv("DATABASE_PATH"))
# A Postgres connection string switches the whole app off SQLite. Unset
# (the default everywhere but a real deployment) keeps the zero-config
# SQLite file above -- see database.py's module docstring for why a
# serverless deployment must not run on the SQLite fallback.
DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
# A ceiling on the answer, in tokens. Nothing bounded this before, so a single
# question could bill for a two-thousand-word essay that a farmer on a phone was
# never going to read. The prompt asks for about 120 words; this is roughly
# three times that, so it caps the pathological case without truncating a normal
# answer mid-sentence.
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "400"))
# Low, not zero. These are questions with correct answers -- planting windows,
# what a rainfall figure means -- and invention is the failure mode that matters.
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
# Shorter than the 30s this used to allow. On a serverless host the platform
# kills the invocation on its own schedule, and a fallback answer served at 20s
# is worth more than a platform error page at 30.
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
# Speech to text for the composer mic.
TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")

# The chat quota. See `rate_limit.py` for what these can and cannot promise.
CHAT_RATE_LIMIT = int(os.getenv("CHAT_RATE_LIMIT", "20"))
CHAT_RATE_WINDOW_SECONDS = int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "900"))
CHAT_DAILY_LIMIT = int(os.getenv("CHAT_DAILY_LIMIT", "120"))
KINDWISE_API_KEY = os.getenv("KINDWISE_API_KEY", "")
KINDWISE_CROP_HEALTH_API_KEY = os.getenv("KINDWISE_CROP_HEALTH_API_KEY", KINDWISE_API_KEY)
KINDWISE_PLANT_ID_API_KEY = os.getenv("KINDWISE_PLANT_ID_API_KEY", KINDWISE_API_KEY)
KINDWISE_CROP_HEALTH_URL = os.getenv("KINDWISE_CROP_HEALTH_URL", "https://crop.kindwise.com")
KINDWISE_PLANT_ID_URL = os.getenv("KINDWISE_PLANT_ID_URL", "https://api.plant.id/v3")
AMBEE_API_KEY = os.getenv("AMBEE_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
AMBEE_BASE_URL = os.getenv("AMBEE_BASE_URL", "https://api.ambeedata.com")
