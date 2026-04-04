from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from .auth import create_access_token, decode_access_token, hash_password, verify_password
from .database import decode_payload, encode_payload, get_connection, init_db, row_to_dict, set_database_path
from .diagnosis import (
    SUPPORTED_IMAGE_ANALYSIS_TYPES,
    diagnose_crop_image,
    format_image_analysis_response,
)
from .domain import (
    build_activity_rows,
    build_advisory_payload,
    generate_sample_activities,
    infer_calendar_type,
    infer_crop_name,
    infer_weeks,
    json_dumps,
    parse_json_list,
    serialize_advisory,
    serialize_calendar,
    serialize_calendar_activity,
    serialize_cycle,
)
from .schemas import (
    ChatRequest,
    CommodityResponse,
    CommodityTrendResponse,
    CropDiagnosisRequest,
    FAQResponse,
    HealthResponse,
    ImageAnalysisRequest,
    MarketCenterResponse,
    ProductionCycleCreateRequest,
    ProductionCycleUpdateRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
    UserResponse,
)
from .spreadsheet_parser import (
    build_calendar_preview_payload,
    build_advisory_preview_payload,
    build_committed_calendar_payload,
    build_committed_advisory_payload,
    discard_preview_payload,
    get_preview_payload,
)


def load_local_env() -> None:
    backend_root = Path(__file__).resolve().parent.parent
    env_candidates = [backend_root / ".env", backend_root / ".env.example"]

    for env_path in env_candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()


APP_NAME = os.getenv("APP_NAME", "AgroMet Backend")
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
FRONTEND_ORIGINS = [origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if origin.strip()]
DATABASE_PATH = os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parent.parent / "agromet.db"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
KINDWISE_API_KEY = os.getenv("KINDWISE_API_KEY", "")
KINDWISE_CROP_HEALTH_API_KEY = os.getenv("KINDWISE_CROP_HEALTH_API_KEY", KINDWISE_API_KEY)
KINDWISE_PLANT_ID_API_KEY = os.getenv("KINDWISE_PLANT_ID_API_KEY", KINDWISE_API_KEY)
KINDWISE_CROP_HEALTH_URL = os.getenv("KINDWISE_CROP_HEALTH_URL", "https://crop.kindwise.com")
KINDWISE_PLANT_ID_URL = os.getenv("KINDWISE_PLANT_ID_URL", "https://api.plant.id/v3")
GHANANLP_PRIMARY_KEY = os.getenv("GHANANLP_PRIMARY_KEY", "")
GHANANLP_SECONDARY_KEY = os.getenv("GHANANLP_SECONDARY_KEY", "")
GHANANLP_TRANSLATE_URL = os.getenv("GHANANLP_TRANSLATE_URL", "https://translation-api.ghananlp.org/v1")
GHANANLP_TTS_URL = os.getenv("GHANANLP_TTS_URL", "https://translation-api.ghananlp.org/tts/v1")
AMBEE_API_KEY = os.getenv("AMBEE_API_KEY", "")
AMBEE_BASE_URL = os.getenv("AMBEE_BASE_URL", "https://api.ambeedata.com")

set_database_path(DATABASE_PATH)
init_db()

app = FastAPI(title=APP_NAME, debug=DEBUG)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAQ_MESSAGES = {
    "when-to-plant-maize": "In Ghana, maize is typically planted at the start of the rains. Target April to June for the major season and September to November for the minor season, depending on local rainfall onset.",
    "when-to-plant-rice": "Rice planting depends on irrigation and region. Rainfed systems usually start with the first dependable rains, while irrigated rice can be staggered year-round.",
    "maize-fertilizer": "Use a soil test where possible. A practical starting point is a balanced basal NPK application followed by a nitrogen top-dress at early vegetative growth.",
    "rainy-season-farming": "Prepare fields early, use drainage where needed, and match planting windows to local rainfall onset instead of fixed calendar dates.",
}

GHANANLP_TTS_SPEAKERS = [
    {"id": "twi_speaker_4", "language": "tw", "name": "Twi Speaker 4"},
    {"id": "twi_speaker_5", "language": "tw", "name": "Twi Speaker 5"},
    {"id": "twi_speaker_6", "language": "tw", "name": "Twi Speaker 6"},
    {"id": "twi_speaker_7", "language": "tw", "name": "Twi Speaker 7"},
    {"id": "twi_speaker_8", "language": "tw", "name": "Twi Speaker 8"},
    {"id": "twi_speaker_9", "language": "tw", "name": "Twi Speaker 9"},
    {"id": "ewe_speaker_3", "language": "ee", "name": "Ewe Speaker 3"},
    {"id": "ewe_speaker_4", "language": "ee", "name": "Ewe Speaker 4"},
    {"id": "kikuyu_speaker_1", "language": "ki", "name": "Kikuyu Speaker 1"},
    {"id": "kikuyu_speaker_5", "language": "ki", "name": "Kikuyu Speaker 5"},
]


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_access_token(token, SECRET_KEY)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")

    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, email, name, created_at FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    user = row_to_dict(row)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated user no longer exists.")
    return user


def serialize_user(user: dict) -> UserResponse:
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user.get("name"),
        created_at=user["created_at"],
    )


def get_optional_user(authorization: str | None) -> dict | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    try:
        return get_current_user(token)
    except HTTPException:
        return None


def save_diagnosis_record(owner_id: int, diagnosis: dict, crop: str | None, region: str | None) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO diagnosis_records(
                owner_id, provider_product, status, plant, disease, confidence, severity,
                source, crop_context, region_context, result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_id,
                diagnosis.get("providerProduct"),
                diagnosis.get("status"),
                diagnosis.get("plant"),
                diagnosis.get("disease"),
                diagnosis.get("confidence"),
                diagnosis.get("severity"),
                diagnosis.get("source"),
                crop,
                region,
                encode_payload(diagnosis),
            ),
        )
    return cursor.lastrowid


def serialize_diagnosis_record(record: dict) -> dict:
    result = decode_payload(record.get("result_json"))
    return {
        "id": record["id"],
        "providerProduct": record.get("provider_product"),
        "status": record.get("status"),
        "plant": record.get("plant"),
        "disease": record.get("disease"),
        "confidence": record.get("confidence"),
        "severity": record.get("severity"),
        "source": record.get("source"),
        "cropContext": record.get("crop_context"),
        "regionContext": record.get("region_context"),
        "createdAt": record.get("created_at"),
        "result": result,
    }


def normalize_record(record: dict) -> dict:
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


def get_ghananlp_keys() -> list[str]:
    return [key for key in [GHANANLP_PRIMARY_KEY, GHANANLP_SECONDARY_KEY] if key]


async def ghananlp_request(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
    response_type: str = "json",
    timeout: float = 20.0,
):
    keys = get_ghananlp_keys()
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GhanaNLP integration is unavailable because no API key is configured.",
        )

    last_error: Exception | None = None
    for index, api_key in enumerate(keys):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method,
                    path,
                    headers={
                        "Content-Type": "application/json",
                        "Ocp-Apim-Subscription-Key": api_key,
                    },
                    json=json_body,
                    params=params,
                )
                response.raise_for_status()
                if response_type == "bytes":
                    return response.content
                if not response.content:
                    return {}
                content_type = response.headers.get("content-type", "").lower()
                if "json" in content_type:
                    return response.json()
                return response.text
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code not in {401, 403} or index == len(keys) - 1:
                break
        except Exception as exc:  # pragma: no cover - defensive proxy handling
            last_error = exc
            break

    detail = str(last_error) if last_error else "Unknown GhanaNLP error."
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)


async def ambee_request(path: str, *, params: dict, timeout: float = 20.0) -> dict:
    if not AMBEE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ambee integration is unavailable because the API key is not configured.",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{AMBEE_BASE_URL.rstrip('/')}/{path.lstrip('/')}",
                headers={
                    "Content-type": "application/json",
                    "x-api-key": AMBEE_API_KEY,
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


async def build_chat_reply(message: str, user_context: dict) -> str:
    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/responses",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENAI_MODEL,
                        "input": [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "You are AgroMet AI, a practical agricultural assistant focused on Ghanaian weather and farming support."
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": message,
                                    }
                                ],
                            },
                        ],
                    },
                )
                response.raise_for_status()
                payload = response.json()
                output = payload.get("output", [])
                for item in output:
                    for content in item.get("content", []):
                        text = content.get("text")
                        if text:
                            return text
        except Exception:
            pass

    context_region = user_context.get("region") or "your area"
    return (
        f"I do not have a live AI provider configured yet, so here is a practical fallback. "
        f"For {context_region}, focus on rainfall timing, field drainage, seed quality, and pest monitoring. "
        f"You asked: {message}"
    )


@app.get("/api/health", response_model=HealthResponse)
def api_health():
    return HealthResponse(status="healthy", app=APP_NAME)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", app=APP_NAME)


@app.get("/api/integrations/status")
def integrations_status():
    return {
        "success": True,
        "data": {
            "kindwise": {
                "cropHealthConfigured": bool(KINDWISE_CROP_HEALTH_API_KEY),
                "plantIdConfigured": bool(KINDWISE_PLANT_ID_API_KEY),
            },
            "ghananlp": {
                "primaryConfigured": bool(GHANANLP_PRIMARY_KEY),
                "secondaryConfigured": bool(GHANANLP_SECONDARY_KEY),
            },
            "ambee": {
                "configured": bool(AMBEE_API_KEY),
                "baseUrl": AMBEE_BASE_URL,
            },
        },
    }


@app.post("/api/v1/auth/register", response_model=RegisterResponse)
def register(payload: RegisterRequest):
    with get_connection() as connection:
        existing = connection.execute("SELECT id FROM users WHERE email = ?", (payload.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")

        cursor = connection.execute(
            "INSERT INTO users(email, name, password_hash) VALUES (?, ?, ?)",
            (payload.email, payload.name, hash_password(payload.password)),
        )
        user_id = cursor.lastrowid
        row = connection.execute(
            "SELECT id, email, name, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    return RegisterResponse(user=serialize_user(row_to_dict(row)))


@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, email, name, password_hash, created_at FROM users WHERE email = ?",
            (form_data.username,),
        ).fetchone()

    user = row_to_dict(row)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")

    token = create_access_token(user["email"], SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES)
    return TokenResponse(
        access_token=token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=serialize_user(user),
    )


@app.get("/api/v1/auth/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user)):
    return serialize_user(current_user)


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    reply = await build_chat_reply(payload.message, payload.userContext)
    return {"success": True, "message": reply}


@app.post("/api/v1/translate")
async def translate_text(request: Request):
    payload = dict(await request.json())
    text = str(payload.get("in") or "").strip()
    lang = str(payload.get("lang") or "").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Translation input text is required.")
    if "-" not in lang:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Translation language pair must be in the form 'en-tw'.")

    translated = await ghananlp_request(
        "POST",
        f"{GHANANLP_TRANSLATE_URL.rstrip('/')}/translate",
        json_body={"in": text, "lang": lang},
    )

    if isinstance(translated, str):
        output = translated
    else:
        output = translated.get("out") or translated.get("translation") or translated.get("text")
    if not output:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="GhanaNLP translation did not return any translated text.")
    return {"out": output, "translation": output}


@app.get("/api/tts/languages")
async def list_tts_languages():
    languages = await ghananlp_request(
        "GET",
        f"{GHANANLP_TRANSLATE_URL.rstrip('/')}/languages",
    )
    return languages


@app.get("/api/tts/speakers")
def list_tts_speakers():
    return GHANANLP_TTS_SPEAKERS


@app.post("/api/tts/tts")
@app.post("/api/tts/synthesize")
async def synthesize_speech(request: Request):
    payload = dict(await request.json())
    text = str(payload.get("text") or "").strip()
    language = str(payload.get("language") or "en").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="TTS text is required.")

    audio = await ghananlp_request(
        "POST",
        f"{GHANANLP_TTS_URL.rstrip('/')}/tts",
        json_body={"text": text, "language": language},
        response_type="bytes",
        timeout=30.0,
    )
    return Response(content=audio, media_type="audio/mpeg")


@app.get("/api/ambee/weather/latest/by-lat-lng")
async def get_ambee_latest_weather(
    lat: float = Query(...),
    lng: float = Query(...),
):
    payload = await ambee_request(
        "/weather/latest/by-lat-lng",
        params={"lat": lat, "lng": lng},
    )
    return payload


@app.get("/api/ambee/weather/forecast/by-lat-lng")
async def get_ambee_forecast_weather(
    lat: float = Query(...),
    lng: float = Query(...),
):
    payload = await ambee_request(
        "/weather/forecast/by-lat-lng",
        params={"lat": lat, "lng": lng},
    )
    return payload


@app.post("/api/crop-diagnosis")
async def crop_diagnosis(payload: CropDiagnosisRequest, authorization: str | None = Header(default=None)):
    context = dict(payload.context)
    context.setdefault("language", payload.language)
    diagnosis = await diagnose_crop_image(
        KINDWISE_CROP_HEALTH_API_KEY,
        KINDWISE_CROP_HEALTH_URL,
        KINDWISE_PLANT_ID_API_KEY,
        KINDWISE_PLANT_ID_URL,
        payload.image,
        crop=payload.crop,
        region=payload.region,
        language=payload.language,
        context=context,
    )
    current_user = get_optional_user(authorization)
    if current_user and diagnosis["status"] == "ok":
        record_id = save_diagnosis_record(current_user["id"], diagnosis, payload.crop, payload.region)
        diagnosis["historyId"] = record_id
    return diagnosis


@app.post("/api/image-analysis")
async def image_analysis(payload: ImageAnalysisRequest, authorization: str | None = Header(default=None)):
    if payload.analysisType not in SUPPORTED_IMAGE_ANALYSIS_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported analysisType '{payload.analysisType}'. Supported types: {', '.join(sorted(SUPPORTED_IMAGE_ANALYSIS_TYPES))}.",
        )

    diagnosis = await diagnose_crop_image(
        KINDWISE_CROP_HEALTH_API_KEY,
        KINDWISE_CROP_HEALTH_URL,
        KINDWISE_PLANT_ID_API_KEY,
        KINDWISE_PLANT_ID_URL,
        payload.image,
        crop=payload.context.get("crop"),
        region=payload.context.get("region"),
        language=payload.context.get("language"),
        context=payload.context,
    )
    current_user = get_optional_user(authorization)
    if current_user and diagnosis["status"] == "ok":
        save_diagnosis_record(current_user["id"], diagnosis, payload.context.get("crop"), payload.context.get("region"))
    if diagnosis["status"] != "ok":
        return {
            "analysis": None,
            "source": diagnosis["source"],
            "providerProduct": diagnosis.get("providerProduct"),
            "status": diagnosis["status"],
            "message": diagnosis["remedy"],
        }
    return format_image_analysis_response(diagnosis)


@app.get("/api/diagnosis-history")
def list_diagnosis_history(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    status_filter: str | None = Query(default=None, alias="status"),
    crop: str | None = Query(default=None),
):
    clauses = ["owner_id = ?"]
    params: list[object] = [current_user["id"]]
    if status_filter:
        clauses.append("status = ?")
        params.append(status_filter)
    if crop:
        clauses.append("LOWER(crop_context) = LOWER(?)")
        params.append(crop)
    params.append(limit)
    query = f"""
        SELECT * FROM diagnosis_records
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
    """
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    data = [serialize_diagnosis_record(row_to_dict(row)) for row in rows]
    return {"success": True, "data": data, "total": len(data)}


@app.get("/api/diagnosis-history/{record_id}")
def get_diagnosis_history_item(record_id: int, current_user: dict = Depends(get_current_user)):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM diagnosis_records WHERE id = ? AND owner_id = ?",
            (record_id, current_user["id"]),
        ).fetchone()
    record = row_to_dict(row)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis history item not found.")
    return {"success": True, "data": serialize_diagnosis_record(record)}


@app.delete("/api/diagnosis-history/{record_id}")
def delete_diagnosis_history_item(record_id: int, current_user: dict = Depends(get_current_user)):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM diagnosis_records WHERE id = ? AND owner_id = ?",
            (record_id, current_user["id"]),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis history item not found.")
        connection.execute(
            "DELETE FROM diagnosis_records WHERE id = ? AND owner_id = ?",
            (record_id, current_user["id"]),
        )
    return {"success": True, "message": "Diagnosis history item deleted successfully."}


@app.get("/api/faq/{topic}", response_model=FAQResponse)
def faq(topic: str):
    message = FAQ_MESSAGES.get(topic)
    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="FAQ topic not found.")
    return FAQResponse(success=True, message=message)


@app.post("/api/agricultural-data/upload")
async def upload_agricultural_data(
    request: Request,
    dataType: str = Form(...),
    file: UploadFile | None = File(default=None),
    authorization: str | None = Header(default=None),
):
    owner_id = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            current_user = get_current_user(token)
            owner_id = current_user["id"]
        except HTTPException:
            owner_id = None

    payload = {}
    if file:
        payload["originalFilename"] = file.filename

    form = await request.form()
    for key, value in form.multi_items():
        if key == "dataType" or key == "file":
            continue
        payload[key] = value

    record_payload = payload
    if file:
        contents = await file.read()
        record_payload["fileSize"] = len(contents)
        record_payload["fileName"] = file.filename
        record_payload["fileContentType"] = file.content_type

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                owner_id,
                dataType,
                encode_payload(record_payload),
                file.filename if file else None,
                file.content_type if file else None,
                record_payload.get("fileSize"),
            ),
        )
        record_id = cursor.lastrowid
        row = connection.execute(
            "SELECT * FROM agricultural_records WHERE id = ?",
            (record_id,),
        ).fetchone()
        calendar = insert_calendar_from_record(connection, record_id, dataType, record_payload)

    response = {"success": True, "data": normalize_record(row_to_dict(row))}
    if calendar:
        response["calendar"] = calendar
        response["calendarType"] = calendar["calendarType"]
        response["commodity"] = calendar["commodity"]
        response["activities"] = len(calendar.get("sampleActivities", []))
        response["message"] = "Agricultural data uploaded and calendar processed successfully."
    return response


@app.get("/api/agricultural-data/{data_type}")
def list_agricultural_data(data_type: str):
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM agricultural_records WHERE data_type = ? ORDER BY created_at DESC, id DESC",
            (data_type,),
        ).fetchall()

    return {"success": True, "data": [normalize_record(row_to_dict(row)) for row in rows]}


@app.delete("/api/agricultural-data/{data_type}/{record_id}")
def delete_agricultural_data(data_type: str, record_id: int):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM agricultural_records WHERE id = ? AND data_type = ?",
            (record_id, data_type),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agricultural record not found.")
        connection.execute(
            "DELETE FROM agricultural_records WHERE id = ? AND data_type = ?",
            (record_id, data_type),
        )

    return {"success": True, "message": "Agricultural record deleted successfully."}


@app.post("/api/weekly-advisories/upload")
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


async def _preview_calendar_upload(file: UploadFile, metadata: dict, calendar_type: str):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Spreadsheet file is required.")
    contents = await file.read()
    preview = build_calendar_preview_payload(contents, metadata, calendar_type)
    return {"success": True, "data": preview}


@app.post("/api/crop-calendars/preview")
async def preview_crop_calendar(
    file: UploadFile = File(...),
    region: str = Form(...),
    district: str = Form(...),
    crop: str = Form(...),
    title: str = Form(default=""),
    description: str = Form(default=""),
    year: int | None = Form(default=None),
):
    return await _preview_calendar_upload(
        file,
        {
            "region": region,
            "district": district,
            "crop": crop,
            "title": title or f"{crop} Calendar",
            "description": description,
            "year": year,
        },
        "crop-calendar",
    )


@app.post("/api/poultry-calendars/preview")
async def preview_poultry_calendar(
    file: UploadFile = File(...),
    region: str = Form(...),
    district: str = Form(...),
    poultryType: str = Form(...),
    title: str = Form(default=""),
    description: str = Form(default=""),
    year: int | None = Form(default=None),
):
    return await _preview_calendar_upload(
        file,
        {
            "region": region,
            "district": district,
            "poultryType": poultryType,
            "title": title or f"{poultryType} Calendar",
            "description": description,
            "year": year,
        },
        "poultry-calendar",
    )


@app.post("/api/crop-calendars/commit")
async def commit_crop_calendar(parseToken: str = Form(...)):
    preview_payload = get_preview_payload(parseToken)
    if not preview_payload or preview_payload.get("entityType") != "crop-calendar":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview token not found or expired.")
    payload = build_committed_calendar_payload(preview_payload)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "crop-calendar", encode_payload(payload), None, None, None),
        )
        record_id = cursor.lastrowid
        calendar = insert_calendar_from_parsed_payload(connection, record_id, "crop-calendar", payload)
    discard_preview_payload(parseToken)
    return {"success": True, "data": calendar, "message": "Crop calendar committed successfully."}


@app.post("/api/poultry-calendars/commit")
async def commit_poultry_calendar(parseToken: str = Form(...)):
    preview_payload = get_preview_payload(parseToken)
    if not preview_payload or preview_payload.get("entityType") != "poultry-calendar":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview token not found or expired.")
    payload = build_committed_calendar_payload(preview_payload)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "poultry-calendar", encode_payload(payload), None, None, None),
        )
        record_id = cursor.lastrowid
        calendar = insert_calendar_from_parsed_payload(connection, record_id, "poultry-calendar", payload)
    discard_preview_payload(parseToken)
    return {"success": True, "data": calendar, "message": "Poultry calendar committed successfully."}


async def _preview_advisory_upload(file: UploadFile, metadata: dict, advisory_type: str):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Spreadsheet file is required.")
    contents = await file.read()
    preview = build_advisory_preview_payload(contents, metadata, advisory_type)
    return {"success": True, "data": preview}


@app.post("/api/weekly-advisories/preview")
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


@app.post("/api/poultry-advisories/preview")
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


@app.post("/api/weekly-advisories/commit")
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


@app.post("/api/poultry-advisories/commit")
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


@app.get("/api/weekly-advisories")
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


@app.delete("/api/weekly-advisories/{advisory_id}")
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


@app.get("/api/weekly-advisories/activities")
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


@app.get("/api/weekly-advisories/{advisory_id}")
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


@app.get("/api/enhanced-calendars/metadata")
def get_enhanced_calendar_metadata():
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM calendars ORDER BY created_at DESC, id DESC").fetchall()

    calendars = [serialize_calendar(row_to_dict(row)) for row in rows]
    return {
        "commodities": sorted({item["commodity"] for item in calendars if item.get("commodity")}),
        "regions": sorted({item["regionCode"] for item in calendars if item.get("regionCode")}),
        "districts": sorted({item["districtCode"] for item in calendars if item.get("districtCode")}),
        "calendarTypes": sorted({item["calendarType"] for item in calendars if item.get("calendarType")}),
        "totalCalendars": len(calendars),
    }


@app.get("/api/enhanced-calendars")
def list_enhanced_calendars(
    calendarType: str | None = Query(default=None),
    commodity: str | None = Query(default=None),
    regionCode: str | None = Query(default=None),
    districtCode: str | None = Query(default=None),
    year: int | None = Query(default=None),
    search: str | None = Query(default=None),
):
    clauses = ["1=1"]
    params: list[object] = []
    if calendarType:
        clauses.append("calendar_type = ?")
        params.append(calendarType)
    if commodity:
        clauses.append("LOWER(crop) = LOWER(?)")
        params.append(commodity)
    if regionCode:
        clauses.append("region_code = ?")
        params.append(regionCode)
    if districtCode:
        clauses.append("district_code = ?")
        params.append(districtCode)
    if year is not None:
        clauses.append("year = ?")
        params.append(year)
    if search:
        clauses.append("(LOWER(title) LIKE LOWER(?) OR LOWER(crop) LIKE LOWER(?))")
        params.extend([f"%{search}%", f"%{search}%"])

    query = f"SELECT * FROM calendars WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, id DESC"
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()

    data = [serialize_calendar(row_to_dict(row)) for row in rows]
    return {
        "success": True,
        "data": data,
        "total": len(data),
        "filters": {
            "calendarType": calendarType,
            "commodity": commodity,
            "regionCode": regionCode,
            "districtCode": districtCode,
            "year": year,
            "search": search,
        },
        "summary": {
            "calendarTypes": sorted({item["calendarType"] for item in data}),
            "commodities": sorted({item["commodity"] for item in data}),
        },
    }


@app.get("/api/enhanced-calendars/{calendar_id}/activities")
def get_enhanced_calendar_activities(
    calendar_id: int,
    currentWeek: int | None = Query(default=None),
    startWeek: int | None = Query(default=None),
    endWeek: int | None = Query(default=None),
):
    with get_connection() as connection:
        calendar_row = connection.execute("SELECT * FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
        if not calendar_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calendar not found.")
        activities = fetch_calendar_activities(connection, calendar_id, currentWeek, startWeek, endWeek)
    return {
        "success": True,
        "data": {
            "activities": activities,
            "schedule": activities,
        },
        "filters": {"currentWeek": currentWeek, "startWeek": startWeek, "endWeek": endWeek},
    }


@app.get("/api/enhanced-calendars/{calendar_id}")
def get_enhanced_calendar(calendar_id: int):
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calendar not found.")
        activities = fetch_calendar_activities(connection, calendar_id)
    calendar = serialize_calendar(row_to_dict(row))
    calendar["activities"] = activities
    return {"success": True, "data": calendar}


@app.delete("/api/enhanced-calendars/{calendar_id}")
def delete_enhanced_calendar(calendar_id: int):
    with get_connection() as connection:
        row = connection.execute("SELECT source_record_id FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calendar not found.")
        source_record_id = row["source_record_id"]
        connection.execute("DELETE FROM calendars WHERE id = ?", (calendar_id,))
        if source_record_id:
            connection.execute("DELETE FROM agricultural_records WHERE id = ?", (source_record_id,))
    return {"success": True, "message": "Calendar deleted successfully."}


@app.post("/api/crop-calendars/create")
async def create_crop_calendar(request: Request):
    payload = dict(await request.json())
    payload.setdefault("title", payload.get("crop") or "Crop Calendar")
    payload.setdefault("regionCode", payload.get("region"))
    payload.setdefault("districtCode", payload.get("district"))
    payload.setdefault("crop", payload.get("crop") or payload.get("commodity"))
    payload.setdefault("year", payload.get("year"))

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agricultural_records(owner_id, data_type, payload_json, file_name, file_content_type, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "crop-calendar", encode_payload(payload), None, None, None),
        )
        record_id = cursor.lastrowid
        calendar = insert_calendar_from_record(connection, record_id, "crop-calendar", payload)
    return {"success": True, "data": calendar}


@app.get("/api/crop-calendars/district/{district}")
def get_crop_calendars_by_district(district: str, year: int | None = Query(default=None), crop: str | None = Query(default=None)):
    params: dict[str, object] = {"districtCode": district}
    if year is not None:
        params["year"] = year
    if crop:
        params["commodity"] = crop
    return list_enhanced_calendars(calendarType="seasonal", **params)


@app.get("/api/crop-calendars/search")
def search_crop_calendars(search: str | None = Query(default=None), region: str | None = Query(default=None), district: str | None = Query(default=None)):
    return list_enhanced_calendars(
        calendarType="seasonal",
        commodity=search,
        regionCode=region,
        districtCode=district,
        search=search,
    )


@app.get("/api/crop-calendars/stats")
def get_crop_calendar_stats():
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM calendars WHERE calendar_type = 'seasonal' ORDER BY created_at DESC").fetchall()
    calendars = [serialize_calendar(row_to_dict(row)) for row in rows]
    return {
        "success": True,
        "data": {
            "totalCalendars": len(calendars),
            "regions": sorted({item["regionCode"] for item in calendars if item.get("regionCode")}),
            "commodities": sorted({item["commodity"] for item in calendars if item.get("commodity")}),
        },
    }


@app.get("/api/production-cycles")
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


@app.post("/api/production-cycles")
def create_production_cycle(payload: ProductionCycleCreateRequest):
    with get_connection() as connection:
        calendar_row = connection.execute("SELECT * FROM calendars WHERE id = ?", (payload.calendarId,)).fetchone()
        if not calendar_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calendar not found.")
        calendar = row_to_dict(calendar_row)
        cursor = connection.execute(
            """
            INSERT INTO production_cycles(calendar_id, batch_name, commodity, start_date, status, initial_quantity, current_quantity, notes)
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (
                payload.calendarId,
                payload.batchName,
                calendar["crop"],
                payload.startDate,
                payload.initialQuantity,
                payload.initialQuantity,
                payload.notes,
            ),
        )
        cycle_id = cursor.lastrowid
        row = connection.execute("SELECT * FROM production_cycles WHERE id = ?", (cycle_id,)).fetchone()
    return {"success": True, "data": serialize_cycle(row_to_dict(row))}


@app.put("/api/production-cycles/{cycle_id}")
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


@app.get("/api/production-cycles/{cycle_id}/current-activities")
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


@app.get("/api/user/dashboard/stats")
@app.get("/user/dashboard/stats")
def get_dashboard_stats():
    return {"success": True, "data": build_dashboard_stats()}
