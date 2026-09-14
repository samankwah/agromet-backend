"""AgroMet AI: the chat endpoint and everything only it needs -- the OpenAI
call, the Google Translate fallback used by both `/api/v1/translate` and the
assistant's own grounding, transcription, and the TTS stubs.

Reads `config.OPENAI_API_KEY` etc. as `config.NAME` (module attribute
access), not `from ..config import NAME` -- a test that patches
`backend.app.config.OPENAI_API_KEY` needs that patch to reach whichever
module actually reads it at call time, and attribute access is what makes
that true regardless of which router ends up owning this code. Same reason
`chat_context` is imported as a module below rather than importing
`build_context_block` by name.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile, status

from .. import chat_context, config
from ..chat_prompt import build_chat_input
from ..rate_limit import Limiter, client_ip, client_keys
from ..schemas import ChatReply, ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# --- Google Translate fallback (free, high-quality Ghanaian language support) ---

GOOGLE_LANG_MAP = {
    "en": "en",
    "tw": "ak",   # Twi/Akan
    "ee": "ee",   # Ewe
    "gaa": "gaa", # Ga
    "dag": "dag", # Dagbani
    "ha": "ha",   # Hausa
    "fat": "ak",  # Fante → Akan (closest)
    "nzi": "ak",  # Nzema → Akan (closest)
    "ki": "ki",   # Kikuyu
}


async def google_translate_with_client(client: httpx.AsyncClient, text: str, src_lang: str, tgt_lang: str) -> str | None:
    """Translate using Google Translate free endpoint. High quality for Ghanaian languages."""
    src_code = GOOGLE_LANG_MAP.get(src_lang, src_lang)
    tgt_code = GOOGLE_LANG_MAP.get(tgt_lang, tgt_lang)
    try:
        response = await client.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": src_code, "tl": tgt_code, "dt": "t", "q": text},
        )
        response.raise_for_status()
        data = response.json()
        # Response format: [[["translated text","original text",...],...],...]
        if isinstance(data, list) and data and isinstance(data[0], list):
            # Concatenate all translated segments
            result = "".join(segment[0] for segment in data[0] if segment and segment[0])
            if result and result.lower() != text.lower():
                return result
    except Exception as exc:
        print(f"[GoogleTranslate] Translation failed: {exc}")
    return None


async def google_translate_fallback(text: str, src_lang: str, tgt_lang: str) -> str | None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await google_translate_with_client(client, text, src_lang, tgt_lang)


# One limiter for the process. Module scope on purpose: a limiter rebuilt per
# request counts nothing.
chat_limiter = Limiter(
    limit=config.CHAT_RATE_LIMIT,
    window_seconds=config.CHAT_RATE_WINDOW_SECONDS,
    daily_limit=config.CHAT_DAILY_LIMIT,
)


class ChatOutcome:
    """What one attempt at an answer produced.

    A tuple did for two values. It stopped doing when there were four, and the
    third and fourth are the point of this: `reason` is what turns "the
    assistant is degraded" into something an operator can act on, and `usage` is
    the only number that makes the bill visible.
    """

    __slots__ = ("text", "degraded", "reason", "usage")

    def __init__(self, text: str, degraded: bool, reason: str | None = None, usage: dict | None = None) -> None:
        self.text = text
        self.degraded = degraded
        self.reason = reason
        self.usage = usage


def fallback_reply(message: str, region: str | None) -> str:
    """The answer served when the model cannot be reached.

    Kept because a chat box that answers something beats a 502, and worded so it
    never pretends to have read the question: the client labels it, and this
    text has to survive being read without that label.
    """
    where = region or "your area"
    return (
        f"I cannot reach the AgroMet assistant right now, so this is general guidance rather than "
        f"an answer to your question. For {where}, watch the rainfall timing, keep field drainage "
        f"clear, use good seed, and check your crop for pests weekly. Please ask me again shortly."
    )


def extract_reply_text(payload: dict) -> str | None:
    """The assistant's words out of a Responses API payload."""
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if text and text.strip():
                return text
    return None


async def build_chat_reply(
    message: str,
    conversation_history: list[dict],
    user_context: dict | None = None,
    context_block: str | None = None,
) -> ChatOutcome:
    """The reply, and whether it is the real thing.

    Every failure here ends in the same fallback and an HTTP 200, which is
    deliberate -- see `fallback_reply`. What changed is that the *reason* now
    survives: `no_key`, `timeout`, `upstream_error` and `empty_output` used to be
    one indistinguishable degraded answer, and the first of those is the one it
    usually was not.
    """
    context = user_context if isinstance(user_context, dict) else {}
    region = context.get("region")

    if not config.OPENAI_API_KEY:
        logger.warning("Chat asked for an answer with no OPENAI_API_KEY configured.")
        return ChatOutcome(fallback_reply(message, region), True, "no_key")

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=config.OPENAI_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.OPENAI_MODEL,
                    "input": build_chat_input(message, conversation_history, context_block),
                    "max_output_tokens": config.OPENAI_MAX_OUTPUT_TOKENS,
                    "temperature": config.OPENAI_TEMPERATURE,
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException:
        logger.warning(
            "Chat completion timed out after %.1fs; serving the fallback reply.",
            time.perf_counter() - started,
        )
        return ChatOutcome(fallback_reply(message, region), True, "timeout")
    except Exception:
        # Falling through to the canned reply is deliberate: a chat box that
        # answers something beats a 502. Swallowing the reason was not.
        logger.exception("Chat completion failed; serving the fallback reply.")
        return ChatOutcome(fallback_reply(message, region), True, "upstream_error")

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    text = extract_reply_text(payload)

    if not text:
        # A 200 with nothing in it. Rare, and it used to be silent: the fallback
        # went out looking exactly like a missing key.
        logger.warning("Chat completion returned no text; serving the fallback reply.")
        return ChatOutcome(fallback_reply(message, region), True, "empty_output", usage)

    logger.info(
        "Chat answered in %.2fs (model=%s, input_tokens=%s, output_tokens=%s)",
        time.perf_counter() - started,
        config.OPENAI_MODEL,
        (usage or {}).get("input_tokens"),
        (usage or {}).get("output_tokens"),
    )
    return ChatOutcome(text, False, None, usage)


@router.post("/api/chat", response_model=ChatReply)
async def chat(
    payload: ChatRequest,
    request: Request,
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
):
    """Answer one question from a farmer.

    Three things happen before the model is called, in this order because each
    is cheaper than the next: the quota is checked, the live figures for this
    farmer's area are gathered, and only then is anything billed.

    The 429 is the one path here that is not a 200. It has to be: an answer that
    said "you have asked too many questions" in the assistant's own voice would
    be indistinguishable from the assistant refusing to help, and the client
    needs to tell those apart to know whether retrying is worth anything.
    """
    keys = client_keys(x_device_id, client_ip(request.headers, request.client.host if request.client else None))
    decision = chat_limiter.check(keys)
    if not decision.allowed:
        logger.info("Chat request refused by the quota (%s).", decision.reason)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=decision.message,
            headers={"Retry-After": str(decision.retry_after)},
        )
    chat_limiter.record(keys)

    context = payload.userContext
    context_block = await chat_context.build_context_block(
        payload.message,
        region=context.region,
        district=context.district,
        town=context.town,
        crops=context.crops,
    )

    outcome = await build_chat_reply(
        payload.message,
        payload.conversationHistory,
        context.model_dump(),
        context_block,
    )

    # `degraded` says the answer is the built-in fallback rather than the
    # model's. Still a 200 with `success: True`, because the farmer did get
    # usable words back, but the client can now say where they came from
    # instead of presenting canned advice as an answer to their question.
    return ChatReply(
        success=True,
        message=outcome.text,
        degraded=outcome.degraded,
        degradedReason=outcome.reason,
        usage=outcome.usage,
    )


# A minute of speech is a long question. The cap exists because the upload
# happens on a rural connection and the transcription is billed by duration,
# not because a longer clip would break anything.
MAX_TRANSCRIPT_AUDIO_BYTES = 10 * 1024 * 1024


@router.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Speech to text, so a farmer can ask by speaking instead of typing.

    Deliberately returns the text rather than an answer: the transcript goes
    into the composer's draft for the farmer to correct before sending.
    Transcription of accented English over a poor connection is not reliable
    enough to send unread, and a wrong question answered confidently is worse
    than no question at all.
    """
    if not config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice questions need a transcription provider, which is not configured.",
        )

    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The recording was empty.")
    if len(payload) > MAX_TRANSCRIPT_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="That recording is too long. Ask a shorter question.",
        )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                files={"file": (audio.filename or "question.m4a", payload, audio.content_type or "audio/m4a")},
                data={"model": config.TRANSCRIPTION_MODEL},
            )
            response.raise_for_status()
            text = str(response.json().get("text") or "").strip()
    except Exception:
        # Unlike the chat fallback there is nothing sensible to invent here: a
        # made-up transcript would put words in the farmer's mouth.
        logger.exception("Transcription failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not turn that recording into text. Try again, or type your question.",
        )

    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No speech was found in that recording.",
        )

    return {"success": True, "text": text}


@router.post("/api/v1/translate")
async def translate_text(request: Request):
    payload = dict(await request.json())
    text = str(payload.get("in") or "").strip()
    lang = str(payload.get("lang") or "").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Translation input text is required.")
    if "-" not in lang:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Translation language pair must be in the form 'en-tw'.")

    src_lang, tgt_lang = lang.split("-", 1)

    # Keep the route stable for older callers and translate through the current provider.
    google_result = await google_translate_fallback(text, src_lang, tgt_lang)
    if google_result:
        return {"out": google_result, "translation": google_result}

    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Translation service failed.")


@router.post("/api/v1/translate/batch")
async def translate_text_batch(request: Request):
    payload = dict(await request.json())
    texts = payload.get("texts")
    lang = str(payload.get("lang") or "").strip()

    if not isinstance(texts, list) or not texts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Translation texts must be a non-empty list.")
    if len(texts) > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Translation batch is limited to 100 texts.")
    if "-" not in lang:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Translation language pair must be in the form 'en-tw'.")

    src_lang, tgt_lang = lang.split("-", 1)
    normalized_texts = [str(text or "").strip() for text in texts]
    unique_texts = list(dict.fromkeys(text for text in normalized_texts if text))
    translated_by_text: dict[str, str] = {}
    semaphore = asyncio.Semaphore(8)

    async with httpx.AsyncClient(timeout=10.0) as client:
        async def translate_one(text: str) -> tuple[str, str | None]:
            async with semaphore:
                return text, await google_translate_with_client(client, text, src_lang, tgt_lang)

        pairs = await asyncio.gather(*(translate_one(text) for text in unique_texts))

    for text, translated in pairs:
        translated_by_text[text] = translated or text

    translations = [translated_by_text.get(text, text) for text in normalized_texts]
    return {
        "translations": translations,
        "provider": "google-translate-fallback",
        "count": len(translations),
    }


@router.get("/api/tts/languages")
async def list_tts_languages():
    return [
        {"code": "en", "language": "en", "name": "English", "source": "browser"},
        {"code": "tw", "language": "tw", "name": "Twi / Akan", "source": "browser"},
        {"code": "gaa", "language": "gaa", "name": "Ga", "source": "browser"},
        {"code": "ee", "language": "ee", "name": "Ewe", "source": "browser"},
        {"code": "dag", "language": "dag", "name": "Dagbani", "source": "browser"},
        {"code": "ha", "language": "ha", "name": "Hausa", "source": "browser"},
    ]


@router.get("/api/tts/speakers")
def list_tts_speakers():
    return []


@router.post("/api/tts/tts")
@router.post("/api/tts/synthesize")
async def synthesize_speech(request: Request):
    payload = dict(await request.json())
    text = str(payload.get("text") or "").strip()
    language = str(payload.get("language") or "en").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="TTS text is required.")

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "success": False,
            "fallback": "browser",
            "message": "Server text-to-speech is disabled. Use browser speech synthesis.",
            "language": language,
        },
    )
