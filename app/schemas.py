from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str | None = None


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    name: str | None = None
    role: str = "administrator"
    created_at: datetime | str


class RegisterResponse(BaseModel):
    success: bool = True
    message: str = "Account created successfully."
    user: UserResponse


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


#: The longest question the assistant will accept. The mobile composer stops at
#: 1000 characters, so this is double what any client should ever send: room for
#: a pasted paragraph, and a ceiling on what an unauthenticated caller can put
#: through a metered model.
MAX_CHAT_MESSAGE_CHARS = 2000

#: How many prior turns a client may send. Trimmed to CHAT_HISTORY_LIMIT again
#: when the model input is assembled; this outer bound is about the size of the
#: request body, not about how much context the model gets.
MAX_CHAT_HISTORY_ENTRIES = 24

#: The longest single prior turn kept. Longer ones are truncated rather than
#: rejected -- see the validator.
MAX_CHAT_HISTORY_CHARS = 2000


class ChatUserContext(BaseModel):
    """What the assistant is told about who is asking.

    Extra keys are ignored rather than rejected: the mobile and web clients ship
    on their own schedules, and a field one of them adds early must not 422 the
    other's users.
    """

    model_config = ConfigDict(extra="ignore")

    region: str | None = Field(default=None, max_length=80)
    #: The town the farmer selected on Home. Kept distinct from `district`
    #: because it is a town: the mobile app has no district picker, and calling
    #: Tamale a district in an answer would put the farmer somewhere they did
    #: not say they were.
    town: str | None = Field(default=None, max_length=80)
    district: str | None = Field(default=None, max_length=80)
    crops: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("crops", mode="after")
    @classmethod
    def tidy_crops(cls, crops: list[str]) -> list[str]:
        return [crop.strip()[:60] for crop in crops if isinstance(crop, str) and crop.strip()]


class ChatRequest(BaseModel):
    """A question for the assistant.

    Two different postures here, deliberately.

    ``message`` is *rejected* when it is empty or over-long: it is the thing the
    caller meant to send, every client already caps it well below this ceiling,
    and an unbounded string on an unauthenticated route is a bill someone else
    pays.

    ``conversationHistory`` is *sanitised* instead. It is bookkeeping the client
    replays rather than something a person typed, it arrives from clients this
    server does not version-control, and refusing the whole question because one
    old turn is malformed would lose the farmer their answer. Entries are capped
    and truncated here; roles are filtered again in ``chat_prompt`` where the
    model input is built.
    """

    message: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)
    # `list[Any]`, not `list[dict]`, so that a stray number or string in the
    # replayed history is dropped by the validator below instead of failing type
    # coercion and taking the farmer's question down with it.
    conversationHistory: list[Any] = Field(default_factory=list)
    userContext: ChatUserContext = Field(default_factory=ChatUserContext)

    @field_validator("message", mode="after")
    @classmethod
    def require_words(cls, message: str) -> str:
        stripped = message.strip()
        if not stripped:
            raise ValueError("Ask a question first.")
        return stripped

    @field_validator("conversationHistory", mode="after")
    @classmethod
    def trim_history(cls, history: list[Any]) -> list[dict]:
        trimmed: list[dict] = []
        for entry in history[-MAX_CHAT_HISTORY_ENTRIES:]:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if isinstance(content, str) and len(content) > MAX_CHAT_HISTORY_CHARS:
                entry = {**entry, "content": content[:MAX_CHAT_HISTORY_CHARS]}
            trimmed.append(entry)
        return trimmed


class ChatReply(BaseModel):
    """The assistant's answer.

    Note there is no ``data`` key: this envelope is flat, and both clients read
    ``message`` off the top level. ``http.ts`` in the mobile app documents the
    same quirk from the other side.
    """

    success: bool = True
    message: str
    #: True when ``message`` is the built-in fallback rather than the model's
    #: answer. Served with success, because the farmer did get usable words, so
    #: this is the only thing telling the client to label them.
    degraded: bool = False
    #: Which failure caused the fallback: no_key, timeout, upstream_error,
    #: empty_output. Absent on a real answer.
    degradedReason: str | None = None
    #: Token counts as the provider reported them, for metering. Absent when the
    #: provider was not reached.
    usage: dict | None = None


class FAQResponse(BaseModel):
    success: bool
    message: str


class ContactMessageRequest(BaseModel):
    """A message from the Contact screen.

    Validated here rather than only in the clients, because a public POST is
    reachable by anything. The length ceilings are what stops the table being
    filled with megabytes by a script; the floors are what stops an accidental
    empty submission being stored as if it were a real question.
    """

    name: str = Field(min_length=1, max_length=120)
    subject: str = Field(min_length=1, max_length=160)
    message: str = Field(min_length=1, max_length=4000)
    # At least one of these is required -- see the validator below. Optional
    # individually because a farmer may have a phone but no email address.
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=40)
    source: str = Field(default="mobile", max_length=40)

    @model_validator(mode="after")
    def require_a_reply_channel(self) -> "ContactMessageRequest":
        if not self.email and not (self.phone or "").strip():
            raise ValueError("Provide an email address or a phone number so we can reply.")
        return self


class ContactMessageResponse(BaseModel):
    success: bool
    message: str
    reference: int


class LegalSection(BaseModel):
    title: str
    body: str
    # Optional bullets under `body`. Some sections are a paragraph, some are a
    # lead-in followed by a list, so the field is absent rather than empty.
    items: list[str] | None = None


class LegalDocumentResponse(BaseModel):
    """A legal document, served as structured sections rather than HTML.

    The web app renders these with its own components and the mobile app with
    native ones, so shipping markup would force one of them to parse it. Both
    render the same sections their own way, and the wording lives in one place.
    """

    success: bool
    slug: str
    title: str
    summary: str
    updated: str
    sections: list[LegalSection]


class HealthResponse(BaseModel):
    status: str
    app: str


class ProductionCycleCreateRequest(BaseModel):
    calendarId: int
    startDate: str
    batchName: str
    initialQuantity: int = 0
    notes: str | None = None


class ProductionCycleUpdateRequest(BaseModel):
    status: str | None = None
    batchName: str | None = None
    initialQuantity: int | None = None
    currentQuantity: int | None = None
    notes: str | None = None


class CropDiagnosisRequest(BaseModel):
    image: str
    crop: str | None = None
    region: str | None = None
    language: str | None = None
    context: dict = Field(default_factory=dict)


class ImageAnalysisRequest(BaseModel):
    image: str
    analysisType: str
    context: dict = Field(default_factory=dict)


# ── Market schemas ──────────────────────────────────────────────────────────

class CommodityResponse(BaseModel):
    slug: str
    name: str
    category: str
    price: float
    unit: str
    trend: str
    demand: str


class CommodityTrendResponse(BaseModel):
    commodity_slug: str
    month_prices: list[float] = Field(alias="6months", default_factory=list)
    seasonal_pattern: str | None = None
    peak_months: list[int] = Field(default_factory=list)
    low_months: list[int] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class MarketCenterResponse(BaseModel):
    region: str
    major_markets: list[str] = Field(default_factory=list)
    transport_access: str
    price_premium: float


class HazardOverrideRequest(BaseModel):
    """A GMet bulletin that supersedes the computed index for one region."""

    region: str
    hazard: Literal["flood", "drought"]
    band: str
    headline: str | None = None
    advisories: list[str] = Field(default_factory=list)
    issuedBy: str = "GMet"
    effectiveFrom: str | None = None
    effectiveTo: str | None = None
