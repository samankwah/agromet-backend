from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


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


class ChatRequest(BaseModel):
    message: str
    conversationHistory: list[dict] = Field(default_factory=list)
    userContext: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    success: bool
    message: str
    usage: dict | None = None


class FAQResponse(BaseModel):
    success: bool
    message: str


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
