from __future__ import annotations

from pydantic import BaseModel, Field


class EmailHitOut(BaseModel):
    email: str
    confidence: float = Field(ge=0.0, le=1.0)
    occurrences: int = Field(ge=1, le=10_000)


class ExtractMeta(BaseModel):
    width: int
    height: int
    elapsed_ms: float
    engine: str
    model: str
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ExtractResponse(BaseModel):
    request_id: str
    emails: list[EmailHitOut]
    count: int
    meta: ExtractMeta
    text: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    engine: str
    model: str
    ocr_ready: bool
    language: str
    auth_required: bool
