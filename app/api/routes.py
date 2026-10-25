from __future__ import annotations

import logging
import uuid
from io import BytesIO
from typing import NoReturn

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.security import require_api_key
from app.core.config import settings
from app.models.schemas import EmailHitOut, ExtractMeta, ExtractResponse
from app.services.ocr import (
    OcrConfigError,
    OcrError,
    OcrExecutionError,
    OcrTimeoutError,
)
from app.services.pipeline import ExtractionPipeline, get_pipeline
from app.services.preprocess import ImageTooLargeError, ImageValidationError

router = APIRouter()
logger = logging.getLogger(__name__)

_READ_CHUNK = 1024 * 1024


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MB"
    if value >= 1024:
        return f"{value / 1024:.0f} KB"
    return f"{value} bytes"


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    buffer = BytesIO()
    total = 0
    while True:
        chunk = await upload.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"La imagen supera el limite de {_format_bytes(limit)}.",
            )
        buffer.write(chunk)
    return buffer.getvalue()


def _fail(
    request_id: str,
    status_code: int,
    exc: Exception,
    headers: dict[str, str] | None = None,
) -> NoReturn:
    logger.warning(
        "extract_failed request_id=%s status=%d error=%s",
        request_id,
        status_code,
        exc,
    )
    raise HTTPException(status_code=status_code, detail=str(exc), headers=headers)


@router.post(
    "/v1/extract",
    response_model=ExtractResponse,
    tags=["extraction"],
    summary="Extrae direcciones de correo de una imagen",
    dependencies=[Depends(require_api_key)],
)
async def extract(
    request: Request,
    file: UploadFile = File(..., description="Imagen JPEG, PNG, WEBP, BMP o TIFF"),
    include_text: bool = Form(False, description="Incluir el texto completo detectado por OCR"),
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> ExtractResponse:
    request_id = getattr(request.state, "request_id", "") or uuid.uuid4().hex
    try:
        data = await _read_limited(file, settings.max_upload_bytes)
    finally:
        await file.close()
    if not data:
        _fail(request_id, 400, ImageValidationError("El archivo esta vacio."))

    logger.info("extract_start request_id=%s bytes=%d", request_id, len(data))

    try:
        result = await run_in_threadpool(pipeline.process_bytes, data, include_text)
    except ImageTooLargeError as exc:
        _fail(request_id, 413, exc)
    except ImageValidationError as exc:
        _fail(request_id, 400, exc)
    except OcrConfigError as exc:
        _fail(request_id, 503, exc)
    except OcrTimeoutError as exc:
        _fail(request_id, 504, exc)
    except OcrExecutionError as exc:
        _fail(request_id, 502, exc)
    except OcrError as exc:
        _fail(request_id, 502, exc)

    emails = [
        EmailHitOut(
            email=hit.email,
            confidence=min(max(hit.confidence, 0.0), 1.0),
            occurrences=min(max(hit.occurrences, 1), 10_000),
        )
        for hit in result.emails
    ]

    logger.info(
        "extract_ok request_id=%s emails=%d elapsed_ms=%.1f engine=%s model=%s",
        request_id,
        len(emails),
        result.elapsed_ms,
        result.engine,
        result.model,
    )

    return ExtractResponse(
        request_id=request_id,
        emails=emails,
        count=len(emails),
        meta=ExtractMeta(
            width=result.width,
            height=result.height,
            elapsed_ms=result.elapsed_ms,
            engine=result.engine,
            model=result.model,
            ocr_confidence=result.ocr_confidence,
        ),
        text=result.text if include_text else None,
    )
