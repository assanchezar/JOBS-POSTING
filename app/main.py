from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.models.schemas import HealthResponse
from app.services.ocr import OcrError
from app.services.pipeline import close_pipeline, get_pipeline

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_MULTIPART_OVERHEAD = 256 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    close_pipeline()


app = FastAPI(
    title="Email OCR API",
    description=("Extrae direcciones de correo desde imagenes con OCR local (Tesseract)."),
    version=settings.version,
    lifespan=lifespan,
)


def _with_response_headers(response: Response, request_id: str) -> Response:
    response.headers["X-Request-ID"] = request_id
    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
    request.state.request_id = request_id

    # Rechaza cuerpos enormes antes de que Starlette parsee el multipart.
    if request.method == "POST" and request.url.path == "/v1/extract":
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                content_length = int(raw_length)
            except ValueError:
                content_length = 0
            if content_length > settings.max_upload_bytes + _MAX_MULTIPART_OVERHEAD:
                return _with_response_headers(
                    JSONResponse(
                        status_code=413,
                        content={"detail": "La solicitud es demasiado grande."},
                    ),
                    request_id,
                )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled_error request_id=%s", request_id)
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Error interno del servidor.",
                "request_id": request_id,
            },
        )
    return _with_response_headers(response, request_id)


# Se agrega despues del middleware anterior para que CORS quede en la capa externa
# y aplique tambien a respuestas tempranas (413) y errores 500.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Retry-After"],
)

app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    pipeline = get_pipeline()
    language = ""
    if pipeline.ready:
        try:
            language = pipeline.engine.language
        except OcrError:
            language = ""

    ready = pipeline.ready and bool(language)
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=settings.version,
        engine=pipeline.engine.name,
        model=pipeline.engine.model,
        ocr_ready=pipeline.ready,
        language=language,
        auth_required=bool(settings.api_keys),
    )
