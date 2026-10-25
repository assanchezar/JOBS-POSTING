from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Carga .env desde la raiz del proyecto sin sobrescribir variables ya definidas.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_VERSION = "0.6.0"
DEFAULT_LANG = "spa+eng"
DEFAULT_PSM = 6
DEFAULT_FALLBACK_PSM = 11
DEFAULT_OEM = 1
DEFAULT_TIMEOUT = 30.0
DEFAULT_FALLBACK_CONFIDENCE = 0.5
DEFAULT_MAX_CONCURRENCY = 1
MAX_OCR_CONCURRENCY = 8


def _clamp_int(value: int, minimum: int, maximum: int | None = None) -> int:
    value = max(minimum, value)
    return value if maximum is None else min(value, maximum)


def _clamp_float(value: float, minimum: float, maximum: float | None = None) -> float:
    value = max(minimum, value)
    return value if maximum is None else min(value, maximum)


def _finite_or(value: float, default: float) -> float:
    return value if math.isfinite(value) else default


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return _clamp_int(value, minimum, maximum)


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return _clamp_float(value, minimum, maximum)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_lang(raw: str | None) -> str:
    """Normaliza una lista Tesseract como 'spa+eng' sin duplicados."""
    parts = [part.strip().lower() for part in (raw or "").split("+") if part.strip()]
    if not parts:
        return DEFAULT_LANG
    return "+".join(dict.fromkeys(parts))


@dataclass
class Settings:
    """Configuracion de la aplicacion. Se puede sobrescribir con variables de entorno."""

    version: str = DEFAULT_VERSION
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_side: int = 1600
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    api_keys: list[str] = field(default_factory=list)

    # OCR local con Tesseract. En PythonAnywhere el binario ya esta en /usr/bin/tesseract.
    tesseract_cmd: str = ""
    tesseract_lang: str = DEFAULT_LANG
    tesseract_psm: int = DEFAULT_PSM
    tesseract_fallback_psm: int = DEFAULT_FALLBACK_PSM
    tesseract_oem: int = DEFAULT_OEM
    tesseract_timeout: float = DEFAULT_TIMEOUT
    tesseract_tessdata: str = ""

    # Control de recursos. Tesseract es CPU-bound y conviene limitar la concurrencia.
    ocr_max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    ocr_upscale: bool = True
    ocr_upscale_min_side: int = 1000
    ocr_border: int = 20

    # Calidad de OCR: enderezado, binarizacion y reduccion de ruido.
    ocr_deskew: bool = True
    ocr_deskew_max_angle: float = 8.0
    ocr_deskew_step: float = 0.5
    ocr_binarize_fallback: bool = True
    ocr_denoise_fallback: bool = True
    ocr_bottom_strip_fallback: bool = True
    ocr_bottom_strip_ratio: float = 0.30
    tesseract_fallback_confidence: float = DEFAULT_FALLBACK_CONFIDENCE

    def __post_init__(self) -> None:
        """Normaliza valores invalidos para que un .env mal escrito no rompa la API."""
        self.version = (self.version or DEFAULT_VERSION).strip() or DEFAULT_VERSION
        self.max_upload_bytes = _clamp_int(self.max_upload_bytes, 1)
        self.max_image_pixels = _clamp_int(self.max_image_pixels, 1)
        self.max_side = _clamp_int(self.max_side, 64)
        self.cors_origins = [
            origin.strip() for origin in self.cors_origins if origin and origin.strip()
        ]

        raw_cmd = (self.tesseract_cmd or "").strip()
        self.tesseract_cmd = os.path.expanduser(raw_cmd) if raw_cmd else ""
        self.tesseract_lang = normalize_lang(self.tesseract_lang)
        raw_tessdata = (self.tesseract_tessdata or "").strip()
        self.tesseract_tessdata = os.path.expanduser(raw_tessdata) if raw_tessdata else ""
        self.tesseract_psm = _clamp_int(self.tesseract_psm, 0, 13)
        self.tesseract_fallback_psm = _clamp_int(self.tesseract_fallback_psm, 0, 13)
        self.tesseract_oem = _clamp_int(self.tesseract_oem, 0, 3)
        self.tesseract_timeout = _clamp_float(
            _finite_or(self.tesseract_timeout, DEFAULT_TIMEOUT),
            1.0,
            300.0,
        )

        self.ocr_max_concurrency = _clamp_int(self.ocr_max_concurrency, 1, MAX_OCR_CONCURRENCY)
        self.ocr_upscale_min_side = _clamp_int(self.ocr_upscale_min_side, 256, 4000)
        self.ocr_border = _clamp_int(self.ocr_border, 0, 100)
        self.ocr_deskew_max_angle = _clamp_float(self.ocr_deskew_max_angle, 0.0, 15.0)
        self.ocr_deskew_step = _clamp_float(self.ocr_deskew_step, 0.25, 5.0)
        self.ocr_bottom_strip_ratio = _clamp_float(self.ocr_bottom_strip_ratio, 0.10, 0.60)
        self.tesseract_fallback_confidence = _clamp_float(
            _finite_or(self.tesseract_fallback_confidence, DEFAULT_FALLBACK_CONFIDENCE),
            0.0,
            1.0,
        )

        # Conserva el orden de configuracion y elimina duplicados exactos.
        self.api_keys = list(
            dict.fromkeys(key.strip() for key in self.api_keys if key and key.strip())
        )


def load_settings() -> Settings:
    return Settings(
        version=os.getenv("APP_VERSION", DEFAULT_VERSION),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024, minimum=1),
        max_image_pixels=_env_int("MAX_IMAGE_PIXELS", 40_000_000, minimum=1),
        max_side=_env_int("MAX_IMAGE_SIDE", 1600, minimum=64),
        cors_origins=[
            origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()
        ],
        api_keys=[key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()],
        tesseract_cmd=os.getenv("TESSERACT_CMD", ""),
        tesseract_lang=os.getenv("TESSERACT_LANG", DEFAULT_LANG),
        tesseract_psm=_env_int("TESSERACT_PSM", DEFAULT_PSM, minimum=0, maximum=13),
        tesseract_fallback_psm=_env_int(
            "TESSERACT_FALLBACK_PSM",
            DEFAULT_FALLBACK_PSM,
            minimum=0,
            maximum=13,
        ),
        tesseract_oem=_env_int("TESSERACT_OEM", DEFAULT_OEM, minimum=0, maximum=3),
        tesseract_timeout=_env_float(
            "TESSERACT_TIMEOUT",
            DEFAULT_TIMEOUT,
            minimum=1.0,
            maximum=300.0,
        ),
        tesseract_tessdata=os.getenv("TESSERACT_TESSDATA", ""),
        tesseract_fallback_confidence=_env_float(
            "TESSERACT_FALLBACK_CONFIDENCE",
            DEFAULT_FALLBACK_CONFIDENCE,
            minimum=0.0,
            maximum=1.0,
        ),
        ocr_max_concurrency=_env_int(
            "OCR_MAX_CONCURRENCY",
            DEFAULT_MAX_CONCURRENCY,
            minimum=1,
            maximum=MAX_OCR_CONCURRENCY,
        ),
        ocr_upscale=_env_bool("OCR_UPSCALE", True),
        ocr_upscale_min_side=_env_int("OCR_UPSCALE_MIN_SIDE", 1000, minimum=256, maximum=4000),
        ocr_border=_env_int("OCR_BORDER", 20, minimum=0, maximum=100),
        ocr_deskew=_env_bool("OCR_DESKEW", True),
        ocr_deskew_max_angle=_env_float(
            "OCR_DESKEW_MAX_ANGLE",
            8.0,
            minimum=0.0,
            maximum=15.0,
        ),
        ocr_deskew_step=_env_float("OCR_DESKEW_STEP", 0.5, minimum=0.25, maximum=5.0),
        ocr_binarize_fallback=_env_bool("OCR_BINARIZE_FALLBACK", True),
        ocr_denoise_fallback=_env_bool("OCR_DENOISE_FALLBACK", True),
        ocr_bottom_strip_fallback=_env_bool("OCR_BOTTOM_STRIP_FALLBACK", True),
        ocr_bottom_strip_ratio=_env_float(
            "OCR_BOTTOM_STRIP_RATIO",
            0.30,
            minimum=0.10,
            maximum=0.60,
        ),
    )


settings = load_settings()
