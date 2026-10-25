from __future__ import annotations

import logging
import math
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import pytesseract
from PIL import Image

from app.core.config import normalize_lang, settings

logger = logging.getLogger(__name__)


class OcrError(RuntimeError):
    """Error generico del motor OCR."""


class OcrConfigError(OcrError):
    """Falta el binario de Tesseract o sus datos de idioma."""


class OcrTimeoutError(OcrError):
    """Tesseract no termino dentro del tiempo configurado."""


class OcrExecutionError(OcrError):
    """Tesseract fallo durante el procesamiento."""


@dataclass(slots=True)
class OcrLine:
    """Una linea detectada, con su confianza media entre 0 y 1."""

    text: str
    confidence: float


@dataclass(slots=True)
class OcrResult:
    lines: list[OcrLine]
    engine: str
    model: str
    elapsed_ms: float

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def confidence(self) -> float | None:
        if not self.lines:
            return None
        return round(sum(line.confidence for line in self.lines) / len(self.lines), 3)


@runtime_checkable
class OcrEngine(Protocol):
    name: str
    model: str

    @property
    def ready(self) -> bool: ...

    @property
    def language(self) -> str: ...

    def extract(self, image: Image.Image) -> OcrResult: ...

    def close(self) -> None: ...


def _to_float(value: Any, default: float = -1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _value(data: dict[str, Any], key: str, index: int, default: Any) -> Any:
    values = data.get(key)
    if isinstance(values, (list, tuple)) and index < len(values):
        return values[index]
    return default


def lines_from_data(data: dict[str, Any]) -> list[OcrLine]:
    """Agrupa el TSV de Tesseract por bloque/parrafo/linea."""
    words = data.get("text")
    if not isinstance(words, (list, tuple)):
        return []

    grouped: OrderedDict[tuple[int, int, int, int], list[tuple[str, float]]] = OrderedDict()
    for index, raw_word in enumerate(words):
        word = str(raw_word or "").strip()
        if not word:
            continue
        key = (
            _to_int(_value(data, "page_num", index, 0)),
            _to_int(_value(data, "block_num", index, 0)),
            _to_int(_value(data, "par_num", index, 0)),
            _to_int(_value(data, "line_num", index, 0)),
        )
        confidence = _to_float(_value(data, "conf", index, -1.0))
        grouped.setdefault(key, []).append((word, confidence))

    lines: list[OcrLine] = []
    for line_words in grouped.values():
        text = " ".join(word for word, _ in line_words)
        confidences = [
            confidence
            for _, confidence in line_words
            if confidence >= 0 and math.isfinite(confidence)
        ]
        average = sum(confidences) / len(confidences) if confidences else 0.0
        lines.append(
            OcrLine(
                text=text,
                confidence=round(min(max(average / 100.0, 0.0), 1.0), 3),
            )
        )
    return lines


class TesseractEngine:
    """Motor OCR local basado en el binario de Tesseract y pytesseract."""

    name = "tesseract"

    def __init__(
        self,
        cmd: str | None = None,
        lang: str | None = None,
        *,
        psm: int | None = None,
        fallback_psm: int | None = None,
        oem: int | None = None,
        timeout: float | None = None,
        tessdata_dir: str | None = None,
        fallback_confidence: float | None = None,
    ) -> None:
        raw_cmd = (cmd if cmd is not None else settings.tesseract_cmd).strip()
        self.cmd = os.path.expanduser(raw_cmd) if raw_cmd else ""
        if self.cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cmd

        self.requested_language = normalize_lang(
            lang if lang is not None else settings.tesseract_lang
        )
        self.psm = min(max(psm if psm is not None else settings.tesseract_psm, 0), 13)
        self.fallback_psm = min(
            max(fallback_psm if fallback_psm is not None else settings.tesseract_fallback_psm, 0),
            13,
        )
        self.oem = min(max(oem if oem is not None else settings.tesseract_oem, 0), 3)
        timeout_value = timeout if timeout is not None else settings.tesseract_timeout
        self.timeout = min(max(timeout_value, 1.0), 300.0) if math.isfinite(timeout_value) else 30.0
        confidence_value = (
            fallback_confidence
            if fallback_confidence is not None
            else settings.tesseract_fallback_confidence
        )
        self.fallback_confidence = (
            min(max(confidence_value, 0.0), 1.0) if math.isfinite(confidence_value) else 0.5
        )
        raw_tessdata = (
            tessdata_dir if tessdata_dir is not None else settings.tesseract_tessdata
        ).strip()
        self.tessdata_dir = os.path.expanduser(raw_tessdata) if raw_tessdata else ""

        self._version: str | None = None
        self._language: str | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def ready(self) -> bool:
        return self._binary_path() is not None

    @property
    def model(self) -> str:
        if self._version is None:
            if not self.ready:
                return "tesseract"
            try:
                self._version = str(pytesseract.get_tesseract_version())
            except Exception:
                logger.warning("No se pudo obtener la version de Tesseract.", exc_info=True)
                return "tesseract"
        return f"tesseract-{self._version}"

    @property
    def language(self) -> str:
        if self._language is not None:
            return self._language

        available = self._available_languages()
        requested = [part for part in self.requested_language.split("+") if part]
        selected = [part for part in requested if part in available]
        if not selected:
            if "eng" in available:
                selected = ["eng"]
            elif available:
                selected = [available[0]]

        if not selected:
            raise OcrConfigError(
                "Tesseract no tiene datos de idioma disponibles. "
                "Define TESSERACT_TESSDATA o instala spa.traineddata."
            )

        self._language = "+".join(selected)
        if selected != requested:
            logger.warning(
                "Idiomas solicitados no disponibles: %s. Usando: %s",
                self.requested_language,
                self._language,
            )
        return self._language

    def close(self) -> None:
        """pytesseract no mantiene recursos persistentes."""

    def extract(self, image: Image.Image) -> OcrResult:
        if not self.ready:
            raise OcrConfigError(
                "Tesseract no esta instalado o TESSERACT_CMD apunta a una ruta invalida."
            )

        started = time.perf_counter()
        lines = self._run(image, self.psm)
        if self.fallback_psm != self.psm and self._needs_fallback(lines):
            logger.info(
                "Tesseract uso psm=%d con confianza baja o sin texto; reintentando con psm=%d.",
                self.psm,
                self.fallback_psm,
            )
            fallback_lines = self._run(image, self.fallback_psm)
            if fallback_lines and (
                not lines or self._quality(fallback_lines) > self._quality(lines)
            ):
                lines = fallback_lines

        return OcrResult(
            lines=lines,
            engine=self.name,
            model=self.model,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _needs_fallback(self, lines: list[OcrLine]) -> bool:
        if not lines:
            return True
        if self.fallback_confidence <= 0:
            return False
        average = sum(line.confidence for line in lines) / len(lines)
        return average < self.fallback_confidence

    @staticmethod
    def _quality(lines: list[OcrLine]) -> float:
        """Prefiere el resultado con mas texto y mejor confianza."""
        return sum(line.confidence * max(1, len(line.text)) for line in lines)

    def _binary_path(self) -> str | None:
        command = self.cmd or "tesseract"
        path = Path(command)
        if path.is_absolute() or path.parent != Path("."):
            return str(path) if path.is_file() else None
        return shutil.which(command)

    def _tessdata_config(self) -> str:
        return f"--tessdata-dir {self.tessdata_dir}" if self.tessdata_dir else ""

    def _available_languages(self) -> list[str]:
        if hasattr(self, "_cached_languages") and self._cached_languages is not None:
            return self._cached_languages
        try:
            languages = pytesseract.get_languages(config=self._tessdata_config())
        except Exception:
            logger.warning("No se pudieron listar los idiomas de Tesseract.", exc_info=True)
            self._cached_languages = []
            return []
        result = [
            language
            for language in (str(item).strip() for item in languages)
            if language and language not in {"osd", "equ"}
        ]
        self._cached_languages = result
        return result

    def _config(self, psm: int) -> str:
        parts = [f"--oem {self.oem}", f"--psm {psm}"]
        if self.tessdata_dir:
            parts.append(f"--tessdata-dir {self.tessdata_dir}")
        return " ".join(parts)

    def _run(self, image: Image.Image, psm: int) -> list[OcrLine]:
        # Resolver el idioma fuera del try para que OcrConfigError llegue como 503.
        language = self.language
        try:
            data = pytesseract.image_to_data(
                image,
                lang=language,
                config=self._config(psm),
                output_type=pytesseract.Output.DICT,
                timeout=self.timeout,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrConfigError(
                "No se encontro el ejecutable de Tesseract. Define TESSERACT_CMD."
            ) from exc
        except pytesseract.TesseractError as exc:
            message = " ".join(str(exc).split())[:300]
            raise OcrExecutionError(f"Tesseract fallo: {message}") from exc
        except RuntimeError as exc:
            if "timeout" in str(exc).lower():
                raise OcrTimeoutError("Tesseract excedio el tiempo de espera.") from exc
            raise OcrExecutionError("Tesseract fallo durante el OCR.") from exc
        except (OSError, ValueError) as exc:
            raise OcrExecutionError("No se pudo ejecutar Tesseract.") from exc

        return lines_from_data(data)
