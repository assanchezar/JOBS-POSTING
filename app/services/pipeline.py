from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Self

from PIL import Image

from app.core.config import settings
from app.services.email_extract import (
    AT_CONFUSION_RE,
    EMAIL_CUE_RE,
    LOOSE_EMAIL_RE,
    EmailHit,
    clean_email,
    clean_ocr_text,
    extract_emails,
    merge_email_hits,
    normalize_text,
    repair_ocr_confusions,
)
from app.services.ocr import OcrEngine, OcrError, OcrResult, TesseractEngine
from app.services.preprocess import load_image, prepare_for_ocr

logger = logging.getLogger(__name__)

# Senales de que la imagen es una oferta laboral. Se usan solo para decidir si
# vale la pena una pasada extra sobre la franja inferior, donde suele estar el
# contacto en los posters.
JOB_CUE_RE = re.compile(
    r"\b("
    r"buscamos|equipo|vendedor(?:a)?|empleo|trabajo|postul\w*|cv|curriculum|"
    r"requisitos|sueldo|jornada|whatsapp|contacto|enviar|envia|"
    r"seleccionamos|convocatoria|reclutamiento|personal|puesto|horario|turno"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class PipelineResult:
    emails: list[EmailHit]
    width: int
    height: int
    elapsed_ms: float
    text: str = ""
    engine: str = "tesseract"
    model: str = ""
    ocr_confidence: float | None = None


class ExtractionPipeline:
    """Orquesta validacion, preprocesamiento y OCR local."""

    def __init__(self, engine: OcrEngine | None = None) -> None:
        self.engine = engine or TesseractEngine()
        # Tesseract es CPU-bound; en PythonAnywhere conviene una sola extraccion a la vez.
        self._slots = threading.BoundedSemaphore(settings.ocr_max_concurrency)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def ready(self) -> bool:
        return self.engine.ready

    def close(self) -> None:
        self.engine.close()

    @staticmethod
    def _emails_from_ocr(ocr: OcrResult) -> list[EmailHit]:
        base_confidence = ocr.confidence if ocr.confidence is not None else 0.5
        line_hits: list[EmailHit] = []
        for line in ocr.lines:
            score = line.confidence if line.confidence > 0 else base_confidence
            line_hits.extend(extract_emails(line.text, confidence=score))

        # Red de seguridad para emails partidos entre lineas.
        text_hits = extract_emails(ocr.text, confidence=base_confidence)
        return merge_email_hits(line_hits, text_hits)

    def _extract_attempt(
        self,
        image: Image.Image,
        *,
        binarize: bool = False,
        denoise: bool = False,
    ) -> tuple[OcrResult, list[EmailHit]]:
        """Ejecuta una variante de preprocesado y devuelve OCR + emails."""
        prepared = prepare_for_ocr(image, binarize=binarize, denoise=denoise)
        try:
            ocr = self.engine.extract(prepared)
        finally:
            prepared.close()

        return ocr, self._emails_from_ocr(ocr)

    def _extract_bottom_strip(self, image: Image.Image) -> tuple[OcrResult, list[EmailHit]]:
        """OCR de la franja inferior: alli suele estar el contacto de los posters."""
        width, height = image.size
        top = int(height * (1.0 - settings.ocr_bottom_strip_ratio))
        crop = image.crop((0, top, width, height))
        try:
            prepared = prepare_for_ocr(crop, denoise=True)
            try:
                ocr = self.engine.extract(prepared)
            finally:
                prepared.close()
        finally:
            crop.close()

        return ocr, self._emails_from_ocr(ocr)

    @staticmethod
    def _smart_retry_text(ocr: OcrResult, emails: list[EmailHit]) -> list[EmailHit]:
        """Smart retry seguro: repara candidatos laxos sin hacer nuevo OCR.
        Solo activa si hay contexto de email y candidatos sin extraer.
        No relentiza porque opera solo sobre texto existente."""
        text = ocr.text
        if not text:
            return emails
        # Detecta candidatos laxos (como en bench_images)
        candidates = sorted({
            match.lower().strip(".,;:()[]<>\"'")
            for match in LOOSE_EMAIL_RE.findall(normalize_text(repair_ocr_confusions(text)))
        })
        # Si no hay candidatos nuevos, no hace nada
        existing = {email.email.lower() for email in emails}
        missing_candidates = [c for c in candidates if c not in existing]
        if not missing_candidates:
            return emails
        # Solo intenta si hay contexto de email o @ presente
        has_context = bool(EMAIL_CUE_RE.search(text)) or bool(AT_CONFUSION_RE.search(text))
        if not has_context:
            return emails
        # Intenta extraer los candidatos faltantes con un umbral mas laxo
        new_hits = extract_emails(text, confidence=0.4)
        # Combina sin duplicar, manteniendo confianza alta si ya existia
        combined = merge_email_hits(emails, new_hits)
        # Si el retry no agrego valor (mismos o menos emails confiables), conserva original
        if len(combined) <= len(emails) and all(
            hit.email.lower() in existing for hit in combined if hit.email.lower() not in existing
        ):
            # Solo conserva los nuevos si mejoran la confianza media
            if combined and sum(h.confidence for h in combined) / len(combined) >= sum(h.confidence for h in emails) / len(emails) if emails else True:
                return combined
            return emails
        return combined

    @staticmethod
    def _attempt_score(
        attempt: tuple[OcrResult, list[EmailHit]],
    ) -> tuple[int, float, float, int]:
        ocr, emails = attempt
        best_email = max((hit.confidence for hit in emails), default=0.0)
        return (len(emails), best_email, ocr.confidence or 0.0, len(ocr.text))

    def process_bytes(self, data: bytes, include_text: bool = False) -> PipelineResult:
        started = time.perf_counter()
        # El limite cubre decodificacion, preprocesamiento y OCR para acotar memoria.
        with self._slots:
            image = load_image(data)
            width, height = image.size
            try:
                attempts = [self._extract_attempt(image)]
                best_email = max(
                    (hit.confidence for hit in attempts[0][1]),
                    default=0.0,
                )
                # Si el primer pase no encuentra emails o son poco fiables, prueba
                # binarizacion y luego reduccion de ruido. Son las mejoras del
                # articulo que mas ayudan en escaneos con fondo irregular o ruido.
                retry = not attempts[0][1] or best_email < settings.tesseract_fallback_confidence
                if retry and settings.ocr_binarize_fallback:
                    try:
                        attempts.append(self._extract_attempt(image, binarize=True))
                    except OcrError:
                        logger.warning(
                            "Fallo la pasada con binarizacion; se conserva el primer resultado.",
                            exc_info=True,
                        )
                if (
                    retry
                    and settings.ocr_denoise_fallback
                    and not any(emails for _, emails in attempts[1:])
                ):
                    try:
                        attempts.append(self._extract_attempt(image, denoise=True))
                    except OcrError:
                        logger.warning(
                            "Fallo la pasada con reduccion de ruido; "
                            "se conserva el primer resultado.",
                            exc_info=True,
                        )
                if (
                    settings.ocr_bottom_strip_fallback
                    and not any(emails for _, emails in attempts)
                    and image.height >= image.width
                    and JOB_CUE_RE.search("\n".join(ocr.text for ocr, _ in attempts))
                ):
                    try:
                        bottom_attempt = self._extract_bottom_strip(image)
                        if bottom_attempt[1]:
                            attempts.append(bottom_attempt)
                    except OcrError:
                        logger.warning(
                            "Fallo la pasada sobre la franja inferior; "
                            "se conserva el resultado anterior.",
                            exc_info=True,
                        )
            finally:
                image.close()

        ocr, emails = max(attempts, key=self._attempt_score)
        # Smart retry seguro: solo sobre el texto ya extraido, sin nuevo OCR.
        # Se activa solo si hay candidatos laxos sin extraer y hay contexto de email.
        emails = self._smart_retry_text(ocr, emails)

        return PipelineResult(
            emails=emails,
            width=width,
            height=height,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            text=clean_ocr_text(ocr.text) if include_text else "",
            engine=ocr.engine,
            model=ocr.model,
            ocr_confidence=ocr.confidence,
        )


_pipeline: ExtractionPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> ExtractionPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ExtractionPipeline()
    return _pipeline


def close_pipeline() -> None:
    """Cierra el motor OCR compartido (se llama al apagar la aplicacion)."""
    global _pipeline
    with _pipeline_lock:
        pipeline = _pipeline
        _pipeline = None
    if pipeline is not None:
        pipeline.close()
