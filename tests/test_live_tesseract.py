from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.services.ocr import TesseractEngine
from app.services.pipeline import ExtractionPipeline

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not TesseractEngine().ready,
    reason="Define RUN_LIVE_TESTS=1 y ten Tesseract instalado para la prueba real.",
)


def test_live_extract_clean_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "generated" / "clean.png"
    with ExtractionPipeline() as pipeline:
        result = pipeline.process_bytes(fixture.read_bytes(), include_text=True)

    emails = {hit.email.lower() for hit in result.emails}
    assert "ventas@acme.com" in emails
    assert result.model
    assert result.elapsed_ms > 0
