from __future__ import annotations

import io

import pytest
from PIL import Image

from app.core.config import settings
from app.services.ocr import OcrLine, OcrResult
from app.services.pipeline import ExtractionPipeline


class FakeEngine:
    name = "tesseract"
    model = "tesseract-test"

    def __init__(self, lines: list[OcrLine] | None = None, ready: bool = True) -> None:
        self.lines = lines or []
        self.ready = ready
        self.closed = False
        self.image_mode: str | None = None
        self.image_size: tuple[int, int] | None = None

    @property
    def language(self) -> str:
        return "spa+eng"

    def extract(self, image: Image.Image) -> OcrResult:
        self.image_mode = image.mode
        self.image_size = image.size
        return OcrResult(
            lines=self.lines,
            engine=self.name,
            model=self.model,
            elapsed_ms=12.5,
        )

    def close(self) -> None:
        self.closed = True


class FallbackEngine(FakeEngine):
    """Devuelve texto sin email hasta la pasada indicada."""

    def __init__(self, email_on_call: int = 2) -> None:
        super().__init__()
        self.email_on_call = email_on_call
        self.calls = 0

    def extract(self, image: Image.Image) -> OcrResult:
        self.calls += 1
        self.image_mode = image.mode
        self.image_size = image.size
        lines = (
            [OcrLine("ventas@acme.com", 0.9)]
            if self.calls >= self.email_on_call
            else [OcrLine("Solo texto de relleno", 0.9)]
        )
        return OcrResult(
            lines=lines,
            engine=self.name,
            model=self.model,
            elapsed_ms=12.5,
        )


class BottomStripEngine(FakeEngine):
    """Simula un poster cuyo email solo aparece al recortar la franja inferior."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def extract(self, image: Image.Image) -> OcrResult:
        self.calls += 1
        self.image_mode = image.mode
        self.image_size = image.size
        lines = (
            [OcrLine("reclutamiento.mallfit@gmail.com", 0.9)]
            if self.calls >= 4
            else [OcrLine("Buscamos vendedor con experiencia", 0.9)]
        )
        return OcrResult(
            lines=lines,
            engine=self.name,
            model=self.model,
            elapsed_ms=12.5,
        )


def png_bytes(width: int = 300, height: int = 120) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_pipeline_extracts_emails_and_cleans_text() -> None:
    engine = FakeEngine(
        [
            OcrLine("Ventas: ventas@acme.com", 0.95),
            OcrLine("Soporte: soporte@acme.com", 0.8),
            OcrLine("Extra: extra@acme.org", 0.9),
        ]
    )
    pipeline = ExtractionPipeline(engine=engine)

    output = pipeline.process_bytes(png_bytes(), include_text=True)

    assert [hit.email for hit in output.emails] == [
        "ventas@acme.com",
        "extra@acme.org",
        "soporte@acme.com",
    ]
    assert output.emails[0].confidence == 0.95
    assert output.emails[0].occurrences == 1
    assert output.width == 300
    assert output.height == 120
    assert output.engine == "tesseract"
    assert output.model == "tesseract-test"
    assert output.ocr_confidence == 0.883
    assert output.text.startswith("Ventas:")
    assert "@" not in output.text.splitlines()
    assert "---" not in output.text.splitlines()
    assert engine.image_mode == "L"
    assert engine.image_size is not None


def test_pipeline_hides_text_when_not_requested() -> None:
    engine = FakeEngine([OcrLine("Ventas: ventas@acme.com", 0.9)])
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes())
    assert output.text == ""


def test_pipeline_cleans_decorative_text_noise() -> None:
    engine = FakeEngine(
        [
            OcrLine("Ventas: ventas@acme.com", 0.9),
            OcrLine("@", 0.2),
            OcrLine("@@@", 0.2),
            OcrLine("---", 0.2),
        ]
    )
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes(), include_text=True)
    assert output.text == "Ventas: ventas@acme.com"


def test_pipeline_uses_full_text_for_split_email() -> None:
    engine = FakeEngine(
        [
            OcrLine("Escribe a info@", 0.8),
            OcrLine("acme.org para mas informacion", 0.8),
        ]
    )
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes())
    assert [hit.email for hit in output.emails] == ["info@acme.org"]
    assert output.emails[0].confidence == 0.8


def test_pipeline_returns_no_emails_for_text_without_email() -> None:
    engine = FakeEngine([OcrLine("Solo texto de relleno", 0.9)])
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes())
    assert output.emails == []


def test_pipeline_retries_with_binarization_when_no_email() -> None:
    engine = FallbackEngine(email_on_call=2)
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes())
    assert [hit.email for hit in output.emails] == ["ventas@acme.com"]
    assert engine.calls == 2


def test_pipeline_retries_with_denoise_if_binarization_fails() -> None:
    engine = FallbackEngine(email_on_call=3)
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes())
    assert [hit.email for hit in output.emails] == ["ventas@acme.com"]
    assert engine.calls == 3


def test_pipeline_uses_bottom_strip_for_poster_without_email() -> None:
    engine = BottomStripEngine()
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes(120, 300))
    assert [hit.email for hit in output.emails] == ["reclutamiento.mallfit@gmail.com"]
    assert engine.calls == 4


def test_pipeline_can_disable_bottom_strip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_bottom_strip_fallback", False)
    engine = BottomStripEngine()
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes(120, 300))
    assert output.emails == []
    assert engine.calls == 3


def test_pipeline_can_disable_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_binarize_fallback", False)
    monkeypatch.setattr(settings, "ocr_denoise_fallback", False)
    engine = FallbackEngine(email_on_call=2)
    output = ExtractionPipeline(engine=engine).process_bytes(png_bytes())
    assert output.emails == []
    assert engine.calls == 1


def test_pipeline_ready_reflects_engine() -> None:
    assert ExtractionPipeline(engine=FakeEngine(ready=True)).ready is True
    assert ExtractionPipeline(engine=FakeEngine(ready=False)).ready is False


def test_pipeline_close_delegates_to_engine() -> None:
    engine = FakeEngine()
    ExtractionPipeline(engine=engine).close()
    assert engine.closed is True


def test_close_pipeline_releases_global(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.pipeline as pipeline_module

    engine = FakeEngine()
    fake = ExtractionPipeline(engine=engine)
    monkeypatch.setattr(pipeline_module, "_pipeline", fake)

    pipeline_module.close_pipeline()

    assert engine.closed is True
    assert pipeline_module._pipeline is None
