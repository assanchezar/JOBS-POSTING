from __future__ import annotations

import pytesseract
import pytest
from PIL import Image

from app.services.ocr import (
    OcrConfigError,
    OcrExecutionError,
    OcrLine,
    OcrTimeoutError,
    TesseractEngine,
    lines_from_data,
)


def test_lines_from_data_groups_words_by_line() -> None:
    data = {
        "text": ["Hola", "mundo", "", "ventas@acme.com"],
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 2, 2],
        "conf": ["90", "80", "-1", "95"],
    }

    lines = lines_from_data(data)

    assert [(line.text, line.confidence) for line in lines] == [
        ("Hola mundo", 0.85),
        ("ventas@acme.com", 0.95),
    ]


def test_lines_from_data_handles_missing_columns() -> None:
    lines = lines_from_data({"text": ["solo", "texto"]})
    assert [line.text for line in lines] == ["solo texto"]
    assert lines[0].confidence == 0.0


def test_engine_is_not_ready_with_invalid_command() -> None:
    engine = TesseractEngine(cmd="ruta/que/no/existe")
    assert engine.ready is False
    with pytest.raises(OcrConfigError):
        engine.extract(Image.new("L", (10, 10)))


def test_engine_model_is_plain_without_binary() -> None:
    assert TesseractEngine(cmd="ruta/que/no/existe").model == "tesseract"


def test_engine_normalizes_requested_language() -> None:
    assert TesseractEngine(lang=" SPA + eng + spa ").requested_language == "spa+eng"


def test_language_uses_requested_available_languages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TesseractEngine(lang="spa+eng")
    monkeypatch.setattr(engine, "_available_languages", lambda: ["spa", "eng"])
    assert engine.language == "spa+eng"


def test_language_falls_back_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TesseractEngine(lang="spa+eng")
    monkeypatch.setattr(engine, "_available_languages", lambda: ["eng"])
    assert engine.language == "eng"


def test_language_raises_without_language_data(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TesseractEngine(lang="spa+eng")
    monkeypatch.setattr(engine, "_available_languages", list)
    with pytest.raises(OcrConfigError):
        _ = engine.language


def test_extract_uses_fallback_psm(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TesseractEngine(lang="eng", psm=6, fallback_psm=11)
    engine._version = "4.1.1"
    monkeypatch.setattr(TesseractEngine, "ready", property(lambda self: True))
    monkeypatch.setattr(TesseractEngine, "language", property(lambda self: "eng"))

    calls: list[int] = []

    def fake_run(image: Image.Image, psm: int) -> list[OcrLine]:
        calls.append(psm)
        if psm == 6:
            return []
        return [OcrLine("ventas@acme.com", 0.9)]

    monkeypatch.setattr(engine, "_run", fake_run)

    result = engine.extract(Image.new("L", (10, 10)))

    assert calls == [6, 11]
    assert result.text == "ventas@acme.com"
    assert result.model == "tesseract-4.1.1"
    assert result.confidence == 0.9


def test_extract_does_not_fallback_when_text_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TesseractEngine(lang="eng", psm=6, fallback_psm=11)
    engine._version = "4.1.1"
    monkeypatch.setattr(TesseractEngine, "ready", property(lambda self: True))
    monkeypatch.setattr(TesseractEngine, "language", property(lambda self: "eng"))

    calls: list[int] = []

    def fake_run(image: Image.Image, psm: int) -> list[OcrLine]:
        calls.append(psm)
        return [OcrLine("texto", 0.95)]

    monkeypatch.setattr(engine, "_run", fake_run)

    engine.extract(Image.new("L", (10, 10)))

    assert calls == [6]


def test_extract_uses_fallback_when_confidence_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TesseractEngine(lang="eng", psm=6, fallback_psm=11, fallback_confidence=0.8)
    engine._version = "4.1.1"
    monkeypatch.setattr(TesseractEngine, "ready", property(lambda self: True))
    monkeypatch.setattr(TesseractEngine, "language", property(lambda self: "eng"))

    calls: list[int] = []

    def fake_run(image: Image.Image, psm: int) -> list[OcrLine]:
        calls.append(psm)
        if psm == 6:
            return [OcrLine("texto dudoso", 0.3)]
        return [OcrLine("texto claro", 0.95)]

    monkeypatch.setattr(engine, "_run", fake_run)

    result = engine.extract(Image.new("L", (10, 10)))

    assert calls == [6, 11]
    assert result.text == "texto claro"


def test_extract_keeps_primary_when_fallback_is_worse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TesseractEngine(lang="eng", psm=6, fallback_psm=11, fallback_confidence=0.8)
    engine._version = "4.1.1"
    monkeypatch.setattr(TesseractEngine, "ready", property(lambda self: True))
    monkeypatch.setattr(TesseractEngine, "language", property(lambda self: "eng"))

    def fake_run(image: Image.Image, psm: int) -> list[OcrLine]:
        if psm == 6:
            return [OcrLine("texto principal", 0.4)]
        return [OcrLine("x", 0.2)]

    monkeypatch.setattr(engine, "_run", fake_run)

    result = engine.extract(Image.new("L", (10, 10)))

    assert result.text == "texto principal"


def test_extract_propagates_language_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TesseractEngine(lang="spa")
    monkeypatch.setattr(TesseractEngine, "ready", property(lambda self: True))
    monkeypatch.setattr(engine, "_available_languages", list)

    with pytest.raises(OcrConfigError):
        engine.extract(Image.new("L", (10, 10)))


def test_run_maps_tesseract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TesseractEngine(lang="eng")
    monkeypatch.setattr(TesseractEngine, "ready", property(lambda self: True))
    monkeypatch.setattr(TesseractEngine, "language", property(lambda self: "eng"))
    monkeypatch.setattr(
        pytesseract,
        "image_to_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pytesseract.TesseractError(1, "fallo interno")
        ),
    )

    with pytest.raises(OcrExecutionError):
        engine.extract(Image.new("L", (10, 10)))


def test_run_maps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TesseractEngine(lang="eng")
    monkeypatch.setattr(TesseractEngine, "ready", property(lambda self: True))
    monkeypatch.setattr(TesseractEngine, "language", property(lambda self: "eng"))
    monkeypatch.setattr(
        pytesseract,
        "image_to_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Tesseract process timeout")),
    )

    with pytest.raises(OcrTimeoutError):
        engine.extract(Image.new("L", (10, 10)))
