from __future__ import annotations

from app.core.config import (
    DEFAULT_FALLBACK_CONFIDENCE,
    DEFAULT_FALLBACK_PSM,
    DEFAULT_LANG,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_OEM,
    DEFAULT_PSM,
    DEFAULT_TIMEOUT,
    MAX_OCR_CONCURRENCY,
    Settings,
    load_settings,
    normalize_lang,
)


def test_settings_clamps_invalid_numeric_values() -> None:
    settings = Settings(
        max_upload_bytes=0,
        max_image_pixels=-1,
        max_side=1,
        tesseract_psm=99,
        tesseract_fallback_psm=-4,
        tesseract_oem=9,
        tesseract_timeout=0,
        ocr_max_concurrency=99,
        ocr_upscale_min_side=1,
        ocr_border=999,
        ocr_deskew_max_angle=99,
        ocr_deskew_step=0,
        ocr_bottom_strip_ratio=9,
        tesseract_fallback_confidence=9,
    )

    assert settings.max_upload_bytes == 1
    assert settings.max_image_pixels == 1
    assert settings.max_side == 64
    assert settings.tesseract_psm == 13
    assert settings.tesseract_fallback_psm == 0
    assert settings.tesseract_oem == 3
    assert settings.tesseract_timeout == 1.0
    assert settings.ocr_max_concurrency == MAX_OCR_CONCURRENCY
    assert settings.ocr_upscale_min_side == 256
    assert settings.ocr_border == 100
    assert settings.ocr_deskew_max_angle == 15.0
    assert settings.ocr_deskew_step == 0.25
    assert settings.ocr_bottom_strip_ratio == 0.60
    assert settings.tesseract_fallback_confidence == 1.0


def test_settings_normalizes_language() -> None:
    assert Settings(tesseract_lang=" SPA + eng + spa ").tesseract_lang == "spa+eng"
    assert Settings(tesseract_lang="").tesseract_lang == DEFAULT_LANG
    assert normalize_lang(None) == DEFAULT_LANG
    assert normalize_lang("eng+eng") == "eng"


def test_settings_defaults_are_safe() -> None:
    settings = Settings()
    assert settings.tesseract_lang == DEFAULT_LANG
    assert settings.tesseract_psm == DEFAULT_PSM
    assert settings.tesseract_fallback_psm == DEFAULT_FALLBACK_PSM
    assert settings.tesseract_oem == DEFAULT_OEM
    assert settings.tesseract_timeout == DEFAULT_TIMEOUT
    assert settings.tesseract_fallback_confidence == DEFAULT_FALLBACK_CONFIDENCE
    assert settings.ocr_max_concurrency == DEFAULT_MAX_CONCURRENCY
    assert settings.ocr_deskew is True
    assert settings.ocr_binarize_fallback is True
    assert settings.ocr_denoise_fallback is True
    assert settings.ocr_bottom_strip_fallback is True
    assert settings.ocr_bottom_strip_ratio == 0.30
    assert settings.api_keys == []


def test_settings_rejects_non_finite_timeout() -> None:
    assert Settings(tesseract_timeout=float("inf")).tesseract_timeout == DEFAULT_TIMEOUT
    assert Settings(tesseract_timeout=float("nan")).tesseract_timeout == DEFAULT_TIMEOUT


def test_settings_caps_timeout() -> None:
    assert Settings(tesseract_timeout=9999).tesseract_timeout == 300.0


def test_settings_strips_paths() -> None:
    settings = Settings(
        tesseract_cmd=" /usr/bin/tesseract ",
        tesseract_tessdata=" /home/user/tessdata ",
    )
    assert settings.tesseract_cmd == "/usr/bin/tesseract"
    assert settings.tesseract_tessdata == "/home/user/tessdata"


def test_settings_filters_empty_api_keys() -> None:
    settings = Settings(api_keys=[" uno ", "", "   ", "dos"])
    assert settings.api_keys == ["uno", "dos"]


def test_settings_deduplicates_api_keys() -> None:
    settings = Settings(api_keys=["clave", " clave ", "otra", "otra"])
    assert settings.api_keys == ["clave", "otra"]


def test_empty_boolean_env_uses_default(monkeypatch) -> None:
    monkeypatch.setenv("OCR_UPSCALE", "")
    assert load_settings().ocr_upscale is True


def test_unknown_boolean_env_uses_default(monkeypatch) -> None:
    monkeypatch.setenv("OCR_UPSCALE", "quizas")
    assert load_settings().ocr_upscale is True


def test_false_boolean_env(monkeypatch) -> None:
    monkeypatch.setenv("OCR_UPSCALE", "false")
    assert load_settings().ocr_upscale is False


def test_load_settings_reads_tesseract_env(monkeypatch) -> None:
    monkeypatch.setenv("TESSERACT_CMD", "/usr/bin/tesseract")
    monkeypatch.setenv("TESSERACT_LANG", "ENG+spa")
    monkeypatch.setenv("TESSERACT_PSM", "11")
    monkeypatch.setenv("TESSERACT_FALLBACK_PSM", "6")
    monkeypatch.setenv("TESSERACT_OEM", "1")
    monkeypatch.setenv("TESSERACT_TIMEOUT", "45")
    monkeypatch.setenv("TESSERACT_TESSDATA", "/home/user/tessdata")
    monkeypatch.setenv("TESSERACT_FALLBACK_CONFIDENCE", "0.7")
    monkeypatch.setenv("OCR_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("OCR_BORDER", "10")
    monkeypatch.setenv("OCR_DESKEW", "false")
    monkeypatch.setenv("OCR_DESKEW_MAX_ANGLE", "6")
    monkeypatch.setenv("OCR_DESKEW_STEP", "0.25")
    monkeypatch.setenv("OCR_BINARIZE_FALLBACK", "false")
    monkeypatch.setenv("OCR_DENOISE_FALLBACK", "false")
    monkeypatch.setenv("OCR_BOTTOM_STRIP_FALLBACK", "false")
    monkeypatch.setenv("OCR_BOTTOM_STRIP_RATIO", "0.4")

    settings = load_settings()
    assert settings.tesseract_cmd == "/usr/bin/tesseract"
    assert settings.tesseract_lang == "eng+spa"
    assert settings.tesseract_psm == 11
    assert settings.tesseract_fallback_psm == 6
    assert settings.tesseract_oem == 1
    assert settings.tesseract_timeout == 45.0
    assert settings.tesseract_tessdata == "/home/user/tessdata"
    assert settings.tesseract_fallback_confidence == 0.7
    assert settings.ocr_max_concurrency == 2
    assert settings.ocr_border == 10
    assert settings.ocr_deskew is False
    assert settings.ocr_deskew_max_angle == 6.0
    assert settings.ocr_deskew_step == 0.25
    assert settings.ocr_binarize_fallback is False
    assert settings.ocr_denoise_fallback is False
    assert settings.ocr_bottom_strip_fallback is False
    assert settings.ocr_bottom_strip_ratio == 0.4
