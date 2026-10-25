from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings
from app.services.preprocess import (
    ImageTooLargeError,
    ImageValidationError,
    binarize_image,
    estimate_skew_angle,
    load_image,
    otsu_threshold,
    prepare_for_ocr,
)


def png_bytes(width: int = 200, height: int = 100, mode: str = "RGB") -> bytes:
    image = Image.new(mode, (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_accepts_png() -> None:
    image = load_image(png_bytes())
    assert image.mode == "RGB"
    assert image.size == (200, 100)


def test_load_image_rejects_text() -> None:
    with pytest.raises(ImageValidationError):
        load_image(b"esto no es una imagen")


def test_load_image_rejects_empty() -> None:
    with pytest.raises(ImageValidationError):
        load_image(b"")


def test_load_image_rejects_gif() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buffer, format="GIF")
    with pytest.raises(ImageValidationError):
        load_image(buffer.getvalue())


def test_load_image_rejects_too_many_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_image_pixels", 100)
    with pytest.raises(ImageTooLargeError):
        load_image(png_bytes(50, 50))


def test_load_image_downscales_large_side(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_side", 100)
    image = load_image(png_bytes(400, 200))
    assert max(image.size) == 100


def test_load_image_handles_transparent_png() -> None:
    image = load_image(png_bytes(120, 60, mode="RGBA"))
    assert image.mode == "RGB"


def test_prepare_for_ocr_returns_grayscale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_upscale", False)
    monkeypatch.setattr(settings, "ocr_border", 0)
    image = load_image(png_bytes())
    prepared = prepare_for_ocr(image)
    assert prepared.mode == "L"


def test_prepare_for_ocr_upscales_small_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_upscale", True)
    monkeypatch.setattr(settings, "ocr_upscale_min_side", 1000)
    monkeypatch.setattr(settings, "ocr_border", 0)
    image = load_image(png_bytes(200, 100))
    prepared = prepare_for_ocr(image)
    assert prepared.size == (400, 200)


def test_prepare_for_ocr_adds_white_border(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_upscale", False)
    monkeypatch.setattr(settings, "ocr_border", 10)
    image = load_image(png_bytes(200, 100))
    prepared = prepare_for_ocr(image)
    assert prepared.size == (220, 120)


def test_prepare_for_ocr_respects_max_side(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_side", 300)
    monkeypatch.setattr(settings, "ocr_upscale", True)
    monkeypatch.setattr(settings, "ocr_upscale_min_side", 1000)
    monkeypatch.setattr(settings, "ocr_border", 0)
    image = load_image(png_bytes(200, 100))
    prepared = prepare_for_ocr(image)
    assert prepared.size == (300, 150)


def text_image(angle: float = 0.0) -> Image.Image:
    image = Image.new("L", (600, 220), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    draw.text((40, 40), "ventas@acme.com", font=font, fill=0)
    draw.text((40, 110), "soporte@acme.com", font=font, fill=0)
    if angle:
        return image.rotate(angle, expand=True, fillcolor=255)
    return image


def test_otsu_threshold_separates_black_and_white() -> None:
    image = Image.new("L", (20, 10), 0)
    image.paste(255, (0, 0, 10, 10))
    threshold = otsu_threshold(image)
    assert 0 <= threshold < 255


def test_binarize_image_returns_only_black_and_white() -> None:
    image = Image.new("L", (20, 10), 30)
    image.paste(220, (0, 0, 10, 10))
    binary = binarize_image(image)
    histogram = binary.histogram()
    assert sum(histogram[1:255]) == 0
    assert histogram[0] + histogram[255] == binary.width * binary.height
    assert binary.getextrema() == (0, 255)


def test_estimate_skew_angle_detects_correction() -> None:
    detected = estimate_skew_angle(text_image(3.0), max_angle=6.0, step=0.5)
    assert abs(detected + 3.0) <= 0.75


def test_estimate_skew_angle_keeps_straight_text() -> None:
    assert estimate_skew_angle(text_image(0.0), max_angle=6.0, step=0.5) == 0.0


def test_prepare_for_ocr_deskews_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_upscale", False)
    monkeypatch.setattr(settings, "ocr_border", 0)
    original = text_image(3.0)
    prepared = prepare_for_ocr(original)
    assert prepared.size != original.size
