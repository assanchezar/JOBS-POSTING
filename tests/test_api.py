from __future__ import annotations

import io
import re

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app
from app.services.email_extract import EmailHit
from app.services.ocr import OcrConfigError, OcrError, OcrExecutionError, OcrTimeoutError
from app.services.pipeline import PipelineResult, get_pipeline

client = TestClient(app)


class FakeEngine:
    name = "tesseract"
    model = "tesseract-test"

    @property
    def language(self) -> str:
        return "spa+eng"


class FakePipeline:
    ready = True

    def __init__(self) -> None:
        self.engine = FakeEngine()

    def process_bytes(self, data: bytes, include_text: bool = False) -> PipelineResult:
        return PipelineResult(
            emails=[EmailHit(email="ventas@acme.com", confidence=0.95, occurrences=2)],
            width=800,
            height=400,
            elapsed_ms=42.5,
            text="Ventas: ventas@acme.com" if include_text else "",
            engine="tesseract",
            model="tesseract-test",
            ocr_confidence=0.93,
        )


class NotReadyPipeline(FakePipeline):
    ready = False


class NoLanguageEngine(FakeEngine):
    @property
    def language(self) -> str:
        raise OcrConfigError("sin idioma")


class NoLanguagePipeline(FakePipeline):
    def __init__(self) -> None:
        self.engine = NoLanguageEngine()


class RaisingPipeline:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def process_bytes(self, data: bytes, include_text: bool = False) -> PipelineResult:
        raise self.exc


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def png_bytes(width: int = 40, height: int = 20) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def post_image(pipeline: object | None = None, **kwargs):
    if pipeline is not None:
        app.dependency_overrides[get_pipeline] = lambda: pipeline
    return client.post(
        "/v1/extract",
        files={"file": ("test.png", png_bytes(), "image/png")},
        **kwargs,
    )


def test_extract_ok() -> None:
    response = post_image(FakePipeline())
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["emails"][0] == {
        "email": "ventas@acme.com",
        "confidence": 0.95,
        "occurrences": 2,
    }
    assert body["text"] is None
    assert body["meta"]["engine"] == "tesseract"
    assert body["meta"]["model"] == "tesseract-test"
    assert body["meta"]["elapsed_ms"] == 42.5
    assert body["meta"]["ocr_confidence"] == 0.93
    assert "total_tokens" not in body["meta"]
    assert body["request_id"]
    assert response.headers["x-request-id"] == body["request_id"]


def test_extract_includes_text_when_requested() -> None:
    response = post_image(FakePipeline(), data={"include_text": "true"})
    assert response.status_code == 200
    assert response.json()["text"] == "Ventas: ventas@acme.com"


def test_extract_rejects_oversized_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 50)
    response = post_image(FakePipeline())
    assert response.status_code == 413


def test_extract_rejects_invalid_image() -> None:
    response = client.post(
        "/v1/extract",
        files={"file": ("fake.png", b"esto no es una imagen", "image/png")},
    )
    assert response.status_code == 400
    assert "imagen" in response.json()["detail"].lower()


def test_extract_requires_file() -> None:
    response = client.post("/v1/extract")
    assert response.status_code == 422


def test_extract_missing_tesseract_maps_to_503() -> None:
    response = post_image(RaisingPipeline(OcrConfigError("sin tesseract")))
    assert response.status_code == 503


def test_extract_timeout_maps_to_504() -> None:
    response = post_image(RaisingPipeline(OcrTimeoutError("timeout")))
    assert response.status_code == 504


def test_extract_ocr_failure_maps_to_502() -> None:
    response = post_image(RaisingPipeline(OcrExecutionError("fallo")))
    assert response.status_code == 502


def test_extract_generic_ocr_error_maps_to_502() -> None:
    response = post_image(RaisingPipeline(OcrError("error")))
    assert response.status_code == 502


def test_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr("app.main.get_pipeline", lambda: FakePipeline())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["engine"] == "tesseract"
    assert body["model"] == "tesseract-test"
    assert body["ocr_ready"] is True
    assert body["language"] == "spa+eng"
    assert body["auth_required"] is False


def test_health_reports_auth_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", ["secret"])
    monkeypatch.setattr("app.main.get_pipeline", lambda: FakePipeline())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["auth_required"] is True


def test_extract_requires_key_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", ["secret"])
    response = post_image(FakePipeline())
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["x-request-id"]


def test_extract_accepts_bearer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", ["secret"])
    response = post_image(FakePipeline(), headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200


def test_extract_accepts_x_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", ["secret"])
    response = post_image(FakePipeline(), headers={"X-API-Key": "secret"})
    assert response.status_code == 200


def test_extract_rejects_wrong_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", ["secret"])
    response = post_image(FakePipeline(), headers={"Authorization": "Bearer incorrecta"})
    assert response.status_code == 401


def test_health_reports_degraded_without_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr("app.main.get_pipeline", lambda: NotReadyPipeline())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ocr_ready"] is False


def test_health_reports_degraded_without_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr("app.main.get_pipeline", lambda: NoLanguagePipeline())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ocr_ready"] is True
    assert body["language"] == ""


def test_response_includes_security_headers() -> None:
    response = post_image(FakePipeline())
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-request-id"]


def test_invalid_request_id_is_replaced() -> None:
    response = post_image(FakePipeline(), headers={"X-Request-ID": "bad value!"})
    assert response.status_code == 200
    request_id = response.headers["x-request-id"]
    assert request_id != "bad value!"
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert response.json()["request_id"] == request_id


def test_unhandled_error_returns_json_500() -> None:
    response = post_image(RaisingPipeline(RuntimeError("boom")))
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Error interno del servidor."
    assert body["request_id"] == response.headers["x-request-id"]


def test_extract_rejects_oversized_request_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 1)
    response = client.post(
        "/v1/extract",
        files={"file": ("big.bin", b"x" * (256 * 1024 + 1024), "application/octet-stream")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "La solicitud es demasiado grande."
