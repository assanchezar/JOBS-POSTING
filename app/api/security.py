from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException, Request

from app.core.config import settings

logger = logging.getLogger(__name__)


def _token_from_headers(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None


def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Protege la API con API_KEYS. Si la lista esta vacia, queda abierta (desarrollo)."""
    if not settings.api_keys:
        return

    token = _token_from_headers(authorization, x_api_key)
    matches = (
        [
            secrets.compare_digest(token.encode("utf-8"), key.encode("utf-8"))
            for key in settings.api_keys
        ]
        if token is not None
        else []
    )
    if any(matches):
        return

    logger.warning("auth_failed request_id=%s", getattr(request.state, "request_id", ""))
    raise HTTPException(
        status_code=401,
        detail="API key invalida o ausente.",
        headers={"WWW-Authenticate": "Bearer"},
    )
