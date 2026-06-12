"""Lightweight API-key auth.

A shared-secret check on the write/expensive endpoints (`/v1/query`, `/v1/ingest`)
so a public demo URL can't be used to spend our LLM credits. Enabled only when
`API_KEY` is configured; otherwise it's a no-op (convenient for local dev).
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from insightrag.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce the X-API-Key header when an API key is configured."""
    settings = get_settings()
    if settings.api_key is None:
        return  # auth disabled
    if not x_api_key or x_api_key != settings.api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
