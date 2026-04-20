import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import API_KEY, API_KEY_HEADER

logger = logging.getLogger(__name__)


def require_api_key(
    api_key_header: Optional[str] = Header(None, alias=API_KEY_HEADER),
) -> None:
    """FastAPI dependency. Reject requests whose X-API-Key does not match env."""
    if API_KEY is None:
        logger.warning("API_KEY env var is unset; rejecting protected request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )

    if not api_key_header or api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
