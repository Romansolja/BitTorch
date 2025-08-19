from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from app.auth import auth_service
from app.middleware import rate_limiter

security = HTTPBearer(auto_error=False)


async def get_api_key(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """Extract API key from request"""
    if credentials:
        return credentials.credentials

    # Check header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key

    # Check query parameter (not recommended for production)
    api_key = request.query_params.get("api_key")
    return api_key


async def require_api_key(
        request: Request,
        api_key: Optional[str] = Depends(get_api_key)
) -> str:
    """Require valid API key"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_obj = auth_service.verify_api_key(api_key)
    if not key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check rate limit
    if not await rate_limiter.check_rate_limit(request, api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {key_obj.rate_limit} requests per hour",
            headers={"Retry-After": "3600"}
        )

    return api_key


async def require_write_permission(api_key: str = Depends(require_api_key)):
    """Require write permission"""
    key_obj = auth_service.verify_api_key(api_key)
    if key_obj.permissions not in ["write", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write permission required"
        )
    return api_key


async def require_admin(api_key: str = Depends(require_api_key)):
    """Require admin permission"""
    key_obj = auth_service.verify_api_key(api_key)
    if key_obj.permissions != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    return api_key