from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Tuple, Optional
import time
import asyncio
from app.database import SessionLocal, RateLimitTracker, APIKey
import hashlib


class RateLimiter:
    def __init__(self):
        # In-memory storage for fast access (production would use Redis)
        self.requests: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = 300  # Clean old entries every 5 minutes
        self.last_cleanup = time.time()

    def _get_client_id(self, request: Request, api_key: Optional[str] = None) -> Tuple[str, int]:
        """Get client identifier and rate limit"""
        if api_key:
            # Use API key for authenticated requests
            return api_key, self._get_api_key_limit(api_key)
        else:
            # Use IP for anonymous requests
            client_ip = request.client.host
            return f"ip:{client_ip}", 10  # 10 requests/hour for anonymous

    def _get_api_key_limit(self, api_key: str) -> int:
        """Get rate limit for API key from database"""
        db = SessionLocal()
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            db_key = db.query(APIKey).filter(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True
            ).first()
            return db_key.rate_limit if db_key else 10
        finally:
            db.close()

    def _cleanup_old_requests(self):
        """Remove requests older than 1 hour"""
        if time.time() - self.last_cleanup > self.cleanup_interval:
            current_time = datetime.now()
            for client_id in list(self.requests.keys()):
                self.requests[client_id] = [
                    req_time for req_time in self.requests[client_id]
                    if current_time - req_time < timedelta(hours=1)
                ]
                if not self.requests[client_id]:
                    del self.requests[client_id]
            self.last_cleanup = time.time()

    async def check_rate_limit(self, request: Request, api_key: Optional[str] = None) -> bool:
        """Check if request is within rate limit"""
        self._cleanup_old_requests()

        client_id, limit = self._get_client_id(request, api_key)
        current_time = datetime.now()

        # Get requests in the last hour
        hour_ago = current_time - timedelta(hours=1)
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > hour_ago
        ]

        # Check limit
        if len(self.requests[client_id]) >= limit:
            # Log to database for analytics
            db = SessionLocal()
            try:
                tracker = RateLimitTracker(
                    key_hash=client_id if api_key else None,
                    ip_address=request.client.host,
                    endpoint=str(request.url.path),
                    timestamp=current_time
                )
                db.add(tracker)
                db.commit()
            finally:
                db.close()
            return False

        # Add current request
        self.requests[client_id].append(current_time)
        return True


rate_limiter = RateLimiter()