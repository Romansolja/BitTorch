from fastapi import HTTPException, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import hashlib
from sqlalchemy.orm import Session
from app.database import SessionLocal, APIKey

# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
API_KEY_LENGTH = 32

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class AuthService:
    def create_api_key(self, name: str, permissions: str = "read", rate_limit: int = 100) -> str:
        """Create a new API key"""
        # Generate random key
        api_key = secrets.token_urlsafe(API_KEY_LENGTH)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Save to database
        db = SessionLocal()
        try:
            db_key = APIKey(
                key_hash=key_hash,
                name=name,
                permissions=permissions,
                rate_limit=rate_limit
            )
            db.add(db_key)
            db.commit()
        finally:
            db.close()

        return api_key

    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify API key and return key object"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        db = SessionLocal()
        try:
            db_key = db.query(APIKey).filter(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True
            ).first()

            if db_key:
                # Update last used
                db_key.last_used = datetime.utcnow()
                db.commit()

            return db_key
        finally:
            db.close()

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None


auth_service = AuthService()