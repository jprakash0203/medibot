"""Password hashing and JWT create/verify.

Uses bcrypt for passwords (same as Spring Security's BCryptPasswordEncoder)
and HS256 JWTs (symmetric signing — fine for a single-service app; use
RS256 if you split into multiple services).
"""
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Called by seed_users.py at startup — never called per request."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def create_access_token(*, subject: str, role: str) -> str:
    """Build a JWT with `sub` (username) and `role` claims.

    Role is embedded in the token so /chat can trust it without another
    lookup — this is the whole point of a self-contained token.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify signature + expiry, return the claims. Raises JWTError if invalid."""
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])