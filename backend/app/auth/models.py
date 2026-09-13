"""In-memory user store + auth DTOs.

For a real system you'd back this with a database; here we hard-code the
five demo users the rubric requires. The password hashes are computed
once at import time — same as if they were seeded into a DB.
"""
from pydantic import BaseModel

from app.auth.security import hash_password
from app.core.rbac import Role


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    collections: list[str]


class AuthenticatedUser(BaseModel):
    """What get_current_user returns — used across routes."""
    username: str
    role: Role


class UserRecord(BaseModel):
    """Server-side user record. Never leaked to the client."""
    username: str
    hashed_password: str
    role: Role


# ── Demo users required by the rubric ────────────────────────────────────
# Passwords are hashed at import time. Do NOT store plaintext.
_RAW_USERS: list[tuple[str, str, Role]] = [
    ("dr.mehta",     "doctor",            Role.DOCTOR),
    ("nurse.priya",  "nurse",             Role.NURSE),
    ("billing.ravi", "billing_executive", Role.BILLING_EXECUTIVE),
    ("tech.anand",   "technician",        Role.TECHNICIAN),
    ("admin.sys",    "admin",             Role.ADMIN),
]

USERS: dict[str, UserRecord] = {
    username: UserRecord(
        username=username,
        hashed_password=hash_password(password),
        role=role,
    )
    for username, password, role in _RAW_USERS
}


def get_user(username: str) -> UserRecord | None:
    return USERS.get(username)