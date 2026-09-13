"""Auth router: /login endpoint + get_current_user dependency."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from app.auth.models import (
    AuthenticatedUser,
    LoginRequest,
    LoginResponse,
    get_user,
)
from app.auth.security import create_access_token, decode_access_token, verify_password
from app.core.rbac import Role, collections_for_role


router = APIRouter(prefix="", tags=["auth"])

# OAuth2PasswordBearer knows how to read `Authorization: Bearer <token>`.
# `tokenUrl` is what Swagger UI uses to power the Authorize button.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    """Verify credentials, return a role-tagged JWT."""
    user = get_user(payload.username)
    if not user or not verify_password(payload.password, user.hashed_password):
        # Deliberately vague error — do NOT reveal whether the username exists.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token(subject=user.username, role=user.role.value)
    return LoginResponse(
        access_token=token,
        role=user.role.value,
        collections=collections_for_role(user.role),
    )


def get_current_user(token: str = Depends(oauth2_scheme)) -> AuthenticatedUser:
    """Extract & verify the JWT, return the authenticated user.

    This is the FastAPI equivalent of Spring's SecurityContextHolder —
    add `user: AuthenticatedUser = Depends(get_current_user)` to any
    route parameter and the auth check runs before the handler.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        claims = decode_access_token(token)
    except JWTError:
        raise credentials_exc

    username = claims.get("sub")
    role_str = claims.get("role")
    if not username or not role_str:
        raise credentials_exc

    try:
        role = Role(role_str)
    except ValueError:
        # The token was signed by us but its role isn't in our enum anymore
        # — schema drift. Reject.
        raise credentials_exc

    # Defense in depth: confirm the user still exists.
    if not get_user(username):
        raise credentials_exc

    return AuthenticatedUser(username=username, role=role)