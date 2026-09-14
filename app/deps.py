"""FastAPI dependencies shared across routers -- who's asking.

`get_current_user` is a `Depends()` target for any route that requires a
signed-in user; `get_optional_user` is the same check for a route (crop
diagnosis, image analysis) that works either way but attaches a diagnosis
record to the account when one is present. Both live here, once, rather than
in whichever router happened to need auth first.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from . import config
from .auth import decode_access_token
from .database import get_connection, row_to_dict
from .schemas import UserResponse

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_access_token(token, config.SECRET_KEY)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")

    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, email, name, created_at FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    user = row_to_dict(row)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated user no longer exists.")
    return user


def serialize_user(user: dict) -> UserResponse:
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user.get("name"),
        created_at=user["created_at"],
    )


def get_optional_user(authorization: str | None) -> dict | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    try:
        return get_current_user(token)
    except HTTPException:
        return None
