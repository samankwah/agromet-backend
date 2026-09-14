"""Register, log in, and read your own profile."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from .. import config
from ..auth import create_access_token, hash_password, verify_password
from ..database import get_connection, row_to_dict
from ..deps import get_current_user, serialize_user
from ..schemas import RegisterRequest, RegisterResponse, TokenResponse, UserResponse

router = APIRouter(prefix="/api/v1/auth", tags=["accounts"])


@router.post("/register", response_model=RegisterResponse)
def register(payload: RegisterRequest):
    with get_connection() as connection:
        existing = connection.execute("SELECT id FROM users WHERE email = ?", (payload.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")

        cursor = connection.execute(
            "INSERT INTO users(email, name, password_hash) VALUES (?, ?, ?)",
            (payload.email, payload.name, hash_password(payload.password)),
        )
        user_id = cursor.lastrowid
        row = connection.execute(
            "SELECT id, email, name, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    return RegisterResponse(user=serialize_user(row_to_dict(row)))


@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, email, name, password_hash, created_at FROM users WHERE email = ?",
            (form_data.username,),
        ).fetchone()

    user = row_to_dict(row)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")

    token = create_access_token(user["email"], config.SECRET_KEY, config.ACCESS_TOKEN_EXPIRE_MINUTES)
    return TokenResponse(
        access_token=token,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=serialize_user(user),
    )


@router.get("/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user)):
    return serialize_user(current_user)
