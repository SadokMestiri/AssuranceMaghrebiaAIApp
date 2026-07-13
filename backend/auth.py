"""
Local auth for the dashboard app — password hashing, JWT issuance/verification,
and the `get_current_user` / `require_role` FastAPI dependencies.

This is separate from the insurance data warehouse: it only touches the
`users` table (see sql/init_schema.sql).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET_KEY
from db import get_db

# role_key -> french display label. The dashboard gates features by role_key.
ROLES: dict[str, str] = {
    "ceo":       "Direction / CEO",
    "agent":     "Agent commercial",
    "sinistres": "Gestionnaire Sinistres",
    "analyst":   "Data Analyst / MLOps",
    "admin":     "Administrateur",
}

# Roles a user can grant themselves via POST /auth/register. "ceo" and
# "admin" are deliberately excluded — those get assigned by an existing
# admin (e.g. a direct DB update or a future admin-only promotion endpoint),
# never through open self-service signup.
SELF_REGISTER_ROLES: dict[str, str] = {
    key: label for key, label in ROLES.items() if key not in {"ceo", "admin"}
}

_bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(*, user_id: int, email: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentification requise.")

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session invalide ou expiree.")

    row = db.execute(
        text("SELECT id_user, email, nom, prenom, role, is_active FROM users WHERE id_user = :id"),
        {"id": int(payload["sub"])},
    ).mappings().first()

    if row is None or not row["is_active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Compte introuvable ou desactive.")

    return dict(row)


def require_role(*allowed_roles: str):
    """FastAPI dependency factory: require_role("ceo", "admin") gates a route to those roles."""

    def _check(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Acces refuse pour ce role.")
        return current_user

    return _check
