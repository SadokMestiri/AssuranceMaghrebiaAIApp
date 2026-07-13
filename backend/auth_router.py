from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from auth import (
    ROLES,
    SELF_REGISTER_ROLES,
    create_access_token,
    get_current_user,
    hash_password,
    require_role,
    verify_password,
)
from db import get_db

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=150, pattern=EMAIL_PATTERN)
    password: str = Field(min_length=8, max_length=128)
    nom: str = Field(min_length=1, max_length=100)
    prenom: str = Field(min_length=1, max_length=100)
    role: str = Field(default="agent")


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=150, pattern=EMAIL_PATTERN)
    password: str = Field(min_length=1, max_length=128)


class UpdateRoleRequest(BaseModel):
    role: str


class UpdateActiveRequest(BaseModel):
    is_active: bool


def _serialize_user(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id_user"],
        "email": row["email"],
        "nom": row["nom"],
        "prenom": row["prenom"],
        "role": row["role"],
        "role_label": ROLES.get(row["role"], row["role"]),
    }


@router.get("/roles")
def list_roles() -> dict[str, Any]:
    # Self-registration only — "ceo"/"admin" are excluded on purpose
    # (see auth.SELF_REGISTER_ROLES). Existing accounts with those roles
    # still work fine; this just controls what /register will accept.
    return {"roles": [{"key": key, "label": label} for key, label in SELF_REGISTER_ROLES.items()]}


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.role not in SELF_REGISTER_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Role invalide pour une inscription. Roles disponibles: {', '.join(SELF_REGISTER_ROLES)}",
        )

    existing = db.execute(
        text("SELECT id_user FROM users WHERE email = :email"), {"email": payload.email}
    ).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Un compte existe deja avec cet email.")

    row = db.execute(
        text("""
            INSERT INTO users (email, password_hash, nom, prenom, role)
            VALUES (:email, :password_hash, :nom, :prenom, :role)
            RETURNING id_user, email, nom, prenom, role, is_active
        """),
        {
            "email": payload.email,
            "password_hash": hash_password(payload.password),
            "nom": payload.nom,
            "prenom": payload.prenom,
            "role": payload.role,
        },
    ).mappings().first()
    db.commit()

    user = dict(row)
    token = create_access_token(user_id=user["id_user"], email=user["email"], role=user["role"])
    return {"token": token, "user": _serialize_user(user)}


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.execute(
        text("SELECT id_user, email, password_hash, nom, prenom, role, is_active FROM users WHERE email = :email"),
        {"email": payload.email},
    ).mappings().first()

    if row is None or not row["is_active"] or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")

    user = dict(row)
    token = create_access_token(user_id=user["id_user"], email=user["email"], role=user["role"])
    return {"token": token, "user": _serialize_user(user)}


@router.get("/me")
def me(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"user": _serialize_user(current_user)}


# ── Admin-only user management ──────────────────────────────────────────────
# Unlike /register (self-service, "ceo"/"admin" excluded), these accept the
# full ROLES set — this is the intended path to actually grant those roles.

@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    _admin: dict[str, Any] = Depends(require_role("admin")),
) -> dict[str, Any]:
    rows = db.execute(
        text("""
            SELECT id_user, email, nom, prenom, role, is_active, created_at
            FROM users ORDER BY created_at DESC
        """)
    ).mappings().all()
    return {
        "users": [
            {**_serialize_user(dict(row)), "is_active": row["is_active"], "created_at": str(row["created_at"])}
            for row in rows
        ]
    }


@router.patch("/users/{user_id}/role")
def update_user_role(
    user_id: int,
    payload: UpdateRoleRequest,
    db: Session = Depends(get_db),
    _admin: dict[str, Any] = Depends(require_role("admin")),
) -> dict[str, Any]:
    if payload.role not in ROLES:
        raise HTTPException(status_code=400, detail=f"Role invalide. Roles disponibles: {', '.join(ROLES)}")

    row = db.execute(
        text("""
            UPDATE users SET role = :role, updated_at = NOW()
            WHERE id_user = :id
            RETURNING id_user, email, nom, prenom, role, is_active
        """),
        {"role": payload.role, "id": user_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    db.commit()
    return {"user": _serialize_user(dict(row))}


@router.patch("/users/{user_id}/active")
def update_user_active(
    user_id: int,
    payload: UpdateActiveRequest,
    db: Session = Depends(get_db),
    admin: dict[str, Any] = Depends(require_role("admin")),
) -> dict[str, Any]:
    if user_id == admin["id_user"] and not payload.is_active:
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas desactiver votre propre compte.")

    row = db.execute(
        text("""
            UPDATE users SET is_active = :is_active, updated_at = NOW()
            WHERE id_user = :id
            RETURNING id_user, email, nom, prenom, role, is_active
        """),
        {"is_active": payload.is_active, "id": user_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    db.commit()
    return {"user": _serialize_user(dict(row))}
