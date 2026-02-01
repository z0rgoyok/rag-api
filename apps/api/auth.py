from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, status

from core.db import Db, fetch_one


@dataclass(frozen=True)
class Principal:
    api_key: str
    tier: str
    citations_enabled: bool


def authenticate(db: Db, authorization: Optional[str], *, allow_anonymous: bool) -> Principal:
    if not authorization:
        if allow_anonymous:
            return Principal(api_key="anonymous", tier="anonymous", citations_enabled=False)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")

    with db.connect() as conn:
        row = fetch_one(
            conn,
            "select api_key, tier, citations_enabled from api_keys where api_key = %(k)s",
            {"k": token},
        )
        if not row:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        return Principal(api_key=row["api_key"], tier=row["tier"], citations_enabled=bool(row["citations_enabled"]))


def auth_dependency(db: Db, allow_anonymous: bool):
    def _dep(authorization: Optional[str] = Header(default=None)) -> Principal:
        return authenticate(db, authorization, allow_anonymous=allow_anonymous)

    return _dep

