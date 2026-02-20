from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, status
from sqlalchemy import select

from core.db import Db
from core.db_models import ApiKey


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

    with db.session() as session:
        row = session.execute(select(ApiKey).where(ApiKey.api_key == token)).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        return Principal(api_key=row.api_key, tier=row.tier, citations_enabled=bool(row.citations_enabled))


def auth_dependency(db: Db, allow_anonymous: bool):
    def _dep(authorization: Optional[str] = Header(default=None)) -> Principal:
        return authenticate(db, authorization, allow_anonymous=allow_anonymous)

    return _dep
