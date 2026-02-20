from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


@dataclass
class Db:
    url: str
    _engine: Engine = field(init=False, repr=False)
    _session_factory: sessionmaker[Session] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        effective_url = self.url
        if effective_url.startswith("postgresql://"):
            effective_url = effective_url.replace("postgresql://", "postgresql+psycopg://", 1)
        self._engine = create_engine(effective_url, future=True, pool_pre_ping=True)
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )

    @property
    def engine(self) -> Engine:
        return self._engine

    def session(self) -> Session:
        return self._session_factory()
