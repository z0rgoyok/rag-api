from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import psycopg


@dataclass(frozen=True)
class Db:
    url: str

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.url, autocommit=True)

    def connect_tx(self) -> psycopg.Connection:
        return psycopg.connect(self.url, autocommit=False)


def fetch_one(conn: psycopg.Connection, query: str, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(query, params or {})
        return cur.fetchone()


def fetch_all(conn: psycopg.Connection, query: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(query, params or {})
        return list(cur.fetchall())


def execute(conn: psycopg.Connection, query: str, params: Optional[dict[str, Any]] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(query, params or {})


def execute_many(conn: psycopg.Connection, query: str, param_rows: Iterable[dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(query, list(param_rows))
