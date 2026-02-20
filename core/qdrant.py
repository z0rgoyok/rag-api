from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient


@dataclass(frozen=True)
class Qdrant:
    url: str
    api_key: str | None
    collection: str

    def connect(self) -> QdrantClient:
        return QdrantClient(url=self.url, api_key=self.api_key, timeout=10.0)
