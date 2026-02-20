from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field


load_dotenv()


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str] = Field(default_factory=list)
    top_n: int | None = Field(default=None, ge=1)


class RerankResultOut(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankResultOut]


_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_DEFAULT_MODEL = (os.getenv("RERANK_HOST_DEFAULT_MODEL") or "BAAI/bge-reranker-v2-m3").strip()
_BATCH_SIZE = max(1, int(os.getenv("RERANK_HOST_BATCH_SIZE", "64")))
_API_KEY = (os.getenv("RERANK_HOST_API_KEY") or "").strip() or None


@lru_cache(maxsize=4)
def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=_resolve_device())


def _check_auth(authorization: str | None) -> None:
    if not _API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != _API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


app = FastAPI(title="rag-rerank-host", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": True, "device": _resolve_device()}


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest, authorization: str | None = Header(default=None)) -> RerankResponse:
    _check_auth(authorization)

    documents = req.documents or []
    if not documents:
        return RerankResponse(results=[])

    model_name = (req.model or _DEFAULT_MODEL).strip()
    query = req.query or ""
    top_n = req.top_n

    def _infer() -> list[RerankResultOut]:
        model = _load_model(model_name)
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs, batch_size=_BATCH_SIZE)

        results = [
            RerankResultOut(index=i, relevance_score=float(score))
            for i, score in enumerate(scores)
        ]
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        if top_n is not None:
            return results[:top_n]
        return results

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(_EXECUTOR, _infer)
    return RerankResponse(results=results)
