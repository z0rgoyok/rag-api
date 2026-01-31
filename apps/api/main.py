from __future__ import annotations

import asyncio
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import Principal, auth_dependency
from .config import load_settings
from .db import Db
from .embeddings_client import build_embeddings_client
from .lmstudio import LmStudioClient
from .retrieval import build_context, retrieve_top_k
from .schema import ensure_schema, get_schema_info
from .models import ChatCompletionsRequest


load_dotenv()
settings = load_settings()
db = Db(settings.database_url)
chat_client = LmStudioClient(settings.chat_base_url, api_key=settings.chat_api_key)
embed_client = build_embeddings_client(settings)

app = FastAPI(title="rag-api", version="0.1.0")
auth_dep = auth_dependency(db, settings.allow_anonymous)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.on_event("startup")
async def _startup() -> None:
    info = get_schema_info(db)
    if info is None:
        if settings.embedding_dim is not None:
            ensure_schema(db, embedding_dim=settings.embedding_dim)
            return
        # Best effort: if LM Studio isn't up yet, keep API running; schema can be
        # initialized by running ingestion once (it will probe dim).
        try:
            dim = await embed_client.probe_embedding_dim(model=settings.embeddings_model)
        except Exception:
            return
        ensure_schema(db, embedding_dim=dim)


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionsRequest,
    principal: Principal = Depends(auth_dep),
) -> Any:
    # Server-enforced paid feature:
    include_sources = bool(req.citations) and bool(principal.citations_enabled)

    user_text = ""
    for m in reversed(req.messages):
        if m.role == "user":
            user_text = m.content
            break

    context_text = ""
    sources: list[dict[str, Any]] = []
    if req.rag and user_text.strip():
        qvec = (await embed_client.embeddings(model=settings.embeddings_model, input_texts=[user_text], input_type="RETRIEVAL_QUERY"))[0]
        segments = retrieve_top_k(db, query_embedding=qvec, k=settings.top_k)
        context_text, sources = build_context(segments, max_chars=settings.max_context_chars, include_sources=include_sources)

    messages = [m.model_dump() for m in req.messages]
    if context_text:
        messages = [
            {
                "role": "system",
                "content": "You answer using the provided CONTEXT. If the user asks about the corpus, use it. If context is insufficient, say so.\n\nCONTEXT:\n"
                + context_text,
            },
            *messages,
        ]

    payload: dict[str, Any] = {
        "model": req.model or settings.chat_model,
        "messages": messages,
        "stream": bool(req.stream),
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    if not req.stream:
        # Non-streaming passthrough (still adds sources if allowed).
        data = await chat_client.chat_completions(payload)
        if include_sources:
            data["sources"] = sources
        return JSONResponse(data)

    async def _sse() -> Any:
        injected = False
        async for chunk in chat_client.stream_chat_completions(payload):
            if include_sources and (not injected) and b"data: [DONE]" in chunk:
                before, after = chunk.split(b"data: [DONE]", 1)
                if before:
                    yield before
                import json

                meta = {"sources": sources}
                yield f"data: {json.dumps(meta)}\n\n".encode("utf-8")
                yield b"data: [DONE]"
                if after:
                    yield after
                injected = True
                continue
            yield chunk

    return StreamingResponse(_sse(), media_type="text/event-stream")
