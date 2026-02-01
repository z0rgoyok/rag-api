from __future__ import annotations

import asyncio
import logging
import os
import json
from typing import Any
import uuid

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import load_settings
from core.db import Db
from core.embeddings_client import build_embeddings_client
from core.schema import ensure_schema, get_schema_info

from .auth import Principal, auth_dependency
from .chat_client import build_chat_client
from .logging_config import configure_logging
from .models import AgentChatRequest, AgentChatResponse, ChatCompletionsRequest
from .retrieval import build_context, retrieve_top_k

from apps.agent.agent import Agent, AgentConfig


load_dotenv()
configure_logging()
settings = load_settings()
db = Db(settings.database_url)
chat_client = build_chat_client(settings)
embed_client = build_embeddings_client(settings)

log = logging.getLogger("rag_api")

app = FastAPI(title="rag-api", version="0.1.0")
auth_dep = auth_dependency(db, settings.allow_anonymous)

@app.middleware("http")
async def _request_logging(request: Request, call_next: Any) -> Any:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    started = asyncio.get_running_loop().time()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_id=%s http %s %s unhandled_error", request_id, request.method, request.url.path)
        raise
    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000.0
    response.headers["x-request-id"] = request_id
    log.info(
        "request_id=%s http %s %s status=%s dur_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        getattr(response, "status_code", "?"),
        elapsed_ms,
    )
    return response


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
    request: Request,
    principal: Principal = Depends(auth_dep),
) -> Any:
    request_id = getattr(getattr(request, "state", None), "request_id", None) or str(uuid.uuid4())
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
        log.info(
            "request_id=%s rag_retrieve k=%s hits=%s include_sources=%s",
            request_id,
            settings.top_k,
            len(segments),
            include_sources,
        )
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

    upstream_model = req.model or settings.chat_model
    # Keep OpenAI-compatible clients working even when using LiteLLM backends
    # that require provider-prefixed model strings (e.g. `vertex_ai/gemini-2.0-flash`).
    if settings.chat_backend == "litellm" and upstream_model and "/" not in upstream_model:
        upstream_model = settings.chat_model

    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(req.stream),
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    if (os.getenv("LOG_PROMPTS") or "").strip().lower() in {"1", "true", "yes", "on"}:
        def _truncate(s: str, n: int = 800) -> str:
            return s if len(s) <= n else s[: n - 3] + "..."

        safe_msgs: list[dict[str, Any]] = []
        for m in payload.get("messages", []):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, str):
                safe_msgs.append({"role": role, "content": _truncate(content)})
            else:
                safe_msgs.append({"role": role, "content": "<non-str>"})

        log.info(
            "request_id=%s upstream_chat model=%s stream=%s sources_count=%s",
            request_id,
            payload.get("model"),
            payload.get("stream"),
            len(sources) if include_sources else 0,
        )
        pretty_msgs = json.dumps(safe_msgs, ensure_ascii=False, indent=2)
        log.info("request_id=%s upstream_chat_messages\n%s", request_id, pretty_msgs)

    if not req.stream:
        # Non-streaming passthrough (still adds sources if allowed).
        try:
            data = await chat_client.chat_completions(payload)
        except Exception as e:
            # Never leak upstream secrets (e.g. API keys embedded in URLs).
            log.error("request_id=%s upstream_error=%s", request_id, type(e).__name__)
            return JSONResponse({"error": {"message": "Upstream chat provider failed"}}, status_code=502)
        if include_sources:
            data["sources"] = sources
        if (os.getenv("LOG_COMPLETIONS") or "").strip().lower() in {"1", "true", "yes", "on"}:
            def _pick_text(resp: Any) -> str:
                if not isinstance(resp, dict):
                    return ""
                choices = resp.get("choices")
                if not isinstance(choices, list) or not choices:
                    return ""
                c0 = choices[0] or {}
                msg = c0.get("message") if isinstance(c0, dict) else None
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
                # Some providers return "text" at choice-level.
                if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                    return c0["text"]
                return ""

            def _truncate(s: str, n: int = 4000) -> str:
                return s if len(s) <= n else s[: n - 3] + "..."

            assistant_text = _pick_text(data)
            log.info("request_id=%s completion model=%s", request_id, payload.get("model"))
            text = _truncate(assistant_text) if assistant_text else "<empty>"
            log.info("request_id=%s completion_text\n%s", request_id, text)
        return JSONResponse(data)

    async def _sse() -> Any:
        injected = False
        chunks = 0
        bytes_out = 0
        try:
            async for chunk in chat_client.stream_chat_completions(payload):
                chunks += 1
                bytes_out += len(chunk)
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
        except Exception as e:
            log.error("request_id=%s upstream_stream_error=%s", request_id, type(e).__name__)
            # Return a terminal chunk in SSE format.
            yield b"data: [DONE]\n\n"
        finally:
            if (os.getenv("LOG_COMPLETIONS") or "").strip().lower() in {"1", "true", "yes", "on"}:
                log.info(
                    "request_id=%s completion_stream model=%s chunks=%s bytes=%s",
                    request_id,
                    payload.get("model"),
                    chunks,
                    bytes_out,
                )

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/v1/agent/chat", response_model=AgentChatResponse)
async def agent_chat(
    req: AgentChatRequest,
    request: Request,
    principal: Principal = Depends(auth_dep),
) -> AgentChatResponse:
    """Agentic RAG endpoint - multi-step reasoning with iterative retrieval."""
    request_id = getattr(getattr(request, "state", None), "request_id", None) or str(uuid.uuid4())
    include_sources = bool(req.citations) and bool(principal.citations_enabled)

    log.info(
        "request_id=%s agent_chat query=%s max_iterations=%d include_sources=%s",
        request_id,
        req.query[:100],
        req.max_iterations,
        include_sources,
    )

    config = AgentConfig(
        max_iterations=req.max_iterations,
        top_k=settings.top_k,
        include_sources=include_sources,
    )

    agent = Agent(
        db=db,
        embed_client=embed_client,
        chat_client=chat_client,
        settings=settings,
        config=config,
    )

    try:
        result = await agent.run(req.query)
    except Exception as e:
        log.error("request_id=%s agent_error=%s", request_id, type(e).__name__)
        return JSONResponse(
            {"error": {"message": "Agent execution failed"}},
            status_code=502,
        )

    log.info(
        "request_id=%s agent_complete iterations=%d searches=%d answer_len=%d",
        request_id,
        result.iterations,
        result.search_count,
        len(result.answer),
    )

    return AgentChatResponse(
        answer=result.answer,
        sources=result.sources if include_sources else [],
        reasoning_steps=result.reasoning_steps,
        search_count=result.search_count,
        iterations=result.iterations,
    )
