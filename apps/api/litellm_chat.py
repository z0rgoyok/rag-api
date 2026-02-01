from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, AsyncIterator
import re


_RE_QUERY_KEY = re.compile(r"([?&]key=)([^&\\s]+)")


def _sanitize_error_message(s: str) -> str:
    # Gemini API keys often appear as `?key=...` in upstream URLs. Never leak them.
    return _RE_QUERY_KEY.sub(r"\\1REDACTED", s)


def _to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()  # type: ignore[no-any-return]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()  # type: ignore[no-any-return]
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        return {"value": obj}


class LiteLLMChatClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        vertex_credentials: str | None = None,
    ) -> None:
        self._api_key = api_key or None
        self._vertex_project = vertex_project or None
        self._vertex_location = vertex_location or None
        self._vertex_credentials = vertex_credentials or None

    def _litellm_kwargs_for_model(self, model: str) -> dict[str, Any]:
        # LiteLLM uses special kwargs for Vertex AI routing; keep them explicit and testable.
        if model.startswith("vertex_ai/"):
            if not self._vertex_project or not self._vertex_location:
                raise RuntimeError(
                    "Vertex chat model selected but CHAT_VERTEX_PROJECT and CHAT_VERTEX_LOCATION are not set"
                )
            kwargs: dict[str, Any] = {"vertex_project": self._vertex_project, "vertex_location": self._vertex_location}
            if self._vertex_credentials:
                kwargs["vertex_credentials"] = self._vertex_credentials
            return kwargs
        return {}

    async def chat_completions(self, payload: dict[str, Any], *, timeout_s: float = 120.0) -> dict[str, Any]:
        import litellm  # type: ignore[import-not-found]

        model = payload.get("model")
        messages = payload.get("messages")
        if not model or not isinstance(messages, list):
            raise ValueError("Invalid chat payload: expected 'model' and 'messages'")

        params = {k: v for k, v in payload.items() if k not in {"model", "messages"}}
        params.setdefault("stream", False)
        params.update(self._litellm_kwargs_for_model(model))

        acompletion = getattr(litellm, "acompletion", None)
        if callable(acompletion):
            try:
                resp = await asyncio.wait_for(
                    acompletion(model=model, messages=messages, api_key=self._api_key, timeout=timeout_s, **params),
                    timeout=timeout_s,
                )
                return _to_dict(resp)
            except Exception as e:
                raise RuntimeError(_sanitize_error_message(str(e))) from None

        def _sync() -> dict[str, Any]:
            resp = litellm.completion(model=model, messages=messages, api_key=self._api_key, timeout=timeout_s, **params)
            return _to_dict(resp)

        try:
            return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=timeout_s)
        except Exception as e:
            raise RuntimeError(_sanitize_error_message(str(e))) from None

    async def stream_chat_completions(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        import litellm  # type: ignore[import-not-found]

        model = payload.get("model")
        messages = payload.get("messages")
        if not model or not isinstance(messages, list):
            raise ValueError("Invalid chat payload: expected 'model' and 'messages'")

        params = {k: v for k, v in payload.items() if k not in {"model", "messages"}}
        params["stream"] = True
        params.update(self._litellm_kwargs_for_model(model))

        # LiteLLM streaming yields OpenAI-shaped chunks; we encode them as SSE lines.
        # Bridge the sync generator into an async stream without buffering everything.
        loop = asyncio.get_running_loop()
        done = object()
        q: asyncio.Queue[Any] = asyncio.Queue(maxsize=100)

        def _worker() -> None:
            try:
                for chunk in litellm.completion(model=model, messages=messages, api_key=self._api_key, **params):
                    asyncio.run_coroutine_threadsafe(q.put(_to_dict(chunk)), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(q.put(RuntimeError(_sanitize_error_message(str(e)))), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(done), loop).result()

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            item = await q.get()
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            data = json.dumps(item, ensure_ascii=False)
            yield f"data: {data}\n\n".encode("utf-8")

        yield b"data: [DONE]\n\n"
