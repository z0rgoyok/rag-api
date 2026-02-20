from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from .lmstudio import LmStudioClient

log = logging.getLogger("rag_models")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


@dataclass(frozen=True)
class ModelTarget:
    role: str
    backend: str
    base_url: str
    api_key: str | None
    model: str


def _is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").strip().lower()
    return host in _LOCAL_HOSTS


def _run_lms(args: list[str]) -> str:
    cmd = ["lms", *args]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        out = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{' '.join(cmd)} failed: {out[:1000]}")
    return (proc.stdout or "").strip()


def _lms_loaded_identifiers() -> set[str]:
    raw = _run_lms(["ps", "--json"])
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from `lms ps --json`: {raw[:300]}") from e
    ids: set[str] = set()
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            ident = item.get("identifier") or item.get("modelKey")
            if isinstance(ident, str) and ident.strip():
                ids.add(ident.strip())
    return ids


def _model_search_terms(model: str) -> list[str]:
    raw = (model or "").strip()
    if not raw:
        return []
    terms: list[str] = []
    for candidate in (raw, raw.split("/")[-1], raw.split("@")[0], raw.split("/")[-1].split("@")[0]):
        c = candidate.strip()
        if c and c not in terms:
            terms.append(c)
    return terms


def _ensure_lms_model_loaded(model: str) -> None:
    if shutil.which("lms") is None:
        raise RuntimeError("`lms` CLI is not installed")

    loaded = _lms_loaded_identifiers()
    if model in loaded:
        return

    try:
        _run_lms(["load", model, "--exact", "-y"])
    except Exception:
        pass

    if model in _lms_loaded_identifiers():
        return

    errors: list[str] = []

    # Path/identifier exact flow.
    try:
        _run_lms(["get", model, "-y"])
        _run_lms(["load", model, "--exact", "-y"])
    except Exception as e:
        errors.append(str(e))

    if model in _lms_loaded_identifiers():
        return

    # Fuzzy lookup flow: download by search term and load with explicit alias
    # so OpenAI-compatible API accepts the configured model id.
    for term in _model_search_terms(model):
        try:
            _run_lms(["get", term, "-y"])
            _run_lms(["load", term, "--identifier", model, "-y"])
        except Exception as e:
            errors.append(str(e))
            continue
        if model in _lms_loaded_identifiers():
            return

    joined = " | ".join(errors[-3:])
    raise RuntimeError(f"Model is still not loaded after auto-pull attempts: {model}. {joined[:1200]}")


def _build_targets(
    *,
    chat_backend: str,
    chat_base_url: str,
    chat_api_key: str | None,
    chat_model: str,
    embeddings_backend: str,
    embeddings_base_url: str,
    embeddings_api_key: str | None,
    embeddings_model: str,
    require_chat: bool,
    require_embeddings: bool,
) -> list[ModelTarget]:
    targets: list[ModelTarget] = []
    if require_chat:
        targets.append(
            ModelTarget(
                role="chat",
                backend=(chat_backend or "").strip().lower(),
                base_url=chat_base_url,
                api_key=chat_api_key,
                model=chat_model,
            )
        )
    if require_embeddings:
        targets.append(
            ModelTarget(
                role="embeddings",
                backend=(embeddings_backend or "").strip().lower(),
                base_url=embeddings_base_url,
                api_key=embeddings_api_key,
                model=embeddings_model,
            )
        )
    return targets


async def _fetch_model_ids(*, base_url: str, api_key: str | None) -> set[str]:
    client = LmStudioClient(base_url=base_url, api_key=api_key)
    payload = await client.models()
    data = payload.get("data")
    if not isinstance(data, list):
        return set()
    out: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        ident = item.get("id")
        if isinstance(ident, str) and ident.strip():
            out.add(ident.strip())
    return out


async def ensure_runtime_models(
    *,
    chat_backend: str,
    chat_base_url: str,
    chat_api_key: str | None,
    chat_model: str,
    embeddings_backend: str,
    embeddings_base_url: str,
    embeddings_api_key: str | None,
    embeddings_model: str,
    strict: bool,
    auto_pull: bool,
    require_chat: bool = True,
    require_embeddings: bool = True,
) -> None:
    targets = _build_targets(
        chat_backend=chat_backend,
        chat_base_url=chat_base_url,
        chat_api_key=chat_api_key,
        chat_model=chat_model,
        embeddings_backend=embeddings_backend,
        embeddings_base_url=embeddings_base_url,
        embeddings_api_key=embeddings_api_key,
        embeddings_model=embeddings_model,
        require_chat=require_chat,
        require_embeddings=require_embeddings,
    )

    openai_targets = [t for t in targets if t.backend == "openai_compat" and t.model.strip()]
    if not openai_targets:
        return

    grouped: dict[tuple[str, str | None], list[ModelTarget]] = {}
    for target in openai_targets:
        grouped.setdefault((target.base_url, target.api_key), []).append(target)

    for (base_url, api_key), group in grouped.items():
        required = {t.model for t in group}
        try:
            model_ids = await _fetch_model_ids(base_url=base_url, api_key=api_key)
        except httpx.HTTPError as e:
            if strict:
                raise RuntimeError(f"Model preflight failed for {base_url}: {e}") from e
            log.warning("model_preflight_skip base_url=%s reason=%s", base_url, type(e).__name__)
            continue

        missing = required - model_ids
        if missing and auto_pull:
            if not _is_local_base_url(base_url):
                if strict:
                    raise RuntimeError(
                        f"Cannot auto-pull models for non-local base URL {base_url}. Missing models: {sorted(missing)}"
                    )
                log.warning("model_autopull_skip base_url=%s missing=%s", base_url, sorted(missing))
            else:
                for model in sorted(missing):
                    try:
                        await asyncio.to_thread(_ensure_lms_model_loaded, model)
                    except Exception as e:
                        if strict:
                            raise RuntimeError(f"Failed to auto-pull model `{model}`: {e}") from e
                        log.warning("model_autopull_failed model=%s error=%s", model, type(e).__name__)
                model_ids = await _fetch_model_ids(base_url=base_url, api_key=api_key)
                missing = required - model_ids

        if missing:
            msg = f"Required models are not available at {base_url}: {sorted(missing)}"
            if strict:
                raise RuntimeError(msg)
            log.warning("model_missing_non_strict base_url=%s missing=%s", base_url, sorted(missing))
