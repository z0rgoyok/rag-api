# AGENTS.md — rag-api

Scope: this repository (`~/dev/tools/rag-api`).

## Product constraints

- API-only. Do not add any GUI/web UI.
- Local-first. Prefer local OpenAI-compatible inference servers (e.g. LM Studio) for chat + embeddings.
- Streaming is required: keep `/v1/chat/completions` compatible with OpenAI SSE streaming.
- Citations/sources are a paid feature:
  - Always store source metadata for segments in the DB.
  - Never return source identifiers/paths/pages/timecodes unless the caller entitlement allows it.
  - Never put source identifiers into the LLM prompt unless the caller entitlement allows it.
- Future-proof ingestion: keep ingestion pipelines and retrieval strategies pluggable by `document_type` / `collection`.

## Architecture rules

- Keep a strict separation:
  - `apps/api/`: HTTP transport, auth, entitlements, request/response mapping, streaming proxy.
  - `apps/ingest/`: offline ingestion jobs/CLI.
  - `apps/api/schema.py`: DB schema management; avoid migrations for MVP, but keep schema changes centralized.
- Do not leak infrastructure details into strategy interfaces; prefer small, explicit contracts.
- No hidden globals for auth/entitlements: access checks must be explicit and testable.

## Security

- Never log secrets or credentials (Authorization headers, API keys).
- Treat all user input as untrusted; enforce size limits and timeouts on calls to inference servers.
- Enforce entitlements server-side; client-provided feature toggles are advisory only.

## Repo hygiene

- Keep generated/stateful files inside `var/` and `infra/pgdata/` only.
- Keep diffs minimal and focused.

