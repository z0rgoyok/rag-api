# rag-api (local, streaming, API-only)

API-only RAG backend for a private document corpus.

Goals:
- Local-first: calls a local OpenAI-compatible inference server (e.g. LM Studio) for chat + embeddings.
- Streaming: exposes an OpenAI-compatible `/v1/chat/completions` with SSE streaming.
- Future-proof: ingestion pipelines and retrieval strategies are pluggable (PDF now, audio/video later).
- Source citations are a paid feature (server-enforced).

## Layout

- `var/pdfs/` — drop PDFs here (your corpus source-of-truth; can contain subfolders).
- `apps/api/` — FastAPI service (auth, entitlements, retrieval, streaming proxy to LM Studio).
- `apps/agent/` — Agentic RAG module (ReAct pattern, iterative retrieval).
- `apps/ingest/` — CLI to ingest PDFs or pre-extracted chunks into Qdrant (vector index) with PostgreSQL metadata/state.
- `infra/` — docker compose + DB init.
- `scripts/` — common operational workflows (compose up/down, ingest, etc).

## Docker compose workflow (recommended)

1. Put PDFs into `var/pdfs/`:

```bash
cp /path/to/*.pdf var/pdfs/
```

1. Start services (PostgreSQL + Qdrant + API + NextChat UI):

```bash
./scripts/up.sh
```

1. Run ingest:

```bash
./scripts/ingest.sh
```

`scripts/ingest.sh` mode behavior:
- `INGEST_MODE=extract` -> host/local run (`pdf_extract` in CLI).
- `INGEST_MODE=ingest` -> host/local run (`chunks_full` in CLI).
- `INGEST_MODE=resume` -> host/local resume.

Default mode is `ingest` (fail-fast `--on-error fail`).
Index from already extracted JSONL chunks:

```bash
INGEST_MODE=ingest ./scripts/ingest.sh
```

Extract-only mode (host/local, no embeddings):

```bash
INGEST_MODE=extract ./scripts/ingest.sh
```

Resume an interrupted task:

```bash
INGEST_MODE=resume INGEST_TASK_ID=<task_uuid> INGEST_ON_ERROR=skip ./scripts/ingest.sh
```

Skip broken files and continue:

```bash
INGEST_MODE=ingest INGEST_ON_ERROR=skip ./scripts/ingest.sh
```

`scripts/ingest.sh` env options:
- `INGEST_MODE=extract|ingest|resume` (default `ingest`)
- `INGEST_ON_ERROR=fail|skip` (default `fail`)
- `INGEST_TASK_ID=<task_uuid>` (required for `resume`)
- `INGEST_PDF_DIR=var/pdfs` (used by `INGEST_MODE=extract`)
- `INGEST_CHUNKS_DIR=var/extracted` (used by `INGEST_MODE=ingest`)
- `INGEST_EXTRACT_OUTPUT_DIR=var/extracted` (used by `INGEST_MODE=extract`)
- `PYTHON_BIN=.venv/bin/python` (local Python interpreter)
- `INGEST_FORCE=1` (adds `--force`)

1. Create an API key (prints it to stdout):

```bash
./scripts/create-api-key.sh
```

1. Health check:

```bash
./scripts/health.sh
```

NextChat UI is exposed on `http://localhost:${NEXTCHAT_PORT:-3000}`.
It is preconfigured to call the local API container via `NEXTCHAT_BASE_URL=http://api:8080` (without `/v1`).
Realtime Chat toggle default can be controlled with `NEXTCHAT_ENABLE_REALTIME_DEFAULT` (`1` by default).

When `ALLOW_ANONYMOUS=false`, set `NEXTCHAT_OPENAI_API_KEY` in `.env` to a valid rag-api key
(create one with `./scripts/create-api-key.sh`).

## Local (non-docker) workflow (optional)

1. Start PostgreSQL + Qdrant:

```bash
cd infra
docker compose up -d
```

1. Create Python venv + deps:

```bash
cd ..
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

1. Configure env:

```bash
cp .env.example .env
```

1. Drop PDFs into `var/pdfs/` and ingest:

```bash
rag-ingest ingest --mode pdf_full --pdf-dir var/pdfs --on-error fail
```

If run is interrupted or fails, resume it by task id:

```bash
rag-ingest ingest --mode resume --task-id <task_uuid> --on-error skip
```

Extract-only to JSONL chunks:

```bash
rag-ingest ingest --mode pdf_extract --pdf-dir var/pdfs --extract-output-dir var/extracted --on-error skip
```

Index from pre-extracted chunks:

```bash
rag-ingest ingest --mode chunks_full --chunks-dir var/extracted --on-error skip
```

1. Run API:

```bash
uvicorn apps.api.main:app --reload --port 18080
```

## Host reranker service (Docker API + host inference)

Use this when API runs in Docker, but reranking should run on host (e.g. Apple Silicon MPS):

1. Start host reranker:

```bash
./scripts/rerank-host.sh
```

1. Set environment:

```bash
RERANKING_STRATEGY=http
RERANKING_BASE_URL=http://host.docker.internal:18123
RERANKING_MODEL=BAAI/bge-reranker-v2-m3
```

Optional auth:
- Set `RERANK_HOST_API_KEY` for host service.
- Set `RERANKING_API_KEY` with the same value for API container.

Notes:
- In this mode, API container does not need `sentence-transformers`/`torch`.
- Heavy rerank model + inference stay on host process.

## Configuration

- Inference provider (OpenAI-compatible): LM Studio *or* external providers (OpenAI, etc.)
  - Base URL: `INFERENCE_BASE_URL` (preferred) or legacy `LMSTUDIO_BASE_URL`
    - Local LM Studio (non-docker): `http://localhost:1234/v1`
    - Docker Desktop/compose -> host LM Studio: `http://host.docker.internal:1234/v1` (containers can’t reach your host via `localhost`)
  - API key (optional): `INFERENCE_API_KEY` (preferred) or legacy `LMSTUDIO_API_KEY`
  - Models: `INFERENCE_CHAT_MODEL`, `INFERENCE_EMBEDDING_MODEL` (or legacy `LMSTUDIO_*`)
  - Optional split: set `CHAT_*` and/or `EMBEDDINGS_*` to use different providers for chat vs embeddings
  - Chat adapter: `CHAT_BACKEND` (`openai_compat` default, or `litellm`)
    - Gemini via LiteLLM example:
      - `CHAT_BACKEND=litellm`
      - `CHAT_MODEL=gemini/<model-name>`
      - `CHAT_API_KEY=...` (Gemini API key)
    - Vertex AI chat via LiteLLM example:
      - `CHAT_BACKEND=litellm`
      - `CHAT_MODEL=vertex_ai/<model-name>` (e.g. `vertex_ai/gemini-2.0-flash`)
      - Required: `CHAT_VERTEX_PROJECT`, `CHAT_VERTEX_LOCATION`
      - Auth: Application Default Credentials (recommended) or `CHAT_VERTEX_CREDENTIALS` path
  - Embeddings adapter: `EMBEDDINGS_BACKEND` (`openai_compat` default, or `litellm`)
  - Vertex AI via LiteLLM:
    - Set `EMBEDDINGS_BACKEND=litellm`
    - Model example: `EMBEDDINGS_MODEL=vertex_ai/text-multilingual-embedding-002`
    - Required: `EMBEDDINGS_VERTEX_PROJECT`, `EMBEDDINGS_VERTEX_LOCATION`
    - Auth: Application Default Credentials (recommended) or `EMBEDDINGS_VERTEX_CREDENTIALS` path
- Storage:
  - `DATABASE_URL` configures PostgreSQL for metadata (`api_keys`, `rag_meta`, ingest task state).
  - `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` configure vector storage and retrieval.
- Ports:
  - API: `API_PORT` (default `18080`)
  - Postgres (metadata/state): `PG_PORT` (default `56473`)
  - Qdrant (vector index): `QDRANT_PORT` (default `6333`)
- Retrieval:
  - `TOP_K` controls how many chunks are returned to context.
  - `RETRIEVAL_USE_FTS=1|0` toggles hybrid ranking (vector similarity + lexical score on retrieved candidates).
  - With `RETRIEVAL_USE_FTS=0`, returned `score` is pure vector similarity from Qdrant.
  - Reranking:
    - `RERANKING_STRATEGY=none|lmstudio|cross_encoder|cohere|http`
    - `RERANKING_RETRIEVAL_K` controls how many candidates are pulled before rerank.
    - `RERANKING_MODEL` sets reranker model:
      - `cross_encoder`: HF model id, e.g. `BAAI/bge-reranker-v2-m3`.
      - `lmstudio`: model id exposed by LM Studio `/v1/models`, e.g. `text-embedding-bge-reranker-v2-m3`.
      - `http`: model id forwarded to host reranker service.
    - `RERANKING_BASE_URL` / `RERANKING_API_KEY` optionally override provider endpoint/auth.
      - `lmstudio`: LM Studio base URL and optional API key.
      - `http`: host reranker base URL and optional bearer token.
    - `RERANKING_BATCH_SIZE` controls reranker batch size.
- Ingest chunking strategy:
  - `CHUNKING_STRATEGY=recursive|sliding|semantic|docling_hierarchical|docling_hybrid`
  - `semantic` is the recommended default for PDF books.
  - `docling_*` strategies are for host/local PDF extraction flow.
- Ingest reliability:
  - Ingest is task-based (`ingest_tasks` + `ingest_task_items` in DB).
  - Interrupted runs are resumable via `--task-id`.
  - `--on-error fail|skip` controls per-file failure behavior for each start/resume.
  - Schema metadata validates both embedding dimension and `EMBEDDINGS_MODEL` to avoid mixed vector spaces.
  - `INGEST_EMBED_BATCH_SIZE` controls embeddings request batch size during ingest (useful for large chunk files / provider timeouts).
  - CLI run mode is explicit via `--mode`:
    - `pdf_full` = PDF -> chunks -> embeddings -> Qdrant
    - `pdf_extract` = PDF -> `*.chunks.jsonl` only
    - `chunks_full` = `*.chunks.jsonl` -> embeddings -> Qdrant
    - `resume` = continue existing task by `--task-id`
  - Wrapper script mode (`INGEST_MODE`) is environment-specific:
    - `extract` = host/local extract
    - `ingest` = host/local chunks ingest
    - `resume` = host/local resume
- Chunk sanitization:
  - Base text cleanup runs before chunking in extraction.
  - Chunk sanitizer runs after chunking (before embeddings/Qdrant upsert): `CHUNK_SANITIZE_ENABLED`, `CHUNK_SANITIZE_MIN_WORDS`, `CHUNK_SANITIZE_DEDUP`.

Note: if you run Docker Compose manually with `-f infra/compose.yml`, pass the env file explicitly (our `scripts/*.sh` already do this):
`docker compose --env-file .env -f infra/compose.yml up -d`

## API

- `POST /v1/chat/completions` — OpenAI-compatible chat completions (supports `stream=true`).
- `POST /v1/agent/chat` — Agentic RAG with iterative retrieval (see below).
- `GET /healthz` — health check.

Auth:
- `Authorization: Bearer <api_key>`

### Example: non-streaming

```bash
curl -sS \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","rag":true,"citations":false,"messages":[{"role":"user","content":"Summarize what the corpus says about X"}]}' \
  http://localhost:18080/v1/chat/completions
```

### Example: streaming (SSE)

```bash
curl -N \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","stream":true,"rag":true,"citations":false,"messages":[{"role":"user","content":"What does the corpus say about X?"}]}' \
  http://localhost:18080/v1/chat/completions
```

### Agentic RAG

For complex questions requiring multi-step reasoning or synthesis from multiple sources, use the agentic endpoint:

```bash
curl -sS \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare approaches to error handling across the corpus","max_iterations":3,"citations":true}' \
  http://localhost:18080/v1/agent/chat
```

Response:
```json
{
  "answer": "Based on the documents...",
  "sources": [{"title": "...", "path": "...", "page": 42, "score": 0.89}],
  "reasoning_steps": ["Calling search: {...}", "Refining search..."],
  "search_count": 2,
  "iterations": 2
}
```

Parameters:
- `query` (`string`, required): User's question
- `max_iterations` (`int`, default `3`): Max agent iterations (1-10)
- `citations` (`bool`, default `false`): Include sources (if entitled)

When to use agentic vs standard RAG:
- **Standard RAG** (`/v1/chat/completions`): Simple factual questions, low latency required
- **Agentic RAG** (`/v1/agent/chat`): Comparative analysis, synthesis from multiple sources, ambiguous queries

See `apps/agent/README.md` for detailed documentation.

## Notes

- The service **enforces** entitlements server-side. Client-provided `citations=true` is ignored unless the API key has `citations_enabled=true`.
- Docker API image intentionally excludes host-only ingestion/rerank dependencies (`docling`, `semchunk`, `chonkie`, `sentence-transformers`/`torch`).
- Extract flow is host/local only (`INGEST_MODE=extract`).
- Ingest-from-chunks flow is host/local (`INGEST_MODE=ingest`).
- PDF text extraction:
  - Applies to host/local extract flow (`INGEST_MODE=extract`).
  - `PDF_TEXT_EXTRACTOR=docling` (default)
  - `DOCLING_DO_OCR=1|0` (default `1`, hybrid OCR: page-level auto-detection)
  - `DOCLING_DO_TABLE_STRUCTURE=1|0` (default `0`)
  - `DOCLING_FORCE_FULL_PAGE_OCR=1|0` (default `0`)
  - `DOCLING_FORCE_BACKEND_TEXT=1|0` (default `0`, no force; let docling pick text/OCR path per page)
  - `DOCLING_INCLUDE_PICTURES=1|0` (default `0`, text-only markdown export without picture placeholders)
  - `DOCLING_DO_PICTURE_CLASSIFICATION=1|0` (default `0`)
  - `DOCLING_DO_PICTURE_DESCRIPTION=1|0` (default `0`)
  - `DOCLING_OCR_AUTO=1|0` (default `1`, auto-disable OCR for PDFs with strong text layer)
  - `DOCLING_OCR_AUTO_TEXT_LAYER_THRESHOLD=0..1` (default `0.9`)
  - `DOCLING_OCR_AUTO_MIN_CHARS` (default `20`, page is considered text-layer if extracted chars >= this value)
  - `DOCLING_OCR_AUTO_SAMPLE_PAGES` (default `0`, check all pages; use `N` to sample `N` evenly spaced pages)
  - To debug what gets ingested, set `PDF_DUMP_MD=1` and re-ingest; the extracted markdown-ish text is written under `var/extracted/*.md` (override with `PDF_DUMP_DIR`).

## Logging

- `LOG_LEVEL` (default `INFO`) — set to `DEBUG` to see more.
- `LOG_FORMAT` (default `pretty`) — set to `json` for JSON lines.
- `LOG_PROMPTS=1` — logs the upstream prompt (system+context+messages), truncated.
- `LOG_COMPLETIONS=1` — logs the upstream completion text (non-streaming), truncated.

Notes:
- The API never logs credentials (API keys / Authorization headers).
