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
- `apps/ingest/` — CLI to ingest PDFs into Postgres+pgvector.
- `infra/` — docker compose + DB init.
- `scripts/` — common operational workflows (compose up/down, ingest, etc).

## Docker compose workflow (recommended)

1) Put PDFs into `var/pdfs/`:

```bash
cp /path/to/*.pdf var/pdfs/
```

2) Start services (Postgres+pgvector + API):

```bash
./scripts/up.sh
```

3) Ingest PDFs into the vector index:

```bash
./scripts/ingest.sh
```

4) Create an API key (prints it to stdout):

```bash
./scripts/create-api-key.sh
```

5) Health check:

```bash
./scripts/health.sh
```

## Local (non-docker) workflow (optional)

1) Start Postgres+pgvector:

```bash
cd infra
docker compose up -d
```

2) Create Python venv + deps:

```bash
cd ..
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

3) Configure env:

```bash
cp .env.example .env
```

4) Drop PDFs into `var/pdfs/` and ingest:

```bash
rag-ingest ingest --pdf-dir var/pdfs
```

5) Run API:

```bash
uvicorn apps.api.main:app --reload --port 18080
```

## Configuration

- Inference provider (OpenAI-compatible): LM Studio *or* external providers (OpenAI, etc.)
  - Base URL: `INFERENCE_BASE_URL` (preferred) or legacy `LMSTUDIO_BASE_URL`
    - Local LM Studio (non-docker): `http://localhost:1234/v1`
    - Docker Desktop/compose -> host LM Studio: `http://host.docker.internal:1234/v1` (containers can’t reach your host via `localhost`)
  - API key (optional): `INFERENCE_API_KEY` (preferred) or legacy `LMSTUDIO_API_KEY`
  - Models: `INFERENCE_CHAT_MODEL`, `INFERENCE_EMBEDDING_MODEL` (or legacy `LMSTUDIO_*`)
  - Optional split: set `CHAT_*` and/or `EMBEDDINGS_*` to use different providers for chat vs embeddings
  - Embeddings adapter: `EMBEDDINGS_BACKEND` (`openai_compat` default, or `litellm`)
  - Vertex AI via LiteLLM:
    - Set `EMBEDDINGS_BACKEND=litellm`
    - Model example: `EMBEDDINGS_MODEL=vertex_ai/text-multilingual-embedding-002`
    - Required: `EMBEDDINGS_VERTEX_PROJECT`, `EMBEDDINGS_VERTEX_LOCATION`
    - Auth: Application Default Credentials (recommended) or `EMBEDDINGS_VERTEX_CREDENTIALS` path
- Ports:
  - API: `API_PORT` (default `18080`)
  - Postgres: `PG_PORT` (default `56473`)

## API

- `POST /v1/chat/completions` — OpenAI-compatible chat completions (supports `stream=true`).
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

## Notes

- The service **enforces** entitlements server-side. Client-provided `citations=true` is ignored unless the API key has `citations_enabled=true`.
- PDFs are mounted read-only into the API container at `/data/pdfs`. Ingestion reads from `/data/pdfs`.
