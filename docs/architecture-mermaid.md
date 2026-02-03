# RAG API Architecture Manual

> Production-grade RAG system architecture guide.
> Covers current state, target state, and migration path.

---

## 1. Current Architecture (As-Is)

```mermaid
graph TB
    subgraph "APPS LAYER"
        API["apps/api/<br/>main.py<br/>retrieval.py<br/>auth.py<br/>models.py"]
        AGENT["apps/agent/<br/>agent.py<br/>tools.py<br/>protocol.py"]
        INGEST["apps/ingest/<br/>cli.py<br/>store.py<br/>pdf_extract.py"]
    end

    subgraph "CORE LAYER"
        CONFIG["config.py"]
        DB["db.py"]
        SCHEMA["schema.py"]
        LMSTUDIO["lmstudio.py<br/>embeddings_client.py"]
        CHUNKING["chunking/<br/>sliding, recursive, semantic"]
        RERANKING["reranking/<br/>none, cross_encoder, cohere<br/>⚠️ NOT WIRED"]
    end

    subgraph "INFRASTRUCTURE"
        PG["PostgreSQL + pgvector<br/>documents, segments<br/>segment_embeddings<br/>⚠️ tsvector NOT USED"]
        LLM["LM Studio / LiteLLM<br/>/v1/embeddings<br/>/v1/chat/completions"]
    end

    API --> CONFIG
    API --> DB
    API --> LMSTUDIO
    AGENT --> API
    INGEST --> CONFIG
    INGEST --> DB
    INGEST --> CHUNKING

    DB --> PG
    LMSTUDIO --> LLM

    style RERANKING fill:#fff3cd
    style PG fill:#fff3cd
```

### Current Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Business logic in infrastructure | `retrieval.py`, `store.py` | Hard to test, change |
| Reranker not wired | `core/reranking/` | Missing 10-15% quality |
| tsvector index unused | DB schema | Missing hybrid search |
| No caching | everywhere | Redundant embeddings |
| No domain layer | project-wide | Rules scattered |

---

## 2. Current Retrieval Flow

```mermaid
flowchart TD
    A["User Query<br/>'What is X?'"] --> B["Embed Query"]
    B --> C["Vector Search Only<br/>SELECT ... ORDER BY<br/>embedding <=> query_vec<br/>LIMIT k"]
    C --> D["Build Context<br/>segments → text + sources<br/>truncate to max_chars"]
    D --> E["LLM Call<br/>system + context + user<br/>→ answer"]

    C -.- C1["⚠️ No BM25/FTS"]
    C -.- C2["⚠️ No hybrid fusion"]
    C -.- C3["⚠️ No reranking"]

    D -.- D1["⚠️ Simple concatenation"]
    D -.- D2["⚠️ No deduplication"]

    style C1 fill:#fff3cd,stroke:#ffc107
    style C2 fill:#fff3cd,stroke:#ffc107
    style C3 fill:#fff3cd,stroke:#ffc107
    style D1 fill:#fff3cd,stroke:#ffc107
    style D2 fill:#fff3cd,stroke:#ffc107
```

---

## 3. Target Architecture (To-Be)

```mermaid
graph TB
    subgraph "PRESENTATION"
        API2["apps/api/<br/>HTTP handlers<br/>Request/Response DTOs<br/>Auth middleware"]
        AGENT2["apps/agent/<br/>Agent loop<br/>Tool execution"]
        INGEST2["apps/ingest/<br/>CLI entry point<br/>Progress reporting"]
    end

    subgraph "APPLICATION"
        RETPIPE["Retrieval Pipeline<br/>QueryTransformer<br/>RetrievalStrategy<br/>Reranker<br/>ContextBuilder"]
        INGPIPE["Ingestion Pipeline<br/>DocumentExtractor<br/>TextPreprocessor<br/>ChunkingStrategy<br/>MetadataEnricher<br/>EmbeddingService"]
        USECASES["Use Cases<br/>ChatWithRAG<br/>SearchDocuments<br/>IngestDocument"]
    end

    subgraph "DOMAIN"
        MODELS["Models<br/>Document, Segment<br/>SearchQuery, SearchResult<br/>Conversation, Message"]
        SERVICES["Services<br/>ContextBuilder<br/>CitationResolver<br/>RelevanceScorer"]
        PORTS["Ports (interfaces)<br/>DocumentRepository<br/>VectorStore<br/>EmbeddingService<br/>Reranker"]
    end

    subgraph "INFRASTRUCTURE"
        RETRIEVAL["Retrieval<br/>DenseRetriever<br/>SparseRetriever<br/>HybridRetriever"]
        STORAGE["Storage<br/>PgDocumentRepo<br/>PgVectorStore<br/>PgSearchIndex"]
        EXTERNAL["External APIs<br/>LmStudioClient<br/>LiteLLMClient<br/>CohereReranker"]
        CACHE["Caching<br/>EmbeddingCache<br/>SearchCache"]
        OBS["Observability<br/>PrometheusMetrics<br/>OpenTelemetryTracing"]
    end

    API2 --> USECASES
    AGENT2 --> USECASES
    INGEST2 --> INGPIPE

    USECASES --> RETPIPE
    RETPIPE --> SERVICES
    INGPIPE --> SERVICES

    SERVICES --> PORTS
    MODELS --> PORTS

    RETRIEVAL -.->|implements| PORTS
    STORAGE -.->|implements| PORTS
    EXTERNAL -.->|implements| PORTS
    CACHE -.->|implements| PORTS

    style DOMAIN fill:#d4edda
    style PORTS fill:#d4edda
```

### Target Directory Structure

```text
rag-api/
├── domain/                    # Business rules (pure, no deps)
│   ├── models/
│   ├── services/
│   └── ports/
├── application/               # Use cases (orchestration)
│   ├── commands/
│   ├── queries/
│   └── pipelines/
├── core/                      # Shared infrastructure
│   ├── chunking/
│   ├── reranking/
│   ├── retrieval/
│   ├── cache/
│   └── observability/
├── apps/
│   ├── api/
│   ├── agent/
│   └── ingest/
├── evaluation/
└── tests/
```

---

## 4. Target Retrieval Pipeline

```mermaid
flowchart TD
    A["User Query"] --> B["Query Transformation"]

    subgraph TRANSFORM ["Query Transformation"]
        B1["Original Query"] --> B2["Expanded<br/>+ synonyms<br/>+ context"]
        B2 --> B3["Multi-Query<br/>[Q1, Q2, Q3]<br/>(optional LLM)"]
    end

    B --> C["Parallel Search"]

    subgraph SEARCH ["Parallel Search"]
        direction LR
        C1["DENSE<br/>query → embed<br/>cosine sim<br/>pgvector ANN"]
        C2["SPARSE<br/>query → tokens<br/>BM25/tsvector<br/>GIN index"]
        C3["METADATA<br/>filters:<br/>language, date<br/>topics, tenant"]
    end

    C --> D["RRF Fusion"]

    subgraph FUSION ["Fusion (k=60)"]
        D1["score(d) = Σ 1/(k + rank_i)"]
        D2["Dense [A,B,C] + Sparse [B,D,A]<br/>= Merged [B,A,C,D]"]
    end

    D --> E["Reranking"]

    subgraph RERANK ["Reranking"]
        E1["CrossEncoder / Cohere"]
        E2["For each (query, doc):<br/>score = CrossEncoder(query, doc)"]
        E3["Before: [B,A,C,D]<br/>After: [A,B,D,C]"]
    end

    E --> F["Context Building"]

    subgraph CONTEXT ["Context Building"]
        F1["1. Deduplicate"]
        F2["2. Prioritize by score"]
        F3["3. Fit token budget"]
        F4["4. Add source markers"]
    end

    F --> G["LLM Call<br/>system + context + query<br/>→ answer"]
```

---

## 5. Target Ingestion Pipeline

```mermaid
flowchart TD
    A["Source File<br/>(PDF/HTML/Audio/Video)"] --> B["Document Extraction"]

    subgraph EXTRACT ["Document Extraction"]
        B1["PDF<br/>pymupdf<br/>pymupdf4llm"]
        B2["HTML<br/>trafilatura<br/>readability"]
        B3["Audio<br/>whisper"]
        B4["Video<br/>whisper +<br/>frame OCR"]
    end

    B --> C["Text Preprocessing"]

    subgraph PREPROCESS ["Preprocessing"]
        C1["Normalize Unicode (NFKC)"]
        C2["Remove control chars"]
        C3["Fix encoding"]
        C4["Normalize whitespace"]
        C5["Detect language"]
        C6["Remove boilerplate"]
    end

    C --> D["Chunking"]

    subgraph CHUNK ["Chunking Strategies"]
        D1["Sliding Window<br/>Fast, Simple"]
        D2["Recursive<br/>Balanced, Default"]
        D3["Semantic<br/>Best, Slow"]
    end

    D --> E["Metadata Enrichment"]

    subgraph ENRICH ["Metadata"]
        E1["language"]
        E2["topics (LLM)"]
        E3["entities (NER)"]
        E4["quality_score"]
        E5["token_count"]
    end

    E --> F["Embedding"]

    subgraph EMBED ["Embedding"]
        F1["Embedding Model<br/>LM Studio / Vertex AI"]
        F2["Embedding Cache<br/>content_hash → vector"]
    end

    F --> G["Storage"]

    subgraph STORE ["PostgreSQL"]
        G1["documents<br/>id, source_path, title, sha256"]
        G2["segments<br/>id, document_id, ordinal, page<br/>content, metadata (JSONB), tsv"]
        G3["segment_embeddings<br/>segment_id, embedding (vector)"]
    end
```

---

## 6. Clean Architecture Layers

```mermaid
graph TB
    subgraph DOMAIN ["DOMAIN LAYER"]
        DM["Models"]
        DS["Services"]
        DP["Ports"]
        DM --- DS
        DS --- DP
    end

    subgraph APP ["APPLICATION LAYER"]
        AU["Use Cases"]
        AP["Pipelines"]
        AU --> DP
        AP --> DS
    end

    subgraph INFRA ["INFRASTRUCTURE LAYER"]
        IP["Postgres Adapters<br/>implements: DocRepo, VectorStore"]
        IL["LM Studio Adapters<br/>implements: ChatService, Embeddings"]
        IR["Redis Adapters<br/>implements: Cache"]
    end

    subgraph PRES ["PRESENTATION LAYER"]
        PH["HTTP Handlers"]
        PC["CLI Commands"]
    end

    subgraph COMP ["COMPOSITION ROOT"]
        CW["Wires ports to adapters"]
        CC["Configures dependencies"]
    end

    PH --> AU
    PC --> AU

    IP -.->|implements| DP
    IL -.->|implements| DP
    IR -.->|implements| DP

    COMP --> INFRA
    COMP --> PRES

    style DOMAIN fill:#d4edda
    style APP fill:#cce5ff
    style INFRA fill:#f8d7da
    style PRES fill:#fff3cd
```

### Dependency Rules

| Layer | Can Depend On | Cannot Depend On |
|-------|---------------|------------------|
| Domain | Nothing | Application, Infrastructure, Presentation |
| Application | Domain | Infrastructure, Presentation |
| Infrastructure | Domain, Application | Presentation |
| Presentation | Application | Domain (directly), Infrastructure |

---

## 7. Hybrid Search

```mermaid
flowchart TD
    Q["Query: 'machine learning'"] --> SPLIT

    subgraph SPLIT ["Parallel Retrieval"]
        direction LR
        DENSE["DENSE SEARCH<br/>embed → cosine sim<br/>pgvector"]
        SPARSE["SPARSE SEARCH<br/>to_tsquery<br/>GIN index<br/>ts_rank"]
    end

    DENSE --> DR["Results by cosine<br/>1. doc_A (0.92)<br/>2. doc_B (0.87)<br/>3. doc_C (0.85)<br/>4. doc_E (0.81)"]
    SPARSE --> SR["Results by BM25<br/>1. doc_B (12.5)<br/>2. doc_D (11.2)<br/>3. doc_A (10.1)<br/>4. doc_F (9.8)"]

    DR --> RRF
    SR --> RRF

    subgraph RRF ["RRF FUSION (k=60)"]
        RF1["score(doc) = Σ 1/(k + rank_i)"]
        RF2["doc_A: 1/61 + 1/63 = 0.0322"]
        RF3["doc_B: 1/62 + 1/61 = 0.0326 ← top"]
        RF4["Sorted: [B, A, D, C, E, F]"]
    end

    RRF --> MERGED["Merged Results"]
```

### RRF Formula

```text
RRF(d) = Σ 1 / (k + rank_i(d))
         i

Where:
  - k = 60 (constant, prevents high ranks from dominating)
  - rank_i(d) = position of document d in ranker i (1-indexed)
  - If document not in ranker i, contribution is 0
```

### Why Hybrid?

| Aspect | Dense Only | Sparse Only | Hybrid |
|--------|------------|-------------|--------|
| Semantic similarity | Strong | Weak | Strong |
| Exact keyword match | Weak | Strong | Strong |
| Out-of-vocabulary | Handles | Fails | Handles |
| Rare terms | Weak | Strong | Strong |
| Quality boost | baseline | baseline | +10-20% |

---

## 8. Multi-tenancy

### Option A: Shared Tables + RLS (Recommended)

```mermaid
flowchart LR
    CLIENT["Client"] --> AUTH["Auth/Tenant<br/>Resolver"]
    AUTH --> SET["SET app.tenant_id<br/>= 'abc-123'"]
    SET --> QUERY["Query<br/>(auto-RLS)"]

    subgraph PG ["PostgreSQL"]
        DOC["documents<br/>+ tenant_id UUID"]
        SEG["segments<br/>+ tenant_id UUID"]
        RLS["RLS Policy:<br/>tenant_id = current_setting('app.tenant_id')"]
    end

    QUERY --> PG
```

### Option B: Schema-per-Tenant

```mermaid
flowchart LR
    CLIENT2["Client"] --> AUTH2["Auth/Tenant<br/>Resolver"]
    AUTH2 --> SET2["SET search_path<br/>= 'tenant_acme'"]
    SET2 --> QUERY2["Query<br/>(schema-isolated)"]

    subgraph PG2 ["PostgreSQL"]
        S1["tenant_acme/<br/>documents, segments"]
        S2["tenant_globex/<br/>documents, segments"]
        S3["tenant_initech/<br/>documents, segments"]
    end

    QUERY2 --> PG2
```

### Comparison

| Factor | Shared + RLS | Schema-per-Tenant |
|--------|--------------|-------------------|
| Isolation | Logical | Physical |
| Complexity | Low | Medium |
| Index overhead | 1x | Nx |
| Migrations | Single | Per-schema |
| Cross-tenant queries | Easy | Hard |
| Compliance | May need schema | Preferred |
| Recommended for | SaaS MVP | Enterprise |

---

## 9. Observability

### Request Trace

```mermaid
gantt
    title Request Trace: /v1/chat/completions
    dateFormat X
    axisFormat %L ms

    section Retrieval
    embed_query       :0, 50
    dense_search      :50, 130
    sparse_search     :50, 120
    rrf_fusion        :130, 135
    rerank            :135, 285

    section Generation
    build_context     :285, 295
    llm_call          :295, 1095

    section Total
    total             :milestone, 1095, 0
```

### Key Metrics

```mermaid
mindmap
  root((Metrics))
    Requests
      rag_requests_total
      rag_request_duration_seconds
    Search
      rag_search_latency_seconds
      rag_search_results_count
      rag_rerank_latency_seconds
    LLM
      rag_embedding_latency_seconds
      rag_llm_latency_seconds
      rag_context_tokens
    Cache
      rag_cache_hits_total
      rag_cache_misses_total
    Data
      rag_documents_total
      rag_segments_total
```

---

## 10. What's Missing

### Priority Matrix

```mermaid
quadrantChart
    title Priority vs Effort
    x-axis Low Effort --> High Effort
    y-axis Low Impact --> High Impact
    quadrant-1 Do First
    quadrant-2 Plan
    quadrant-3 Delegate
    quadrant-4 Eliminate

    "Hybrid Search": [0.3, 0.9]
    "Reranker Wire": [0.2, 0.8]
    "Metadata Column": [0.25, 0.7]
    "Domain Layer": [0.6, 0.75]
    "Embedding Cache": [0.4, 0.6]
    "Evaluation": [0.5, 0.65]
    "Multi-tenancy": [0.7, 0.5]
    "Observability": [0.45, 0.55]
    "Tests": [0.55, 0.6]
```

### Gap Summary

| Priority | Gap | Impact | Fix |
|----------|-----|--------|-----|
| P0 | Hybrid retrieval | -15% quality | Add BM25 + RRF |
| P0 | Reranker integration | -10% quality | Wire into /chat |
| P0 | Metadata filtering | No filtering | Add JSONB column |
| P1 | Domain layer | Hard to test | Extract domain/ |
| P1 | Embedding cache | Wasted compute | Add cache layer |
| P1 | Evaluation harness | Blind optimization | Add eval/ |
| P2 | Multi-tenancy | No SaaS | Add tenant_id |
| P2 | Observability | No metrics | Add Prometheus |

---

## 11. Migration Roadmap

```mermaid
gantt
    title Migration Roadmap
    dateFormat YYYY-MM-DD

    section Phase 1: Quality
    Enable BM25 via tsvector      :p1a, 2024-01-01, 7d
    Wire reranker into /chat      :p1b, 2024-01-01, 7d
    Add metadata JSONB column     :p1c, 2024-01-08, 7d

    section Phase 2: Architecture
    Extract domain/ layer         :p2a, 2024-01-15, 7d
    Create application/pipelines/ :p2b, 2024-01-15, 7d
    Move business logic           :p2c, 2024-01-22, 7d

    section Phase 3: Operability
    Prometheus metrics            :p3a, 2024-01-29, 7d
    Embedding cache               :p3b, 2024-01-29, 7d
    Basic evaluation suite        :p3c, 2024-02-05, 7d

    section Phase 4: Scale
    Multi-tenancy (RLS)           :p4a, 2024-02-12, 7d
    Async batch ingestion         :p4b, 2024-02-12, 7d
    Query caching                 :p4c, 2024-02-19, 7d
```

### Phase Details

```mermaid
flowchart LR
    subgraph P1 ["Phase 1: Quality"]
        P1A["BM25 + RRF"]
        P1B["Reranker"]
        P1C["Metadata"]
    end

    subgraph P2 ["Phase 2: Architecture"]
        P2A["domain/"]
        P2B["pipelines/"]
        P2C["Refactor"]
    end

    subgraph P3 ["Phase 3: Ops"]
        P3A["Prometheus"]
        P3B["Cache"]
        P3C["Eval"]
    end

    subgraph P4 ["Phase 4: Scale"]
        P4A["Multi-tenant"]
        P4B["Batch ingest"]
        P4C["Query cache"]
    end

    P1 --> P2 --> P3 --> P4

    style P1 fill:#d4edda
    style P2 fill:#cce5ff
    style P3 fill:#fff3cd
    style P4 fill:#f8d7da
```

---

## Appendix: Quick Reference

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/rag

# Chat backend
CHAT_BACKEND=openai_compat  # or litellm
CHAT_BASE_URL=http://localhost:1234/v1
CHAT_MODEL=local-model

# Embeddings backend
EMBEDDINGS_BACKEND=openai_compat
EMBEDDINGS_BASE_URL=http://localhost:1234/v1
EMBEDDINGS_MODEL=local-embedding-model

# RAG settings
TOP_K=6
MAX_CONTEXT_CHARS=24000

# Chunking
CHUNKING_STRATEGY=recursive  # sliding, recursive, semantic
CHUNKING_CHUNK_SIZE=512

# Reranking (when wired)
RERANKING_STRATEGY=cross_encoder  # none, cross_encoder, cohere
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Security
ALLOW_ANONYMOUS=false
```

### SQL Snippets

```sql
-- Hybrid search (dense + sparse with RRF)
WITH dense AS (
  SELECT segment_id, 1 - (embedding <=> $1) AS score,
         ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
  FROM segment_embeddings
  ORDER BY embedding <=> $1
  LIMIT 20
),
sparse AS (
  SELECT id AS segment_id, ts_rank(tsv, query) AS score,
         ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, query) DESC) AS rank
  FROM segments, plainto_tsquery('simple', $2) query
  WHERE tsv @@ query
  ORDER BY ts_rank(tsv, query) DESC
  LIMIT 20
)
SELECT segment_id,
       COALESCE(1.0/(60 + d.rank), 0) + COALESCE(1.0/(60 + s.rank), 0) AS rrf_score
FROM dense d
FULL OUTER JOIN sparse s USING (segment_id)
ORDER BY rrf_score DESC
LIMIT $3;

-- Add metadata column
ALTER TABLE segments ADD COLUMN metadata JSONB DEFAULT '{}';
CREATE INDEX idx_segments_metadata ON segments USING GIN (metadata);

-- Multi-tenancy RLS
ALTER TABLE documents ADD COLUMN tenant_id UUID;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents
  USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

---

*Last updated: 2024*
