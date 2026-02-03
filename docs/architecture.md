# RAG API Architecture Manual

> Production-grade RAG system architecture guide.
> Covers current state, target state, and migration path.

## Table of Contents

1. [Current Architecture (As-Is)](#1-current-architecture-as-is)
2. [Current Retrieval Flow](#2-current-retrieval-flow)
3. [Target Architecture (To-Be)](#3-target-architecture-to-be)
4. [Target Retrieval Pipeline](#4-target-retrieval-pipeline)
5. [Target Ingestion Pipeline](#5-target-ingestion-pipeline)
6. [Clean Architecture Layers](#6-clean-architecture-layers)
7. [Hybrid Search](#7-hybrid-search)
8. [Multi-tenancy](#8-multi-tenancy)
9. [Observability](#9-observability)
10. [What's Missing](#10-whats-missing)
11. [Migration Roadmap](#11-migration-roadmap)

---

## 1. Current Architecture (As-Is)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPS LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────────┐  │
│  │      apps/api/       │  │     apps/agent/      │  │   apps/ingest/    │  │
│  │                      │  │                      │  │                   │  │
│  │  main.py             │  │  agent.py            │  │  cli.py           │  │
│  │  retrieval.py ◄──────┼──┼── tools.py           │  │  store.py         │  │
│  │  auth.py             │  │  protocol.py         │  │  pdf_extract.py   │  │
│  │  models.py           │  │                      │  │                   │  │
│  │  chat_client.py      │  └──────────────────────┘  └───────────────────┘  │
│  └──────────┬───────────┘                                      │            │
│             │                                                  │            │
│             │  ⚠️ Business logic mixed with infrastructure     │            │
│             │  ⚠️ retrieval.py = SQL + context building        │            │
│             │  ⚠️ store.py = SQL + hashing + dedup logic       │            │
└─────────────┼──────────────────────────────────────────────────┼────────────┘
              │                                                  │
              ▼                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │  config.py │  │   db.py    │  │ schema.py  │  │     lmstudio.py        │ │
│  │            │  │            │  │            │  │  embeddings_client.py  │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐           │
│  │        chunking/            │  │        reranking/           │           │
│  │  ┌─────────┐ ┌──────────┐   │  │  ┌─────────┐ ┌───────────┐  │           │
│  │  │ sliding │ │recursive │   │  │  │  none   │ │cross_enc. │  │  ⚠️ Not  │
│  │  └─────────┘ └──────────┘   │  │  └─────────┘ └───────────┘  │  wired!  │
│  │  ┌──────────┐               │  │  ┌─────────┐                │           │
│  │  │ semantic │               │  │  │ cohere  │                │           │
│  │  └──────────┘               │  │  └─────────┘                │           │
│  └─────────────────────────────┘  └─────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INFRASTRUCTURE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │      PostgreSQL + pgvector     │  │    LM Studio / LiteLLM / Gemini   │ │
│  │                                │  │                                    │ │
│  │  documents ──┐                 │  │  POST /v1/embeddings               │ │
│  │  segments ───┼── CASCADE       │  │  POST /v1/chat/completions         │ │
│  │  segment_embeddings            │  │                                    │ │
│  │  api_keys                      │  │                                    │ │
│  │                                │  │                                    │ │
│  │  ⚠️ tsvector index NOT USED    │  │                                    │ │
│  └────────────────────────────────┘  └────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
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

```text
                    ┌─────────────────┐
                    │   User Query    │
                    │  "What is X?"   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Embed       │
                    │   query → vec   │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      Vector Search (only)    │
              │                              │
              │  SELECT ... ORDER BY         │
              │  embedding <=> query_vec     │
              │  LIMIT k                     │
              │                              │
              │  ⚠️ No BM25/FTS              │
              │  ⚠️ No hybrid fusion         │
              │  ⚠️ No reranking             │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      Build Context           │
              │                              │
              │  segments → text + sources   │
              │  truncate to max_chars       │
              │                              │
              │  ⚠️ Simple concatenation     │
              │  ⚠️ No deduplication         │
              │  ⚠️ No relevance weighting   │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │         LLM Call             │
              │                              │
              │  system + context + user     │
              │         → answer             │
              └──────────────────────────────┘
```

---

## 3. Target Architecture (To-Be)

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │     apps/api/      │  │    apps/agent/     │  │      apps/ingest/          │ │
│  │                    │  │                    │  │                            │ │
│  │  HTTP handlers     │  │  Agent loop        │  │  CLI entry point           │ │
│  │  Request DTOs      │  │  Tool execution    │  │  Progress reporting        │ │
│  │  Response DTOs     │  │                    │  │                            │ │
│  │  Auth middleware   │  │                    │  │                            │ │
│  └─────────┬──────────┘  └─────────┬──────────┘  └─────────────┬──────────────┘ │
│            │                       │                           │                │
└────────────┼───────────────────────┼───────────────────────────┼────────────────┘
             │                       │                           │
             ▼                       ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │         Retrieval Pipeline          │  │      Ingestion Pipeline         │   │
│  │                                     │  │                                 │   │
│  │  QueryTransformer                   │  │  DocumentExtractor              │   │
│  │       ↓                             │  │       ↓                         │   │
│  │  RetrievalStrategy (hybrid)         │  │  TextPreprocessor               │   │
│  │       ↓                             │  │       ↓                         │   │
│  │  Reranker                           │  │  ChunkingStrategy               │   │
│  │       ↓                             │  │       ↓                         │   │
│  │  ContextBuilder                     │  │  MetadataEnricher               │   │
│  │                                     │  │       ↓                         │   │
│  └─────────────────────────────────────┘  │  EmbeddingService               │   │
│                                           │       ↓                         │   │
│  ┌─────────────────────────────────────┐  │  DocumentStore                  │   │
│  │           Use Cases                 │  │                                 │   │
│  │                                     │  └─────────────────────────────────┘   │
│  │  ChatWithRAG(query, options)        │                                        │
│  │  SearchDocuments(query, filters)    │                                        │
│  │  IngestDocument(source, options)    │                                        │
│  │                                     │                                        │
│  └─────────────────────────────────────┘                                        │
│                                                                                  │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DOMAIN LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │       Models        │  │      Services       │  │         Ports           │  │
│  │                     │  │                     │  │      (interfaces)       │  │
│  │  Document           │  │  ContextBuilder     │  │                         │  │
│  │  Segment            │  │    - max_tokens     │  │  DocumentRepository     │  │
│  │  Embedding          │  │    - dedup          │  │  VectorStore            │  │
│  │                     │  │    - prioritize     │  │  EmbeddingService       │  │
│  │  SearchQuery        │  │                     │  │  SearchIndex            │  │
│  │  SearchResult       │  │  CitationResolver   │  │  Reranker               │  │
│  │  RetrievalContext   │  │    - format         │  │  ChatService            │  │
│  │                     │  │    - entitlements   │  │                         │  │
│  │  Conversation       │  │                     │  │                         │  │
│  │  Message            │  │  RelevanceScorer    │  │                         │  │
│  │  Turn               │  │    - weights        │  │                         │  │
│  │                     │  │    - thresholds     │  │                         │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────────┘  │
│                                                                                  │
│                        ✅ Pure Python, no external deps                         │
│                        ✅ All business rules here                               │
│                        ✅ Testable without infrastructure                       │
│                                                                                  │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          INFRASTRUCTURE LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────────┐│
│  │    Retrieval      │  │     Storage       │  │        External APIs          ││
│  │                   │  │                   │  │                               ││
│  │  DenseRetriever   │  │  PgDocumentRepo   │  │  LmStudioClient               ││
│  │  SparseRetriever  │  │  PgVectorStore    │  │  LiteLLMClient                ││
│  │  HybridRetriever  │  │  PgSearchIndex    │  │  CohereReranker               ││
│  │                   │  │                   │  │  VertexAIEmbeddings           ││
│  └───────────────────┘  └───────────────────┘  └───────────────────────────────┘│
│                                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────────┐│
│  │     Chunking      │  │     Caching       │  │       Observability           ││
│  │                   │  │                   │  │                               ││
│  │  SlidingChunker   │  │  EmbeddingCache   │  │  PrometheusMetrics            ││
│  │  RecursiveChunker │  │  SearchCache      │  │  OpenTelemetryTracing         ││
│  │  SemanticChunker  │  │  (Redis/Memory)   │  │  StructuredLogger             ││
│  │                   │  │                   │  │                               ││
│  └───────────────────┘  └───────────────────┘  └───────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Target Directory Structure

```text
rag-api/
├── domain/                    # Business rules (pure Python, no deps)
│   ├── models/
│   │   ├── document.py        # Document, Segment, Embedding
│   │   ├── search.py          # SearchQuery, SearchResult, SearchContext
│   │   └── conversation.py    # Conversation, Message, Turn
│   ├── services/
│   │   ├── context_builder.py # How to assemble context
│   │   └── citation_resolver.py
│   └── ports/                 # Interfaces (implemented in infra)
│       ├── document_store.py
│       ├── vector_store.py
│       └── embedding_service.py
│
├── application/               # Use cases (orchestration)
│   ├── commands/
│   │   ├── ingest_document.py
│   │   └── chat_with_rag.py
│   ├── queries/
│   │   └── search_documents.py
│   └── pipelines/
│       ├── ingestion.py
│       └── retrieval.py
│
├── core/                      # Shared infrastructure
│   ├── config.py
│   ├── db.py
│   ├── schema.py
│   ├── chunking/
│   ├── reranking/
│   ├── retrieval/             # Dense, Sparse, Hybrid strategies
│   ├── cache/                 # Embedding, search caching
│   └── observability/         # Metrics, tracing
│
├── apps/
│   ├── api/
│   ├── agent/
│   └── ingest/
│
├── evaluation/                # Offline eval
│   ├── datasets/
│   ├── metrics/
│   └── runners/
│
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

---

## 4. Target Retrieval Pipeline

```text
                         ┌─────────────────┐
                         │   User Query    │
                         │  "What is X?"   │
                         └────────┬────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY TRANSFORMATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│   │   Original  │    │  Expanded   │    │      Multi-Query        │ │
│   │   "X?"      │ ─► │  + synonyms │ ─► │  [Q1, Q2, Q3]           │ │
│   │             │    │  + context  │    │  (optional, via LLM)    │ │
│   └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│                                                                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  DENSE SEARCH   │    │  SPARSE SEARCH  │    │  METADATA FILTER│
│                 │    │                 │    │                 │
│  query → embed  │    │  query → tokens │    │  filters:       │
│  cosine sim     │    │  BM25/tsvector  │    │   - language    │
│  pgvector ANN   │    │  GIN index      │    │   - date_range  │
│                 │    │                 │    │   - topics      │
│  top-K results  │    │  top-K results  │    │   - tenant_id   │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FUSION (RRF)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Reciprocal Rank Fusion:                                            │
│                                                                      │
│   score(d) = Σ  1 / (k + rank_i(d))                                  │
│              i                                                       │
│                                                                      │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│   │ Dense       │  │ Sparse      │  │ Merged      │                 │
│   │ [A,B,C]     │ +│ [B,D,A]     │ =│ [B,A,C,D]   │                 │
│   └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RERANKING                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Cross-Encoder / Cohere Rerank:                                     │
│                                                                      │
│   ┌─────────────────────────────────────────┐                       │
│   │  For each (query, document) pair:       │                       │
│   │    score = CrossEncoder(query, doc)     │                       │
│   │                                         │                       │
│   │  Sort by score DESC                     │                       │
│   │  Take top-N (N < K)                     │                       │
│   └─────────────────────────────────────────┘                       │
│                                                                      │
│   Before: [B, A, C, D]  (RRF order)                                  │
│   After:  [A, B, D, C]  (relevance order)                            │
│                                                                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CONTEXT BUILDING                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   1. Deduplicate (same content from different sources)               │
│   2. Prioritize by relevance score                                   │
│   3. Fit to token budget (max_context_tokens)                        │
│   4. Format with source markers (if entitled)                        │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │  [SOURCE: doc1.pdf p.5]                                 │       │
│   │  Relevant text from segment A...                        │       │
│   │                                                         │       │
│   │  [SOURCE: doc2.pdf p.12]                                │       │
│   │  Relevant text from segment B...                        │       │
│   │                                                         │       │
│   │  (truncated to fit token budget)                        │       │
│   └─────────────────────────────────────────────────────────┘       │
│                                                                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    LLM Call     │
                    │                 │
                    │  system prompt  │
                    │  + context      │
                    │  + user query   │
                    │       ↓         │
                    │    answer       │
                    └─────────────────┘
```

---

## 5. Target Ingestion Pipeline

```text
┌─────────────────┐
│   Source File   │
│  (PDF/HTML/...) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT EXTRACTION                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │     PDF     │  │    HTML     │  │    Audio    │  │    Video    │ │
│  │             │  │             │  │             │  │             │ │
│  │  pymupdf    │  │ trafilatura │  │  whisper    │  │  whisper +  │ │
│  │  pymupdf4llm│  │ readability │  │             │  │  frame OCR  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                                   │                                  │
│                                   ▼                                  │
│                          list[PageText]                              │
│                          (page_num, text)                            │
│                                                                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TEXT PREPROCESSING                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  1. Normalize Unicode (NFKC)                                 │   │
│  │  2. Remove control characters                                │   │
│  │  3. Fix encoding issues                                      │   │
│  │  4. Normalize whitespace                                     │   │
│  │  5. Detect language                                          │   │
│  │  6. Remove boilerplate (headers/footers)                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CHUNKING                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   Sliding   │  │  Recursive  │  │  Semantic   │                  │
│  │   Window    │  │  (semchunk) │  │  (chonkie)  │                  │
│  │             │  │             │  │             │                  │
│  │  Fast       │  │  Balanced   │  │  Best       │                  │
│  │  Simple     │  │  Default    │  │  Slow       │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│                                                                      │
│                          ↓                                           │
│                   list[Chunk]                                        │
│                   (page, ordinal, content)                           │
│                                                                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     METADATA ENRICHMENT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  For each chunk:                                                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  metadata = {                                                │   │
│  │    "language": detect_language(chunk),                       │   │
│  │    "topics": extract_topics(chunk),       # optional LLM     │   │
│  │    "entities": extract_entities(chunk),   # NER              │   │
│  │    "quality_score": assess_quality(chunk),                   │   │
│  │    "token_count": count_tokens(chunk),                       │   │
│  │  }                                                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐      ┌─────────────────────┐               │
│  │   Embedding Model   │      │   Embedding Cache   │               │
│  │                     │      │                     │               │
│  │   LM Studio (local) │ ◄──► │   content_hash →    │               │
│  │   Vertex AI         │      │   embedding vector  │               │
│  │   OpenAI            │      │                     │               │
│  └─────────────────────┘      └─────────────────────┘               │
│                                                                      │
│  Batch embedding:                                                    │
│    chunks[0:100] → embeddings[0:100]                                 │
│    chunks[100:200] → embeddings[100:200]                             │
│    ...                                                               │
│                                                                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          STORAGE                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                      PostgreSQL                            │     │
│  │                                                            │     │
│  │  documents ◄─────────────────────────────────────────┐     │     │
│  │    id, source_path, title, sha256, tenant_id         │     │     │
│  │                                                      │     │     │
│  │  segments ◄──────────────────────────────────────┐   │     │     │
│  │    id, document_id, ordinal, page, content,      │   │     │     │
│  │    metadata (JSONB), tsv (tsvector)              │   │     │     │
│  │                                                  │   │     │     │
│  │  segment_embeddings                              │   │     │     │
│  │    segment_id, embedding (vector)                │   │     │     │
│  │                                                  │   │     │     │
│  └──────────────────────────────────────────────────┴───┴─────┘     │
│                                                                      │
│  Indexes:                                                            │
│    - IVFFLAT on embeddings (ANN search)                              │
│    - GIN on tsv (full-text search)                                   │
│    - GIN on metadata (JSONB filtering)                               │
│    - B-tree on document_id, tenant_id                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Clean Architecture Layers

```text
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │          DOMAIN LAYER               │
                    │                                     │
                    │   Models    Services    Ports       │
                    │     │          │          ▲         │
                    │     │          │          │         │
                    │     └──────────┴──────────┘         │
                    │                                     │
                    │   ✅ No external dependencies       │
                    │   ✅ Pure business logic            │
                    │                                     │
                    └──────────────────▲──────────────────┘
                                       │
                                       │ depends on
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    │        APPLICATION LAYER            │
                    │                                     │
                    │   Use Cases      Pipelines          │
                    │       │              │              │
                    │       ▼              ▼              │
                    │   Domain Ports   Domain Services    │
                    │                                     │
                    │   ✅ Orchestration only             │
                    │   ✅ Transaction boundaries         │
                    │                                     │
                    └──────────────────▲──────────────────┘
                                       │
                                       │ depends on
                                       │
     ┌─────────────────────────────────┴───────────────────────────────┐
     │                                                                 │
     │                     INFRASTRUCTURE LAYER                        │
     │                                                                 │
     │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
     │  │   Postgres    │  │   LM Studio   │  │    Redis      │       │
     │  │   Adapters    │  │   Adapters    │  │   Adapters    │       │
     │  │               │  │               │  │               │       │
     │  │ implements:   │  │ implements:   │  │ implements:   │       │
     │  │ - DocRepo     │  │ - ChatService │  │ - Cache       │       │
     │  │ - VectorStore │  │ - Embeddings  │  │               │       │
     │  │ - SearchIndex │  │               │  │               │       │
     │  └───────────────┘  └───────────────┘  └───────────────┘       │
     │                                                                 │
     │  ✅ Implements Domain Ports                                     │
     │  ✅ All external dependencies here                              │
     │                                                                 │
     └─────────────────────────────────▲───────────────────────────────┘
                                       │
                                       │ depends on
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    │        PRESENTATION LAYER           │
                    │                                     │
                    │   HTTP Handlers    CLI Commands     │
                    │        │                │           │
                    │        ▼                ▼           │
                    │   Application Use Cases             │
                    │                                     │
                    │   ✅ Only transport concerns        │
                    │   ✅ DTO mapping                    │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       │ uses
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    │        COMPOSITION ROOT             │
                    │                                     │
                    │   - Wires ports to adapters         │
                    │   - Configures dependencies         │
                    │   - Startup/shutdown lifecycle      │
                    │                                     │
                    └─────────────────────────────────────┘
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

```text
                              Query: "machine learning"
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │     DENSE SEARCH      │             │     SPARSE SEARCH     │
        │                       │             │                       │
        │  embed("machine       │             │  to_tsquery(          │
        │   learning")          │             │   'machine & learning'│
        │         ↓             │             │  )                    │
        │  [0.12, -0.34, ...]   │             │         ↓             │
        │         ↓             │             │  GIN index lookup     │
        │  cosine similarity    │             │         ↓             │
        │  via pgvector         │             │  ts_rank scoring      │
        │                       │             │                       │
        └───────────┬───────────┘             └───────────┬───────────┘
                    │                                     │
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │  Results (by cosine)  │             │  Results (by BM25)    │
        │                       │             │                       │
        │  1. doc_A (0.92)      │             │  1. doc_B (12.5)      │
        │  2. doc_B (0.87)      │             │  2. doc_D (11.2)      │
        │  3. doc_C (0.85)      │             │  3. doc_A (10.1)      │
        │  4. doc_E (0.81)      │             │  4. doc_F (9.8)       │
        │                       │             │                       │
        └───────────┬───────────┘             └───────────┬───────────┘
                    │                                     │
                    └───────────────┬─────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────┐
                    │         RRF FUSION (k=60)             │
                    │                                       │
                    │  score(doc) = Σ 1/(k + rank_i)        │
                    │                                       │
                    │  doc_A: 1/61 + 1/63 = 0.0322          │
                    │  doc_B: 1/62 + 1/61 = 0.0326  ◄── top │
                    │  doc_C: 1/63 + 0     = 0.0159         │
                    │  doc_D: 0    + 1/62  = 0.0161         │
                    │  doc_E: 1/64 + 0     = 0.0156         │
                    │  doc_F: 0    + 1/64  = 0.0156         │
                    │                                       │
                    │  Sorted: [B, A, D, C, E, F]           │
                    │                                       │
                    └───────────────────────────────────────┘
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
| Semantic similarity | ✅ Strong | ❌ Weak | ✅ Strong |
| Exact keyword match | ❌ Weak | ✅ Strong | ✅ Strong |
| Out-of-vocabulary | ✅ Handles | ❌ Fails | ✅ Handles |
| Rare terms | ❌ Weak | ✅ Strong | ✅ Strong |
| Typical quality boost | baseline | baseline | +10-20% |

---

## 8. Multi-tenancy

### Option A: Shared Tables + RLS (Recommended)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PostgreSQL                                    │    │
│  │                                                                      │    │
│  │   documents                       segments                           │    │
│  │   ┌──────────────────────┐       ┌──────────────────────────┐       │    │
│  │   │ id          UUID     │       │ id            UUID       │       │    │
│  │   │ tenant_id   UUID  ◄──┼───────┼─ tenant_id    UUID       │       │    │
│  │   │ source_path TEXT     │       │ document_id   UUID       │       │    │
│  │   │ ...                  │       │ ...                      │       │    │
│  │   └──────────────────────┘       └──────────────────────────┘       │    │
│  │                                                                      │    │
│  │   RLS Policy:                                                        │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │  CREATE POLICY tenant_isolation ON documents                │   │    │
│  │   │  USING (tenant_id = current_setting('app.tenant_id')::uuid);│   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Request flow:                                                               │
│  ┌────────┐    ┌─────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ Client │───►│ Auth/Tenant │───►│ SET app.tenant_id│───►│    Query     │  │
│  │        │    │  Resolver   │    │  = 'abc-123'     │    │  (auto-RLS)  │  │
│  └────────┘    └─────────────┘    └──────────────────┘    └──────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pros:**

- Simple schema
- Easy to query across tenants (admin)
- Single index for all tenants

**Cons:**

- Noisy neighbor risk
- Must remember to set session variable

### Option B: Schema-per-Tenant

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PostgreSQL                                    │    │
│  │                                                                      │    │
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │    │
│  │   │ tenant_acme     │  │ tenant_globex   │  │ tenant_initech  │     │    │
│  │   │                 │  │                 │  │                 │     │    │
│  │   │  documents      │  │  documents      │  │  documents      │     │    │
│  │   │  segments       │  │  segments       │  │  segments       │     │    │
│  │   │  embeddings     │  │  embeddings     │  │  embeddings     │     │    │
│  │   │                 │  │                 │  │                 │     │    │
│  │   └─────────────────┘  └─────────────────┘  └─────────────────┘     │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Request flow:                                                               │
│  ┌────────┐    ┌─────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ Client │───►│ Auth/Tenant │───►│ SET search_path  │───►│    Query     │  │
│  │        │    │  Resolver   │    │  = 'tenant_acme' │    │ (schema-iso) │  │
│  └────────┘    └─────────────┘    └──────────────────┘    └──────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pros:**

- Complete isolation
- Per-tenant backup/restore
- Independent scaling

**Cons:**

- More complex migrations
- More indexes to maintain
- Harder cross-tenant queries

### Comparison

| Factor | Shared + RLS | Schema-per-Tenant |
|--------|--------------|-------------------|
| Isolation | Logical | Physical |
| Complexity | Low | Medium |
| Index overhead | 1x | Nx |
| Migrations | Single | Per-schema |
| Cross-tenant queries | Easy | Hard |
| Compliance (SOC2, etc.) | May need schema | Preferred |
| Recommended for | SaaS MVP | Enterprise/regulated |

---

## 9. Observability

### Metrics (Prometheus)

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         METRICS                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │  rag_requests_total{endpoint, status}           counter   │     │
│   │  rag_request_duration_seconds{endpoint}         histogram │     │
│   │  rag_search_latency_seconds{strategy}           histogram │     │
│   │  rag_search_results_count{strategy}             histogram │     │
│   │  rag_rerank_latency_seconds{provider}           histogram │     │
│   │  rag_embedding_latency_seconds{provider}        histogram │     │
│   │  rag_llm_latency_seconds{model}                 histogram │     │
│   │  rag_context_tokens{endpoint}                   histogram │     │
│   │  rag_cache_hits_total{cache_type}               counter   │     │
│   │  rag_cache_misses_total{cache_type}             counter   │     │
│   │  rag_documents_total{tenant}                    gauge     │     │
│   │  rag_segments_total{tenant}                     gauge     │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                      │
│   GET /metrics  →  Prometheus scrape endpoint                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Tracing (OpenTelemetry)

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       TRACING                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Request trace:                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ [chat_completions] ─────────────────────────────────────────│   │
│   │   │                                                         │   │
│   │   ├── [embed_query] ████░░░░░░░░░░░░░░░░░░░░░░░░░  50ms    │   │
│   │   │                                                         │   │
│   │   ├── [dense_search] ██████░░░░░░░░░░░░░░░░░░░░░░  80ms    │   │
│   │   │                                                         │   │
│   │   ├── [sparse_search] █████░░░░░░░░░░░░░░░░░░░░░░  70ms    │   │
│   │   │                                                         │   │
│   │   ├── [rrf_fusion] █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5ms     │   │
│   │   │                                                         │   │
│   │   ├── [rerank] ██████████░░░░░░░░░░░░░░░░░░░░░░░░  150ms   │   │
│   │   │                                                         │   │
│   │   ├── [build_context] █░░░░░░░░░░░░░░░░░░░░░░░░░░  10ms    │   │
│   │   │                                                         │   │
│   │   └── [llm_call] █████████████████████████████████  800ms  │   │
│   │                                                             │   │
│   │   Total: 1165ms                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Structured Logging

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "request_id": "abc-123",
  "tenant_id": "tenant-xyz",
  "event": "search_completed",
  "strategy": "hybrid",
  "dense_results": 10,
  "sparse_results": 8,
  "merged_results": 12,
  "duration_ms": 155
}
```

### Key Dashboards

| Dashboard | Purpose | Key Metrics |
|-----------|---------|-------------|
| Overview | System health | Request rate, error rate, p50/p95 latency |
| Retrieval | Search quality | Results count, rerank improvement, cache hit rate |
| LLM | Inference costs | Token usage, latency by model, error rate |
| Tenants | Per-tenant usage | Documents, segments, requests by tenant |

---

## 10. What's Missing

### P0 (Critical)

| Gap | Current State | Impact | Fix |
|-----|---------------|--------|-----|
| Hybrid retrieval | Vector only | -15% quality | Add BM25 + RRF |
| Reranker integration | Built, not wired | -10% quality | Wire into /chat |
| Metadata filtering | No metadata column | No filtering | Add JSONB column |

### P1 (Important)

| Gap | Current State | Impact | Fix |
|-----|---------------|--------|-----|
| Domain layer | Logic in infra | Hard to test | Extract domain/ |
| Embedding cache | None | Wasted compute | Add cache layer |
| Evaluation harness | None | Blind optimization | Add eval/ |
| Tests | None | Regression risk | Add tests/ |

### P2 (Nice to Have)

| Gap | Current State | Impact | Fix |
|-----|---------------|--------|-----|
| Pipeline abstraction | Inline code | Hard to extend | Add pipelines |
| Multi-tenancy | Single corpus | No SaaS | Add tenant_id |
| Observability | Basic logging | No metrics | Add Prometheus |

---

## 11. Migration Roadmap

### Phase 1: Retrieval Quality

**Goal:** +15-25% retrieval quality with minimal changes.

```text
Week 1-2:
┌─────────────────────────────────────────────────────────────────┐
│  1. Enable BM25 via tsvector (index already exists!)            │
│     - Add sparse_search() function                              │
│     - Add RRF fusion                                            │
│                                                                 │
│  2. Wire reranker into /v1/chat/completions                     │
│     - CrossEncoder or Cohere (already implemented)              │
│     - Config flag to enable/disable                             │
│                                                                 │
│  3. Add metadata JSONB column to segments                       │
│     - Migration script                                          │
│     - GIN index                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Architecture Cleanup

**Goal:** Proper separation of concerns.

```text
Week 3-4:
┌─────────────────────────────────────────────────────────────────┐
│  1. Extract domain/ layer                                       │
│     - Models (Document, Segment, SearchQuery, etc.)             │
│     - Ports (interfaces)                                        │
│     - Services (ContextBuilder, CitationResolver)               │
│                                                                 │
│  2. Create application/pipelines/                               │
│     - IngestionPipeline                                         │
│     - RetrievalPipeline                                         │
│                                                                 │
│  3. Move business logic from retrieval.py, store.py             │
│     - Keep only SQL in infrastructure                           │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Operability

**Goal:** Production-ready observability.

```text
Week 5-6:
┌─────────────────────────────────────────────────────────────────┐
│  1. Prometheus metrics endpoint                                 │
│     - Key counters and histograms                               │
│     - /metrics endpoint                                         │
│                                                                 │
│  2. Embedding cache                                             │
│     - In-memory LRU (dev)                                       │
│     - Redis adapter (prod)                                      │
│                                                                 │
│  3. Basic evaluation suite                                      │
│     - 10-20 golden Q&A pairs                                    │
│     - Recall@K, MRR metrics                                     │
│     - CI integration                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Scale

**Goal:** Multi-tenant, production scale.

```text
Week 7-8:
┌─────────────────────────────────────────────────────────────────┐
│  1. Multi-tenancy (tenant_id + RLS)                             │
│     - Schema migration                                          │
│     - Auth/tenant resolver                                      │
│     - RLS policies                                              │
│                                                                 │
│  2. Async batch ingestion                                       │
│     - Task queue (or simple background workers)                 │
│     - Progress tracking                                         │
│                                                                 │
│  3. Query caching                                               │
│     - Cache search results by query hash                        │
│     - TTL-based invalidation                                    │
└─────────────────────────────────────────────────────────────────┘
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
