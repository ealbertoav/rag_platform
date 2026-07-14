# RAG Platform — Specification-Driven TODO

> **Format:** Each task is a self-contained specification executable by an AI agent.
> Fields: **Goal**, **Inputs**, **Outputs**, **Files**, **Acceptance Criteria**, **Notes**.
> Status: `[ ]` pending · `[~]` in progress · `[x]` done

> **Task numbering:** Phase *N* uses task IDs **T-(N×10)** onward (Phase 0 exception: T-001–T-005). Example: Phase 18 → T-180…T-182; Phase 20 → T-200…T-202.

> **Current focus:** Phase 24 — **T-241** ← next (T-240 ✅). Phases 19–28 follow strict precondition order (see roadmap below).
>
> **Post-merge:** run `./scripts/migrate_ci_checks.sh` and update branch protection to **Quality**, **Unit Tests**, **Extended Tests**.

---

## Phase 0 — Foundation

### T-001 · Core Settings & Configuration loader
- **Status:** `[x]`
- **Goal:** Implement a Pydantic-Settings model that reads from `.env` and `configs/*.yaml`, exposing a single `settings` singleton used across the entire app.
- **Inputs:** `.env.example`, `configs/app.yaml`, `configs/llm.yaml`, `configs/embeddings.yaml`, `configs/retrieval.yaml`, `configs/parsing.yaml`, `configs/logging.yaml`
- **Outputs:** Importable `settings` object with typed fields for every config key.
- **Files:**
  - `src/core/settings.py` — `Settings(BaseSettings)` with nested models per domain
  - `src/core/constants.py` — project-wide constants (collection name, chunk metadata keys, etc.)
- **Acceptance Criteria:**
  - `from src.core.settings import settings` works in any module
  - All env vars override YAML defaults
  - Pydantic validation raises on missing required fields
  - `pytest tests/unit/test_settings.py` passes

---

### T-002 · Structured Logging
- **Status:** `[x]`
- **Goal:** Set up JSON-structured logging with OpenTelemetry trace context injection so every log line carries `trace_id` and `span_id`.
- **Inputs:** `configs/logging.yaml`, `settings.logging`
- **Outputs:** `get_logger(name)` factory usable by all modules.
- **Files:**
  - `src/core/logging.py`
- **Acceptance Criteria:**
  - Log output is valid JSON when `LOG_FORMAT=json`
  - Trace context appears in logs when a span is active
  - Works without OTEL collector running (graceful no-op)

---

### T-003 · Domain Entities
- **Status:** `[x]`
- **Goal:** Define all domain entities as Pydantic v2 models. No business logic here — pure data shapes.
- **Inputs:** Architecture spec (this file), conversation flowchart
- **Outputs:** Typed, immutable dataclasses for the domain layer.
- **Files:**
  - `src/domain/entities/document.py` — `Document(id, source, content, metadata, created_at)`
  - `src/domain/entities/chunk.py` — `Chunk(id, document_id, text, embedding, sparse_vector, metadata)`
  - `src/domain/entities/query.py` — `Query(id, text, expanded_texts, embedding)`
  - `src/domain/entities/answer.py` — `Answer(query_id, text, sources, latency_ms, token_count)`
  - `src/domain/entities/evaluation.py` — `EvalSample(question, expected_answer, retrieved_chunks, generated_answer, scores)`
- **Acceptance Criteria:**
  - All entities serialize/deserialize cleanly with `model.model_dump()` and `Model.model_validate()`
  - No circular imports
  - `pytest tests/unit/test_entities.py` passes

---

### T-004 · Repository Interfaces (Abstract Base Classes)
- **Status:** `[x]`
- **Goal:** Define the abstract repository contracts for each infrastructure concern. Infrastructure implementations must satisfy these interfaces — domain/service layer depends only on these ABCs.
- **Inputs:** T-003 entities
- **Outputs:** Python ABCs with `@abstractmethod` signatures.
- **Files:**
  - `src/domain/repositories/llm_repository.py` — `LLMRepository.generate(prompt, context) -> str`
  - `src/domain/repositories/embedding_repository.py` — `EmbeddingRepository.embed(texts) -> list[DenseVector]; embed_sparse(texts) -> list[SparseVector]`
  - `src/domain/repositories/reranker_repository.py` — `RerankerRepository.rerank(query, chunks, top_k) -> list[Chunk]`
  - `src/domain/repositories/vector_store_repository.py` — `VectorStoreRepository.upsert / search_dense / search_sparse / search_hybrid`
- **Acceptance Criteria:**
  - Importing any repository ABC raises `TypeError` if instantiated directly
  - Type signatures use entities from T-003
  - No infrastructure imports in this layer

---

### T-005 · Custom Exceptions
- **Status:** `[x]`
- **Goal:** Define the exception hierarchy so error handling is consistent across all layers.
- **Files:**
  - `src/core/exceptions.py`
- **Exception tree:**
  ```
  RAGPlatformError
  ├── IngestionError
  │   ├── DocumentLoadError
  │   └── ChunkingError
  ├── RetrievalError
  │   ├── EmbeddingError
  │   └── VectorStoreError
  ├── GenerationError
  │   └── LLMTimeoutError
  └── EvaluationError
  ```
- **Acceptance Criteria:**
  - All exceptions carry `message` and optional `cause`
  - FastAPI exception handlers can catch `RAGPlatformError` as a base

---

## Phase 1 — Ingestion Pipeline

### T-010 · Document Loaders
- **Status:** `[x]`
- **Goal:** Implement one loader per document type. Each loader takes a file path, returns a `Document` entity.
- **Files:**
  - `src/infrastructure/loaders/pdf_loader.py` — uses `pypdf`
  - `src/infrastructure/loaders/docx_loader.py` — uses `python-docx`
  - `src/infrastructure/loaders/html_loader.py` — uses `beautifulsoup4`, strips boilerplate
  - `src/infrastructure/loaders/markdown_loader.py` — uses `markdown` lib
- **Outputs:** Each loader implements a `load(path: Path) -> Document` method.
- **Acceptance Criteria:**
  - Preserves source metadata (`filename`, `page`, `section`) in `Document.metadata`
  - Handles encoding errors gracefully (UTF-8 fallback)
  - `pytest tests/unit/test_loaders.py` passes with fixture files

---

### T-011 · Chunking Strategies
- **Status:** `[x]`
- **Goal:** Implement three chunking strategies. All implement a common `Chunker` protocol: `chunk(document: Document) -> list[Chunk]`.
- **Files:**
  - `src/rag/chunking/recursive_chunker.py` — recursive character text splitter, configurable size/overlap
  - `src/rag/chunking/semantic_chunker.py` — splits on embedding cosine distance drops; use `sentence-transformers`
  - `src/rag/chunking/parent_child_chunker.py` — large parent chunks + small child chunks; store both, retrieve child, return parent context
- **Config:** `configs/retrieval.yaml` → `chunking.strategy`, `chunking.chunk_size`, `chunking.overlap`
- **Acceptance Criteria:**
  - No chunk exceeds `max_tokens` (measured by tiktoken)
  - Parent-child stores parent reference in `Chunk.metadata["parent_id"]`
  - `pytest tests/unit/test_chunking.py` passes

---

### T-012 · BGE-M3 Embedding Provider
- **Status:** `[x]`
- **Goal:** Implement `EmbeddingRepository` for BGE-M3, which produces both dense (1024-dim) and sparse (lexical) vectors in a single forward pass.
- **Files:**
  - `src/infrastructure/embeddings/bge_m3.py`
- **Dependencies:** `FlagEmbedding` library; model downloaded to `models/embeddings/bge-m3`
- **Outputs:** Implements `EmbeddingRepository` from T-004
- **Acceptance Criteria:**
  - `embed_batch(texts)` returns dense vectors; `embed_sparse_batch(texts)` returns sparse dict `{token_id: weight}`
  - Runs on MPS (`device=mps`) without error on Apple Silicon
  - Batch processing handles `batch_size` from config
  - `pytest tests/integration/test_bge_m3.py` passes (requires model present)

---

### T-013 · Qdrant Vector Store
- **Status:** `[x]`
- **Goal:** Implement `VectorStoreRepository` backed by Qdrant. Supports upsert, dense search, sparse search (via `SparseVector`), and hybrid search.
- **Files:**
  - `src/infrastructure/vectordb/qdrant.py`
- **Config:** `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY`
- **Acceptance Criteria:**
  - `upsert(chunks)` stores dense + sparse vectors + payload in one call
  - `search_hybrid(query_dense, query_sparse, alpha, top_k)` uses RRF fusion
  - Collection is auto-created if missing with correct vector config
  - `pytest tests/integration/test_qdrant.py` passes (requires running Qdrant)

---

### T-014 · BM25 Index
- **Status:** `[x]`
- **Goal:** Implement a BM25 retriever that indexes chunk texts in-memory (persisted to disk) using `rank-bm25`.
- **Files:**
  - `src/infrastructure/vectordb/bm25.py`
  - `src/rag/retrieval/bm25_retriever.py`
- **Acceptance Criteria:**
  - Index serializes/deserializes to `data/processed/bm25_index.json`
  - `search(query, top_k)` returns `list[tuple[Chunk, float]]` sorted by score
  - Supports incremental updates (re-index on new chunks)
  - `deferred_rebuild()` context defers rebuilds until exit; `rebuild()` flushes pending changes
  - `iter_chunks()` yields chunks without copying the full corpus _(T-165)_
  - `load_or_create(backend=...)` factory selects memory or disk backend _(T-165)_

---

### T-015 · Ingestion Pipeline
- **Status:** `[x]`
- **Goal:** Orchestrate the full ingestion flow: Loader → Cleaner → Chunker → Embedder → Qdrant + BM25 index.
- **Files:**
  - `src/rag/pipelines/ingestion_pipeline.py`
  - `src/domain/services/ingestion_service.py`
- **Flow:**
  ```
  file_path → Loader → Document
            → Chunker → list[Chunk]
            → BGE-M3 → chunks with dense + sparse vectors
            → Qdrant.upsert()
            → BM25.index()
            → SQLite metadata store
  ```
- **Acceptance Criteria:**
  - Idempotent: re-ingesting same file updates existing chunks (deduplicate by hash)
  - Re-ingest purges superseded chunk IDs from Qdrant + BM25 inside one `deferred_rebuild()` scope _(T-165 hardening)_
  - Hierarchical summaries (T-125) and HyPE questions (T-122) indexed in the same deferred scope when enabled _(T-165)_
  - `ingest_directory()` defers BM25 rebuild until the batch completes (single rebuild per directory)
  - Progress reported via `tqdm` or Rich
  - Errors on individual chunks logged and skipped (pipeline continues)
  - `scripts/ingest.py --source data/raw/` works end-to-end
  - `pytest tests/integration/test_ingestion_pipeline.py` passes

---

### T-016 · Rebuild Embeddings Utility
- **Status:** `[x]`
- **Goal:** Re-embed all chunks from the BM25 index using the current embedding model and sync them back into Qdrant. Used when switching embedding models or recovering a corrupted collection.
- **Files:**
  - `scripts/rebuild_embeddings.py`
- **Flags:** `--batch-size`, `--dry-run`, `--recreate-collection`
- **Acceptance Criteria:**
  - Reads source-of-truth chunks from `BM25Index` via `iter_chunks()` (memory or disk backend — T-165)
  - Embeds with `BGEM3EmbeddingProvider.embed_both()` in configurable batches
  - Upserts into Qdrant; per-batch errors logged and counted without aborting
  - `--dry-run` counts chunks without writing
  - `--recreate-collection` drops the collection first (clean re-index)
  - Exits 1 if any batch fails; exits 0 on full success

---

## Phase 2 — Retrieval Pipeline

### T-020 · Query Expansion
- **Status:** `[x]`
- **Goal:** Given a user query, use the LLM to generate N semantically diverse sub-queries, improving recall for rare/ambiguous questions.
- **Files:**
  - `src/rag/retrieval/query_expansion.py`
  - `src/prompts/retrieval/query_expansion.txt` — system + user prompt template
- **Config:** `retrieval.query_expansion.enabled`, `retrieval.query_expansion.n_variants`
- **Acceptance Criteria:**
  - Returns original query + N variants as `Query.expanded_texts`
  - Disabled by default (no LLM call when `enabled: false`)
  - Cached per query text (avoid repeated LLM calls for same query)

---

### T-021 · Dense Retriever
- **Status:** `[x]`
- **Goal:** Embed the query with BGE-M3, search Qdrant HNSW, return top-K chunks.
- **Files:**
  - `src/rag/retrieval/dense_retriever.py`
- **Acceptance Criteria:**
  - Uses `EmbeddingRepository` and `VectorStoreRepository` interfaces (no direct infra import)
  - Returns `list[tuple[Chunk, float]]` sorted by cosine similarity

---

### T-022 · Hybrid Retriever
- **Status:** `[x]`
- **Goal:** Run dense (Qdrant HNSW) + sparse (BM25) retrieval in parallel, fuse scores with RRF (Reciprocal Rank Fusion), return merged top-K.
- **Files:**
  - `src/rag/retrieval/hybrid_retriever.py`
  - `src/rag/ranking/score_fusion.py` — implements RRF and weighted linear fusion
- **Config:** `retrieval.hybrid_alpha` (0.0=BM25 only, 1.0=dense only)
- **Acceptance Criteria:**
  - Parallelizes dense + sparse calls with `asyncio.gather`
  - RRF formula: `score = Σ 1 / (k + rank_i)` with k=60
  - No duplicate chunks in output (dedup by chunk ID)
  - `pytest tests/unit/test_score_fusion.py` passes with mock data

---

### T-023 · BGE-Reranker
- **Status:** `[x]`
- **Goal:** Cross-encoder reranker that takes (query, chunk) pairs and re-scores them, keeping top-K most relevant.
- **Files:**
  - `src/infrastructure/rerankers/bge_reranker.py`
  - `src/rag/ranking/cross_encoder.py`
- **Dependencies:** `FlagEmbedding`; model at `models/rerankers/bge-reranker-v2-m3`
- **Config:** `reranker.top_k`, `reranker.batch_size`
- **Acceptance Criteria:**
  - Implements `RerankerRepository` from T-004
  - Batches pairs to avoid OOM on long chunk lists
  - Runs on MPS without error
  - `pytest tests/integration/test_reranker.py` passes

---

### T-024 · Contextual Compression
- **Status:** `[x]`
- **Goal:** Given the query and top-K chunks, extract only the sentences/passages within each chunk that are relevant to the query, reducing context tokens sent to the LLM.
- **Files:**
  - `src/rag/compression/contextual_compression.py`
  - `src/prompts/compression/extract_relevant.txt`
- **Config:** `compression.enabled`, `compression.max_tokens`
- **Acceptance Criteria:**
  - Output never exceeds `max_tokens` (checked with tiktoken)
  - Falls back to full chunk if LLM extraction fails
  - Disabled by default (can be turned off to save latency)

---

### T-025 · Retrieval Pipeline
- **Status:** `[x]`
- **Goal:** Orchestrate the full retrieval flow: Query → Expansion → Embedding → Hybrid Search → Reranking → Compression → Final Context.
- **Files:**
  - `src/rag/pipelines/retrieval_pipeline.py`
  - `src/domain/services/retrieval_service.py`
- **Flow:**
  ```
  Query
  → QueryExpansion (optional)
  → BGE-M3 embed
  → HybridRetriever (dense + BM25)
  → score_fusion → Top 50
  → CrossEncoder reranker → Top 10
  → ContextualCompression → Final Context
  ```
- **Acceptance Criteria:**
  - Each step traced with OpenTelemetry spans
  - `latency_ms` logged per step
  - Returns `list[Chunk]` + `context_str` ready for LLM

---

## Phase 3 — Generation & API

### T-030 · llama.cpp LLM Provider
- **Status:** `[x]`
- **Goal:** Implement `LLMRepository` using `llama-cpp-python`. Supports streaming and non-streaming completions.
- **Files:**
  - `src/infrastructure/llm/llama_cpp_provider.py`
- **Config:** `llm.model_path`, `llm.context_size`, `llm.n_gpu_layers`, `llm.temperature`
- **Acceptance Criteria:**
  - `generate(prompt, context, stream=False) -> str`
  - `generate_stream(prompt, context) -> AsyncIterator[str]`
  - Model loaded once at startup, not per-request
  - `n_gpu_layers=-1` offloads all layers to Metal on Apple Silicon
  - `pytest tests/integration/test_llm.py` passes with a small model

---

### T-031 · Chat Pipeline
- **Status:** `[x]`
- **Goal:** End-to-end flow from user question to streamed answer, combining retrieval pipeline + LLM generation.
- **Files:**
  - `src/rag/pipelines/chat_pipeline.py`
  - `src/domain/services/generation_service.py`
  - `src/prompts/system/rag_assistant.txt`
- **Prompt structure:**
  ```
  SYSTEM: You are a helpful assistant. Answer using ONLY the provided context...
  CONTEXT: {compressed_chunks}
  USER: {question}
  ```
- **Acceptance Criteria:**
  - `chat(question: str) -> AsyncIterator[str]` for streaming
  - `Answer.sources` lists chunk IDs used in context
  - No hallucination guard: if context empty, respond "I don't have information about this"

---

### T-032 · FastAPI Application
- **Status:** `[x]`
- **Goal:** Wire up FastAPI app with all routers, dependency injection for services, and lifespan events for model loading.
- **Files:**
  - `src/main.py` — FastAPI app, lifespan, middleware
  - `src/api/dependencies.py` — `get_ingestion_service()`, `get_retrieval_service()`, `get_generation_service()`
  - `src/api/routers/health.py` — `GET /health` → `{"status": "ok", "models_loaded": true}`
  - `src/api/routers/ingest.py` — `POST /ingest` (file upload or path)
  - `src/api/routers/chat.py` — `POST /chat` (streaming SSE response)
  - `src/api/routers/evals.py` — `POST /evals/run`
- **Acceptance Criteria:**
  - `make serve` starts server without error
  - `curl localhost:8000/health` returns 200
  - `POST /chat` streams tokens via `text/event-stream`
  - OpenAPI docs available at `/docs`

---

## Phase 4 — Evaluation Framework

### T-040 · Golden Dataset Builder
- **Status:** `[x]`
- **Goal:** Script to generate synthetic QA pairs from ingested documents using the LLM, saved to `datasets/synthetic/generated_qa.json`.
- **Files:**
  - `scripts/run_evals.py`
  - `src/prompts/evaluation/generate_qa.txt`
- **Output schema:**
  ```json
  {"question": "...", "answer": "...", "relevant_chunks": ["chunk_id_1"]}
  ```
- **Acceptance Criteria:**
  - Generates N pairs per document (configurable)
  - Deduplicates similar questions (cosine similarity threshold)
  - Human-reviewable output format

---

### T-041 · Retrieval Evals (Recall@K, Precision@K, NDCG)
- **Status:** `[x]`
- **Goal:** Implement retrieval metrics against `datasets/goldens/retrieval_dataset.json`.
- **Files:**
  - `src/evals/retrieval/recall_at_k.py`
  - `src/evals/retrieval/precision_at_k.py`
  - `src/evals/retrieval/ndcg.py`
- **Acceptance Criteria:**
  - Each metric is a pure function `metric(retrieved_ids, relevant_ids, k) -> float`
  - Runnable via `pytest tests/benchmarks/` with dataset fixtures
  - Results printed as a summary table

---

### T-042 · Generation Evals (Faithfulness, Relevance, Hallucination)
- **Status:** `[x]`
- **Goal:** LLM-as-judge metrics for generation quality using Ragas and DeepEval.
- **Files:**
  - `src/evals/generation/faithfulness.py` — wraps Ragas `faithfulness`
  - `src/evals/generation/relevance.py` — wraps Ragas `answer_relevancy`
  - `src/evals/generation/hallucination.py` — wraps DeepEval `HallucinationMetric`
- **Config:** `configs/evals.yaml`
- **Acceptance Criteria:**
  - All metrics accept `EvalSample` from T-003
  - Results > threshold pass, <= threshold fail with detailed report
  - `pytest tests/benchmarks/test_generation_evals.py` runnable in CI

---

### T-043 · End-to-End RAG Benchmark
- **Status:** `[x]`
- **Goal:** Full pipeline benchmark: given `qa_dataset.json`, run the entire RAG stack and score end-to-end.
- **Files:**
  - `src/evals/e2e/rag_benchmark.py`
  - `scripts/benchmark.py`
- **Acceptance Criteria:**
  - Runs all QA pairs through `chat_pipeline`
  - Reports Recall@5, Faithfulness, Relevance per run
  - Saves results to `data/exports/benchmark_{timestamp}.json`
  - `make benchmark` exits 0 if all metrics above threshold

---

### T-044 · EvaluationService & Live `/evals/run` Endpoint
- **Status:** `[x]`
- **Goal:** Wire `RAGBenchmark` into the API so `POST /evals/run` executes a real evaluation instead of returning a stub.
- **Files:**
  - `src/domain/services/evaluation_service.py` — orchestrates `RAGBenchmark`, loads golden QA dataset, persists report
  - `src/api/routers/evals.py` — real endpoint wired to `EvaluationService`
- **Flow:**
  ```
  POST /evals/run
    → EvaluationService.run()
    → load datasets/goldens/qa_dataset.json (skip placeholders)
    → RAGBenchmark.run(chat_pipeline, qa_pairs)
    → save data/exports/benchmark_{ts}.json
    → return {status, metrics, passed, report_path}
  ```
- **Acceptance Criteria:**
  - Returns `204` with a clear message when the QA dataset contains only placeholder rows (default state before `make evals`)
  - Returns `200` with full metric summary when real QA pairs are present
  - Thresholds configurable via `EvaluationService.__init__`
  - Placeholder rows detected and filtered (rows whose `relevant_chunks` all start with `chunk_id_`)

---

## Phase 5 — Observability

### T-050 · OpenTelemetry Tracing
- **Status:** `[x]`
- **Goal:** Instrument the retrieval and generation pipelines with OTel spans so every request shows a full trace: query → retrieval steps → LLM → response.
- **Files:**
  - `src/observability/tracing.py` — `TracerProvider` setup, `@traced` decorator
- **Acceptance Criteria:**
  - Every pipeline step wrapped in a named span
  - Span attributes include `chunk_count`, `reranker_score`, `latency_ms`, `token_count`
  - Works without collector (no-op exporter fallback)

---

### T-051 · Prometheus Metrics
- **Status:** `[x]`
- **Goal:** Expose Prometheus metrics for monitoring in production.
- **Files:**
  - `src/observability/metrics.py`
- **Metrics to expose:**
  - `rag_request_latency_seconds` (histogram, labeled by stage)
  - `rag_requests_total` (counter, labeled by status)
  - `rag_retrieval_chunk_count` (histogram)
  - `rag_llm_tokens_total` (counter)
- **Acceptance Criteria:**
  - `GET /metrics` returns Prometheus text format
  - Works with Grafana scrape config

---

## Phase 6 — CI/CD & Quality Gates

### T-060 · Pre-commit & Linting
- **Status:** `[x]`
- **Goal:** Enforce code quality gates on every commit.
- **Files:**
  - `.pre-commit-config.yaml`
- **Hooks:** `ruff check`, `ruff format`, `mypy src`
- **Acceptance Criteria:**
  - `pre-commit install` works
  - `make lint` exits 0 on clean code

---

### T-061 · GitHub Actions CI Pipeline
- **Status:** `[x]`
- **Goal:** CI pipeline that runs on every PR: lint → unit tests → retrieval eval regression check.
- **Files:**
  - `.github/workflows/ci.yml`
- **Jobs:**
  1. Lint (`ruff`, `mypy`)
  2. Unit tests (`pytest tests/unit`)
  3. Integration tests (`pytest tests/integration`) — skipped if no model present
  4. Retrieval eval regression — fail if Recall@5 drops below threshold vs baseline
- **Acceptance Criteria:**
  - Pipeline passes on a clean branch
  - PRs blocked if regression detected

---

## Phase 7 — Graph RAG & Agentic RAG (Library Code)

> **Status:** Core modules implemented (T-070, T-071) but **not wired** into the default API/runtime path. Production wiring is tracked in **Phase 11 (Priority 1)**.

### T-070 · Knowledge Graph Layer (Neo4j)
- **Status:** `[x]`
- **Goal:** Extract entity relationships from ingested documents and store in Neo4j. Add `graph_retriever.py` alongside `hybrid_retriever.py`.
- **Files:** `src/infrastructure/vectordb/neo4j.py`, `src/rag/retrieval/graph_retriever.py`
- **Note:** `HybridRetriever` already accepts an optional `graph_retriever` param (wired to `None` until T-111).

---

### T-071 · Agentic RAG
- **Status:** `[x]`
- **Goal:** Add a tool-calling agent layer that can decide when to retrieve, when to ask clarifying questions, and when to combine multiple retrievals.
- **Files:** `src/rag/pipelines/agent_pipeline.py`
- **Note:** Requires Graph RAG wiring (T-111) for multi-hop reasoning. API exposure tracked in T-114.

---

## Phase 8 — Containerization (Docker Compose)

> **Strategy:** Docker Compose from day 1. Define services thinking about Kubernetes (health checks, env vars, volumes) so the migration to Phase 9 is a straight lift. No Kubernetes yet — it adds unnecessary complexity before real users.
> Services: `api`, `worker`, `qdrant`, `ollama`, `redis`, `prometheus`, `otel-collector`.

### T-080 · Dockerfile — Backend API (multi-stage)
- **Status:** `[x]`
- **Goal:** Create a production-quality multi-stage Dockerfile for the FastAPI backend. Builder stage installs all Python deps via `uv`; runtime stage is a slim Python 3.12 image with only the app code and installed packages.
- **Inputs:** `pyproject.toml`, `uv.lock`, `src/`, `configs/`
- **Outputs:** A Docker image that starts `uvicorn src.main:app --host 0.0.0.0 --port 8000`
- **Files:**
  - `docker/Dockerfile.api`
- **Key constraints:**
  - Model files (`models/`) and data (`data/`) are **mounted as volumes** — never baked into the image (they are 16 GB+ GGUF files)
  - All config via env vars using existing `LLM__*`, `EMBEDDINGS__*`, `QDRANT__*` naming convention
  - Final image must be `< 2 GB` (no model weights included)
- **Acceptance Criteria:**
  - `docker build -f docker/Dockerfile.api -t rag-api .` completes without error
  - `docker run --env-file .env rag-api` starts the server on port 8000
  - `GET /health` returns `200` from inside the container

---

### T-081 · Dockerfile — Ingestion Worker
- **Status:** `[x]`
- **Goal:** Create a Dockerfile for the ingestion worker that shares the same base layer as T-080 but runs `scripts/ingest.py` instead of the API server.
- **Inputs:** T-080 base image, `scripts/ingest.py`, `src/`
- **Outputs:** An image runnable as a one-shot job (`docker run`) or long-running watcher
- **Files:**
  - `docker/Dockerfile.worker`
- **Acceptance Criteria:**
  - `docker build -f docker/Dockerfile.worker -t rag-worker .` completes
  - `docker run --env-file .env -v $(pwd)/data:/app/data rag-worker` ingests documents from `/app/data/raw`

---

### T-082 · docker-compose.yml — Full local stack
- **Status:** `[x]`
- **Goal:** Define the full local development stack as a single Compose file. Replaces the bare `docker run qdrant/qdrant` in the Makefile with a complete, reproducible environment.
- **Inputs:** T-080, T-081, `.env.example`, existing `make qdrant-up` command
- **Outputs:** A `docker-compose.yml` that brings up all services with a single `docker compose up`
- **Files:**
  - `docker-compose.yml` (project root)
- **Services to define:**
  ```
  api           → docker/Dockerfile.api        → port 8000
  worker        → docker/Dockerfile.worker     → no port (job)
  qdrant        → qdrant/qdrant:latest         → ports 6333, 6334
  ollama        → ollama/ollama:latest          → port 11434
  redis         → redis:7-alpine               → port 6379
  prometheus    → prom/prometheus:latest        → port 9090
  otel-collector→ otel/opentelemetry-collector  → port 4317
  ```
- **Named volumes:** `qdrant_data`, `models`, `ollama_data`, `raw_docs`
- **Health checks:** `api` waits for `qdrant` healthcheck before starting
- **Acceptance Criteria:**
  - `docker compose up -d` starts all services without error
  - `curl http://localhost:8000/health` returns `200`
  - `curl http://localhost:6333/healthz` returns `200`
  - `docker compose down -v` cleanly removes containers

---

### T-083 · .dockerignore + build hygiene
- **Status:** `[x]`
- **Goal:** Add a `.dockerignore` file to exclude heavy, unnecessary files from the Docker build context, keeping builds fast.
- **Outputs:** Sub-second Docker build context transfer
- **Files:**
  - `.dockerignore`
- **Must exclude:** `.venv/`, `models/`, `data/`, `tests/`, `.mypy_cache/`, `.ruff_cache/`, `*.gguf`, `*.pkl`, `*.bin`, `datasets/`
- **Acceptance Criteria:**
  - `docker build` context size is `< 5 MB`
  - No model weights or test fixtures appear inside the built image

---

### T-084 · Makefile targets for Docker workflow
- **Status:** `[x]`
- **Goal:** Add Docker-specific targets to the existing `Makefile` so the team can manage the full stack without remembering raw `docker compose` flags.
- **Inputs:** T-082 `docker-compose.yml`, existing Makefile targets
- **Files:**
  - `Makefile` (add targets to existing file)
- **New targets:**
  ```makefile
  docker-build   ## Build all service images
  docker-up      ## Start full stack in detached mode
  docker-down    ## Stop and remove containers (keep volumes)
  docker-logs    ## Follow logs from the api service
  docker-ingest  ## Run a one-shot ingestion job against data/raw/
  docker-clean   ## docker compose down --volumes (destroys data)
  ```
- **Acceptance Criteria:**
  - `make docker-up` starts the stack
  - `make docker-ingest SOURCE=data/raw/` processes documents
  - `make docker-down` stops gracefully

---

### T-085 · docker-compose.override.yml (development hot-reload)
- **Status:** `[x]`
- **Goal:** Create a Compose override file for local development that mounts `src/` as a live volume and enables `uvicorn --reload`, so code changes are reflected instantly without rebuilding.
- **Inputs:** T-082 base Compose file
- **Files:**
  - `docker-compose.override.yml`
- **Key differences from base:**
  - `api` mounts `./src:/app/src` and `./configs:/app/configs` as live volumes
  - `api` CMD overridden to `uvicorn src.main:app --reload --host 0.0.0.0 --port 8000`
  - `LLM__PROVIDER=ollama` (lighter than llama.cpp for dev iteration)
- **Acceptance Criteria:**
  - Editing a file in `src/` triggers an automatic reload (log line visible in `make docker-logs`)
  - Override file is automatically picked up by `docker compose up` without extra flags

---

## Phase 9 — Kubernetes & Production (EKS + Helm + Lens)

> **Strategy:** When the MVP has real users and needs autoscaling, migrate to AWS EKS. Helm charts parameterise the K8s manifests; Lens provides visual cluster management. The existing `/health` endpoint, Prometheus metrics, and env-var-driven config make this a near-zero-code migration from Phase 8.

### T-090 · Helm chart scaffold
- **Status:** `[x]`
- **Goal:** Create the Helm chart skeleton for `rag-platform`. No templates yet — just the chart metadata and a fully-documented `values.yaml` that defines all tunables for Phase 9 tasks.
- **Inputs:** `pyproject.toml` (for `appVersion`), T-082 service definitions
- **Files:**
  - `helm/rag-platform/Chart.yaml`
  - `helm/rag-platform/values.yaml`
  - `helm/rag-platform/templates/.gitkeep`
- **`values.yaml` top-level keys:** `image`, `replicaCount`, `resources`, `env`, `ingress`, `autoscaling`, `persistence`, `serviceAccount`
- **Acceptance Criteria:**
  - `helm lint helm/rag-platform` passes with no errors
  - `helm template rag-platform helm/rag-platform` renders without error (no templates yet, but chart is valid)

---

### T-091 · Deployment + Service manifests (api and worker)
- **Status:** `[x]`
- **Goal:** Create Kubernetes Deployment and Service manifests for the `api` and `worker` services, wired up to the existing `/health` endpoint for liveness/readiness probes.
- **Inputs:** T-090 chart scaffold, `GET /health` endpoint (T-032)
- **Files:**
  - `helm/rag-platform/templates/deployment-api.yaml`
  - `helm/rag-platform/templates/deployment-worker.yaml`
  - `helm/rag-platform/templates/service-api.yaml`
- **Probe config (api):**
  ```yaml
  livenessProbe:
    httpGet: { path: /health, port: 8000 }
    initialDelaySeconds: 30
    periodSeconds: 15
  readinessProbe:
    httpGet: { path: /health, port: 8000 }
    initialDelaySeconds: 10
    periodSeconds: 5
  ```
- **Acceptance Criteria:**
  - `helm template` renders valid YAML for both deployments
  - `kubectl apply --dry-run=client` passes against a local cluster (k3d or kind)

---

### T-092 · ConfigMaps and Secrets
- **Status:** `[x]`
- **Goal:** Map the existing `__`-delimited env var config system to Kubernetes ConfigMaps (non-sensitive) and Secrets (sensitive). No app code changes needed — the settings system already reads from env vars.
- **Inputs:** `.env.example`, T-090 `values.yaml`
- **Files:**
  - `helm/rag-platform/templates/configmap.yaml`
  - `helm/rag-platform/templates/secret.yaml`
- **ConfigMap keys (non-sensitive):** `LOGGING__LEVEL`, `API__HOST`, `API__PORT`, `RETRIEVAL__TOP_K_FINAL`, `RETRIEVAL__HYBRID_ALPHA`
- **Secret keys (sensitive):** `QDRANT__API_KEY` (empty for local, populated in prod)
- **Acceptance Criteria:**
  - `helm template` renders both resources
  - Deployment spec references ConfigMap via `envFrom.configMapRef` and Secret via `envFrom.secretRef`

---

### T-093 · PersistentVolumeClaims (Qdrant data + model storage)
- **Status:** `[x]`
- **Goal:** Define PVCs for the two stateful data concerns: Qdrant vector store and the GGUF model files.
- **Inputs:** T-091 deployments, EKS storage class `gp3`
- **Files:**
  - `helm/rag-platform/templates/pvc-qdrant.yaml`
  - `helm/rag-platform/templates/pvc-models.yaml`
- **Sizes (configurable via `values.yaml`):**
  - `qdrant`: 50Gi, `ReadWriteOnce`
  - `models`: 30Gi, `ReadOnlyMany` (shared across api replicas)
- **Acceptance Criteria:**
  - `helm template` renders both PVCs
  - `storageClassName` is parameterised (default `gp3`; can be overridden to `standard` for local clusters)

---

### T-094 · Horizontal Pod Autoscaler
- **Status:** `[x]`
- **Goal:** Configure HPA for the `api` deployment so it scales horizontally under load, using the existing Prometheus metrics as the signal.
- **Inputs:** T-091 `deployment-api.yaml`, metrics-server installed on cluster
- **Files:**
  - `helm/rag-platform/templates/hpa-api.yaml`
- **Config (in `values.yaml`):**
  - `autoscaling.enabled: true`
  - `autoscaling.minReplicas: 2`
  - `autoscaling.maxReplicas: 10`
  - `autoscaling.targetCPUUtilizationPercentage: 70`
- **Acceptance Criteria:**
  - `helm template` renders HPA only when `autoscaling.enabled=true`
  - `kubectl describe hpa` shows correct target after `helm install`

---

### T-095 · AWS ALB Ingress
- **Status:** `[x]`
- **Goal:** Expose the API to the internet via AWS Application Load Balancer with TLS termination. Controlled by `ingress.enabled` flag in `values.yaml` — off for local, on for prod.
- **Inputs:** T-091 `service-api.yaml`, AWS Load Balancer Controller installed on EKS
- **Files:**
  - `helm/rag-platform/templates/ingress.yaml`
- **Required annotations:**
  ```yaml
  kubernetes.io/ingress.class: alb
  alb.ingress.kubernetes.io/scheme: internet-facing
  alb.ingress.kubernetes.io/target-type: ip
  alb.ingress.kubernetes.io/certificate-arn: <ACM_ARN>  # from values.yaml
  ```
- **Acceptance Criteria:**
  - `helm template --set ingress.enabled=true` renders the Ingress resource
  - `helm template --set ingress.enabled=false` omits it entirely
  - ALB annotation keys are parameterised via `values.yaml`, not hardcoded

---

### T-096 · Resource limits and requests
- **Status:** `[x]`
- **Goal:** Set appropriate CPU/memory requests and limits for all containers so the K8s scheduler can place pods correctly on EKS node groups.
- **Inputs:** T-091 deployments, T-090 `values.yaml`
- **Files:**
  - `helm/rag-platform/values.yaml` (update `resources` section)
- **Default values:**
  ```yaml
  resources:
    api:
      requests: { cpu: "500m", memory: "2Gi" }
      limits:   { cpu: "2",    memory: "8Gi" }
    worker:
      requests: { cpu: "1",    memory: "4Gi" }
      limits:   { cpu: "4",    memory: "16Gi" }
  ```
- **Note:** For GPU inference (llama.cpp with CUDA), add `nodeSelector: { accelerator: nvidia-gpu }` and extend limits. For Apple Silicon node pools (future), use `nodeSelector: { kubernetes.io/arch: arm64 }`.
- **Acceptance Criteria:**
  - All Deployment templates reference `{{ .Values.resources.api }}` (not hardcoded values)
  - `helm template` renders resource blocks correctly

---

### T-097 · AWS EKS cluster setup guide + Lens integration
- **Status:** `[x]`
- **Goal:** Document the end-to-end steps to provision a production-ready EKS cluster, install required add-ons, deploy the Helm chart, and connect Lens for visual management.
- **Inputs:** T-090–T-096 Helm chart, AWS CLI, eksctl, existing Terraform familiarity
- **Files:**
  - `infra/eks/README.md`
- **Sections to cover:**
  1. **Cluster provisioning** via `eksctl` (faster than Terraform for first cluster; can be imported into Terraform later)
     ```bash
     eksctl create cluster --name rag-platform-prod \
       --region us-east-1 --nodegroup-name standard \
       --node-type m7g.2xlarge --nodes 3 --nodes-min 2 --nodes-max 10
     ```
  2. **Add-on installation**: AWS Load Balancer Controller, metrics-server, EBS CSI driver (for gp3 PVCs)
  3. **Helm deploy**:
     ```bash
     helm install rag-platform helm/rag-platform \
       --namespace rag-platform --create-namespace \
       --set ingress.enabled=true \
       --set ingress.certificateArn=<ACM_ARN>
     ```
  4. **Lens setup**: Import kubeconfig (`~/.kube/config`); Lens auto-discovers clusters. Navigate to Workloads → Deployments to see `api` and `worker`. Use Lens terminal for `kubectl exec` into pods.
  5. **Teardown**: `eksctl delete cluster --name rag-platform-prod`
- **Acceptance Criteria:**
  - A developer with AWS credentials can follow the guide from zero to a running cluster in one session
  - Lens connection steps are explicit (not just "add kubeconfig")

---

## Phase 10 — Embedding Provider Expansion (API + Self-Hosted Switching)

> **Motivation:** The current platform supports only self-hosted embedding models. This phase adds four API-based providers (OpenAI, Voyage AI, Cohere, Gemini), a Redis embedding cache to control API costs, and embedding model versioning in Qdrant payload to prevent silent vector corruption when switching providers.
>
> **Key risk addressed:** Vectors from different embedding models cannot be mixed in the same Qdrant collection. Without versioning, switching providers silently corrupts search results.

---

### T-100 · Embedding Settings Expansion
- **Status:** `[x]`
- **Goal:** Extend `EmbeddingSettings` to support API-based providers and a Redis embedding cache. No infrastructure code yet — just the settings model.
- **Files:**
  - `src/core/settings.py` — extend `EmbeddingSettings`
  - `configs/embeddings.yaml` — add API provider sections and cache block
  - `.env.example` — add `OPENAI_API_KEY`, `VOYAGE_API_KEY`, `COHERE_API_KEY`, `GEMINI_API_KEY`
- **Changes to `EmbeddingSettings`:**
  ```python
  provider: Literal[
      "bge_m3", "nomic", "qwen_embedding",   # existing
      "openai", "voyage", "cohere", "gemini"  # new
  ] = "bge_m3"

  # per-provider config blocks (all optional — only needed when that provider is active)
  openai: OpenAIEmbeddingConfig | None = None
  voyage: VoyageEmbeddingConfig | None = None
  cohere: CohereEmbeddingConfig | None = None
  gemini: GeminiEmbeddingConfig | None = None
  cache: EmbeddingCacheSettings = EmbeddingCacheSettings()
  ```
- **New nested models:**
  - `OpenAIEmbeddingConfig(api_key, model, dimensions)` — model default `text-embedding-3-large`, dims `3072`
  - `VoyageEmbeddingConfig(api_key, model, dimensions)` — model default `voyage-large-2`, dims `1536`
  - `CohereEmbeddingConfig(api_key, model, dimensions)` — model default `embed-english-v3.0`, dims `1024`
  - `GeminiEmbeddingConfig(api_key, model, dimensions)` — model default `text-embedding-004`, dims `768`
  - `EmbeddingCacheSettings(enabled: bool = True, ttl_seconds: int = 604800)`
- **Acceptance Criteria:**
  - `from src.core.settings import settings` still works with no `.env` changes (all new fields optional)
  - `EMBEDDINGS__PROVIDER=openai OPENAI_API_KEY=sk-...` correctly populates settings
  - `pytest tests/unit/test_settings.py` passes

---

### T-101 · OpenAI Embedding Provider
- **Status:** `[x]`
- **Goal:** Implement `EmbeddingRepository` for OpenAI's embedding API. Dense only — sparse falls back to BM25 (returns `{}`).
- **Files:**
  - `src/infrastructure/embeddings/openai_provider.py`
- **Dependencies:** `openai>=1.0.0` (add to `pyproject.toml`)
- **Supported models:** `text-embedding-3-large` (3072-dim), `text-embedding-3-small` (1536-dim), `text-embedding-ada-002` (1536-dim)
- **Key details:**
  - `text-embedding-3` family supports dimension truncation via `dimensions` param — wire to `settings.embeddings.openai.dimensions`
  - Batch texts into chunks of 2048 items (OpenAI limit)
  - Retry on HTTP 429 with exponential backoff (max 5 retries)
  - `embed_sparse()` always returns `[{} for _ in texts]`
- **Acceptance Criteria:**
  - Implements `EmbeddingRepository` from `src/domain/repositories/embedding_repository.py`
  - Unit tests mock `openai.OpenAI` — no real API calls in CI
  - `pytest tests/unit/test_openai_embedding.py` passes

---

### T-102 · Voyage AI Embedding Provider
- **Status:** `[x]`
- **Goal:** Implement `EmbeddingRepository` for Voyage AI's embedding API. Dense only.
- **Files:**
  - `src/infrastructure/embeddings/voyage_provider.py`
- **Dependencies:** `voyageai>=0.3.0` (add to `pyproject.toml`)
- **Supported models:** `voyage-large-2` (1536-dim), `voyage-code-2` (1536-dim, optimized for code/technical docs)
- **Key details:**
  - Max 128 texts per batch (Voyage limit)
  - Retry on HTTP 429 with exponential backoff
  - `embed_sparse()` returns `[{} for _ in texts]`
- **Acceptance Criteria:**
  - Implements `EmbeddingRepository`
  - Unit tests mock `voyageai.Client`
  - `pytest tests/unit/test_voyage_embedding.py` passes

---

### T-103 · Cohere Embedding Provider
- **Status:** `[x]`
- **Goal:** Implement `EmbeddingRepository` for Cohere's embedding API. Dense only. Notable: Cohere requires an `input_type` flag (`search_document` vs `search_query`).
- **Files:**
  - `src/infrastructure/embeddings/cohere_provider.py`
- **Dependencies:** `cohere>=7.0.0` (add to `pyproject.toml`; `ClientV2` with `embedding_types` requires v7+)
- **Supported models:** `embed-english-v3.0` (1024-dim), `embed-multilingual-v3.0` (1024-dim)
- **Key details:**
  - `embed(texts)` uses `input_type="search_document"` (for ingestion)
  - Override `embed_query(text)` to use `input_type="search_query"` (for retrieval queries)
  - Max 96 texts per batch
  - Retry on 429
- **Acceptance Criteria:**
  - Implements `EmbeddingRepository`
  - `input_type` is correctly set for document vs query calls
  - Unit tests mock `cohere.Client`
  - `pytest tests/unit/test_cohere_embedding.py` passes

---

### T-104 · Gemini Embedding Provider
- **Status:** `[x]`
- **Goal:** Implement `EmbeddingRepository` for Google Gemini's embedding API. Dense only (768-dim).
- **Files:**
  - `src/infrastructure/embeddings/gemini_provider.py`
- **Dependencies:** `google-generativeai>=0.5.0` (add to `pyproject.toml`)
- **Supported models:** `text-embedding-004` (768-dim)
- **Key details:**
  - `task_type="RETRIEVAL_DOCUMENT"` for ingestion, `task_type="RETRIEVAL_QUERY"` for query embedding
  - Max 100 texts per batch
  - `embed_sparse()` returns `[{} for _ in texts]`
- **Acceptance Criteria:**
  - Implements `EmbeddingRepository`
  - Unit tests mock `google.generativeai`
  - `pytest tests/unit/test_gemini_embedding.py` passes

---

### T-105 · Embedding Model Versioning (Qdrant Payload)
- **Status:** `[x]`
- **Goal:** Track which embedding model generated each vector by storing `embedding_model_name` and `embedding_model_version` in each chunk's Qdrant payload. Detect model mismatch on startup to prevent silent vector corruption.
- **Files:**
  - `src/infrastructure/vectordb/qdrant.py` — modify `upsert()` and add `_validate_embedding_model()`
  - `src/domain/entities/chunk.py` — add optional `embedding_model: str | None = None` field
- **Changes to `upsert()`:**
  ```python
  # Add to every point's payload:
  payload["embedding_model_name"] = settings.embeddings.provider
  payload["embedding_model_version"] = self._get_model_version()
  ```
- **New `_validate_embedding_model()` method:**
  - Sample a few existing points from the collection
  - If `embedding_model_name` in payload differs from `settings.embeddings.provider`, raise `VectorStoreError` with message: `"Embedding model mismatch: collection was built with '{existing}' but current config is '{current}'. Run rebuild_embeddings.py --recreate-collection to re-index."`
  - Called during `QdrantVectorStore.__init__` (after collection auto-create check)
  - Skip validation if collection is empty or has no `embedding_model_name` in payload (legacy data)
- **Acceptance Criteria:**
  - Switching `EMBEDDINGS__PROVIDER` with an existing non-empty collection raises `VectorStoreError` at startup
  - `rebuild_embeddings.py --recreate-collection` succeeds after the error
  - `pytest tests/unit/test_qdrant_versioning.py` passes (mock Qdrant client)

---

### T-106 · Redis Embedding Cache
- **Status:** `[x]`
- **Goal:** Implement a transparent caching layer for any `EmbeddingRepository`. Caches dense vectors in Redis to avoid redundant API calls (and costs). Uses the decorator pattern — wraps any provider without modifying it.
- **Files:**
  - `src/infrastructure/embeddings/cached_embedding_provider.py`
- **Dependencies:** `redis>=5.0.0` (already in `pyproject.toml` for the existing Redis service); `src/core/settings.py` `RedisSettings` (already exists)
- **Cache key:** `sha256(text + "|" + model_name + "|" + model_version)` → hex string
- **Storage:** Redis hash or string per key; value = JSON-serialized `list[float]`
- **TTL:** `settings.embeddings.cache.ttl_seconds` (default 7 days = 604800 s)
- **Interface:**
  ```python
  class CachedEmbeddingProvider(EmbeddingRepository):
      def __init__(self, inner: EmbeddingRepository, redis_client: Redis, ttl: int): ...
  ```
- **Behavior:**
  - `embed(texts)`: for each text, check cache; call `inner.embed()` only for misses; populate cache on miss
  - `embed_sparse(texts)`: pass through to inner (sparse vectors are not cached — they are BM25-based or cheap)
  - `embed_both(texts)`: cache dense part; call inner for misses; combine
  - Log cache hit/miss count per batch at DEBUG level
  - Prometheus counter: `rag_embedding_cache_hits_total`, `rag_embedding_cache_misses_total`
- **Acceptance Criteria:**
  - Second call with same texts returns from cache without calling inner provider
  - TTL is set correctly (verify with Redis `TTL` command in tests)
  - Provider works correctly when Redis is unavailable (log warning, fall through to inner)
  - `pytest tests/unit/test_cached_embedding_provider.py` passes (mock Redis)

---

### T-107 · Factory & Config Wiring
- **Status:** `[x]`
- **Goal:** Extend `get_embedding_provider()` factory to instantiate all new providers (T-101–T-104) and optionally wrap with `CachedEmbeddingProvider` (T-106). Single entry point — no other code needs to know which provider is active.
- **Files:**
  - `src/infrastructure/embeddings/__init__.py`
- **Logic:**
  ```python
  def get_embedding_provider(settings: EmbeddingSettings) -> EmbeddingRepository:
      provider = _create_provider(settings)           # routes by settings.provider
      if settings.cache.enabled:
          provider = CachedEmbeddingProvider(provider, redis_client, settings.cache.ttl_seconds)
      return provider
  ```
- **Error handling:** If an API provider is selected but its `api_key` is `None`, raise `ConfigurationError` with message: `"Provider '{name}' requires an API key. Set {ENV_VAR} in your environment."`
- **Acceptance Criteria:**
  - `get_embedding_provider(settings)` returns the correct type for all 7 providers
  - `ConfigurationError` raised when API key is missing
  - Cache is applied when `settings.cache.enabled = True`
  - `pytest tests/unit/test_embedding_factory.py` passes

---

### T-108 · Rebuild Embeddings — Multi-Provider Hardening
- **Status:** `[x]`
- **Goal:** Extend `scripts/rebuild_embeddings.py` to work correctly with API providers and to catch dimension/model mismatches before they corrupt the collection.
- **Files:**
  - `scripts/rebuild_embeddings.py`
- **New pre-flight checks (run before any embedding):**
  1. If provider is API-based, verify API key is set → abort with clear message if missing
  2. If `--recreate-collection` is NOT passed: call `_validate_embedding_model()` (T-105); if mismatch detected, print error and exit 1 with hint to use `--recreate-collection`
  3. Verify `settings.embeddings.dense_dim` matches the provider's documented output dimension → warn if mismatch
- **API-aware batching:** For API providers, reduce default batch size to 32 (OpenAI/Voyage limits) and add per-batch sleep of 0.1s to stay under rate limits. Keep existing batch_size flag.
- **Acceptance Criteria:**
  - `--dry-run` with API provider prints provider name and estimated API call count
  - Running with wrong provider and existing collection exits 1 with model mismatch message
  - Running with `--recreate-collection` after mismatch succeeds

---

### T-109 · Embedding Provider Comparison Script
- **Status:** `[x]`
- **Goal:** Script to benchmark multiple embedding providers against the golden QA dataset and produce a side-by-side quality + cost comparison table. Mirrors `compare_models.py` but for embedding providers.
- **Files:**
  - `scripts/compare_embedding_providers.py`
- **Usage:**
  ```bash
  uv run python scripts/compare_embedding_providers.py \
    --providers bge_m3 openai voyage \
    --max-samples 50
  ```
- **Output table:**
  ```
  ┌───────────────────┬──────────┬──────────┬──────────┬─────────────┬───────────┐
  │ Provider          │ Recall@5 │ NDCG@5   │ Latency  │ Cost/1K tok │ Status    │
  ├───────────────────┼──────────┼──────────┼──────────┼─────────────┼───────────┤
  │ bge_m3 (local)    │  0.843   │  0.871   │  18 ms   │  $0.00      │ PASS ✓    │
  │ openai-3-large    │  0.861   │  0.889   │  210 ms  │  $0.13      │ PASS ✓    │
  │ voyage-large-2    │  0.878   │  0.902   │  185 ms  │  $0.12      │ PASS ✓    │
  └───────────────────┴──────────┴──────────┴──────────┴─────────────┴───────────┘
  ```
- **Flow:** For each provider: load config → embed golden queries → retrieve → compute Recall@5 + NDCG@5 → record latency + estimated cost
- **Acceptance Criteria:**
  - Runs with `--providers bge_m3` (self-hosted only, no API key needed) to verify mechanics
  - Results saved to `data/exports/embedding_comparison_{timestamp}.json`
  - Skips API providers gracefully if API key not set (prints warning, continues)

---

## Phase 11 — Wire Existing Code (Priority 1)

> **Motivation:** Several high-value modules are implemented but not connected to the default runtime path. This phase closes the gap between library code and production behavior — inspired by RAG_Techniques patterns already partially present (fusion retrieval, query transformations, graph RAG, agentic RAG).
>
> **Reference repo:** `/Users/eduardo.albornoz/Projects/Personal/Self Training/RAG_Techniques`
>
> **Depends on:** Phases 1–3 (ingestion, retrieval, API), Phase 7 (T-070, T-071 library code)

---

### T-110 · Multi-Query Retrieval Fusion
- **Status:** `[x]`
- **Goal:** Use `Query.expanded_texts` variants in retrieval, not just the original query. Run hybrid retrieval for each query variant and fuse results with RRF — matching RAG_Techniques **query transformations** and **MemoRAG multi-query retrieval**.
- **Inputs:** T-020 (`QueryExpander`), T-022 (`HybridRetriever`, `rrf_fuse`), T-025 (`RetrievalService`)
- **Outputs:** Retrieval pipeline that searches with `[query.text] + query.expanded_texts` and returns deduplicated, fused top-K chunks.
- **Files:**
  - `src/domain/services/retrieval_service.py` — iterate variants, fuse with RRF
  - `src/rag/retrieval/hybrid_retriever.py` — optional `retrieve_multi()` helper (or keep logic in service)
  - `tests/unit/test_retrieval_service.py` — multi-query fusion cases
  - `tests/integration/test_retrieval_pipeline.py` — end-to-end with mocked expander
- **Flow:**
  ```
  Query
  → QueryExpander → expanded_texts populated
  → For each variant in [query.text] + expanded_texts:
      → embed variant
      → HybridRetriever.retrieve(variant_query, top_k)
  → rrf_fuse(all result lists) → dedup by chunk ID → top_k_retrieval
  → Reranker → Compressor → context
  ```
- **Config:** Reuse `query_expansion.enabled`, `query_expansion.n_variants`; no new keys required.
- **Acceptance Criteria:**
  - When `query_expansion.enabled=true`, retrieval runs at least once per variant (verified via mock call count)
  - Fused output contains no duplicate chunk IDs
  - When `query_expansion.enabled=false`, behavior is identical to current single-query path
  - OTel span `retrieval.multi_query_fusion` records variant count and fused chunk count
  - `pytest tests/unit/test_retrieval_service.py` passes

---

### T-111 · Graph RAG Production Wiring
- **Status:** `[x]`
- **Goal:** Wire `GraphRetriever` into `RetrievalPipeline.from_settings()` so the default hybrid path includes graph retrieval when Neo4j is configured — inspired by RAG_Techniques `graph_rag.py`.
- **Inputs:** T-070 (`GraphRetriever`), T-022 (`HybridRetriever.graph_retriever` param), T-112 (Neo4j settings)
- **Outputs:** `HybridRetriever` instantiated with `graph_retriever=GraphRetriever(...)` when enabled.
- **Files:**
  - `src/rag/pipelines/retrieval_pipeline.py` — conditional graph wiring in `from_settings()`
  - `src/rag/pipelines/chat_pipeline.py` — ensure graph-enabled retrieval propagates
  - `tests/unit/test_retrieval_pipeline.py` — graph on/off factory tests
- **Config:** `neo4j.enabled: false` (default off; graceful degradation when disabled)
- **Acceptance Criteria:**
  - When `neo4j.enabled=false`, `HybridRetriever.graph` is `None` (current behavior preserved)
  - When `neo4j.enabled=true` and Neo4j is reachable, graph results participate in RRF fusion
  - When Neo4j is unreachable, pipeline logs warning and continues with dense + BM25 only
  - `pytest tests/unit/test_graph_rag.py` passes

---

### T-112 · Neo4j Settings & Configuration
- **Status:** `[x]`
- **Goal:** Add typed `Neo4jSettings` to the settings model. Currently `Neo4jGraphRepository.from_settings()` uses `getattr(settings, "neo4j", None)` with hardcoded defaults — make configuration explicit and env-overridable.
- **Inputs:** T-001 (`Settings`), `.env.example`, T-070 (`Neo4jGraphRepository`)
- **Outputs:** `settings.neo4j` with URI, credentials, database name, and enable flag.
- **Files:**
  - `src/core/settings.py` — add `Neo4jSettings` nested model
  - `configs/retrieval.yaml` — add `neo4j:` block (or `configs/neo4j.yaml`)
  - `.env.example` — add `NEO4J__URI`, `NEO4J__USER`, `NEO4J__PASSWORD`, `NEO4J__ENABLED`
  - `src/infrastructure/vectordb/neo4j.py` — read from `settings.neo4j` (remove `getattr` fallback)
  - `tests/unit/test_settings.py` — Neo4j settings validation
- **Config schema:**
  ```yaml
  neo4j:
    enabled: false
    uri: bolt://localhost:7687
    user: neo4j
    password: ""          # required when enabled=true
    database: neo4j
    max_hops: 2           # graph traversal depth
  ```
- **Acceptance Criteria:**
  - `NEO4J__ENABLED=true` with missing password raises Pydantic validation error
  - `from src.core.settings import settings` works with no Neo4j env vars (defaults to disabled)
  - `pytest tests/unit/test_settings.py` passes

---

### T-113 · Graph Entity Extraction During Ingestion
- **Status:** `[x]`
- **Goal:** Populate Neo4j during ingestion so graph retrieval has data at query time — inspired by RAG_Techniques `graph_rag.py` entity/relationship extraction.
- **Inputs:** T-015 (`IngestionPipeline`), T-070 (`EntityExtractor`, `Neo4jGraphRepository`), T-112
- **Outputs:** Ingestion optionally extracts entities/relationships per document and upserts to Neo4j.
- **Files:**
  - `src/rag/pipelines/ingestion_pipeline.py` — call entity extraction after chunking
  - `src/domain/services/ingestion_service.py` — optional graph enrichment step
  - `src/rag/retrieval/graph_retriever.py` — ensure `EntityExtractor` is reusable from ingestion
  - `tests/integration/test_ingestion_pipeline.py` — graph extraction with mocked Neo4j
- **Flow:**
  ```
  Document → Chunker → Embed → Qdrant + BM25
                       ↓ (if neo4j.enabled)
              EntityExtractor → Neo4jGraphRepository.upsert_triplets()
  ```
- **Acceptance Criteria:**
  - When `neo4j.enabled=false`, ingestion path unchanged (no LLM/Neo4j calls)
  - When enabled, entities and relationships from each document appear in Neo4j
  - Entity extraction failure on one document logs warning and continues pipeline
  - Re-ingesting same document updates (not duplicates) graph nodes by document ID

---

### T-114 · Agentic RAG API Endpoint
- **Status:** `[x]`
- **Goal:** Expose `AgentPipeline` via FastAPI so clients can opt into multi-step retrieval — inspired by RAG_Techniques `Agentic_RAG.ipynb`, `self_rag.py`, and `crag.py`.
- **Inputs:** T-071 (`AgentPipeline`), T-032 (FastAPI app), T-111 (graph wiring for `GRAPH_LOOKUP` action)
- **Outputs:** New endpoint(s) for agentic chat with streaming and full-response modes.
- **Files:**
  - `src/api/routers/chat.py` — add `POST /chat/agent` and `POST /chat/agent/full`
  - `src/api/dependencies.py` — `get_agent_pipeline()` factory
  - `src/main.py` — mount agent pipeline in lifespan / app.state
  - `src/api/schemas/chat.py` — request/response models (if not inline)
  - `tests/unit/test_agent_pipeline.py` — existing tests remain green
  - `tests/integration/test_chat_agent.py` — new endpoint smoke tests
- **API contract:**
  ```
  POST /chat/agent
    Body: { "question": "...", "max_iterations": 3 }
    Response: text/event-stream (SSE tokens)

  POST /chat/agent/full
    Body: { "question": "...", "max_iterations": 3 }
    Response: { "answer": "...", "sources": [...], "iterations": 2, "actions": [...] }
  ```
- **Acceptance Criteria:**
  - `POST /chat` behavior unchanged (standard `ChatPipeline`)
  - Agent endpoint supports `RETRIEVE_MORE`, `GRAPH_LOOKUP`, `CLARIFY`, `ANSWER` actions
  - `max_iterations` capped at 5 regardless of client value
  - OpenAPI docs list both standard and agent endpoints
  - Integration test mocks LLM and verifies at least one agent iteration

---

### T-115 · Config Drift Resolution
- **Status:** `[x]`
- **Goal:** Align configuration keys with actual runtime behavior. Several settings are defined but unused, causing operator confusion.
- **Inputs:** T-001 (`Settings`), T-025 (`RetrievalService`), `configs/retrieval.yaml`
- **Outputs:** Every retrieval config key affects runtime behavior or is removed.
- **Files:**
  - `src/domain/services/retrieval_service.py` — wire `top_k_final` after reranking
  - `src/rag/pipelines/retrieval_pipeline.py` — pass `top_k_final` from settings
  - `src/rag/retrieval/hybrid_retriever.py` — document RRF vs `hybrid_alpha`; optionally implement weighted linear fusion as alternative strategy
  - `src/rag/ranking/score_fusion.py` — expose fusion mode selector if implementing alpha-weighted path
  - `configs/retrieval.yaml` — add comments clarifying each key's effect
  - `tests/unit/test_settings.py`, `tests/unit/test_retrieval_service.py`
- **Drift items to resolve:**
  | Key | Current state | Target behavior |
  |-----|---------------|-----------------|
  | `retrieval.top_k_final` | Defined, unused | Cap final chunks after rerank/compression (default: 5) |
  | `retrieval.hybrid_alpha` | Stored, RRF ignores | Either implement weighted fusion mode OR rename/document as legacy |
  | `reranker.top_k` vs `top_k_final` | Both exist | Reranker selects top N; `top_k_final` trims after compression |
- **Acceptance Criteria:**
  - `top_k_final=5` limits chunks passed to generation (verified in unit test)
  - README and `configs/retrieval.yaml` document fusion mode (RRF default)
  - No silent no-op config keys remain in `RetrievalSettings`

---

### T-116 · Idempotent Re-Ingest by Content Hash
- **Status:** `[x]`
- **Goal:** Complete the idempotent ingestion spec from T-015. `content_hash` is computed but deduplication is not enforced — re-ingesting identical files should skip or update, not duplicate.
- **Inputs:** T-015 (`IngestionPipeline`, `IngestionResult.skipped`), T-003 (`Chunk`, `Document` metadata)
- **Outputs:** Ingestion returns `skipped=True` for unchanged documents; updates chunks when content changes.
- **Files:**
  - `src/domain/services/ingestion_service.py` — hash comparison logic
  - `src/rag/pipelines/ingestion_pipeline.py` — skip/update branch
  - `src/infrastructure/vectordb/qdrant.py` — delete stale chunks by document ID before re-upsert
  - `src/infrastructure/vectordb/bm25.py` — remove old chunks for document before re-index
  - `tests/unit/test_ingestion_service.py` — skip on same hash, update on changed hash
- **Hash strategy:** `sha256(normalized_text + source_path)` stored in `Document.metadata["content_hash"]` and Qdrant payload.
- **Acceptance Criteria:**
  - Re-ingesting identical file → `IngestionResult.skipped=True`, zero new Qdrant upserts
  - Re-ingesting modified file → old chunks removed, new chunks upserted
  - `scripts/ingest.py` logs "skipped (unchanged)" per file
  - `pytest tests/unit/test_ingestion_service.py` passes

---

### T-117 · SQLite Metadata Store
- **Status:** `[x]`
- **Goal:** Implement the metadata store referenced in T-015 flow diagram. Track document ingestion history, content hashes, chunk counts, and timestamps for operational visibility and dedup support.
- **Inputs:** T-015, T-116 (content hash), `aiosqlite` (already in dependencies)
- **Outputs:** Persistent SQLite DB at `data/processed/metadata.db` with document and ingestion run records.
- **Files:**
  - `src/infrastructure/metadata/sqlite_store.py` — CRUD for documents and ingestion runs
  - `src/domain/repositories/metadata_repository.py` — ABC interface
  - `src/rag/pipelines/ingestion_pipeline.py` — write metadata after each ingest
  - `scripts/ingest.py` — `--list` flag to show ingested documents
  - `tests/unit/test_sqlite_metadata.py`
- **Schema:**
  ```sql
  documents(id, source_path, content_hash, chunk_count, ingested_at, updated_at)
  ingestion_runs(id, document_id, status, chunks_added, chunks_skipped, duration_ms, error)
  ```
- **Acceptance Criteria:**
  - Every successful ingest creates/updates a row in `documents`
  - T-116 dedup reads hash from SQLite (not only Qdrant payload)
  - `scripts/ingest.py --list` prints ingested files with timestamps
  - Works without SQLite file present (auto-creates on first ingest)

---

## Phase 12 — Index-Time Enrichment (Priority 2)

> **Motivation:** Improve recall and context quality at indexing time — inspired by RAG_Techniques **contextual chunk headers**, **document augmentation**, **HyPE**, **RSE**, and **hierarchical indices**.
>
> **Reference techniques:**
> - `contextual_chunk_headers.ipynb`
> - `document_augmentation.py`
> - `HyPE_Hypothetical_Prompt_Embeddings.py`
> - `relevant_segment_extraction.ipynb`
> - `context_enrichment_window_around_chunk.py`
> - `hierarchical_indices.py`
>
> **Depends on:** Phase 11 (T-115 config alignment), Phase 1 ingestion pipeline

---

### T-120 · Contextual Chunk Headers (CCH)
- **Status:** `[x]`
- **Goal:** Prepend document title, section, and page metadata to each chunk before embedding — inspired by RAG_Techniques `contextual_chunk_headers.ipynb`. Low cost, often large recall gain.
- **Inputs:** T-010 (loaders preserve metadata), T-011 (chunkers), T-012 (embedding)
- **Outputs:** Chunks embedded with contextual header prefix; header excluded from LLM context optionally.
- **Files:**
  - `src/rag/chunking/contextual_headers.py` — `prepend_headers(document, chunk) -> str`
  - `src/rag/chunking/__init__.py` — wrap any chunker with CCH decorator
  - `src/prompts/ingestion/chunk_header_template.txt` — header format template
  - `configs/retrieval.yaml` — add `chunking.contextual_headers.enabled: false`
  - `tests/unit/test_contextual_headers.py`
- **Header format example:**
  ```
  [Document: Annual Report 2023 | Section: Revenue | Page: 42]
  {chunk_text}
  ```
- **Acceptance Criteria:**
  - Disabled by default; no behavior change when `enabled=false`
  - Embedded text includes header; `Chunk.metadata["raw_text"]` preserves text without header for display
  - Headers derived from loader metadata (`filename`, `section`, `page`)
  - `pytest tests/unit/test_contextual_headers.py` passes

---

### T-121 · Document Augmentation (Synthetic Questions)
- **Status:** `[x]`
- **Goal:** At ingest time, generate N synthetic questions per chunk and store them as additional indexable content — inspired by RAG_Techniques `document_augmentation.py`.
- **Inputs:** T-015 (ingestion), T-030 (LLM), T-013 (Qdrant upsert)
- **Outputs:** Each chunk may have companion "question chunks" indexed alongside the source chunk.
- **Files:**
  - `src/rag/enrichment/document_augmentation.py` — `generate_questions(chunk, llm) -> list[str]`
  - `src/prompts/ingestion/generate_chunk_questions.txt`
  - `src/rag/pipelines/ingestion_pipeline.py` — optional augmentation step
  - `configs/retrieval.yaml` — add `chunking.augmentation.enabled`, `chunking.augmentation.n_questions`
  - `tests/unit/test_document_augmentation.py`
- **Index strategy:** Store question text as separate Qdrant points with `metadata["type"]="synthetic_question"` and `metadata["source_chunk_id"]`.
- **Acceptance Criteria:**
  - Disabled by default (no extra LLM calls during ingest)
  - When enabled, each chunk produces up to N questions indexed in Qdrant + BM25
  - Retrieval returns source chunk (not question chunk) via `source_chunk_id` resolution
  - Augmentation failure on one chunk logs warning and continues

---

### T-122 · HyPE — Hypothetical Prompt Embeddings
- **Status:** `[x]`
- **Goal:** Precompute hypothetical questions per chunk at index time and embed them for question-question matching at query time — inspired by RAG_Techniques `HyPE_Hypothetical_Prompt_Embeddings.py`. Strong for FAQ-style corpora.
- **Inputs:** T-121 (question generation — reuse or extend), T-012 (embedding), T-021 (dense retrieval)
- **Outputs:** HyPE index alongside standard chunk index; retrieval mode selectable via config.
- **Files:**
  - `src/rag/enrichment/hype_indexer.py` — build HyPE vectors per chunk
  - `src/rag/retrieval/hype_retriever.py` — embed query, search HyPE index, resolve to source chunks
  - `src/rag/retrieval/hybrid_retriever.py` — optional fourth RRF source: HyPE results
  - `configs/retrieval.yaml` — add `retrieval.hype.enabled: false`
  - `tests/unit/test_hype_retriever.py`
- **Flow:**
  ```
  Ingest: chunk → generate hypothetical questions → embed questions → store in Qdrant (hype collection or typed payload)
  Query:  question → embed → search hype vectors → map to source chunks → fuse via RRF
  ```
- **Acceptance Criteria:**
  - HyPE disabled by default; zero overhead when off
  - When enabled, HyPE results participate in RRF fusion with dense + BM25 (+ graph)
  - Benchmark script can compare HyPE-on vs HyPE-off (feeds T-150)

---

### T-123 · Relevant Segment Extraction (RSE)
- **Status:** `[x]`
- **Goal:** After retrieval, merge adjacent relevant chunks into longer coherent segments — inspired by RAG_Techniques `relevant_segment_extraction.ipynb`. Complements `ParentChildChunker`.
- **Inputs:** T-025 (retrieval pipeline), T-011 (`parent_child_chunker.py`)
- **Outputs:** Post-retrieval step that expands retrieved child chunks into merged parent segments.
- **Files:**
  - `src/rag/enrichment/relevant_segment_extraction.py` — `merge_adjacent(chunks) -> list[Chunk]`
  - `src/domain/services/retrieval_service.py` — call RSE after reranking, before compression
  - `configs/retrieval.yaml` — add `retrieval.rse.enabled: false`, `retrieval.rse.max_segment_tokens`
  - `tests/unit/test_relevant_segment_extraction.py`
- **Merge rules:**
  - Adjacent chunks from same document with consecutive `metadata["chunk_index"]` merge
  - Respect `max_segment_tokens` cap (tiktoken)
  - Never merge chunks from different documents
- **Acceptance Criteria:**
  - Disabled by default
  - When enabled, adjacent retrieved chunks merge into single segment
  - Merged segment never exceeds `max_segment_tokens`
  - OTel span `retrieval.rse` records merge count

---

### T-124 · Context Window Enhancement (Parent Context on Retrieve)
- **Status:** `[x]`
- **Goal:** When retrieving child chunks, include parent chunk text (and optional sibling context) in the context sent to the LLM — inspired by RAG_Techniques `context_enrichment_window_around_chunk.py`.
- **Inputs:** T-011 (`ParentChildChunker`), T-123 (RSE — complementary)
- **Outputs:** Retrieval resolves child → parent context before compression/generation.
- **Files:**
  - `src/rag/enrichment/parent_context_resolver.py` — lookup parent by `metadata["parent_id"]`
  - `src/infrastructure/vectordb/bm25.py` — parent chunk lookup by ID
  - `src/domain/services/retrieval_service.py` — expand context after retrieval
  - `configs/retrieval.yaml` — add `retrieval.parent_context.enabled: false`
  - `tests/unit/test_parent_context_resolver.py`
- **Acceptance Criteria:**
  - Only active when `chunking.strategy=parent_child` and `parent_context.enabled=true`
  - Retrieved child chunks replaced/enriched with parent text for LLM context
  - `Answer.sources` still references original retrieved child chunk IDs
  - Falls back to child text when parent not found

---

### T-125 · Hierarchical Index Summaries
- **Status:** `[x]`
- **Goal:** Build two-tier index: document-level summary nodes + detail chunks — inspired by RAG_Techniques `hierarchical_indices.py` and `raptor.py` (lightweight variant).
- **Inputs:** T-015 (ingestion), T-030 (LLM for summary generation), T-013 (Qdrant)
- **Outputs:** Summary vectors indexed alongside detail chunks; retrieval can match summaries first then drill down.
- **Files:**
  - `src/rag/enrichment/hierarchical_indexer.py` — generate + embed document summaries
  - `src/rag/retrieval/hierarchical_retriever.py` — two-stage: summary search → detail search within matched docs
  - `src/prompts/ingestion/generate_document_summary.txt`
  - `configs/retrieval.yaml` — add `chunking.hierarchical.enabled: false`
  - `tests/unit/test_hierarchical_retriever.py`
- **Flow:**
  ```
  Ingest: document → generate summary → embed summary → store as type="summary"
          document → detail chunks → embed → store as type="detail" with document_id
  Query:  search summaries (top 3 docs) → search details within those docs → RRF fuse
  ```
- **Acceptance Criteria:**
  - Disabled by default
  - Summary points stored with `metadata["type"]="summary"`
  - Two-stage retrieval returns detail chunks, not summary text, to the LLM
  - Works with existing hybrid retriever via RRF fusion of hierarchical results

---

### T-126 · Proposition Chunking
- **Status:** `[x]`
- **Goal:** LLM extracts atomic factual propositions from document text and indexes each proposition as a separate chunk — inspired by RAG_Techniques `proposition_chunking.ipynb`. Best for dense factual corpora (policies, contracts).
- **Inputs:** T-011 (chunking protocol), T-030 (LLM), T-015 (ingestion)
- **Outputs:** New chunking strategy `proposition` available via config.
- **Files:**
  - `src/rag/chunking/proposition_chunker.py` — extract + quality-grade propositions
  - `src/prompts/ingestion/extract_propositions.txt`
  - `src/rag/chunking/__init__.py` — register `proposition` strategy
  - `configs/retrieval.yaml` — add `proposition` to strategy enum comment
  - `tests/unit/test_proposition_chunker.py`
- **Acceptance Criteria:**
  - `chunking.strategy=proposition` selects proposition chunker
  - Each proposition is a standalone factual statement
  - Low-quality propositions (LLM score below threshold) discarded
  - Ingestion latency documented in README (significantly slower than recursive)

---

## Phase 13 — Query Intelligence (Priority 3)

> **Motivation:** Improve retrieval quality at query time with advanced transformation and routing strategies — inspired by RAG_Techniques **HyDE**, **adaptive retrieval**, **query transformations**, **multi-faceted filtering**, and **dartboard retrieval**.
>
> **Reference techniques:**
> - `HyDe_Hypothetical_Document_Embedding.py`
> - `adaptive_retrieval.py`
> - `query_transformations.py`
> - `dartboard.ipynb`
>
> **Depends on:** Phase 11 (T-110 multi-query fusion), Phase 2 retrieval pipeline

---

### T-130 · HyDE — Hypothetical Document Embedding
- **Status:** `[x]`
- **Goal:** At query time, generate a hypothetical answer document, embed it, and retrieve using that embedding — inspired by RAG_Techniques `HyDe_Hypothetical_Document_Embedding.py`. Helps vague or underspecified questions.
- **Inputs:** T-021 (`DenseRetriever`), T-030 (LLM), T-110 (multi-query fusion pattern)
- **Outputs:** Optional HyDE retrieval path selectable via config; results fused with standard retrieval via RRF.
- **Files:**
  - `src/rag/retrieval/hyde_retriever.py` — `generate_hypothetical_doc(query, llm) -> str; retrieve(query) -> list[SearchResult]`
  - `src/prompts/retrieval/hyde_generate.txt`
  - `src/domain/services/retrieval_service.py` — optional HyDE branch before/alongside hybrid
  - `configs/retrieval.yaml` — add `retrieval.hyde.enabled: false`
  - `tests/unit/test_hyde_retriever.py`
- **Flow:**
  ```
  Query → LLM generates hypothetical passage → embed passage → dense search → RRF fuse with standard results
  ```
- **Acceptance Criteria:**
  - Disabled by default (no extra LLM call per query)
  - When enabled, HyDE results merged via RRF with hybrid results
  - HyDE LLM failure falls back to standard retrieval only
  - OTel span `retrieval.hyde` records hypothetical doc length

---

### T-131 · Adaptive Query Classification
- **Status:** `[x]`
- **Goal:** Classify incoming queries into categories (Factual, Analytical, Opinion, Contextual) to drive retrieval strategy selection — inspired by RAG_Techniques `adaptive_retrieval.py`.
- **Inputs:** T-030 (LLM with structured output), T-003 (`Query` entity)
- **Outputs:** `Query.metadata["category"]` populated before retrieval.
- **Files:**
  - `src/rag/retrieval/adaptive/query_classifier.py` — Pydantic structured LLM classification
  - `src/prompts/retrieval/query_classification.txt`
  - `src/domain/entities/query.py` — add optional `metadata: dict` field (if not present)
  - `configs/retrieval.yaml` — add `retrieval.adaptive.enabled: false`
  - `tests/unit/test_query_classifier.py`
- **Categories:**
  ```python
  class QueryCategory(StrEnum):
      FACTUAL = "factual"
      ANALYTICAL = "analytical"
      OPINION = "opinion"
      CONTEXTUAL = "contextual"
  ```
- **Acceptance Criteria:**
  - Classification uses structured LLM output (Pydantic model, not free-text parsing)
  - Disabled by default; no LLM call when `adaptive.enabled=false`
  - Classification result attached to Query and visible in OTel span attributes
  - Invalid/unparseable classification defaults to `FACTUAL`

---

### T-132 · Adaptive Retrieval Strategies
- **Status:** `[x]`
- **Goal:** Apply category-specific retrieval parameters — inspired by RAG_Techniques `adaptive_retrieval.py` strategy pattern.
- **Inputs:** T-131 (query classification), T-025 (retrieval service)
- **Outputs:** Strategy objects that tune k, expansion count, compression, and HyDE per query category.
- **Files:**
  - `src/rag/retrieval/adaptive/strategies.py` — `BaseRetrievalStrategy` + per-category implementations
  - `src/rag/retrieval/adaptive/__init__.py` — strategy registry
  - `src/domain/services/retrieval_service.py` — select strategy based on `Query.metadata["category"]`
  - `configs/retrieval.yaml` — per-category overrides under `retrieval.adaptive.strategies`
  - `tests/unit/test_adaptive_strategies.py`
- **Strategy defaults:**
  | Category | top_k | n_variants | hyde | compression |
  |----------|-------|------------|------|-------------|
  | Factual | 30 | 1 | false | true |
  | Analytical | 50 | 3 | true | true |
  | Opinion | 20 | 2 | false | false |
  | Contextual | 40 | 2 | false | true |
- **Acceptance Criteria:**
  - Each category maps to a strategy with distinct parameters
  - Unknown category falls back to Factual strategy
  - Strategies configurable via YAML without code changes
  - `pytest tests/unit/test_adaptive_strategies.py` passes

---

### T-133 · Step-Back Query Transformation
- **Status:** `[x]`
- **Goal:** Generate a broader "step-back" query alongside the original to retrieve background context — inspired by RAG_Techniques `query_transformations.ipynb` (step-back prompting).
- **Inputs:** T-020 (`QueryExpander` — extend or parallel module), T-110 (multi-query fusion)
- **Outputs:** Step-back variant added to `Query.expanded_texts` or separate `Query.metadata["step_back"]`.
- **Files:**
  - `src/rag/retrieval/step_back.py` — `generate_step_back(query, llm) -> str`
  - `src/prompts/retrieval/step_back.txt`
  - `src/rag/retrieval/query_expansion.py` — optionally invoke step-back when enabled
  - `configs/retrieval.yaml` — add `query_expansion.step_back.enabled: false`
  - `tests/unit/test_step_back.py`
- **Acceptance Criteria:**
  - Disabled by default
  - When enabled, step-back query included in multi-query RRF fusion (T-110)
  - Step-back failure does not block standard retrieval
  - Analytical queries benefit (documented in strategy T-132 config)

---

### T-134 · Multi-Faceted Qdrant Filtering
- **Status:** `[x]`
- **Goal:** Apply metadata filters, similarity thresholds, and document scope constraints at retrieval time — inspired by RAG_Techniques **multi-faceted filtering** (README; notebook missing from reference repo).
- **Inputs:** T-013 (Qdrant), T-021 (`DenseRetriever`), T-003 (`Chunk.metadata`)
- **Outputs:** Retrieval accepts optional filter parameters; Qdrant payload filters applied.
- **Files:**
  - `src/rag/retrieval/filters.py` — `RetrievalFilter` dataclass + Qdrant filter builder
  - `src/infrastructure/vectordb/qdrant.py` — accept `query_filter` in `search_dense()`
  - `src/domain/entities/query.py` — add optional `filters: RetrievalFilter | None`
  - `src/api/routers/chat.py` — accept optional `document_ids`, `metadata_filters` in request body
  - `tests/unit/test_retrieval_filters.py`
- **Filter types:**
  - `document_ids: list[str]` — scope to specific documents
  - `metadata: dict[str, str]` — exact-match payload filters (e.g. `section`, `source`)
  - `min_score: float` — discard results below similarity threshold
- **Acceptance Criteria:**
  - No filters → current behavior unchanged
  - `document_ids` filter restricts results to specified documents only
  - `min_score` filter applied post-search, before RRF fusion
  - API request schema documented in OpenAPI

---

### T-135 · Diversity Retrieval (MMR / Dartboard-lite)
- **Status:** `[x]`
- **Goal:** Reduce redundant chunks in final results by optimizing relevance + diversity — inspired by RAG_Techniques `dartboard.ipynb` (lightweight MMR implementation, not full RIG optimization).
- **Inputs:** T-023 (reranker output), T-025 (retrieval service)
- **Outputs:** Optional diversity re-ranking step after cross-encoder, before compression.
- **Files:**
  - `src/rag/ranking/diversity.py` — `mmr_select(chunks, embeddings, lambda_, top_k) -> list[Chunk]`
  - `src/domain/services/retrieval_service.py` — optional diversity step after reranking
  - `configs/retrieval.yaml` — add `retrieval.diversity.enabled: false`, `retrieval.diversity.lambda: 0.7`
  - `tests/unit/test_diversity.py`
- **Acceptance Criteria:**
  - Disabled by default
  - When enabled, final chunks maximize MMR score (relevance − similarity_to_selected)
  - Works with reranker output (does not replace cross-encoder)
  - `lambda=1.0` degrades to pure relevance ranking (no diversity penalty)

---

## Phase 14 — Quality Gates & Explainability (Priority 4)

> **Motivation:** Add runtime quality gates so the system refuses to hallucinate, self-corrects weak retrieval, explains its decisions, and learns from user relevance feedback — inspired by RAG_Techniques **Reliable RAG**, **Self-RAG**, **CRAG**, **explainable retrieval**, and **retrieval with feedback loop**.
>
> **Reference techniques:**
> - `reliable_rag.ipynb`
> - `self_rag.py`
> - `crag.py`
> - `explainable_retrieval.py`
> - `retrieval_with_feedback_loop.py`
>
> **Depends on:** Phase 11 (T-114 agent endpoint), Phase 13 (adaptive strategies optional)

---

### T-140 · Reliable RAG — Document Relevancy Grading
- **Status:** `[x]`
- **Goal:** After reranking, grade each chunk's relevancy to the query using structured LLM output. Filter irrelevant chunks before compression/generation — inspired by RAG_Techniques `reliable_rag.ipynb`.
- **Inputs:** T-023 (reranker output), T-030 (LLM), T-025 (retrieval service)
- **Outputs:** Chunks below relevancy threshold discarded; empty context triggers "insufficient information" response.
- **Files:**
  - `src/rag/quality/reliable_rag.py` — `grade_relevance(query, chunks, llm) -> list[Chunk]`
  - `src/prompts/quality/relevance_grading.txt`
  - `src/domain/services/retrieval_service.py` — call grading after rerank, before compression
  - `configs/retrieval.yaml` — add `quality.reliable_rag.enabled: false`, `quality.reliable_rag.min_score: 0.5`
  - `tests/unit/test_reliable_rag.py`
- **Structured output:**
  ```python
  class ChunkRelevance(BaseModel):
      chunk_id: str
      relevance_score: float  # 0.0–1.0
      supporting: bool
  ```
- **Acceptance Criteria:**
  - Disabled by default
  - Chunks with `relevance_score < min_score` excluded from context
  - All chunks filtered → generation returns "I don't have information about this"
  - OTel span `retrieval.relevance_grading` records pass/fail counts

---

### T-141 · Self-RAG Decision Loop
- **Status:** `[x]`
- **Goal:** Extend `AgentPipeline` with Self-RAG gates: decide whether to retrieve, check answer support, and score utility — inspired by RAG_Techniques `self_rag.py`.
- **Inputs:** T-071 (`AgentPipeline`), T-140 (relevance grading), T-114 (agent API)
- **Outputs:** Agent loop with explicit retrieve/generate/critique steps and structured decision output.
- **Files:**
  - `src/rag/quality/self_rag.py` — `RetrievalDecision`, `SupportCheck`, `UtilityScore` Pydantic models + LLM chains
  - `src/prompts/quality/self_rag_decision.txt`, `self_rag_support.txt`, `self_rag_utility.txt`
  - `src/rag/pipelines/agent_pipeline.py` — integrate Self-RAG gates into iteration loop
  - `configs/retrieval.yaml` — add `quality.self_rag.enabled: false`
  - `tests/unit/test_self_rag.py`
- **Self-RAG flow:**
  ```
  Query → Need retrieval? (yes/no)
        → Retrieve → Relevance grade (T-140)
        → Generate draft → Supported by context? (yes/no)
        → Utility score → Accept / Re-retrieve / Refuse
  ```
- **Acceptance Criteria:**
  - Disabled by default; agent uses current behavior when off
  - When enabled, agent refuses to answer if support check fails after max iterations
  - `/chat/agent/full` response includes `self_rag_decisions` array
  - Structured LLM output via Pydantic (no regex parsing)

---

### T-142 · Corrective RAG (CRAG) — Web Search Fallback
- **Status:** `[x]`
- **Goal:** Score overall retrieval quality; when context is weak, fall back to web search and refine knowledge before generation — inspired by RAG_Techniques `crag.py`.
- **Inputs:** T-140 (relevance grading), T-031 (`ChatPipeline`), T-030 (LLM)
- **Outputs:** Optional CRAG pipeline branch with web search fallback and knowledge refinement.
- **Files:**
  - `src/rag/quality/crag.py` — `score_retrieval_quality()`, `refine_knowledge()`, thresholds
  - `src/infrastructure/search/web_search.py` — DuckDuckGo or Tavily wrapper (domain ABC)
  - `src/domain/repositories/web_search_repository.py` — ABC interface
  - `src/prompts/quality/crag_knowledge_refinement.txt`
  - `src/rag/pipelines/chat_pipeline.py` — optional CRAG branch
  - `configs/retrieval.yaml` — add `quality.crag.enabled: false`, `quality.crag.lower_threshold: 0.3`, `quality.crag.upper_threshold: 0.7`
  - `tests/unit/test_crag.py`
- **Threshold behavior (from RAG_Techniques):**
  - Score > upper_threshold → use retrieved context as-is
  - Score between thresholds → combine retrieved + web results, refine with LLM
  - Score < lower_threshold → discard retrieval, web search only
- **Acceptance Criteria:**
  - Disabled by default (no web search calls)
  - Web search provider swappable via settings (`web_search.provider: duckduckgo|tavily|none`)
  - Missing API key / unreachable search → fall back to "insufficient information"
  - CRAG decisions logged and visible in OTel spans

---

### T-143 · Explainable Retrieval API
- **Status:** `[x]`
- **Goal:** Return human-readable explanations for why each chunk was retrieved and how it relates to the query — inspired by RAG_Techniques `explainable_retrieval.py`.
- **Inputs:** T-025 (retrieval result), T-030 (LLM), T-032 (API)
- **Outputs:** Optional `explanations` field in chat response with per-chunk reasoning.
- **Files:**
  - `src/rag/quality/explainable_retrieval.py` — `explain_chunks(query, chunks, llm) -> list[ChunkExplanation]`
  - `src/prompts/quality/explain_retrieval.txt`
  - `src/domain/entities/answer.py` — add optional `explanations: list[ChunkExplanation]`
  - `src/api/routers/chat.py` — `explain=true` query param on `/chat/full`
  - `tests/unit/test_explainable_retrieval.py`
- **Response schema addition:**
  ```json
  {
    "answer": "...",
    "sources": ["chunk_id_1"],
    "explanations": [
      { "chunk_id": "chunk_id_1", "reason": "Contains revenue figures for Q3 2023..." }
    ]
  }
  ```
- **Acceptance Criteria:**
  - `explain=false` (default) → no extra LLM calls, response unchanged
  - `explain=true` → one explanation per source chunk
  - Explanation generation failure omits explanations (does not fail the request)

---

### T-144 · Source Highlighting in Answers
- **Status:** `[x]`
- **Goal:** Identify and return the specific sentences within each chunk that support the generated answer — extends Reliable RAG (T-140) for user-facing transparency.
- **Inputs:** T-140 (relevance grading), T-031 (generation), T-143 (explainable retrieval)
- **Outputs:** `Answer.highlights` with chunk ID → supporting sentence spans.
- **Files:**
  - `src/rag/quality/source_highlighting.py` — `extract_highlights(answer, chunks, llm) -> dict[str, list[str]]`
  - `src/prompts/quality/source_highlighting.txt`
  - `src/domain/entities/answer.py` — add `highlights: dict[str, list[str]]`
  - `tests/unit/test_source_highlighting.py`
- **Acceptance Criteria:**
  - Disabled by default; enabled via `quality.source_highlighting.enabled`
  - Each highlight is a verbatim substring of the source chunk text
  - `/chat/full` response includes highlights when enabled
  - No highlights generated → field omitted (not empty dict)

---

### T-145 · Retrieval Feedback Loop
- **Status:** `[x]`
- **Goal:** Collect user relevance feedback on retrieved chunks and persist scores in chunk metadata for future retrieval boosting — inspired by RAG_Techniques `retrieval_with_feedback_loop.py`.
- **Inputs:** T-013 (Qdrant payload updates), T-117 (SQLite metadata), T-032 (API)
- **Outputs:** Feedback API + metadata-boosted retrieval scoring.
- **Files:**
  - `src/rag/quality/feedback_loop.py` — `record_feedback`, `apply_feedback_boost`, `score_from_relevant`, `merge_chunk_views`, `resolve_feedback_score`
  - `src/api/routers/feedback.py` — `POST /feedback` endpoint (204 / 404 / 502)
  - `src/infrastructure/vectordb/qdrant.py` — `accumulate_feedback_score` with CAS retries; upsert/rollback preserves `feedback_score` / `feedback_revision`
  - `src/domain/repositories/vector_store_repository.py` — `get_feedback_score(s)`, `accumulate_feedback_score` ABCs
  - `src/rag/retrieval/hybrid_retriever.py` — RRF boost + expanded candidate pool when boost enabled
  - `src/rag/ranking/cross_encoder.py` — re-applies boost after reranking
  - `src/domain/services/retrieval_service.py` — wires boost multiplier + vector store
  - `src/rag/pipelines/retrieval_pipeline.py` — settings wiring
  - `src/rag/ranking/score_fusion.py` — dedup merges non-feedback metadata only
  - `configs/retrieval.yaml` — `quality.feedback_loop` block
  - `src/core/settings.py` — `FeedbackLoopSettings`
  - `src/core/constants.py` — `FEEDBACK_SCORE_KEY`, `FEEDBACK_REVISION_KEY`
  - `tests/unit/test_feedback_loop.py`, `tests/unit/test_qdrant.py`, `tests/unit/test_repositories.py`, `tests/unit/test_hybrid_retriever.py`, `tests/unit/test_reranker.py`
- **API contract:**
  ```
  POST /feedback
    Body: { "query_id": "...", "chunk_id": "...", "relevant": true }
  ```
- **Acceptance Criteria:**
  - Feedback persisted to Qdrant chunk payload (`feedback_score: float`, `feedback_revision: int`)
  - Positive votes add `+1.0`; negative votes subtract `1.0` (accumulated score)
  - Chunks with positive feedback receive additive RRF / reranker boost (`boost_multiplier × feedback_score`)
  - Boost re-applied after cross-encoder so feedback survives reranking
  - Feedback endpoint returns 204 on success; 404 when chunk missing; 502 on vector store errors
  - No feedback boost configured → retrieval ranking unchanged (feedback may still be recorded)
  - Qdrant is write source of truth; no BM25 disk write on feedback path
  - Re-ingest upsert preserves existing feedback scores under CAS
- **Notes:** Core multi-replica hardening (CAS accumulation, Qdrant-only writes, live score lookup) landed with T-145. Remaining production gaps (ops docs, rate limiting, concurrency benchmarks, optional Redis backend) tracked in **T-146**.

---

### T-146 · Feedback Loop Production Hardening _(follow-up to T-145)_
- **Status:** `[x]` — ops docs, rate limits, pluggable backends, BM25 dirty-save, concurrency benchmark
- **Goal:** Close remaining production gaps in the T-145 feedback loop identified during Bugbot review and multi-replica deployment analysis — without blocking local/single-replica usage.
- **Inputs:** T-145 (feedback API + boost), T-013 (Qdrant), T-032 (API), T-095 (Helm HPA), T-160 (rate limiting, optional)
- **Outputs:** Documented gap tracker, hardened feedback persistence for horizontal scale, and CI/load-test coverage before HPA ≥ 2.
- **Motivation:** Bugbot flagged per-request BM25 disk writes and non-atomic feedback accumulation. Code review further identified per-pod BM25 drift and missing `/feedback` rate limits under Helm defaults (`replicaCount.api: 2`, HPA min 2).
- **Gap tracker:**

  | Gap | Severity | Status | Trigger to address | Owner task |
  |---|---|---|---|---|
  | Full BM25 JSON rewrite on every `POST /feedback` | High | **Fixed** — deferred to lifespan `save_indexes()` | — | T-145 hardening |
  | Non-atomic read-modify-write on `feedback_score` | Medium | **Fixed** — Qdrant CAS retry in `accumulate_feedback_score` | — | T-145 hardening |
  | Per-pod BM25 metadata drift under multi-replica | Medium | **Fixed** — Qdrant is write source of truth; boost reads `vector_store.get_feedback_scores` | — | T-145 hardening |
  | CAS retry insufficient under extreme same-chunk contention | Low | **Mitigated** — use `backend: redis` for HINCRBYFLOAT | Feedback drives ranking in prod **and** load tests show lost increments | T-146 |
  | No rate limit on `/feedback` | Medium | **Fixed** — T-160 middleware includes `/feedback` when enabled | Public API or abuse observed | **T-160** |
  | No multi-pod feedback load test / baseline | Low | **Fixed** — `test_feedback_concurrency.py` (scenario 5) + T-172 infra suite | Before enabling Helm HPA in prod | T-172 ✅ |
  | Shared BM25 PVC last-writer-wins on shutdown save | Low | **Mitigated** — skip BM25 save when unchanged | Multiple API replicas share BM25 persistence volume | T-146 |
  | True atomic increment (Redis / Postgres) | Low | **Fixed** — `quality.feedback_loop.backend: redis \| postgres` | Business-critical feedback under heavy multi-pod load | T-146 |

- **Files:**
  - `src/infrastructure/vectordb/qdrant.py` — `accumulate_feedback_score`, `_try_set_feedback_score_if_current`, upsert feedback preservation _(done)_
  - `src/rag/quality/feedback_loop.py` — Qdrant-only `record_feedback`; live `get_feedback_scores` at boost time _(done)_
  - `src/api/routers/feedback.py` — remove BM25 coupling; require API key when configured _(done)_
  - `tests/unit/test_qdrant.py` — CAS retry + concurrent accumulation + upsert/rollback tests _(done)_
  - `tests/unit/test_feedback_loop.py` — feedback loop + boost tests _(done)_
  - `README.md` — T-145 usage, API contract, pipeline position _(done)_
  - `src/infrastructure/vectordb/feedback_store.py` — pluggable Redis / SQL atomic increment backend _(done)_
  - `src/core/settings.py` — `quality.feedback_loop.backend: qdrant | redis | postgres` _(done)_
  - `configs/app.yaml` — feedback backend + Redis URL + `api.rate_limit` block
  - `tests/benchmarks/test_feedback_concurrency.py` — multi-process lost-increment regression _(done)_
  - `docs/operations/feedback-multi-replica.md` — deployment guidance _(done)_
- **Completed (PR #28):**
  1. **Multi-replica ops:** `docs/operations/feedback-multi-replica.md` documents safe deployment modes (1 replica, HPA ≥ 2 with CAS, optional Redis backend).
  2. **Rate limiting:** `/feedback` included in `src/api/rate_limit.py` protected routes when `api.rate_limit.enabled=true` (**T-160**).
  3. **Pluggable backend:** `FeedbackStore` with Redis `HINCRBYFLOAT` or SQL `UPSERT … score += delta` behind `accumulate_feedback_score`.
  4. **Shared BM25 PVC:** skip BM25 save on shutdown when unchanged (`BM25Index._dirty`); disk-backed segmented index with bounded search RAM addressed in **T-165** (PR #37).
- **Acceptance Criteria:**
  - [x] No `bm25_index.save()` on feedback path
  - [x] `accumulate_feedback_score` uses compare-and-set retries (not process-local lock only)
  - [x] `record_feedback` writes Qdrant only; retrieval boost reads live Qdrant scores
  - [x] README documents T-145 API contract, config, pipeline position, T-146 deployment caveats, and T-160 rate limiting
  - [x] `docs/operations/feedback-multi-replica.md` documents safe deployment modes (1 replica, HPA ≥ 2 with CAS, optional Redis backend)
  - [x] **T-160** updated to rate-limit `/feedback` when enabled (`src/api/rate_limit.py`)
  - [x] **T-172** adds scenario: 10 concurrent `POST /feedback` on same `chunk_id` across simulated pods — zero lost increments (`tests/benchmarks/test_feedback_concurrency.py`)
  - [x] Redis/SQL feedback backend selectable via settings; default remains Qdrant CAS
- **Safe without closing T-146:**
  - Local dev, Docker Compose single `api` container, `uvicorn --workers 1`
  - Production with `replicaCount.api: 1` and normal human feedback volume
- **Do not deploy without T-146 + T-160 progress:**
  - Public-facing API with Helm HPA (`minReplicas ≥ 2`) and business-critical feedback-driven ranking **without** `api.rate_limit.enabled=true` and a concurrency baseline (`test_feedback_concurrency.py` or T-172 `make benchmark-infra`)

---

## Phase 15 — Evaluation Operationalization (Priority 5)

> **Motivation:** Operationalize the evaluation framework from Phase 4 — benchmark RAG techniques side-by-side, tune chunk sizes, and enforce CI regression gates with real golden data.
>
> **Depends on:** Phase 4 (T-040–T-043), Phase 11–14 technique flags
>
> **Progress:** T-150 complete (PR #29) · T-151 complete (PR #30) · T-152 complete (PR #31)

---

### T-150 · Evaluation-Driven Technique Benchmark
- **Status:** `[x]`
- **Goal:** Benchmark script that compares RAG techniques side-by-side (baseline vs expansion vs HyDE vs CCH vs Self-RAG vs feedback loop) — inspired by RAG_Techniques `choose_chunk_size.py` and `evaluation/` notebooks.
- **Inputs:** T-043 (`RAGBenchmark`), T-040 (golden dataset), Phases 11–14 technique flags (incl. T-145 `quality.feedback_loop`, T-146 backend selection)
- **Outputs:** Comparison table with Recall@5, Faithfulness, Relevance, and latency per technique configuration.
- **Files:**
  - `scripts/benchmark_techniques.py` — CLI to run technique matrix
  - `scripts/_benchmark_utils.py` — shared QA loading (`prepare_qa_pairs`, placeholder filter)
  - `src/evals/e2e/technique_benchmark.py` — orchestrates config permutations
  - `configs/evals.yaml` — `technique_benchmark.configs` list
  - `tests/unit/test_technique_benchmark.py` — unit coverage
  - `tests/benchmarks/test_technique_benchmark.py` — skip on placeholder data
  - `tests/unit/test_benchmark_utils.py` — shared CLI helper tests
- **Usage:**
  ```bash
  make benchmark-techniques
  uv run python scripts/benchmark_techniques.py \
    --techniques baseline,multi_query,hyde,cch,reliable_rag,self_rag,feedback_loop \
    --max-samples 50
  ```
- **Output:** `data/exports/technique_benchmark_{timestamp}.json` + Rich summary table
- **Acceptance Criteria:**
  - Runs baseline with zero new techniques enabled
  - Each technique toggled independently via config override (no code changes between runs)
  - `feedback_loop` technique uses `temporary_feedback_seed` to pre-seed chunk scores and compares Recall@5 with boost on vs off at identical fusion pool size
  - `self_rag` technique runs via `AgentPipeline` adapter
  - Skips gracefully when golden dataset contains only placeholders
  - `make benchmark-techniques` Makefile target added
- **Notes:** `temporary_config` applies env overrides and reloads settings per technique. Pipelines and benchmarks share `build_vector_store_from_settings`. Generation metrics accept optional `parametric_answer` on `EvalSample` for Self-RAG runs. Feedback-loop benchmark disables `expand_candidate_pool` so A/B compares boost effect only.

---

### T-151 · Chunk Size Optimization Sweep
- **Status:** `[x]`
- **Goal:** Automate chunk size tuning by sweeping `chunk_size` values and measuring faithfulness/relevancy/latency — inspired by RAG_Techniques `choose_chunk_size.py`.
- **Inputs:** T-011 (chunkers), T-043 (benchmark), T-040 (golden dataset)
- **Outputs:** Script recommending optimal chunk size for the current corpus.
- **Files:**
  - `scripts/benchmark_chunk_sizes.py` — CLI (`--sizes`, `--ingest-source`, `--dry-run`, `--force-rechunk`, `--llm-config`)
  - `src/evals/e2e/chunk_size_sweep.py` — `ChunkSizeSweep` orchestrator, per-size indexing, weighted recommendation
  - `src/evals/e2e/benchmark_samples.py` — shared sample scoring helpers extracted from technique benchmark
  - `configs/evals.yaml` — `chunk_size_sweep.sizes: [256, 500, 768, 1024]` and `chunk_size_sweep.weights`
  - `src/infrastructure/vectordb/qdrant.py` — `recreate_collection()` for per-size index resets; embedding model validation on existing collections
  - `tests/unit/test_chunk_size_sweep.py` — unit coverage (850+ lines)
  - `tests/unit/test_benchmark_samples.py` — shared helper tests
  - `tests/benchmarks/test_chunk_size_sweep.py` — integration skip on placeholder data
  - `Makefile` — `benchmark-chunk-sizes` target
- **Usage:**
  ```bash
  make benchmark-chunk-sizes
  uv run python scripts/benchmark_chunk_sizes.py --dry-run
  uv run python scripts/benchmark_chunk_sizes.py \
    --ingest-source data/raw/ \
    --sizes 256,500,768,1024 \
    --force-rechunk \
    --max-samples 50
  ```
- **Output:** `data/exports/chunk_size_sweep_{timestamp}.json` + Rich comparison table with ★ on recommended size
- **Acceptance Criteria:**
  - Sweeps configured chunk sizes with isolated Qdrant collections (`rag_documents_cs{size}`)
  - Per-size chunk cache at `data/chunks/{size}/chunks.json` and BM25 at `data/chunks/{size}/bm25_index.json`
  - `--ingest-source` chunks documents when cache is missing; `--force-rechunk` ignores cache
  - `recreate_collection()` clears Qdrant before each per-size re-index (dense + BM25 stay aligned)
  - Reports Recall@5, Faithfulness, Relevance, and avg latency per size
  - Prints recommended size based on weighted score (`recall`, `faithfulness`, `relevance`, `latency` weights in config)
  - Remaps `relevant_chunks` when chunk boundaries shift between sizes (text overlap fallback)
  - `--dry-run` lists planned sweep steps without executing
  - Skips gracefully when golden dataset contains only placeholders
  - `make benchmark-chunk-sizes` Makefile target added
- **Notes:** `temporary_config` applies per-size env overrides (`CHUNKING__CHUNK_SIZE`, `QDRANT__COLLECTION`). Shared scoring logic lives in `benchmark_samples.py` and is reused by `technique_benchmark.py`. Qdrant `_model_validated` resets on collection recreate so per-size collections validate embedding model independently.

---

### T-152 · Golden Dataset Population & CI Gate Hardening
- **Status:** `[x]`
- **Goal:** Replace placeholder golden dataset rows with real QA pairs and enforce eval regression gates in CI — closes the gap identified vs RAG_Techniques eval operationalization. Static analysis gate hardening tracked separately in **T-171**.
- **Inputs:** T-040 (`SyntheticDatasetBuilder`), T-044 (`/evals/run`), T-061 (CI pipeline)
- **Outputs:** Populated `datasets/goldens/qa_dataset.json` and `retrieval_dataset.json`; modular CI regression gate that skips on placeholder-only data and fails on metric/sync regressions with real data.
- **Files:**
  - `src/evals/golden_dataset.py` — placeholder detection (`is_placeholder_*`), evaluable QA filtering (`is_evaluable_qa_pair`, `filter_real_qa_pairs`), QA→retrieval conversion (`qa_dicts_to_retrieval_rows`), sync (`sync_retrieval_from_qa`, `retrieval_rows_match_qa`), chunk expansion (`generate_until_min_pairs`, `resolve_max_chunks`, `resolve_retrieval_output_path`)
  - `src/evals/regression_gate.py` — `check_regression_gate()`: min sample counts, QA/retrieval sync, per-row oracle Recall@5 via `oracle_recall_at_k`
  - `src/evals/retrieval/recall_at_k.py` — `oracle_recall_at_k` for multi-chunk ground-truth recall
  - `scripts/run_evals.py` — adaptive chunk iteration, minimum pair enforcement, auto-sync retrieval output
  - `scripts/sync_retrieval_golden.py` — CLI for `make sync-retrieval-goldens`
  - `scripts/check_regression_gate.py` — CI entrypoint (exit 1 on failure)
  - `datasets/goldens/qa_dataset.json` — populated real QA pairs (≥ 20)
  - `datasets/goldens/retrieval_dataset.json` — synced retrieval rows with `relevant_chunk_ids`
  - `datasets/goldens/retrieval_baseline.json` — committed thresholds (`min_samples`, `min_recall_at_5`)
  - `configs/evals.yaml` — `min_qa_pairs`, `retrieval_baseline_path`, `retrieval.regression.min_recall_at_5`
  - `.github/workflows/ci.yml` — `retrieval-regression` job calls `check_regression_gate.py` (replaces inline assertions)
  - `Makefile` — `evals` and `sync-retrieval-goldens` targets
  - `tests/unit/test_golden_dataset.py` — placeholder filtering, sync, chunk expansion
  - `tests/unit/test_regression_gate.py` — gate pass/fail/skip scenarios
  - `tests/unit/test_sync_retrieval_golden.py` — sync CLI coverage
  - `tests/unit/test_run_evals.py` — chunk iteration and path resolution
  - `tests/unit/test_committed_goldens.py` — committed dataset invariants
  - `tests/benchmarks/test_retrieval_evals.py` — live retrieval benchmark with baseline comparison
- **Usage:**
  ```bash
  make ingest SOURCE=data/raw/
  make evals
  make sync-retrieval-goldens   # after manual QA edits
  uv run python scripts/check_regression_gate.py
  ```
- **Acceptance Criteria:**
  - `make evals` generates ≥ 20 evaluable QA pairs from ingested documents (placeholder rows filtered)
  - `generate_until_min_pairs` expands chunk coverage when dedup leaves fewer than `min_pairs`
  - `make sync-retrieval-goldens` rebuilds retrieval rows from QA without LLM regeneration
  - `retrieval_rows_match_qa` fails regression gate when datasets are out of sync
  - `POST /evals/run` returns 200 (not 204) after evals
  - CI `retrieval-regression` job runs `check_regression_gate.py`; skips on placeholder-only data; enforces min samples, sync, and oracle Recall@5 floors with real data
  - README documents eval setup workflow, mermaid flows, and links to T-145 feedback loop + [docs/operations/feedback-multi-replica.md](../docs/operations/feedback-multi-replica.md) for human-in-the-loop eval extensions
- **Notes:** Oracle Recall@5 uses ground-truth `relevant_chunk_ids` (not live retrieval). Gate failure message recommends `make sync-retrieval-goldens` when sync check fails. `baseline_int` / `baseline_float` coerce committed baseline values safely.

---

## Phase 16 — Production Hardening & Scalability (Priority 6)

> **Motivation:** Close gaps identified in `CODE_ANALYSIS_REPORT.md` that are outside the RAG-technique roadmap (Phases 12–14). These are infrastructure, security, and scalability improvements required before high-traffic production deployment.
>
> **Reference:** `CODE_ANALYSIS_REPORT.md` — Security checklist, Performance bottlenecks, Known vulnerabilities
>
> **Depends on:** Phase 3 (T-032 API), Phase 6 (T-061 CI), Phase 8 (T-082 Docker), Phase 9 (T-095 Ingress)
>
> **Progress:** T-160 complete · T-161 complete · T-162 complete (PR #34) · T-163 complete · T-164 complete (PR #36) · T-165 complete (PR #37) · **T-172** infra baselines (PR #41) for measuring optimizations

---

### T-160 · API Rate Limiting Middleware
- **Status:** `[x]`
- **Goal:** Protect sensitive endpoints (`/ingest`, `/chat`, `/chat/agent`, `/evals/run`, `/feedback`) from abuse with configurable per-IP or per-API-key rate limits — closes the gap flagged in the code analysis security checklist. `/feedback` inclusion closes **T-146** gap tracker item.
- **Inputs:** T-032 (`src/api/security.py`, routers), T-051 (Prometheus metrics)
- **Outputs:** FastAPI middleware that returns `429 Too Many Requests` when limits are exceeded; metrics counter for throttled requests.
- **Files:**
  - `src/api/rate_limit.py` — sliding-window limiter: Redis sorted-set Lua script when available, `InMemoryRateLimiter` fallback _(done)_
  - `src/core/settings.py` — `APIRateLimitSettings` nested under `APISettings` _(done)_
  - `configs/app.yaml` — `api.rate_limit` block _(done)_
  - `.env.example` — `API__RATE_LIMIT__ENABLED`, `API__RATE_LIMIT__REQUESTS_PER_MINUTE`, `API__RATE_LIMIT__BURST` _(done)_
  - `src/main.py` — register `RateLimitHTTPMiddleware` + `CORSMiddleware` _(done)_
  - `src/infrastructure/cache/redis_client.py` — shared Redis client for rate limit + feedback backend _(done)_
  - `src/observability/metrics.py` — `rag_rate_limit_rejected_total{path}` counter _(done)_
  - `tests/unit/test_rate_limit.py` — protected routes (all five prefixes), exempt/public paths, burst allowance, per-key isolation, Redis/in-memory backends, 429 body + `Retry-After`, CORS on 429, autouse config reset, middleware state isolation regression _(done)_
  - `tests/unit/test_redis_client.py` — Redis client helper _(done)_
  - `README.md` — T-160 config, middleware flow, metrics _(done)_
- **Config schema:**
  ```yaml
  api:
    rate_limit:
      enabled: false
      requests_per_minute: 60
      burst: 10
  ```
- **Acceptance Criteria:**
  - [x] Disabled by default (`enabled=false`) — no behavior change for local dev
  - [x] When enabled, exceeding limit returns `429` with `Retry-After` header
  - [x] `/health` and `/metrics` exempt from rate limiting
  - [x] `/feedback` included in protected routes when rate limiting enabled (closes T-146 gap)
  - [x] Redis unavailable → in-memory limiter with warning log (graceful degradation)
  - [x] `pytest tests/unit/test_rate_limit.py` passes (38 tests)
  - [x] Client key prefers `X-API-Key`, then `X-Forwarded-For`, then direct IP
  - [x] `OPTIONS` preflight exempt from rate limiting (CORS compatibility)
  - [x] Burst allows `requests_per_minute + burst` requests per window (in-memory limiter)
  - [x] Separate client keys (`X-API-Key` / IP) have independent quotas
  - [x] Middleware test harness pins in-memory limiter — no cross-test Redis key leakage
  - [x] Sequential test apps do not inherit another app's client quota (regression)
  - [x] 429 response body is `{"detail": "Rate limit exceeded"}`

---

### T-161 · Automated Dependency Scanning (CI)
- **Status:** `[x]`
- **Goal:** Replace manual CVE tracking with automated dependency scanning on every PR — addresses the code analysis finding that dependency scanning is currently manual.
- **Inputs:** T-061 (CI pipeline), `pyproject.toml`, `uv.lock`
- **Outputs:** CI job that fails on high/critical CVEs in direct and transitive dependencies.
- **Files:**
  - `.github/workflows/ci.yml` — add `dependency-scan` job
  - `scripts/check_dependencies.sh` — wrapper around `uv pip audit` or `pip-audit`
  - `docs/dependency-policy.md` — document allowlist process for unfixable CVEs
- **Acceptance Criteria:**
  - CI runs dependency scan on every PR
  - Known unfixable CVEs (e.g. diskcache) documented in allowlist file with expiry/review date
  - Scan completes in < 2 minutes
  - `make audit-deps` runs locally with same tool as CI

---

### T-162 · Transitive Dependency CVE Mitigation (diskcache)
- **Status:** `[x]` _(PR #34)_
- **Goal:** Formalize monitoring and mitigation for CVE-2025-69872 in `diskcache` (transitive via `llama-cpp-python`). No PyPI fix available as of 2025-06 — track upstream and apply compensating controls.
- **Inputs:** T-161 (dependency scanning), T-030 (`llama_cpp_provider.py`), `pyproject.toml` CVE comment
- **Outputs:** Documented risk acceptance, optional cache disable switch, automated upstream version check.
- **Files:**
  - `docs/security-advisories.md` — diskcache CVE entry with impact assessment, compensating controls, and review schedule _(done)_
  - `src/core/diskcache_cve_check.py` — PyPI monitor + version comparison (`exceeds_vulnerable_version_line`, post-release semantics) _(done)_
  - `src/core/settings.py` — `llm.disable_disk_cache: bool = False` _(done)_
  - `src/infrastructure/llm/llama_cpp_provider.py` — `_apply_prompt_cache_policy` (RAM-only or disabled via settings) _(done)_
  - `configs/cve-allowlist.yaml` — allowlist reason links to `docs/security-advisories.md` _(done)_
  - `.github/dependabot.yml` — weekly `llama-cpp-python` update PRs _(done)_
  - `scripts/check_diskcache_cve.sh` / `scripts/check_diskcache_cve.py` — CI/local upstream monitor entrypoints _(done)_
  - `.env.example` — `LLM__DISABLE_DISK_CACHE` _(done)_
  - `tests/unit/test_diskcache_cve_check.py` — monitor logic, post-release versions, CLI entrypoint _(done)_
  - `tests/unit/test_llm.py` — cache policy + `from_settings` forwarding _(done)_
  - `tests/unit/test_settings.py` — `disable_disk_cache` env override _(done)_
  - `README.md` — T-161/T-162 security docs _(done)_
- **Acceptance Criteria:**
  - [x] CVE documented with CVSS, exposure path, and quarterly review date
  - [x] `LLM__DISABLE_DISK_CACHE=true` disables llama.cpp prompt caching when exploit becomes active
  - [x] T-161 allowlist entry references T-162 doc with expiry date
  - [x] Script exits 0 when no fix available, exits 2 when fix is available but not applied
- **Usage:**
  ```bash
  ./scripts/check_diskcache_cve.sh          # exit 0 = no upstream fix yet; exit 2 = upgrade required
  LLM__DISABLE_DISK_CACHE=true make serve # emergency kill switch
  ```

---

### T-163 · Async llama.cpp Streaming
- **Status:** `[x]`
- **Goal:** Replace the thread + queue streaming pattern in `LlamaCppProvider` with native async bindings (when available) or `asyncio.to_thread` isolation — addresses the code analysis performance bottleneck under concurrent load.
- **Inputs:** T-030 (`llama_cpp_provider.py`), T-031 (`ChatPipeline`)
- **Outputs:** Non-blocking streaming that does not contend with the FastAPI event loop under concurrent requests.
- **Files:**
  - `src/infrastructure/llm/llama_cpp_provider.py` — refactor `_stream_in_thread` to async-safe pattern
  - `tests/unit/test_llama_cpp_provider.py` — concurrent stream smoke test
  - `tests/integration/test_llm.py` — verify streaming still works end-to-end
- **Approach (evaluate in order):**
  1. Use `asyncio.to_thread` + bounded queue if llama-cpp-python adds async API
  2. Process pool for model inference (heavier but fully isolated)
  3. Document `LLM__PROVIDER=ollama` as recommended for high-concurrency deployments
- **Acceptance Criteria:**
  - 10 concurrent streaming requests complete without event-loop starvation (verified in test)
  - Single-request latency unchanged within 5%
  - Existing `generate_stream` API signature unchanged
  - OTel span `llm.stream` records queue wait time

---

### T-164 · Neo4j Async Driver Integration
- **Status:** `[x]` — done in PR #36 (`feat/t-164-neo4j-async-driver`)
- **Goal:** Migrate graph repository calls from synchronous Neo4j driver to `AsyncGraphDatabase` so graph retrieval does not block the event loop when `neo4j.enabled=true`.
- **Inputs:** T-070 (`neo4j_graph.py`), T-111 (graph wiring), T-112 (Neo4j settings)
- **Outputs:** Async graph queries compatible with the async hybrid retriever path; sync wrappers for CLI/ingestion via `async_bridge`.
- **Files:**
  - `src/infrastructure/vectordb/neo4j_graph.py` — `AsyncGraphDatabase` driver; async `upsert` / `search_by_entities` / `close`; `upsert_sync` / `close_sync` wrappers _(done)_
  - `src/core/async_bridge.py` — background-loop `run_async()` for sync callers that already have a running event loop _(done)_
  - `src/core/settings.py` + `configs/neo4j.yaml` — `neo4j.max_connection_pool_size` (default 100) _(done)_
  - `src/rag/retrieval/graph_retriever.py` — async `search()` _(done)_
  - `src/rag/retrieval/hybrid_retriever.py` — await native async graph branch in `asyncio.gather` (dense/BM25 still `to_thread`) _(done)_
  - `src/rag/ingestion/graph_indexer.py` — sync `index_chunks` → `run_async` → async upsert _(done)_
  - `src/rag/pipelines/agent_pipeline.py` — async `_graph_lookup` / `GRAPH_LOOKUP` _(done)_
  - `tests/unit/test_graph_rag.py`, `test_async_bridge.py`, related unit tests _(done)_
- **Acceptance Criteria:**
  - [x] Graph retrieval runs concurrently with dense + BM25 via `asyncio.gather`
  - [x] Sync driver removed; sync wrappers isolate CLI/ingestion via `async_bridge.run_async`
  - [x] Neo4j unreachable → same graceful degradation as T-111 (warning + continue)
  - [x] Connection pooling configured via `neo4j.max_connection_pool_size` in settings
- **Notes:** Public graph API is async-first. Sync entry points (`upsert_sync`, `close_sync`, `GraphIndexer.index_chunks`) bridge through a dedicated daemon event loop so FastAPI request handlers and CLI scripts both work.

---

### T-165 · Disk-Backed BM25 Index (Scale)
- **Status:** `[x]` — PR #37
- **Goal:** Extend the in-memory BM25 index (T-014) with a disk-backed mode for corpora exceeding 1M chunks — addresses the code analysis scalability note without replacing the current default.
- **Inputs:** T-014 (`bm25.py`), T-015 (ingestion pipeline), T-146 (BM25 dirty-tracking skip-save on shutdown — partial mitigation for shared PVC)
- **Outputs:** Configurable BM25 backend: `memory` (default) or `disk` (mmap/segmented index).
- **Files:**
  - `src/infrastructure/vectordb/bm25_disk.py` — segmented/mmap disk-backed index; Okapi scoring matched to `rank_bm25` _(done)_
  - `src/infrastructure/vectordb/bm25.py` — `BM25Index.load_or_create(backend=...)`, `iter_chunks()`, soft-view stats caching _(done)_
  - `src/rag/pipelines/ingestion_pipeline.py` — single-doc `deferred_rebuild()`, atomic purge of superseded chunks on re-ingest, hierarchical/HyPE in same scope _(done)_
  - `src/rag/retrieval/bm25_retriever.py` — accepts memory/disk index; `from_disk()` pins memory backend for eval caches _(done)_
  - `src/core/settings.py` — `retrieval.bm25.backend: Literal["memory", "disk"]`, `disk_path`, `segment_size` _(done)_
  - `configs/retrieval.yaml` — `bm25.backend`, `bm25.disk_path`, `bm25.segment_size` _(done)_
  - `.env.example` — `RETRIEVAL__BM25__*` overrides _(done)_
  - `scripts/rebuild_embeddings.py`, `scripts/run_evals.py`, `scripts/compare_embedding_providers.py` — stream via `iter_chunks()` _(done)_
  - `src/evals/golden_dataset.py` — iterator-based chunk windowing for eval generation _(done)_
  - `tests/unit/test_bm25_disk.py` — unit + 100K scale + memory-bound checks _(done)_
  - `tests/unit/test_ingestion.py`, `tests/unit/ingestion_helpers.py` — re-ingest purge + deferred scope coverage _(done)_
- **Acceptance Criteria:**
  - [x] Default `backend=memory` — zero behavior change
  - [x] `backend=disk` indexes and searches correctly for 100K+ chunk fixture
  - [x] Incremental updates work (re-ingest adds/removes chunks)
  - [x] Memory usage for disk backend stays bounded regardless of corpus size
  - [x] README documents when to switch backends
- **Notes:** Disk mode stores chunk JSONL + memmapped lengths + per-segment postings under `disk_path`. Search RAM is IDF/DF + id map + one segment's postings. Prefer `memory` below ~1M chunks / typical enterprise corpora; switch to `disk` when BM25 RSS becomes a problem. Eval sweep caches (`data/chunks/{size}/bm25_index.json`) remain JSON memory indexes regardless of global backend.

---

## Phase 17 — Code Quality & Type Safety (Priority 7)

> **Motivation:** Restore and maintain the Phase 6 quality gate (`T-060`: `make lint` exits 0) beyond the immediate mypy fixes applied during Phase 12. Reduce the 56 `type: ignore` comments flagged in the code analysis and harden CI enforcement.
>
> **Reference:** `CODE_ANALYSIS_REPORT.md` — Type Safety Gaps, Code Quality
>
> **Depends on:** Phase 6 (T-060, T-061), Phase 12 (T-120–T-124 — source of recent type regressions)
>
> **Progress:** T-170 ✅ · T-171 ✅ (PR #39) · T-172 ✅ (PR #41) · T-173 ✅

---

### T-170 · Type Ignore Audit & Reduction
- **Status:** `[x]`
- **Goal:** Audit all 56 `type: ignore` comments, remove unnecessary ones, and replace fixable suppressions with proper types or targeted `mypy` overrides — brings type safety from grade B to A.
- **Inputs:** T-060 (mypy strict config), current `src/` codebase
- **Outputs:** Reduced `type: ignore` count (target: < 20), documented justification for each remaining suppression.
- **Files:**
  - `pyproject.toml` — tighten per-module overrides where possible; enable `warn_unused_ignores = true`
  - `src/infrastructure/llm/llama_cpp_provider.py` — reduce ignores (5 current)
  - `src/infrastructure/vectordb/qdrant.py` — reduce ignores (5 current)
  - `src/rag/pipelines/ingestion_pipeline.py` — reduce ignores (5 current)
  - `src/rag/pipelines/retrieval_pipeline.py` — reduce ignores (5 current)
  - `docs/type-safety.md` — table of remaining ignores with reason and removal plan
- **Acceptance Criteria:**
  - `uv run mypy src` exits 0 with zero errors
  - `type: ignore` count ≤ 20 (down from 56)
  - Each remaining ignore documented in `docs/type-safety.md`
  - `warn_unused_ignores = true` enabled without new warnings

---

### T-171 · Mypy CI Gate Hardening
- **Status:** `[x]` — PR #39
- **Goal:** Ensure CI blocks PRs on any mypy regression — extends T-152 eval gate hardening to static analysis. Closes the gap where Phase 12 feature work can reintroduce type errors.
- **Inputs:** T-061 (CI pipeline), T-170 (clean baseline), T-152 (gate hardening pattern)
- **Outputs:** CI fails if `mypy src` reports any error; pre-commit hook matches CI exactly; config drift detectable before merge.
- **Files:**
  - `.github/workflows/ci.yml` — lint job: ruff check → ruff format --check → mypy → basedpyright (no `continue-on-error`, no CLI `--ignore-missing-imports`)
  - `.pre-commit-config.yaml` — mypy hook targets `^src/`, relies on `pyproject.toml` settings
  - `Makefile` — `make lint` runs same four commands in same order as CI
  - `pyproject.toml` — `[tool.basedpyright]` `reportMissingImports = false` for optional deps
  - `src/core/lint_gate.py` — canonical command list + CI/Makefile/pre-commit drift checks
  - `scripts/check_lint_gate.py` — CLI entrypoint for local/CI verification
  - `src/type_regression/compression.py`, `src/type_regression/contextual_headers.py` — typed smoke modules analyzed by mypy
  - `tests/unit/test_lint_gate.py`, `tests/unit/test_type_regression.py` — gate + regression coverage
  - `tests/unit/test_compression.py`, `tests/unit/test_contextual_headers.py` — runtime checks for type_regression modules
  - `docs/type-safety.md` — CI gate section (T-171)
  - `README.md` — lint workflow for contributors + CI mermaid update
- **Acceptance Criteria:**
  - [x] PR with intentional mypy error is blocked by CI
  - [x] `make lint` and CI use identical commands (ruff check → ruff format --check → mypy → basedpyright)
  - [x] Pre-commit mypy hook catches errors before commit (targets `^src/`, no disallowed CLI args)
  - [x] `scripts/check_lint_gate.py` detects config drift across CI, Makefile, and pre-commit
  - [x] Typed smoke modules under `src/type_regression/` catch compression/contextual-header API regressions
  - [x] README documents lint workflow for contributors
- **Notes:** Removed CLI `--ignore-missing-imports` from CI (T-170 config lives in `pyproject.toml`). Added basedpyright as fourth lint step aligned with Makefile. `lint_gate.py` parses CI workflow step blocks in isolation so `continue-on-error` on unrelated jobs does not false-positive. CI uses `typeCheckingMode = "basic"` and `failOnWarnings = false` so merges block on **errors** only — incremental warning cleanup tracked in **T-173**.

---

### T-173 · Basedpyright Warning Burn-Down & Mode Progression
- **Status:** `[x]`
- **Goal:** Incrementally fix actionable basedpyright warnings and tighten `[tool.basedpyright]` from `basic` toward `standard`/`recommended` — without blocking PRs on third-party stub noise or duplicating mypy strict enforcement. Closes the gap left when T-171 suppressed ~427 Linux CI warnings to unblock the lint gate.
- **Inputs:** T-171 (basedpyright in CI at `--level error`), T-170 (mypy strict baseline), current `pyproject.toml` `[tool.basedpyright]` config
- **Outputs:** Committed basedpyright baseline (optional interim), reduced warning count per phase, documented rule enablement plan, CI remains green on `make lint`.
- **Files:**
  - `pyproject.toml` — progressive `[tool.basedpyright]` rule enablement; `typeCheckingMode` ladder (`basic` → `standard` → `recommended`)
  - `.basedpyright/baseline.json` — optional committed baseline while burning down (basedpyright `baselineFile` setting)
  - `docs/type-safety.md` — basedpyright section: current mode, enabled rules, burn-down progress table
  - `src/` — targeted fixes per phase (see **Phases** below)
  - `tests/unit/test_lint_gate.py` — extend if canonical basedpyright command or config checks change
- **Phases:** _(execute in order; each phase ends with `make lint` exit 0)_
  1. **Inventory & baseline** — capture current warning counts by rule:
     ```bash
     uv run basedpyright --level warning src 2>&1 | rg -o '\(report[A-Za-z]+\)' | sort | uniq -c | sort -rn
     ```
     Record counts in `docs/type-safety.md`. Optionally commit `.basedpyright/baseline.json` so new warnings fail CI while existing debt burns down incrementally.
  2. **Quick wins (~≤15 fixes)** — zero-risk, low-churn:
     - `reportUnusedImport` — remove dead imports
     - `reportUnreachable` — delete or refactor dead branches
     - `reportUnnecessaryCast` / `reportUnnecessaryComparison` / `reportUnnecessaryIsInstance` — simplify
     - `reportUnusedParameter` — prefix with `_` or remove
  3. **Override annotations (~83 fixes)** — add `typing.override` to methods that override base classes:
     - `reportImplicitOverride` — FastAPI middleware (`dispatch`), Pydantic settings sources, Ragas metric subclasses, logging `Formatter.format`, exception `__str__`, etc.
  4. **Intentional discard (~47 fixes)** — `reportUnusedCallResult`:
     - Assign to `_` when return value is intentionally ignored (e.g. side-effect-only calls)
     - Do **not** blanket-suppress the rule — fix call sites
  5. **Class attribute annotations (~250 fixes)** — `reportUnannotatedClassAttribute`:
     - Annotate instance/class attributes set in `__init__` or at class body (e.g. `self._llm: LLMRepository`, `model_config: ClassVar[...]` on Pydantic models where applicable)
     - Prefer `@final` on leaf classes only when inheritance is not intended
     - Batch by package: `core/` → `api/` → `domain/` → `infrastructure/` → `rag/` → `evals/`
  6. **Mode progression** — after phases 2–5 drive warning count near zero (excluding excluded rules):
     - Set `typeCheckingMode = "standard"`; fix newly surfaced errors
     - Optionally advance to `"recommended"` with `failOnWarnings = false` until debt cleared, then flip `failOnWarnings = true`
- **Out of scope (do not fix in src/):**
  - `reportMissingTypeStubs` (~551) — third-party libraries without stubs; keep suppressed or use `allowedUntypedLibraries` per module
  - `reportAny` / `reportExplicitAny` / `reportUnknown*Type` — already aligned with mypy overrides in T-170; re-enable only when upstream stubs improve
  - Chasing neo4j / ragas / llama_cpp stub completeness — tracked in `docs/type-safety.md` removal plan
- **Acceptance Criteria:**
  - [x] Warning inventory committed in `docs/type-safety.md` with per-rule counts and target mode
  - [x] Phase 2 complete: quick-win rules at zero warnings (or baselined with ticket refs)
  - [x] Phase 3 complete: `reportImplicitOverride` at zero warnings in `src/`
  - [x] Phase 4 complete: `reportUnusedCallResult` at zero warnings in `src/`
  - [x] Phase 5 complete: `reportUnannotatedClassAttribute` at zero warnings in `src/`
  - [x] `typeCheckingMode` advanced to at least `"standard"` with `make lint` exit 0 on macOS **and** Linux CI
  - [x] `failOnWarnings` remains `false` until Phase 5 done; document flip criteria in `docs/type-safety.md`
  - [x] No new `# type: ignore` in `src/`; mypy strict remains exit 0
  - [x] README lint workflow mentions basedpyright mode progression (link to `docs/type-safety.md`)
- **Notes:** T-171 intentionally kept basedpyright as an **error-level complement** to mypy — not a second strict gate. Linux CI failed with `0 errors, 427 warnings, exit 1` because basedpyright defaults to `recommended` + `failOnWarnings = true`; most CI warnings were `reportUnannotatedClassAttribute`, not runtime bugs. Completed in parallel with **T-172** (PR #41). Reuse T-152 baseline pattern: commit `.basedpyright/baseline.json`, burn down, shrink baseline each PR.

---

### T-172 · Performance Baseline & Regression Benchmark
- **Status:** `[x]` _(PR #41)_
- **Goal:** Establish baseline latency/throughput metrics for the infrastructure bottlenecks flagged in the code analysis (LLM streaming, BM25 memory, Neo4j sync, feedback concurrency) so Phase 16 optimizations can be measured.
- **Inputs:** T-043 (`RAGBenchmark`), T-051 (Prometheus metrics), T-146 (feedback hardening), T-160 (rate limiting), T-163–T-165 (optimization targets)
- **Outputs:** Benchmark script and CI-optional regression check for p50/p95 latency under concurrent load.
- **Files:**
  - `scripts/benchmark_infra.py` — CLI (`--scenarios`, `--compare`, `--save-baseline`, `--llm-config`) _(done)_
  - `src/evals/e2e/infra_benchmark.py` — `InfraBenchmark` orchestrator, baseline merge/save, `compare_to_baseline` _(done)_
  - `src/rag/pipelines/chat_pipeline.py` — `asyncio.to_thread` for blocking LLM calls in `chat_full` (concurrent scenario health) _(done)_
  - `src/rag/pipelines/agent_pipeline.py` — same for agent `chat_full` _(done)_
  - `configs/evals.yaml` — `evals.infra_benchmark` thresholds _(done)_
  - `data/exports/infra_baseline.json` — committed baseline (BM25 @ 100K; merge on `--save-baseline`) _(done)_
  - `.gitignore` — allowlist `infra_baseline.json` under `data/exports/` _(done)_
  - `Makefile` — `benchmark-infra` target _(done)_
  - `tests/benchmarks/test_infra_benchmark.py` — live integration (`RUN_INFRA_BENCHMARK=1`) _(done)_
  - `tests/unit/test_infra_benchmark.py` — unit coverage for scenarios, baseline compare, percentile helpers _(done)_
  - `tests/unit/test_benchmark_infra_cli.py` — CLI exit codes and flag wiring _(done)_
  - `tests/benchmarks/test_feedback_concurrency.py` — scenario 5: concurrent feedback on same `chunk_id` across simulated pods _(done · T-146)_
  - `README.md` — T-172 section, Evaluation Framework mermaid, project tree _(done)_
- **Scenarios:**
  1. Single streaming chat — p50/p95 inter-token latency (`streaming_chat`)
  2. 10 concurrent chats — event-loop health via parallel `chat_full`; failure count (`concurrent_chats`)
  3. BM25 search on 100K chunk fixture — disk index build + search latency + resident memory (`bm25_100k`)
  4. Graph retrieval with Neo4j enabled — entity lookup latency (`graph_retrieval`; skipped when disabled)
  5. Concurrent feedback on same `chunk_id` across simulated API pods — zero-lost increments (**T-146**; validates Qdrant CAS / Redis backend under load) — **implemented** in `test_feedback_concurrency.py`
- **Usage:**
  ```bash
  make benchmark-infra
  uv run python scripts/benchmark_infra.py --compare
  uv run python scripts/benchmark_infra.py --save-baseline
  uv run python scripts/benchmark_infra.py --scenarios bm25_100k
  RUN_INFRA_BENCHMARK=1 uv run pytest tests/benchmarks/test_infra_benchmark.py -v -s
  uv run pytest tests/unit/test_infra_benchmark.py tests/unit/test_benchmark_infra_cli.py -v
  ```
- **Acceptance Criteria:**
  - [x] Baseline captured and committed (`data/exports/infra_baseline.json`)
  - [x] Scenario 5 runnable independently via `pytest tests/benchmarks/test_feedback_concurrency.py`
  - [x] `--compare` reports p95 regression (>10%), failure-count increase, or skipped/errored baselined scenarios (exit 2)
  - [x] `make benchmark-infra` documented in README
  - [x] Results saved to `data/exports/infra_benchmark_{timestamp}.json`
- **Notes:** `InfraBenchmark` caches one `ChatPipeline` per run and shares its LLM with `GraphRetriever` when possible. Concurrent batch timeout scales as `concurrent_chat_timeout_s × concurrent_chat_count` because llama.cpp serializes inference behind a process-wide lock. `bm25_100k` builds a temp `DiskBM25Index` (T-165) with a deterministic 100K-chunk fixture — no Qdrant/LLM required. `--save-baseline` merges new scenario metrics into the committed baseline without dropping unrelated entries. CLI exit codes: `0` success, `1` error, `2` regression detected.

---

## Phase 18 — CI Performance (Priority 8)

> **Motivation:** Run 29032428404 baseline was ~22 min wall clock with five jobs each paying ~3 min for `uv sync`. Target ~10–12 min via job consolidation, `.venv` cache, parallel quality+unit, and unit-test optimizations.

---

### T-180 · CI Workflow Restructure & venv Cache
- **Status:** `[x]`
- **Goal:** Reduce CI installs from 5 to 3, run Quality and Unit Tests in parallel, cache `.venv` across jobs.
- **Inputs:** T-061 (CI pipeline), T-161 (dependency scan), T-171 (lint gate)
- **Outputs:** 3-job workflow, composite setup action, branch-protection migration script.
- **Files:**
  - `.github/actions/setup-python-env/action.yml` — uv + `.venv` cache + `uv sync --frozen --group dev` _(done)_
  - `.github/workflows/ci.yml` — `quality`, `unit-tests`, `extended-tests` jobs _(done)_
  - `src/core/lint_gate.py` — accept `quality:` job (fallback `lint:`) _(done)_
  - `tests/unit/test_lint_gate.py` — quality job coverage _(done)_
  - `scripts/migrate_ci_checks.sh` — inspect/update required check contexts _(done)_
  - `README.md` — CI/CD section _(done)_
- **Acceptance Criteria:**
  - [x] Five jobs merged to three (`Quality`, `Unit Tests`, `Extended Tests`)
  - [x] Unit tests no longer `needs: [dependency-scan, lint]`
  - [x] `.venv` cached keyed on `uv.lock`
  - [x] `check_lint_gate.py` passes on committed workflow
  - [x] `migrate_ci_checks.sh` documents `gh api` migration
- **Notes:** After first green CI run, execute `./scripts/migrate_ci_checks.sh` and update branch protection contexts to Quality / Unit Tests / Extended Tests.

---

### T-181 · Unit Test Suite Performance
- **Status:** `[x]`
- **Goal:** Cut CI unit pytest time by fixing accidental slow tests and parallelizing the default suite.
- **Inputs:** T-180 (CI unit command), T-165 (100K BM25 disk tests), T-172 (`test_benchmark_infra_cli.py`)
- **Outputs:** `@pytest.mark.slow`, xdist, tenacity sleep mocks, faster CLI entrypoint test.
- **Files:**
  - `tests/unit/test_benchmark_infra_cli.py` — patch `asyncio.run` in `test_module_entrypoint` _(done)_
  - `tests/unit/test_bm25_disk.py` — `@pytest.mark.slow` on 100K test _(done)_
  - `tests/unit/test_api_embedding_providers.py` — autouse `no_tenacity_sleep` fixture _(done)_
  - `pyproject.toml` — `pytest-xdist`, slow marker _(done)_
  - `.github/workflows/ci.yml` — `-m "not slow" -n auto --cov-report=xml` _(done)_
  - `Makefile` — `test-slow` target; `test-unit` excludes slow _(done)_
- **Acceptance Criteria:**
  - [x] `test_module_entrypoint` completes in under 5s locally
  - [x] CI excludes `@pytest.mark.slow` by default
  - [x] `make test-slow` runs scale tests locally
  - [x] Tenacity retry tests do not sleep in CI
- **Notes:** Default CI uses `-n auto` (2 workers on `ubuntu-latest`). Full slow suite runs in `ci-slow.yml`.

---

### T-182 · Path Filters & CI Policy
- **Status:** `[x]`
- **Goal:** Skip jobs when diffs cannot affect them; add manual and scheduled slow-test coverage.
- **Inputs:** T-180 (3-job layout)
- **Outputs:** `dorny/paths-filter`, expanded `paths-ignore`, `workflow_dispatch`, weekly slow workflow.
- **Files:**
  - `.github/workflows/ci.yml` — `changes` job, per-job `if:` conditions, `workflow_dispatch` + `run_slow` input _(done)_
  - `.github/workflows/ci-slow.yml` — weekly `pytest -m slow` _(done)_
  - `README.md` — skip behavior, slow-test docs _(done)_
- **Acceptance Criteria:**
  - [x] `changes` job gates quality/unit/extended via path filters
  - [x] `specs/**` and `data/exports/**` in workflow `paths-ignore`
  - [x] `workflow_dispatch` can run full suite with slow tests
  - [x] Weekly `ci-slow.yml` runs `@pytest.mark.slow`
- **Notes:** Extended tests skip when diff is narrowly scoped to `tests/unit/**` without touching `src/**` or golden datasets. Conservative `extended` filter avoids false skips on production code changes.

---

## Multimodal-RAG Phase Roadmap (Phases 19–28)

> Phases are numbered sequentially after Phase 18. **Task IDs follow Phase × 10** (Phase 20 → T-200…). Every precondition is satisfied by an earlier phase.
>
> | Phase | Priority | Tasks | Depends on | Status |
> |-------|----------|-------|------------|--------|
> | **19** | 9 | T-190 | Phases 0–3, 18 | ✅ complete |
> | **20** | 10 | T-200 → T-202 | Phase 19 | T-200 ✅ · T-201 ✅ · T-202 ✅ |
> | **21** | 11 | T-210 | Phases 19–20 | T-210 ✅ |
> | **22** | 12 | T-220 → T-223 | Phases 19–20 | ✅ complete — T-220 ✅ · T-221 ✅ · T-222 ✅ · T-223 ✅ |
> | **23** | 13 | T-230 → T-232 | Phases 20–21 | **complete** — T-230 ✅ → T-231 ✅ → T-232 ✅ |
> | **24** | 14 | T-240 → T-243 | Phases 20–21 | T-240 ✅ · T-241–T-243 pending |
> | **25** | 15 | T-250 → T-253 | Phase 21 | pending |
> | **26** | 16 | T-260 → T-263 | Phase 25 | pending |
> | **27** | 17 | T-270 → T-274 | Phases 21, 24–25 | pending |
> | **28** | 18 | T-280 → T-282 | Phases 25–26 | pending |

## Phase 19 — Multimodal Parsing Contracts (Priority 9)

> **Motivation:** Define parsing/OCR ABCs and chunk-type constants before layout implementations.
>
> **Preconditions:** Phases 0–3, 18
>
> **Status:** complete — T-190 ✅

---

### T-190 · Layout Parser & OCR Repository ABCs
- **Status:** `[x]`
- **Goal:** Define `LayoutParserRepository`, `OcrRepository`, `ParsedDocument`, `ParsingSettings`, and multimodal chunk constants.
- **Inputs:** T-004, T-003
- **Outputs:** ABCs + `configs/parsing.yaml` + constants in `src/core/constants.py`
- **Files:**
  - `src/domain/repositories/layout_parser_repository.py` — `LayoutParserRepository` ABC _(done)_
  - `src/domain/repositories/ocr_repository.py` — `OcrRepository` ABC _(done)_
  - `src/domain/repositories/__init__.py` — export new ABCs _(done)_
  - `src/domain/entities/parsed_document.py` — frozen `ParsedDocument` entity _(done)_
  - `src/domain/entities/__init__.py` — export `ParsedDocument` _(done)_
  - `src/core/constants.py` — `CHUNK_TYPE_TABLE/CAPTION/FIGURE/PAGE`, `TABLE_ID_KEY`, `FIGURE_ID_KEY`, `BBOX_KEY` _(done)_
  - `src/core/settings.py` — `ParsingSettings`, `LayoutParserSettings`, `OcrSettings` _(done)_
  - `configs/parsing.yaml` — feature flags (both disabled by default) _(done)_
  - `src/rag/chunking/contextual_headers.py` — use `CHUNK_PAGE_KEY` / `CHUNK_SECTION_KEY` for layout metadata _(done)_
  - `tests/unit/test_parsing_repositories.py` — ABC rules, constants, domain import hygiene _(done)_
  - `tests/unit/test_settings.py` — parsing YAML defaults + env overrides _(done)_
- **Acceptance Criteria:**
  - [x] Feature-flagged or backward-compatible defaults preserved
  - [x] Unit tests pass for new modules
  - [x] Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Contracts-only phase — no `infrastructure/` parsers or ingestion wiring. `domain/` modules must not import from `infrastructure/`. Layout parsers populate `metadata.page` / `metadata.section` via `CHUNK_PAGE_KEY` / `CHUNK_SECTION_KEY` (not `page_number` / `section_title`). README documents contracts, config, and upcoming Phase 20–28 roadmap.

---


## Phase 20 — Layout-Aware Document Parsing (Priority 10)

> **Motivation:** Docling layout parser, PPTX loader, structured table chunks.
>
> **Preconditions:** Phase 19 (T-190)
>
> **Status:** complete — T-200 ✅ · T-201 ✅ · T-202 ✅

---

### T-200 · Layout-Aware Parser — Docling (Primary)
- **Status:** `[x]`
- **Goal:** Implement `DoclingLayoutParser` for PDF/DOCX.
- **Inputs:** T-190, T-010
- **Outputs:** `DoclingLayoutParser` + loader delegation + chunk metadata filtering
- **Files:**
  - `src/infrastructure/parsers/docling_parser.py` — `DoclingLayoutParser`, `build_docling_metadata()` _(done)_
  - `src/infrastructure/parsers/__init__.py` — `get_layout_parser()`, `clear_layout_parser_cache()`, `parsed_to_document()` _(done)_
  - `src/infrastructure/loaders/__init__.py` — route `.pdf`/`.docx` through layout parser when enabled _(done)_
  - `src/infrastructure/loaders/docx_loader.py` — promote first heading to `CHUNK_SECTION_KEY` _(done)_
  - `src/infrastructure/loaders/markdown_loader.py` — promote first heading to `CHUNK_SECTION_KEY` _(done)_
  - `src/rag/chunking/metadata.py` — `chunk_metadata()` filters `LAYOUT_DOCUMENT_METADATA_KEYS` _(done)_
  - `src/core/constants.py` — `LAYOUT_DOCUMENT_METADATA_KEYS` _(done)_
  - `src/rag/chunking/recursive_chunker.py`, `semantic_chunker.py`, `proposition_chunker.py` — spread `chunk_metadata()` _(done)_
  - `src/rag/enrichment/hierarchical_indexer.py` — use `chunk_metadata()` _(done)_
  - `configs/parsing.yaml` — layout parser flag + Docling install note _(done)_
  - `tests/unit/test_docling_parser.py` — parser, metadata extraction, factory cache, settings reload _(done)_
  - `tests/unit/test_chunk_metadata.py` — metadata filtering and section promotion _(done)_
  - `tests/unit/test_loaders.py`, `test_ingestion.py`, `test_parsing_repositories.py`, `test_technique_benchmark.py` — routing and integration _(done)_
- **Acceptance Criteria:**
  - [x] Feature-flagged or backward-compatible defaults preserved
  - [x] Unit tests pass for new modules
  - [x] Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Disabled by default — plain `PdfLoader`/`DocxLoader` unchanged when `enabled: false`. Docling is an optional runtime dependency (`uv pip install docling`); missing package or unknown provider raises `ConfigurationError` / `DocumentLoadError`. Parser instance is cached keyed on `(enabled, provider)`; `clear_layout_parser_cache()` supports tests and live settings reload. Document-level `tables`/`figures`/`sections`/`headings` stay on `Document.metadata` for T-202 but are excluded from chunk payloads via `chunk_metadata()`. README documents enablement, ingestion routing, metadata keys, and mermaid flows.

---


### T-201 · PPT/PPTX Loader
- **Status:** `[x]`
- **Goal:** Add `.pptx` to `SUPPORTED_EXTENSIONS`.
- **Inputs:** T-190
- **Outputs:** `PptxLoader`
- **Files:** `src/infrastructure/loaders/pptx_loader.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-202 · Structured Table Chunks at Ingest
- **Status:** `[x]`
- **Goal:** Emit `type=table` chunks with `table_id`.
- **Inputs:** T-200, T-190, T-015
- **Outputs:** Table chunks in Qdrant+BM25; skip-path backfill/purge for unchanged documents; embed-failure retention
- **Files:**
  - `src/rag/ingestion/table_chunker.py` — `TableChunker`, `build_table_chunks()`, `table_chunk_id()` (UUIDv5 on `source:table_id`), `is_table_chunk()`, table-specific sync wrappers (`table_chunks_needing_upsert`, `retained_table_chunk_ids_on_embed_failure`, `merged_table_chunk_ids`, `stale_table_ids_safe_to_purge`, `existing_table_chunk_ids`) _(done)_
  - `src/rag/ingestion/structured_chunk_sync.py` — shared build/embed/upsert helpers reused by caption chunks (T-232) _(done)_
  - `src/infrastructure/parsers/docling_parser.py` — table `text` in metadata via `export_to_markdown()` _(done)_
  - `src/rag/pipelines/ingestion_pipeline.py` — `_build_table_chunker()`, full-path index, `_backfill_table_chunks_on_skip()`, retention on purge _(done)_
  - `src/core/settings.py` — `TableChunkSettings` _(done)_
  - `configs/parsing.yaml` — `table_chunks.enabled` flag _(done)_
  - `tests/unit/test_table_chunker.py` — chunker, stable IDs, skip-path backfill/purge, embed-failure retention, pipeline, settings _(done)_
  - `README.md` — enablement, mermaid flows, skip-path / retention notes _(done)_
- **Acceptance Criteria:**
  - [x] Feature-flagged or backward-compatible defaults preserved (`enabled: false`)
  - [x] Unit tests pass for new modules
  - [x] Documented in `configs/parsing.yaml` or relevant config when applicable
  - [x] Stable chunk IDs keyed on `source` (not ephemeral `document.id`) for idempotent reindex
  - [x] Unchanged documents backfill missing/updated table chunks and purge stale ones when sync succeeds
  - [x] Failed table embed/build retains previously indexed table points
- **Notes:** Disabled by default. Best results with T-200 layout `tables[]` (markdown pipe-table fallback when layout text is missing). Table chunks go to Qdrant **and** BM25 (unlike HyPE/summary extras). Skip-path sync is skipped when LLM enrichers (augmentation / HyPE / hierarchical) force a full reindex via `_requires_full_reindex_on_skip()`. Stale purge only runs after successful build+embed (`table_sync_succeeded`). Shared skip-path primitives live in `structured_chunk_sync.py` (also used by T-232).

---


## Phase 21 — Multimodal Domain Model (Priority 11)

> **Motivation:** Extend `Chunk`/`Answer`/`SourceReference` after parsing shapes are proven.
>
> **Preconditions:** Phases 19–20
>
> **Status:** complete — T-210 ✅

---

### T-210 · Multimodal Domain Model Extensions
- **Status:** `[x]`
- **Goal:** Add modality fields and `SourceReference`.
- **Inputs:** T-003, T-190, T-202
- **Outputs:** Extended entities
- **Files:**
  - `src/domain/entities/source_reference.py` — `SourceReference`, `resolve_modality`, `source_references_for_chunks` _(done)_
  - `src/domain/entities/chunk.py` — `modality`, `image_embedding`, `asset_path` _(done)_
  - `src/domain/entities/answer.py` — `source_references` _(done)_
  - `src/domain/entities/__init__.py` — export new symbols _(done)_
  - `src/core/constants.py` — `MODALITY_*`, `KNOWN_MODALITIES`, `CHUNK_TYPE_TO_MODALITY`, `ASSET_PATH_KEY` _(done)_
  - `configs/parsing.yaml` — T-210 domain-model note _(done)_
  - `tests/unit/test_source_reference.py` — modality helpers, SourceReference, Answer wiring _(done)_
  - `tests/unit/test_entities.py`, `test_parsing_repositories.py` — defaults + constants _(done)_
- **Acceptance Criteria:**
  - [x] Feature-flagged or backward-compatible defaults preserved
  - [x] Unit tests pass for new modules
  - [x] Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Domain-only phase — no ingestion/API wiring. `Answer.sources` stays `list[str]`; `source_references` defaults empty. Legacy table chunks (metadata `type=table` only) resolve via `resolve_modality` / `SourceReference.from_chunk`. `image_embedding` / `asset_path` reserved for T-230/T-250+. README documents the model.

---


## Phase 22 — OCR Pipeline (Priority 12)

> **Motivation:** Self-hosted OCR + Azure Document Intelligence (not OCR-Form-Tools UI).
>
> **Preconditions:** Phases 19–20
>
> **Status:** T-220 ✅ · T-221 ✅ · T-222 ✅ · T-223 ✅

---

### T-220 · OCR Provider Abstraction & Settings
- **Status:** `[x]`
- **Goal:** `get_ocr_provider()` factory.
- **Inputs:** T-190
- **Outputs:** OCR factory
- **Files:** `src/infrastructure/ocr/__init__.py`, tests
- **Acceptance Criteria:**
  - [x] Feature-flagged or backward-compatible defaults preserved
  - [x] Unit tests pass for new modules
  - [x] Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Factory mirrors `get_layout_parser` via shared `EnabledProviderCache` — key `(enabled, provider, identity)`, `None` when disabled. Self-hosted providers implemented in T-221; `azure_di` identity fingerprint + disposal in T-222.

---


### T-221 · Self-Hosted OCR Providers
- **Status:** `[x]`
- **Goal:** Tesseract, EasyOCR, Docling OCR.
- **Inputs:** T-220
- **Outputs:** Self-hosted providers
- **Files:** `src/infrastructure/ocr/*_provider.py`, tests
- **Acceptance Criteria:**
  - [x] Feature-flagged or backward-compatible defaults preserved
  - [x] Unit tests pass for new modules
  - [x] Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Docling-backed `OcrRepository` implementations — `TesseractOcrProvider` (Tesseract CLI), `EasyOcrProvider`, `DoclingOcrProvider` (auto engine). Shared logic in `docling_backed.py`. Optional dep: `uv pip install docling`. Factory returns providers when `parsing.ocr.enabled=true`; still `None` when disabled. Wired into ingest by **T-223** (`apply_ocr_fallback`). `azure_di` is T-222.

---


### T-222 · Azure Document Intelligence OCR Provider
- **Status:** `[x]`
- **Goal:** Azure DI REST API (FOTT backend, not labeling UI).
- **Inputs:** T-220
- **Outputs:** `AzureDocumentIntelligenceOcr`
- **Files:** `src/infrastructure/ocr/azure_di_provider.py`, `docs/ocr-providers.md`, factory branch in `src/infrastructure/ocr/__init__.py`, tests, README / `configs/parsing.yaml`
- **Acceptance Criteria:**
  - [x] Feature-flagged or backward-compatible defaults preserved (`parsing.ocr.provider=azure_di` only constructs when enabled + credentials present)
  - [x] Unit tests pass for new modules (mock Azure DI HTTP; no live calls in CI)
  - [x] Documented in `configs/parsing.yaml` / README — settings, credentials, and when to prefer Azure DI vs self-hosted
  - [x] `apply_ocr_fallback` (T-223) works unchanged with the Azure DI provider once registered in `get_ocr_provider()`
- **Notes:** `AzureDocumentIntelligenceOcr` posts local files as `base64Source` to Document Intelligence (`prebuilt-read`), polls `Operation-Location`, returns `analyzeResult.content`. Credentials under `parsing.ocr.azure_di` (`PARSING__OCR__AZURE_DI__*`). Missing credentials raise `ConfigurationError` (T-223 soft-fails). Factory caches by Azure DI identity (endpoint, API key, API version, model ID, timeout, poll interval) and calls `close()` on the previous client when the identity changes or the cache is cleared. See `docs/ocr-providers.md`.

---


### T-223 · Scanned-PDF Detection & OCR Fallback Router
- **Status:** `[x]`
- **Goal:** Route low-text pages through OCR during ingest.
- **Inputs:** T-010, T-221 (self-hosted) / T-222 (Azure DI, optional), T-200
- **Outputs:** OCR fallback
- **Files:** `src/rag/ingestion/ocr_fallback.py`, `src/rag/pipelines/ingestion_pipeline.py` (dual-hash helpers + skip-path preserve), `src/core/constants.py` (`OCR_APPLIED_KEY`), `configs/parsing.yaml` (`min_chars`), `tests/unit/test_ocr_fallback.py`, README
- **Acceptance Criteria:**
  - [x] Feature-flagged or backward-compatible defaults preserved (`parsing.ocr.enabled=false` → no-op)
  - [x] When OCR is enabled, low-text / empty PDF loads call `get_ocr_provider()` and replace content with OCR text
  - [x] Born-digital PDFs with enough extractable text skip OCR
  - [x] Unit tests pass for new modules (include pipeline wiring with mocked provider)
  - [x] Documented in `configs/parsing.yaml` / README — enabling the flag + re-ingest recovers scanned PDFs
  - [x] Dual-hash dedup: skip matches text `content_hash` or PDF `source_file_hash`; successful OCR stores file hash; failed OCR stores pending hash for retry
  - [x] Toggling `enabled` / `min_chars` does not wipe OCR-derived chunks or force whole-file OCR over already-indexed extractable text
  - [x] Skip path preserves index and still backfills table chunks when layout `tables[]` exist (empty OCR candidates without layout tables do not purge prior tables)
- **Notes:** Closes the T-221 gap: providers exist and ingest calls them via `apply_ocr_fallback` after `load_document`. Whole-file OCR only when every page (or overall content) is below `min_chars` non-whitespace chars — mixed born-digital + scanned docs are not overwritten. Provider construction runs only after the low-text check. Soft-fails on `DocumentLoadError`, empty OCR text, and `ConfigurationError` (misconfigured provider / missing Azure DI credentials). Azure DI provider (T-222) is optional — self-hosted engines from T-221 are sufficient.

---


## Phase 23 — VLM Image Understanding (Priority 13)

> **Motivation:** Figure assets, VLM captions, caption chunks.
>
> **Preconditions:** Phases 20–21
>
> **Status:** **complete** — T-230 ✅ → T-231 ✅ → T-232 ✅
>
> **Next:** Phase 24 (T-241)

---

### T-230 · Figure Asset Extraction & Storage
- **Status:** `[x]`
- **Goal:** Persist figures with `figure_id`.
- **Inputs:** T-200, T-201, T-210
- **Outputs:** Asset store
- **Files:** `figure_extractor.py`, `local_asset_store.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Extract figures from layout `figures[]` (T-200) / PPTX (T-201); persist bytes under a local asset store and set `Chunk.asset_path` / `figure_id` (T-210 fields). VLM captions are T-231. `LocalAssetStore` writes under `parsing.figure_assets.store_dir` (default `data/assets`); `apply_figure_assets` runs after OCR on full ingest and skip path; soft-fails per figure. DOCX backfills via `python-docx` when Docling is unavailable, fails, or returns no raster bytes, aligning blobs to figure slots in document order (page/bbox, then remaining). PPTX walks nested group shapes. `build_figure_chunks()` builds modality=figure chunks for T-231/T-232.

---


### T-231 · VLM Captioning at Ingest
- **Status:** `[x]`
- **Goal:** Gemini/OpenAI vision captions at ingest.
- **Inputs:** T-230, T-030
- **Outputs:** Caption text per figure
- **Files:** `figure_captioner.py`, vision providers, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Depends on stored figure assets from T-230. Prefer feature-flagged vision provider selection; keep ingest soft-fail when VLM is unavailable. `VisionRepository` + OpenAI/Gemini providers under `src/infrastructure/vision/`; `apply_figure_captions` in `figure_captioner.py` runs after `apply_figure_assets` on full + skip paths. Persist successful captions as hash-bound `{stem}.caption.txt` sidecars next to assets so skip-path re-ingests reload without re-calling the VLM when bytes are unchanged. Caption chunk indexing is T-232 (`parsing.caption_chunks`).

---


### T-232 · Image Caption Chunks
- **Status:** `[x]`
- **Goal:** Index `type=caption` chunks.
- **Inputs:** T-231, T-202, T-015
- **Outputs:** Caption chunks in Qdrant+BM25; skip-path backfill/purge for unchanged documents; embed-failure retention
- **Files:**
  - `src/rag/ingestion/caption_chunker.py` — `CaptionChunker`, `build_caption_chunks()`, `caption_chunk_id()` (UUIDv5 on `source:figure_id`), caption sync wrappers _(done)_
  - `src/rag/ingestion/structured_chunk_sync.py` — shared build/embed/upsert helpers with table chunks _(done)_
  - `src/rag/pipelines/ingestion_pipeline.py` — `_build_caption_chunker()`, full-path index, `_backfill_caption_chunks_on_skip()`, shared skip backfill helper _(done)_
  - `src/core/settings.py` — `CaptionChunkSettings` _(done)_
  - `configs/parsing.yaml` / `.env.example` — `caption_chunks.enabled` / `PARSING__CAPTION_CHUNKS__ENABLED` _(done)_
  - `tests/unit/test_caption_chunker.py` — chunker, stable IDs, skip-path backfill/purge, embed-failure retention, pipeline _(done)_
  - `tests/unit/ingestion_helpers.py` — shared structured-chunk skip-path fixtures _(done)_
  - `README.md` — enablement, mermaid flows, skip-path / retention notes _(done)_
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Emit `CHUNK_TYPE_CAPTION` chunks linked to `figure_id`; index like table chunks (T-202 pattern) once captions exist. Feature flag `parsing.caption_chunks.enabled` (default off). Full ingest extends indexed chunks; skip path backfills/purges stable UUIDv5 IDs like tables. Only figures with non-empty `figures[].caption` emit chunks (caption removal purges prior caption points when sync succeeds). Failed embeds retain previously indexed caption IDs via `merged_caption_chunk_ids()`.

---


## Phase 24 — Structure-Aware Chunking (Priority 14)

> **Motivation:** Section/page chunking, config wiring, type registry.
>
> **Preconditions:** Phases 20–21
>
> **Status:** **in progress** — T-240 ✅ · **T-241** ← next

---

### T-240 · Section-Boundary Chunker
- **Status:** `[x]`
- **Goal:** Split on headings; set `metadata.section`.
- **Inputs:** T-011, T-200
- **Outputs:** `SectionChunker`
- **Files:**
  - `src/rag/chunking/section_chunker.py` — `SectionChunker` with inner `RecursiveChunker` _(done)_
  - `src/rag/chunking/headings.py` — markdown / outline / slide segment splitters _(done)_
  - `src/core/markdown_headings.py` — shared ATX `HEADING_RE` + `extract_markdown_headings` (used by MarkdownLoader) _(done)_
  - `src/rag/chunking/__init__.py` — register `section` strategy _(done)_
  - `src/core/settings.py` — `ChunkingSettings.strategy` includes `section` _(done)_
  - `configs/retrieval.yaml` / `.env.example` — strategy comment _(done)_
  - `tests/unit/test_section_chunker.py` _(done)_
  - `README.md` — enablement, mermaid, T-240 subsection _(done)_
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable
- **Notes:** Opt-in via `chunking.strategy: section` (default remains `recursive`). Splits on ATX headings first, then DOCX outline whole-line titles, then PPTX `---` slides; oversized sections use `RecursiveChunker`. Per-chunk `CHUNK_SECTION_KEY` (preamble omits it). No new nested settings — `chunk_size` / `overlap` only.

---


### T-241 · Page-Boundary Chunker & Per-Chunk Page Metadata
- **Status:** `[ ]` ← **start here**
- **Goal:** Set `metadata.page` per chunk.
- **Inputs:** T-200, T-240
- **Outputs:** `PageAwareChunker`
- **Files:** `page_chunker.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-242 · IngestionPipeline Chunker Config Wiring Fix
- **Status:** `[ ]`
- **Goal:** Forward all chunking kwargs from settings.
- **Inputs:** T-115, T-011
- **Outputs:** Fixed `from_settings()`
- **Files:** `ingestion_pipeline.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-243 · Modality Chunk Type Registry & Index Routing
- **Status:** `[ ]`
- **Goal:** Centralize BM25/dense routing per type.
- **Inputs:** T-202, T-232, T-015
- **Outputs:** Type registry
- **Files:** `chunk_type_registry.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


## Phase 25 — Multimodal Embeddings & Vector Indexing (Priority 15)

> **Motivation:** CLIP/Voyage embeddings, Qdrant `image_dense`, ingest wiring.
>
> **Preconditions:** Phase 21

---

### T-250 · EmbeddingRepository Image API Extension
- **Status:** `[ ]`
- **Goal:** Add `embed_image()`.
- **Inputs:** T-004, T-210
- **Outputs:** Extended ABC
- **Files:** `embedding_repository.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-251 · Multimodal Embedding Provider (CLIP / Voyage-Multimodal)
- **Status:** `[ ]`
- **Goal:** Image+text embedding provider.
- **Inputs:** T-250, T-108
- **Outputs:** CLIP/Voyage provider
- **Files:** `clip_provider.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-252 · Qdrant Multimodal Schema Extension
- **Status:** `[ ]`
- **Goal:** Add optional `image_dense` vector.
- **Inputs:** T-013, T-251, T-105
- **Outputs:** Schema v2
- **Files:** `qdrant.py`, migration script, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-253 · Multimodal Ingestion Pipeline Wiring
- **Status:** `[ ]`
- **Goal:** End-to-end multimodal ingest.
- **Inputs:** T-200..T-232, T-252, T-015
- **Outputs:** Full ingest path
- **Files:** `ingestion_pipeline.py`, integration tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


## Phase 26 — Multimodal Retrieval & Reranking (Priority 16)

> **Motivation:** Image dense retriever, cross-modal fusion, modality reranking.
>
> **Preconditions:** Phase 25

---

### T-260 · Image Dense Retriever
- **Status:** `[ ]`
- **Goal:** Search `image_dense` named vector.
- **Inputs:** T-251, T-252, T-021
- **Outputs:** `ImageDenseRetriever`
- **Files:** `image_dense_retriever.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-261 · Cross-Modal Hybrid Fusion
- **Status:** `[ ]`
- **Goal:** Add image leg to `HybridRetriever` RRF.
- **Inputs:** T-260, T-022, T-243
- **Outputs:** Fusion leg
- **Files:** `hybrid_retriever.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-262 · Modality-Aware Reranking
- **Status:** `[ ]`
- **Goal:** Table/caption boost; implement `qwen_reranker`.
- **Inputs:** T-023, T-131, T-243
- **Outputs:** Modality rerank
- **Files:** `cross_encoder.py`, `qwen_reranker.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-263 · Per-Leg RRF Weight Configuration
- **Status:** `[ ]`
- **Goal:** Configurable RRF weights per leg.
- **Inputs:** T-022, T-261
- **Outputs:** Weighted RRF
- **Files:** `score_fusion.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


## Phase 27 — Multimodal Generation & Attribution (Priority 17)

> **Motivation:** Mixed-modality prompts, vision LLM, rich citations, chunk API.
>
> **Preconditions:** Phases 21, 24–25

---

### T-270 · Mixed-Modality System Prompts
- **Status:** `[ ]`
- **Goal:** `rag_assistant_multimodal.txt`.
- **Inputs:** T-031, T-202/T-232
- **Outputs:** Multimodal prompt
- **Files:** prompts + `generation_service.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-271 · Vision-Capable LLM Generation Path
- **Status:** `[ ]`
- **Goal:** Pass figure assets to vision LLM.
- **Inputs:** T-231, T-031, T-230
- **Outputs:** Vision generation
- **Files:** LLM + `chat_pipeline.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-272 · Rich SourceReference Model & API Response
- **Status:** `[ ]`
- **Goal:** Structured citations in API.
- **Inputs:** T-210, T-144, T-032
- **Outputs:** `source_references`
- **Files:** entities + API schemas, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-273 · Chunk Lookup API
- **Status:** `[ ]`
- **Goal:** `GET /chunks/{chunk_id}`.
- **Inputs:** T-013, T-032, T-272
- **Outputs:** Chunk API
- **Files:** `routers/chunks.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-274 · Agent & Streaming Multimodal Explain/Highlight
- **Status:** `[ ]`
- **Goal:** T-143/T-144 on agent endpoints.
- **Inputs:** T-114, T-144, T-270
- **Outputs:** Agent explain/highlight
- **Files:** `agent_pipeline.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


## Phase 28 — Multimodal Evaluation (Priority 18)

> **Motivation:** Golden dataset, modality metrics, regression gate.
>
> **Preconditions:** Phases 25–26

---

### T-280 · Multimodal Golden Dataset Builder
- **Status:** `[ ]`
- **Goal:** Table/figure QA samples.
- **Inputs:** T-040, T-253
- **Outputs:** Golden JSONL
- **Files:** `build_multimodal_golden.py`, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-281 · Table & Figure Retrieval Evals
- **Status:** `[ ]`
- **Goal:** `table_recall@k`, `figure_recall@k`.
- **Inputs:** T-280, T-041, T-260
- **Outputs:** Modality metrics
- **Files:** `modality_recall.py`, benchmark CLI, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---


### T-282 · Multimodal Regression Gate
- **Status:** `[ ]`
- **Goal:** CI-optional regression check.
- **Inputs:** T-281, T-152, T-182
- **Outputs:** Regression gate
- **Files:** `regression_gate.py`, CI wiring, tests
- **Acceptance Criteria:**
  - Feature-flagged or backward-compatible defaults preserved
  - Unit tests pass for new modules
  - Documented in `configs/parsing.yaml` or relevant config when applicable

---

## Dependency Graph

```
T-001 ──► T-002
T-001 ──► T-003 ──► T-004 ──► T-005
                    T-004 ──► T-010 ──► T-011 ──► T-012 ──► T-013
                                                            T-014
                                               T-011+T-012+T-013+T-014 ──► T-015
                    T-015 ──► T-020 ──► T-021 ──► T-022 ──► T-023 ──► T-024 ──► T-025
                    T-025 + T-030 ──► T-031 ──► T-032
                    T-015 + T-031 ──► T-040 ──► T-041 ──► T-042 ──► T-043
                    T-031 ──► T-050 ──► T-051
                    T-043 ──► T-060 ──► T-061
T-061 ──► T-080 ──► T-081 ──► T-082 ──► T-083 ──► T-084 ──► T-085
T-082 ──► T-090 ──► T-091 ──► T-092 ──► T-093 ──► T-094 ──► T-095
T-091 ──► T-096
T-095 ──► T-097
T-100 ──► T-101 ──► T-107
T-100 ──► T-102 ──► T-107
T-100 ──► T-103 ──► T-107
T-100 ──► T-104 ──► T-107
T-101+T-102+T-103+T-104 ──► T-106 ──► T-107
T-105 ──► T-108
T-107 + T-108 ──► T-109
T-110 ──► T-115 ──► T-116 ──► T-117
T-112 ──► T-111 ──► T-113
T-111 + T-071 ──► T-114
T-110 + T-025 ──► T-130 ──► T-132
T-131 ──► T-132
T-110 ──► T-133
T-013 ──► T-134
T-023 ──► T-135
T-120 ──► T-121 ──► T-122
T-011 ──► T-123 ──► T-124
T-015 ──► T-125
T-011 ──► T-126
T-140 ──► T-141 ──► T-142
T-140 ──► T-144
T-143 ──► T-144
T-013 + T-117 ──► T-145 ──► T-146
T-146 + T-160 ──► (feedback rate limiting closed)
T-146 ──► T-172 (scenario 5 closed in test_feedback_concurrency.py)
T-043 + T-110..T-145 ──► T-150
T-011 + T-043 ──► T-151
T-040 + T-061 ──► T-152
T-032 + T-051 ──► T-160
T-061 ──► T-161 ──► T-162
T-030 + T-031 ──► T-163
T-111 + T-112 ──► T-164
T-014 + T-015 ──► T-165
T-060 + T-061 ──► T-170 ──► T-171 ──► T-173
T-043 + T-051 ──► T-172
T-146 ──► T-172
T-163 + T-164 + T-165 ──► T-172
T-061 + T-171 ──► T-180 ──► T-181
T-180 ──► T-182
T-004 ──► T-190
T-190 ──► T-200 ──► T-202
T-190 ──► T-201
T-190 + T-202 ──► T-210
T-190 ──► T-220 ──► T-221
T-220 ──► T-222
T-200 + T-220 + T-221 ──► T-223
T-222 ──► T-223  _(optional Azure DI provider)_
T-200 + T-201 + T-210 ──► T-230 ──► T-231 ──► T-232
T-200 + T-210 ──► T-240 ──► T-241
T-011 + T-115 ──► T-242
T-202 + T-232 ──► T-243
T-004 + T-210 ──► T-250 ──► T-251
T-013 + T-105 + T-251 ──► T-252
T-200..T-232 + T-243 + T-252 + T-210 ──► T-253
T-251 + T-252 ──► T-260 ──► T-261
T-023 + T-131 + T-243 ──► T-262
T-022 + T-261 ──► T-263
T-202 + T-232 + T-241 + T-210 ──► T-270
T-231 + T-031 ──► T-271
T-210 + T-144 + T-032 ──► T-272
T-013 + T-272 ──► T-273
T-114 + T-144 + T-270 ──► T-274
T-040 + T-253 ──► T-280 ──► T-281 ──► T-282
T-150 + T-281 ──► T-282
```

## Quick Start Order for an Agent

1. T-001 → T-002 → T-003 → T-004 → T-005 _(Foundation: ~1 session)_
2. T-010 → T-011 → T-012 → T-013 → T-014 → T-015 _(Ingestion: ~2 sessions)_
3. T-020 → T-021 → T-022 → T-023 → T-024 → T-025 _(Retrieval: ~2 sessions)_
4. T-030 → T-031 → T-032 _(Generation & API: ~1 session)_
5. T-040 → T-041 → T-042 → T-043 _(Evals: ~2 sessions)_
6. T-050 → T-051 _(Observability: ~1 session)_
7. T-060 → T-061 _(CI/CD: ~1 session)_
8. T-080 → T-081 → T-082 → T-083 → T-084 → T-085 _(Docker Compose: ~1 session)_
9. T-090 → T-091 → T-092 → T-093 → T-094 → T-095 → T-096 → T-097 _(Kubernetes/EKS: ~2 sessions)_
10. T-100 → T-101 + T-102 + T-103 + T-104 → T-105 → T-106 → T-107 → T-108 → T-109 _(Embedding Provider Expansion: ~2 sessions)_
11. **Phase 11 — Priority 1 (Wire Existing Code):** T-112 → T-110 → T-111 → T-113 → T-114 → T-115 → T-116 → T-117 _(~2 sessions)_
12. **Phase 12 — Priority 2 (Index-Time Enrichment):** T-120 → T-121 → T-122 → T-123 → T-124 → T-125 → T-126 _(~3 sessions)_
13. **Phase 13 — Priority 3 (Query Intelligence):** T-131 → T-132 → T-130 → T-133 → T-134 → T-135 _(~2 sessions)_
14. **Phase 14 — Priority 4 (Quality Gates & Explainability):** T-140 → T-141 → T-142 → T-143 → T-144 → T-145 → **T-146** _(~2 sessions + hardening follow-up)_
15. **Phase 15 — Priority 5 (Evaluation Operationalization):** T-150 ✅ → T-151 ✅ → T-152 ✅ _(complete — PR #29, PR #30, PR #31)_
16. **Phase 16 — Priority 6 (Production Hardening & Scalability):** T-160 ✅ → T-161 ✅ → T-162 ✅ (PR #34) → T-163 ✅ → T-164 ✅ (PR #36) → T-165 ✅ _(~2 sessions; Phase 16 complete)_
17. **Phase 17 — Priority 7 (Code Quality & Type Safety):** T-170 ✅ → T-171 ✅ (PR #39) → T-172 ✅ (PR #41) → T-173 ✅ _(Phase 17 complete)_
18. **Phase 18 — Priority 8 (CI Performance):** T-180 ✅ → T-181 ✅ → T-182 ✅

19. **Phase 19 — Priority 9 (Parsing Contracts):** T-190 ✅ _(complete)_
20. **Phase 20 — Priority 10 (Layout Parsing):** T-200 ✅ → T-201 ✅ → T-202 ✅ _(complete)_
21. **Phase 21 — Priority 11 (Domain Model):** T-210 ✅ _(complete)_
22. **Phase 22 — Priority 12 (OCR):** T-220 ✅ → T-221 ✅ → T-222 ✅ → T-223 ✅ _(complete)_
23. **Phase 23 — Priority 13 (VLM):** T-230 ✅ → T-231 ✅ → T-232 ✅ _(complete)_
24. **Phase 24 — Priority 14 (Structure-Aware Chunking):** T-240 ✅ → **T-241** _(next)_ → T-242 → T-243
25. **Phase 25 — Priority 15 (Multimodal Embeddings):** T-250 → T-251 → T-252 → T-253
26. **Phase 26 — Priority 16 (Multimodal Retrieval):** T-260 → T-261 → T-262 → T-263
27. **Phase 27 — Priority 17 (Multimodal Generation & Attribution):** T-270 → T-271 → T-272 → T-273 → T-274
28. **Phase 28 — Priority 18 (Multimodal Evals):** T-280 → T-281 → T-282
