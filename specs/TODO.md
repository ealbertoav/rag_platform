# RAG Platform — Specification-Driven TODO

> **Format:** Each task is a self-contained specification executable by an AI agent.
> Fields: **Goal**, **Inputs**, **Outputs**, **Files**, **Acceptance Criteria**, **Notes**.
> Status: `[ ]` pending · `[~]` in progress · `[x]` done

---

## Phase 0 — Foundation

### T-001 · Core Settings & Configuration loader
- **Status:** `[x]`
- **Goal:** Implement a Pydantic-Settings model that reads from `.env` and `configs/*.yaml`, exposing a single `settings` singleton used across the entire app.
- **Inputs:** `.env.example`, `configs/app.yaml`, `configs/llm.yaml`, `configs/embeddings.yaml`, `configs/retrieval.yaml`, `configs/logging.yaml`
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
  - Index serializes/deserializes to `data/processed/bm25_index.pkl`
  - `search(query, top_k)` returns `list[tuple[Chunk, float]]` sorted by score
  - Supports incremental updates (re-index on new chunks)

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
  - Reads source-of-truth chunks from `BM25Index` (persisted pickle)
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

## Phase 7 — Future: Graph RAG (Parking Lot)

> These are designed-for but not implemented in MVP. Architecture supports plug-in without refactoring.

### T-070 · Knowledge Graph Layer (Neo4j)
- **Status:** `[x]`
- **Goal:** Extract entity relationships from ingested documents and store in Neo4j. Add `graph_retriever.py` alongside `hybrid_retriever.py`.
- **Files:** `src/infrastructure/vectordb/neo4j.py`, `src/rag/retrieval/graph_retriever.py`
- **Note:** `HybridRetriever` already accepts an optional `graph_retriever` param (wired to `None` until this task).

---

### T-071 · Agentic RAG
- **Status:** `[x]`
- **Goal:** Add a tool-calling agent layer that can decide when to retrieve, when to ask clarifying questions, and when to combine multiple retrievals.
- **Files:** `src/rag/pipelines/agent_pipeline.py`
- **Note:** Requires Graph RAG (T-070) for multi-hop reasoning.

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
