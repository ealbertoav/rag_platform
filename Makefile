.PHONY: install sync serve ingest ingest-eval-corpus evals sync-retrieval-goldens multimodal-golden benchmark lint format test test-unit test-slow test-e2e clean qdrant-up \
        docker-build docker-up docker-up-prod docker-down docker-logs docker-ingest docker-clean benchmark-techniques \
        benchmark-chunk-sizes benchmark-infra benchmark-modality-recall check-multimodal-regression audit-deps

install:
	uv sync --group dev --extra evals

sync:
	uv sync

serve:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

ingest:
	uv run python scripts/ingest.py --source $(SOURCE)

## Re-ingest the committed source of qa_dataset.json / retrieval_dataset.json (#94),
## so check_regression_gate.py's live retrieval check has real data to verify against
## instead of skipping. Committed under datasets/goldens/, not data/raw/ (gitignored).
ingest-eval-corpus:
	uv run python scripts/ingest.py --source datasets/goldens/rag_platform_corpus.md

evals:
	@echo "Generate golden QA + retrieval datasets (requires: make ingest SOURCE=... first)"
	uv run python scripts/run_evals.py --output datasets/goldens/qa_dataset.json

sync-retrieval-goldens:
	@echo "Sync retrieval_dataset.json from qa_dataset.json (no LLM regeneration)"
	uv run python scripts/sync_retrieval_golden.py

multimodal-golden:
	@echo "Generate multimodal (table/figure) QA golden dataset (requires: make ingest with table/figure chunks enabled)"
	uv run python scripts/build_multimodal_golden.py

benchmark:
	uv run python scripts/benchmark.py

benchmark-techniques:
	uv run python scripts/benchmark_techniques.py

benchmark-chunk-sizes:
	uv run python scripts/benchmark_chunk_sizes.py

benchmark-infra:
	uv run python scripts/benchmark_infra.py

benchmark-modality-recall:
	@echo "Table/figure Recall@K against the multimodal golden (requires: make multimodal-golden)"
	uv run python scripts/benchmark_modality_recall.py

check-multimodal-regression:
	@echo "Multimodal (table/figure) regression gate — skips gracefully without a multimodal golden"
	uv run python scripts/check_multimodal_regression_gate.py

audit-deps:
	./scripts/check_dependencies.sh

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src
	uv run basedpyright --level error src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:
	uv run pytest tests/unit tests/integration -v --cov=src --cov-report=term-missing

test-unit:
	uv run pytest tests/unit -m "not slow" -v

test-slow:
	uv run pytest tests/unit -m slow -v

test-e2e:
	uv run pytest tests/e2e -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov

qdrant-up:
	docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
		-v $(PWD)/data/qdrant:/qdrant/storage \
		qdrant/qdrant

# ── Docker Compose ────────────────────────────────────────────────────────────
# Prerequisites: copy .env.example → .env before first run.

docker-build:
	docker compose build

docker-up:
	docker compose up -d

## Base compose only — skips docker-compose.override.yml (dev hot-reload + Ollama swap),
## so the API serves with .env as-is (e.g. LLM__PROVIDER=nvidia_nim) like a real deployment.
docker-up-prod:
	docker compose -f docker-compose.yml up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

## Run a one-shot ingestion job. Override source: make docker-ingest SOURCE=/app/data/raw/report.pdf
docker-ingest:
	docker compose run --rm worker python scripts/ingest.py --source $(if $(SOURCE),$(SOURCE),/app/data/raw)

## Destroys all containers AND named volumes (qdrant data, ollama models). Irreversible.
docker-clean:
	docker compose down --volumes
