.PHONY: install sync serve ingest evals benchmark lint format test test-unit test-e2e clean qdrant-up \
        docker-build docker-up docker-down docker-logs docker-ingest docker-clean

install:
	uv sync --extra dev --extra evals

sync:
	uv sync

serve:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

ingest:
	uv run python scripts/ingest.py --source $(SOURCE)

evals:
	uv run python scripts/run_evals.py

benchmark:
	uv run python scripts/benchmark.py

lint:
	uv run ruff check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:
	uv run pytest tests/unit tests/integration -v --cov=src --cov-report=term-missing

test-unit:
	uv run pytest tests/unit -v

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
