"""Production-like E2E — ingest "AI Engineering.pdf" and chat about it over real HTTP.

Exercises the deployed stack exactly as a client would: upload the PDF to a
running API container, then ask it a question and check the LLM's answer.
Point .env at NVIDIA NIM's free tier (LLM__PROVIDER=nvidia_nim) and bring the
stack up with:

    make docker-up-prod   # base compose only — no dev hot-reload/Ollama swap
    make test-e2e

Skipped automatically when the API isn't reachable, so it's a no-op in CI/unit
runs that don't have the stack running.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from src.core.constants import ROOT

_API_BASE_URL = "http://localhost:8000"
_PDF_PATH = ROOT / "AI Engineering.pdf"


def _reachable() -> bool:
    try:
        return httpx.get(f"{_API_BASE_URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _PDF_PATH.exists(),
        reason=f"Fixture PDF not found at {_PDF_PATH}",
    ),
    pytest.mark.skipif(
        not _reachable(),
        reason=f"API not reachable at {_API_BASE_URL} — run `make docker-up-prod` first",
    ),
]


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    # CPU-embedding a full book's worth of chunks (BGE-M3, ~800 chunks) can take
    # several minutes; the reranker + NIM chat call adds more on top of that.
    with httpx.Client(base_url=_API_BASE_URL, timeout=900) as c:
        yield c


@pytest.fixture(scope="module")
def ingested_document(client: httpx.Client) -> dict:
    with _PDF_PATH.open("rb") as f:
        response = client.post(
            "/ingest/upload",
            files={"file": (_PDF_PATH.name, f, "application/pdf")},
        )
    assert response.status_code == 200, response.text
    return response.json()


class TestIngestAndChat:
    def test_ingest_upload_indexes_chunks(self, ingested_document: dict) -> None:
        assert ingested_document["status"] == "ok"
        assert ingested_document["chunk_count"] > 0

    def test_chat_full_answers_question_about_document(
        self, client: httpx.Client, ingested_document: dict
    ) -> None:
        response = client.post(
            "/chat/full",
            json={"question": "What is this document about? Give a one-sentence summary."},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert isinstance(body["answer"], str)
        assert body["answer"].strip() != ""
        assert len(body["sources"]) > 0
        assert body["token_count"] > 0
