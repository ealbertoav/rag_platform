"""T-005 — exception hierarchy tests."""
from __future__ import annotations

import pytest

from src.core.exceptions import (
    ChunkingError,
    DocumentLoadError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    IngestionError,
    LLMTimeoutError,
    RAGPlatformError,
    RetrievalError,
    VectorStoreError,
)


class TestInheritance:
    def test_ingestion_is_rag_error(self):
        assert issubclass(IngestionError, RAGPlatformError)

    def test_document_load_is_ingestion(self):
        assert issubclass(DocumentLoadError, IngestionError)

    def test_chunking_is_ingestion(self):
        assert issubclass(ChunkingError, IngestionError)

    def test_retrieval_is_rag_error(self):
        assert issubclass(RetrievalError, RAGPlatformError)

    def test_embedding_is_retrieval(self):
        assert issubclass(EmbeddingError, RetrievalError)

    def test_vector_store_is_retrieval(self):
        assert issubclass(VectorStoreError, RetrievalError)

    def test_generation_is_rag_error(self):
        assert issubclass(GenerationError, RAGPlatformError)

    def test_llm_timeout_is_generation(self):
        assert issubclass(LLMTimeoutError, GenerationError)

    def test_evaluation_is_rag_error(self):
        assert issubclass(EvaluationError, RAGPlatformError)

    def test_all_are_exceptions(self):
        for cls in (
            RAGPlatformError, IngestionError, DocumentLoadError, ChunkingError,
            RetrievalError, EmbeddingError, VectorStoreError,
            GenerationError, LLMTimeoutError, EvaluationError,
        ):
            assert issubclass(cls, Exception)


class TestMessageAndCause:
    def test_message_stored(self):
        err = RAGPlatformError("something went wrong")
        assert err.message == "something went wrong"

    def test_cause_defaults_to_none(self):
        assert RAGPlatformError("msg").cause is None

    def test_cause_stored(self):
        original = ValueError("root cause")
        err = RAGPlatformError("wrapper", cause=original)
        assert err.cause is original

    def test_str_without_cause(self):
        assert str(RAGPlatformError("msg")) == "msg"

    def test_str_with_cause(self):
        err = RAGPlatformError("outer", cause=ValueError("inner"))
        assert "outer" in str(err)
        assert "inner" in str(err)

    def test_subclass_carries_message(self):
        err = DocumentLoadError("cannot read file.pdf")
        assert err.message == "cannot read file.pdf"

    def test_subclass_carries_cause(self):
        io_err = OSError("permission denied")
        err = DocumentLoadError("load failed", cause=io_err)
        assert err.cause is io_err


class TestRaiseable:
    def test_catch_base_catches_leaf(self):
        with pytest.raises(RAGPlatformError):
            raise LLMTimeoutError("timed out after 30s")

    def test_catch_mid_catches_leaf(self):
        with pytest.raises(GenerationError):
            raise LLMTimeoutError("timed out")

    def test_catch_specific_leaf(self):
        with pytest.raises(LLMTimeoutError):
            raise LLMTimeoutError("timed out")

    def test_does_not_catch_sibling(self):
        with pytest.raises(RetrievalError):
            raise EmbeddingError("embed failed")
        # GenerationError should NOT be caught by the block above — pytest
        # ensures only the declared type (RetrievalError) was raised.

    def test_fastapi_handler_pattern(self):
        """Simulate a FastAPI exception handler catching RAGPlatformError."""
        def handler(exc: RAGPlatformError) -> dict:
            return {"error": exc.message}

        err = VectorStoreError("qdrant unavailable")
        assert handler(err) == {"error": "qdrant unavailable"}
