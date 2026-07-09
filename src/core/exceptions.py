from __future__ import annotations

from typing import override


class RAGPlatformError(Exception):
    """Base for all application exceptions.

    FastAPI exception handlers should catch this type to return structured
    error responses across every layer of the stack.
    """

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.cause: BaseException | None = cause

    @override
    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


# ── Ingestion ──────────────────────────────────────────────────────────────────


class IngestionError(RAGPlatformError):
    """Raised when any step in the document ingestion pipeline fails."""


class DocumentLoadError(IngestionError):
    """Failed to read or parse a source document."""


class ChunkingError(IngestionError):
    """Failed to split a document into chunks."""


# ── Retrieval ──────────────────────────────────────────────────────────────────


class RetrievalError(RAGPlatformError):
    """Raised when any step in the retrieval pipeline fails."""


class EmbeddingError(RetrievalError):
    """Failed to produce vector representations for a query or chunk."""


class VectorStoreError(RetrievalError):
    """Failed to read from or write to the vector store."""


# ── Generation ─────────────────────────────────────────────────────────────────


class GenerationError(RAGPlatformError):
    """Raised when the LLM generation step fails."""


class LLMTimeoutError(GenerationError):
    """LLM did not respond within the configured time limit."""


# ── Evaluation ─────────────────────────────────────────────────────────────────


class EvaluationError(RAGPlatformError):
    """Raised when an evaluation run fails."""


# ── Configuration ──────────────────────────────────────────────────────────────


class ConfigurationError(RAGPlatformError):
    """Raised when the required configuration is missing or invalid.

    Typically, it means an API key is absent or an unsupported provider is requested.
    """
