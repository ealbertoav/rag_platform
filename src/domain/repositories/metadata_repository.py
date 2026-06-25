from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DocumentRecord:
    """Persisted metadata for an ingested source file."""

    id: str
    source_path: str
    content_hash: str
    chunk_count: int
    ingested_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class IngestionRunRecord:
    """Audit row for a single ingest attempt."""

    id: str
    document_id: str
    status: str
    chunks_added: int
    chunks_skipped: int
    duration_ms: float
    error: str | None = None


class MetadataRepository(ABC):
    """Contract for document ingestion metadata persistence."""

    @abstractmethod
    def get_by_source(self, source_path: str) -> DocumentRecord | None:
        """Return the document record for *source_path*, if any."""

    @abstractmethod
    def get_chunk_ids(self, document_id: str) -> list[str]:
        """Return chunk IDs previously indexed for *document_id*."""

    @abstractmethod
    def upsert_document(
        self,
        source_path: str,
        content_hash: str,
        chunk_ids: list[str],
        *,
        duration_ms: float = 0.0,
        skipped: bool = False,
        error: str | None = None,
    ) -> DocumentRecord:
        """Create or update a document record and its chunk ID list."""

    @abstractmethod
    def list_documents(self) -> list[DocumentRecord]:
        """Return all ingested documents ordered by most recently updated."""
