"""T-015 integration tests — IngestionPipeline (requires BGE-M3 + Qdrant).

Run with:
    make qdrant-up
    # Download BGE-M3 model to models/embeddings/bge-m3
    uv run pytest tests/integration/test_ingestion_pipeline.py -v
"""

from __future__ import annotations

import pytest

from src.core.constants import MODELS_DIR

_MODEL_PATH = MODELS_DIR / "embeddings" / "bge-m3"
_QDRANT_URL = "http://localhost:6333"


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=_QDRANT_URL, timeout=2, check_compatibility=False).get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _MODEL_PATH.exists() or not _qdrant_reachable(),
    reason="Requires BGE-M3 model and running Qdrant",
)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from src.domain.services.ingestion_service import IngestionService
    from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
    from src.infrastructure.vectordb.bm25 import BM25Index
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore
    from src.rag.chunking import get_chunker
    from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

    bm25_path = tmp_path_factory.mktemp("bm25") / "index.pkl"
    chunker = get_chunker("recursive", chunk_size=200, overlap=20)
    embedder = BGEM3EmbeddingProvider(model_path=str(_MODEL_PATH), device="mps", batch_size=4)
    vector_store = QdrantVectorStore(url=_QDRANT_URL, collection="test_ingest", dense_dim=1024)
    bm25 = BM25Index(index_path=bm25_path)
    service = IngestionService(chunker=chunker, embedder=embedder)
    return IngestionPipeline(service=service, vector_store=vector_store, bm25=bm25)


@pytest.fixture(scope="module")
def sample_md_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("docs") / "sample.md"
    path.write_text(
        "# Introduction\n\nThis document describes IAM roles and policies.\n\n"
        "## Section One\n\nKubernetes pod scheduling requires node affinity rules.\n\n"
        "## Section Two\n\nVector databases store embeddings for similarity search.\n"
    )
    return path


class TestIngestionPipelineIntegration:
    def test_ingest_file_returns_result(self, pipeline, sample_md_file):
        from src.rag.pipelines.ingestion_pipeline import IngestionResult

        result = pipeline.ingest_file(sample_md_file)
        assert isinstance(result, IngestionResult)

    def test_ingest_produces_chunks(self, pipeline, sample_md_file):
        result = pipeline.ingest_file(sample_md_file)
        assert result.chunk_count > 0

    def test_content_hash_set(self, pipeline, sample_md_file):
        result = pipeline.ingest_file(sample_md_file)
        assert len(result.content_hash) == 16

    def test_bm25_indexed(self, pipeline, sample_md_file):
        pipeline.ingest_file(sample_md_file)
        results = pipeline._bm25.search("kubernetes pod", top_k=1)
        assert len(results) >= 1

    def test_qdrant_has_chunks(self, pipeline, sample_md_file):
        pipeline.ingest_file(sample_md_file)
        assert pipeline._vector_store.count() > 0

    def test_save_indexes(self, pipeline):
        pipeline.save_indexes()
        assert pipeline._bm25._path.exists()
