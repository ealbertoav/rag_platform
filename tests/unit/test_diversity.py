"""T-135 — MMR diversity retrieval tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.services.retrieval_service import RetrievalService
from src.rag.ranking.diversity import mmr_select


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", text=text)


def _vec(*values: float) -> list[float]:
    return list(values)


class TestMmrSelect:
    def test_empty_input(self):
        assert mmr_select([], [], lambda_=0.7, top_k=5) == []

    def test_length_mismatch_raises(self):
        chunks = [_chunk("c0")]
        with pytest.raises(ValueError, match="length mismatch"):
            mmr_select(chunks, [], lambda_=0.7, top_k=1)

    def test_lambda_one_returns_relevance_order(self):
        chunks = [_chunk("c0"), _chunk("c1"), _chunk("c2")]
        embeddings = [_vec(1, 0), _vec(0.99, 0.01), _vec(0, 1)]
        result = mmr_select(chunks, embeddings, lambda_=1.0, top_k=2)
        assert [c.id for c in result] == ["c0", "c1"]

    def test_lambda_one_with_single_chunk(self):
        chunks = [_chunk("c0")]
        embeddings = [_vec(1.0, 0.0)]
        result = mmr_select(chunks, embeddings, lambda_=1.0, top_k=5)
        assert result == chunks

    def test_prefers_diverse_chunk_over_near_duplicate(self):
        """Second slot should go to the diverse chunk, not a near-duplicate."""
        chunks = [
            _chunk("best", text="primary hit"),
            _chunk("dup", text="near duplicate"),
            _chunk("alt", text="different topic"),
        ]
        embeddings = [
            _vec(1.0, 0.0, 0.0),
            _vec(0.99, 0.01, 0.0),
            _vec(0.0, 1.0, 0.0),
        ]
        result = mmr_select(chunks, embeddings, lambda_=0.5, top_k=2)
        assert [c.id for c in result] == ["best", "alt"]

    def test_top_k_caps_output(self):
        chunks = [_chunk(f"c{i}") for i in range(5)]
        embeddings = [_vec(float(i), 1.0) for i in range(5)]
        result = mmr_select(chunks, embeddings, lambda_=0.7, top_k=3)
        assert len(result) == 3

    def test_zero_top_k_returns_empty(self):
        chunks = [_chunk("c0")]
        embeddings = [_vec(1.0)]
        assert mmr_select(chunks, embeddings, lambda_=0.7, top_k=0) == []


class TestRetrievalServiceDiversity:
    @pytest.fixture
    def reranked_chunks(self) -> list[Chunk]:
        return [
            Chunk(id="c0", document_id="doc", text="topic A", embedding=_vec(1, 0)),
            Chunk(id="c1", document_id="doc", text="topic A duplicate", embedding=_vec(0.99, 0.01)),
            Chunk(id="c2", document_id="doc", text="topic B", embedding=_vec(0, 1)),
        ]

    def _service(
        self,
        *,
        diversity_enabled: bool = False,
        diversity_lambda: float = 0.5,
        reranked_chunks: list[Chunk] | None = None,
    ) -> RetrievalService:
        chunks = reranked_chunks or []
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(c, 0.9) for c in chunks])
        reranker = MagicMock()
        reranker.rerank.return_value = chunks
        return RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=hybrid,
            reranker=reranker,
            top_k_retrieval=10,
            top_k_rerank=3,
            diversity_enabled=diversity_enabled,
            diversity_lambda=diversity_lambda,
        )

    @pytest.mark.asyncio
    async def test_disabled_preserves_reranker_order(self, reranked_chunks):
        svc = self._service(diversity_enabled=False, reranked_chunks=reranked_chunks)
        result = await svc.retrieve(Query(text="test"))
        assert [c.id for c in result.chunks] == ["c0", "c1", "c2"]

    @pytest.mark.asyncio
    async def test_enabled_promotes_diverse_chunk(self, reranked_chunks):
        svc = self._service(
            diversity_enabled=True,
            diversity_lambda=0.5,
            reranked_chunks=reranked_chunks,
        )
        result = await svc.retrieve(Query(text="test"))
        assert [c.id for c in result.chunks] == ["c0", "c2", "c1"]

    @pytest.mark.asyncio
    async def test_lambda_one_skips_diversity_reorder(self, reranked_chunks):
        svc = self._service(
            diversity_enabled=True,
            diversity_lambda=1.0,
            reranked_chunks=reranked_chunks,
        )
        result = await svc.retrieve(Query(text="test"))
        assert [c.id for c in result.chunks] == ["c0", "c1", "c2"]

    @pytest.mark.asyncio
    async def test_embeds_missing_vectors_when_embedder_provided(self):
        chunks = [
            Chunk(id="c0", document_id="doc", text="alpha"),
            Chunk(id="c1", document_id="doc", text="beta"),
        ]
        embedder = MagicMock()
        embedder.embed_passage.return_value = [_vec(1, 0), _vec(0, 1)]
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(c, 0.9) for c in chunks])
        reranker = MagicMock()
        reranker.rerank.return_value = chunks
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=hybrid,
            reranker=reranker,
            top_k_retrieval=10,
            top_k_rerank=2,
            diversity_enabled=True,
            diversity_lambda=0.5,
            embedder=embedder,
        )
        await svc.retrieve(Query(text="test"))
        embedder.embed_passage.assert_called_once_with(["alpha", "beta"])
        embedder.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolves_embeddings_from_chunk_lookup_before_embedder(self):
        chunks = [
            Chunk(id="c0", document_id="doc", text="alpha"),
            Chunk(id="c1", document_id="doc", text="beta"),
        ]
        lookup = MagicMock()
        lookup.get_by_id.side_effect = lambda chunk_id: {
            "c0": Chunk(id="c0", document_id="doc", text="alpha", embedding=_vec(1, 0)),
            "c1": Chunk(id="c1", document_id="doc", text="beta", embedding=_vec(0, 1)),
        }.get(chunk_id)
        embedder = MagicMock()
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(c, 0.9) for c in chunks])
        reranker = MagicMock()
        reranker.rerank.return_value = chunks
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=hybrid,
            reranker=reranker,
            top_k_retrieval=10,
            top_k_rerank=2,
            diversity_enabled=True,
            diversity_lambda=0.5,
            embedder=embedder,
            chunk_lookup=lookup,
        )
        await svc.retrieve(Query(text="test"))
        embedder.embed_passage.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_diversity_when_embeddings_unavailable(self):
        chunks = [
            Chunk(id="c0", document_id="doc", text="alpha"),
            Chunk(id="c1", document_id="doc", text="beta"),
        ]
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(c, 0.9) for c in chunks])
        reranker = MagicMock()
        reranker.rerank.return_value = chunks
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=hybrid,
            reranker=reranker,
            top_k_retrieval=10,
            top_k_rerank=2,
            diversity_enabled=True,
            diversity_lambda=0.5,
        )
        result = await svc.retrieve(Query(text="test"))
        assert [c.id for c in result.chunks] == ["c0", "c1"]


class TestDiversityPassageEmbeddingProviders:
    def test_voyage_diversity_uses_document_input_type(self) -> None:
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        provider = VoyageEmbeddingProvider(api_key="voy-test")
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2]])
        provider._client = mock_client

        provider.embed_passage(["chunk text about kubernetes"])
        mock_client.embed.assert_called_once_with(
            ["chunk text about kubernetes"],
            model="voyage-large-2",
            input_type="document",
        )

    def test_cohere_diversity_uses_search_document_input_type(self) -> None:
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="coh-test")
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=MagicMock(float_=[[0.1, 0.2]]))
        provider._client = mock_client

        provider.embed_passage(["chunk text about kubernetes"])
        mock_client.embed.assert_called_once()
        assert mock_client.embed.call_args.kwargs["input_type"] == "search_document"

    def test_gemini_diversity_uses_retrieval_document_task_type(self) -> None:
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="gem-test")
        with patch(
            "src.infrastructure.embeddings.gemini_provider.GeminiEmbeddingProvider._call_api",
            return_value=[[0.1, 0.2]],
        ) as mock_call:
            provider.embed_passage(["chunk text about kubernetes"])
            mock_call.assert_called_once_with(
                ["chunk text about kubernetes"],
                "RETRIEVAL_DOCUMENT",
            )
