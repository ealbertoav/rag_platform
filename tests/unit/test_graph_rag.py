"""T-070 / T-164 — Knowledge Graph (EntityExtractor, GraphRetriever, Neo4jGraphRepository)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.infrastructure.vectordb.neo4j_graph import GraphRelation, Neo4jGraphRepository
from src.rag.retrieval.graph_retriever import EntityExtractor, GraphRetriever

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=f"text {i}")


def _llm_mock(response: str = "[]") -> MagicMock:
    m = MagicMock()
    m.generate.return_value = response
    return m


def _bm25_mock(chunk: Chunk | None = None) -> MagicMock:
    m = MagicMock()
    m.get_by_id.return_value = chunk or _chunk(0)
    return m


def _neo4j_mock(results: list[tuple[str, float]] | None = None) -> MagicMock:
    m = MagicMock()
    m.search_by_entities = AsyncMock(return_value=results or [("c0", 1.0)])
    return m


def _async_session_context(session: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ── Relation parsing (tested through EntityExtractor.extract_relations) ────────


def _extractor(response: str) -> EntityExtractor:
    return EntityExtractor(_llm_mock(response))


class TestRelationParsing:
    def test_valid_json_parsed(self):
        json_str = (
            '[{"subject":"EKS","relation":"uses","object":"IAM",'
            '"subject_type":"Technology","object_type":"Concept"}]'
        )
        result = _extractor(json_str).extract_relations("text")
        assert len(result) == 1
        assert result[0].subject == "EKS"
        assert result[0].relation == "uses"

    def test_empty_list_returns_empty(self):
        assert _extractor("[]").extract_relations("text") == []

    def test_embedded_json_found(self):
        text = (
            "Here are the triples:\n"
            '[{"subject":"A","relation":"r","object":"B","subject_type":"T","object_type":"T"}]'
        )
        result = _extractor(text).extract_relations("input")
        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        assert _extractor("not json").extract_relations("text") == []

    def test_missing_fields_skipped(self):
        assert _extractor('[{"subject":"A"}]').extract_relations("text") == []

    def test_defaults_entity_type(self):
        json_str = '[{"subject":"A","relation":"r","object":"B"}]'
        result = _extractor(json_str).extract_relations("text")
        assert len(result) == 1
        assert result[0].subject_type == "Entity"


# ── EntityExtractor ────────────────────────────────────────────────────────────


class TestEntityExtractor:
    def test_extract_relations_returns_list(self):
        json_str = (
            '[{"subject":"EKS","relation":"uses","object":"IAM",'
            '"subject_type":"Technology","object_type":"Concept"}]'
        )
        llm = _llm_mock(json_str)
        ex = EntityExtractor(llm)
        result = ex.extract_relations("EKS uses IAM roles.")
        assert len(result) == 1
        assert isinstance(result[0], GraphRelation)

    def test_llm_failure_returns_empty(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        ex = EntityExtractor(llm)
        assert ex.extract_relations("text") == []

    def test_extract_entity_names_capitalised(self):
        ex = EntityExtractor(_llm_mock())
        names = ex.extract_entity_names("Amazon EKS uses IAM roles for authentication")
        assert "Amazon" in names or "EKS" in names

    def test_extract_entity_names_empty_string(self):
        ex = EntityExtractor(_llm_mock())
        assert ex.extract_entity_names("") == []


# ── GraphRetriever ─────────────────────────────────────────────────────────────


class TestGraphRetriever:
    @staticmethod
    def _retriever(
        graph_results: list[tuple[str, float]] | None = None,
        chunk: Chunk | None = None,
    ) -> GraphRetriever:
        ex = EntityExtractor(_llm_mock())
        graph = _neo4j_mock(graph_results)
        bm25 = _bm25_mock(chunk)
        return GraphRetriever(extractor=ex, graph=graph, bm25=bm25)

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        r = self._retriever([("c0", 0.9)])
        results = await r.search("AWS EKS IAM roles", top_k=3)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_no_entities_returns_empty(self):
        r = self._retriever()
        results = await r.search("what is this?", top_k=3)
        # No capitalised tokens → no entities → empty
        assert results == []

    @pytest.mark.asyncio
    async def test_chunk_not_in_bm25_skipped(self):
        ex = EntityExtractor(_llm_mock())
        graph = _neo4j_mock([("ghost-id", 1.0)])
        bm25 = MagicMock()
        bm25.get_by_id.return_value = None  # chunk not found
        retriever = GraphRetriever(extractor=ex, graph=graph, bm25=bm25)
        assert await retriever.search("AWS EKS", top_k=5) == []

    @pytest.mark.asyncio
    async def test_graph_failure_returns_empty(self):
        ex = EntityExtractor(_llm_mock())
        graph = MagicMock()
        graph.search_by_entities = AsyncMock(side_effect=RuntimeError("Neo4j down"))
        bm25 = _bm25_mock()
        retriever = GraphRetriever(extractor=ex, graph=graph, bm25=bm25)
        assert await retriever.search("AWS EKS", top_k=5) == []

    @pytest.mark.asyncio
    async def test_document_id_filter_excludes_out_of_scope_chunks(self):
        chunk = Chunk(id="c0", document_id="doc-a", text="text 0")
        r = self._retriever([("c0", 0.9)], chunk=chunk)
        filt = RetrievalFilter(document_ids=["doc-b"])
        assert await r.search("AWS EKS IAM roles", top_k=3, filters=filt) == []

    @pytest.mark.asyncio
    async def test_document_id_filter_scoped_in_graph_before_limit(self):
        """In-scope chunks must be considered even when globally outranked."""
        chunk_in = Chunk(id="c-in", document_id="doc-a", text="text in")
        ex = EntityExtractor(_llm_mock())
        graph = _neo4j_mock([("c-in", 0.5)])
        bm25 = MagicMock()
        bm25.get_by_id.side_effect = lambda cid: chunk_in if cid == "c-in" else None
        retriever = GraphRetriever(extractor=ex, graph=graph, bm25=bm25)
        filt = RetrievalFilter(document_ids=["doc-a"])

        results = await retriever.search("AWS EKS IAM roles", top_k=1, filters=filt)

        graph.search_by_entities.assert_awaited_once()
        args, kwargs = graph.search_by_entities.await_args  # type: ignore[union-attr]
        assert args[1] == 1
        assert kwargs["filters"] == filt
        assert len(results) == 1
        assert results[0][0].id == "c-in"

    @pytest.mark.asyncio
    async def test_metadata_filter_overfetches_before_limit(self):
        chunk = Chunk(id="c0", document_id="doc", text="text 0", metadata={"lang": "en"})
        ex = EntityExtractor(_llm_mock())
        graph = _neo4j_mock([("c0", 0.9)])
        bm25 = _bm25_mock(chunk)
        retriever = GraphRetriever(extractor=ex, graph=graph, bm25=bm25)
        filt = RetrievalFilter(metadata={"lang": "en"})

        await retriever.search("AWS EKS IAM roles", top_k=3, filters=filt)

        args, kwargs = graph.search_by_entities.await_args  # type: ignore[union-attr]
        assert args[1] == 30  # top_k * _METADATA_OVERFETCH
        assert kwargs["filters"] == filt


# ── HybridRetriever with graph ────────────────────────────────────────────────


class TestHybridRetrieverWithGraph:
    @pytest.mark.asyncio
    async def test_graph_results_included_in_fusion(self):
        from src.domain.entities.query import Query
        from src.rag.retrieval.hybrid_retriever import HybridRetriever

        c0, c1, c2 = _chunk(0), _chunk(1), _chunk(2)
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(c0, 0.9), (c1, 0.7)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [(c1, 1.2)]
        graph_mock = MagicMock()
        graph_mock.search = AsyncMock(return_value=[(c2, 1.0)])

        hr = HybridRetriever(dense=dense_mock, bm25=bm25_mock, graph_retriever=graph_mock)
        results = await hr.retrieve(Query(text="AWS EKS"), top_k=3)
        ids = {c.id for c, _ in results}
        assert "c0" in ids
        assert "c1" in ids
        assert "c2" in ids

    @pytest.mark.asyncio
    async def test_no_graph_still_works(self):
        from src.domain.entities.query import Query
        from src.rag.retrieval.hybrid_retriever import HybridRetriever

        c0 = _chunk(0)
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(c0, 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = []

        hr = HybridRetriever(dense=dense_mock, bm25=bm25_mock)  # no graph_retriever
        results = await hr.retrieve(Query(text="query"), top_k=3)
        assert len(results) == 1


# ── Neo4jGraphRepository ───────────────────────────────────────────────────────


class TestNeo4jGraphRepository:
    def test_from_settings_returns_instance(self):
        assert isinstance(Neo4jGraphRepository.from_settings(), Neo4jGraphRepository)

    def test_connection_failure_raises_retrieval_error(self):
        from src.core.exceptions import RetrievalError

        repo = Neo4jGraphRepository(uri="bolt://localhost:9999")
        with (
            patch(
                "src.infrastructure.vectordb.neo4j_graph.AsyncGraphDatabase",
                create=True,
            ) as mock_gdb,
            pytest.raises(RetrievalError),
        ):
            mock_gdb.driver.side_effect = Exception("refused")
            repo._get_driver()

    @pytest.mark.asyncio
    async def test_upsert_empty_relations_is_noop(self):
        repo = Neo4jGraphRepository()
        repo._driver = MagicMock()  # type: ignore[assignment]
        await repo.upsert([], chunk_id="c0")  # must not raise
        repo._driver.session.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_search_empty_entities_returns_empty(self):
        repo = Neo4jGraphRepository()
        assert await repo.search_by_entities([], top_k=5) == []

    @pytest.mark.asyncio
    async def test_upsert_with_relations(self):
        from src.infrastructure.vectordb.neo4j_graph import GraphRelation

        repo = Neo4jGraphRepository()
        mock_session = MagicMock()
        mock_session.execute_write = AsyncMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value = _async_session_context(mock_session)
        repo._driver = mock_driver

        rel = GraphRelation(subject="EKS", relation="uses", object="IAM")
        await repo.upsert([rel], chunk_id="c0", document_id="doc-1")
        mock_session.execute_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_by_entities_returns_results(self):
        repo = Neo4jGraphRepository()
        mock_session = MagicMock()
        mock_session.execute_read = AsyncMock(return_value=[("c0", 1.0)])
        mock_driver = MagicMock()
        mock_driver.session.return_value = _async_session_context(mock_session)
        repo._driver = mock_driver

        results = await repo.search_by_entities(["EKS"], top_k=3)
        assert results == [("c0", 1.0)]

    @pytest.mark.asyncio
    async def test_close_clears_driver(self):
        repo = Neo4jGraphRepository()
        mock_driver = MagicMock()
        mock_driver.close = AsyncMock()
        repo._driver = mock_driver
        await repo.close()
        mock_driver.close.assert_awaited_once()
        assert repo._driver is None

    def test_from_settings_with_neo4j_config(self):
        mock_settings = MagicMock()
        mock_settings.neo4j = MagicMock(
            uri="bolt://neo4j:7687",
            user="admin",
            password="secret",
            max_connection_pool_size=50,
        )
        with patch("src.core.settings.settings", mock_settings):
            repo = Neo4jGraphRepository.from_settings()
        assert repo.uri == "bolt://neo4j:7687"
        assert repo.user == "admin"
        assert repo.password == "secret"
        assert repo.max_connection_pool_size == 50

    @pytest.mark.asyncio
    async def test_upsert_failure_raises_retrieval_error(self):
        from src.core.exceptions import RetrievalError
        from src.infrastructure.vectordb.neo4j_graph import GraphRelation

        repo = Neo4jGraphRepository()
        mock_driver = MagicMock()
        mock_driver.session.side_effect = RuntimeError("neo4j down")
        repo._driver = mock_driver
        with pytest.raises(RetrievalError, match="upsert"):
            await repo.upsert([GraphRelation("A", "r", "B")], chunk_id="c0")

    @pytest.mark.asyncio
    async def test_search_failure_raises_retrieval_error(self):
        from src.core.exceptions import RetrievalError

        repo = Neo4jGraphRepository()
        mock_driver = MagicMock()
        mock_driver.session.side_effect = RuntimeError("neo4j down")
        repo._driver = mock_driver
        with pytest.raises(RetrievalError, match="search"):
            await repo.search_by_entities(["EKS"], top_k=3)

    def test_upsert_sync_delegates_to_async(self):
        repo = Neo4jGraphRepository()
        with patch.object(repo, "upsert", new_callable=AsyncMock) as mock_upsert:
            repo.upsert_sync([GraphRelation("A", "r", "B")], chunk_id="c0", document_id="doc")
        mock_upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_sync_works_inside_running_loop(self):
        """FastAPI ingest calls sync upsert from an async handler event loop."""
        repo = Neo4jGraphRepository()
        with patch.object(repo, "upsert", new_callable=AsyncMock) as mock_upsert:
            repo.upsert_sync([GraphRelation("A", "r", "B")], chunk_id="c0", document_id="doc")
        mock_upsert.assert_awaited_once()

    def test_upsert_sync_reuses_background_loop_across_calls(self):
        """Multi-chunk CLI ingestion must not break on a cached async driver."""
        repo = Neo4jGraphRepository()
        with patch.object(repo, "upsert", new_callable=AsyncMock) as mock_upsert:
            rel = GraphRelation("A", "r", "B")
            repo.upsert_sync([rel], chunk_id="c0", document_id="doc")
            repo.upsert_sync([rel], chunk_id="c1", document_id="doc")
        assert mock_upsert.await_count == 2

    def test_close_sync_delegates_to_async(self):
        repo = Neo4jGraphRepository()
        with patch.object(repo, "close", new_callable=AsyncMock) as mock_close:
            repo.close_sync()
        mock_close.assert_awaited_once()

    def test_driver_uses_connection_pool_size(self):
        repo = Neo4jGraphRepository(
            uri="bolt://localhost:7687",
            max_connection_pool_size=25,
        )
        mock_driver = MagicMock()
        fake_neo4j = MagicMock()
        fake_neo4j.AsyncGraphDatabase.driver.return_value = mock_driver
        with patch.dict("sys.modules", {"neo4j": fake_neo4j}):
            assert repo._get_driver() is mock_driver
        fake_neo4j.AsyncGraphDatabase.driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "neo4j"),
            max_connection_pool_size=25,
        )


class _AsyncRecordIter:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self._index = 0

    def __aiter__(self) -> _AsyncRecordIter:
        return self

    async def __anext__(self) -> dict[str, object]:
        if self._index >= len(self._rows):
            raise StopAsyncIteration
        row = self._rows[self._index]
        self._index += 1
        return row


class TestNeo4jCypherHelpers:
    @pytest.mark.asyncio
    async def test_upsert_relation_runs_cypher(self):
        from src.infrastructure.vectordb.neo4j_graph import GraphRelation, _upsert_relation

        tx = MagicMock()
        tx.run = AsyncMock()
        rel = GraphRelation(subject="A", relation="rel", object="B")
        await _upsert_relation(tx, rel, "c1", "doc-1")
        tx.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_chunks_returns_scores(self):
        from src.infrastructure.vectordb.neo4j_graph import _search_chunks

        tx = MagicMock()
        tx.run = AsyncMock(return_value=_AsyncRecordIter([{"chunk_id": "c1", "score": 0.75}]))
        results = await _search_chunks(tx, ["A", "B"], top_k=5)
        assert results == [("c1", 0.75)]

    @pytest.mark.asyncio
    async def test_search_chunks_scopes_document_ids_in_cypher(self):
        from src.infrastructure.vectordb.neo4j_graph import _search_chunks

        tx = MagicMock()
        tx.run = AsyncMock(return_value=_AsyncRecordIter([{"chunk_id": "c1", "score": 1.0}]))
        await _search_chunks(tx, ["A"], top_k=2, document_ids=["doc-a"])
        cypher = tx.run.call_args[0][0]
        assert "c.document_id IN $document_ids" in cypher
        assert tx.run.call_args[1]["document_ids"] == ["doc-a"]
