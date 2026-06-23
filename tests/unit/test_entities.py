"""T-003 — domain entity tests.

Covers: round-trip serialisation, auto-generated defaults, immutability,
model_copy updates, and absence of circular imports.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.entities import Answer, Chunk, Document, EvalSample, Query

# ── Document ───────────────────────────────────────────────────────────────────


class TestDocument:
    def test_round_trip(self):
        doc = Document(source="data/raw/manual.pdf", content="hello world")
        restored = Document.model_validate(doc.model_dump())
        assert restored == doc

    def test_id_auto_generated(self):
        a = Document(source="a.pdf", content="x")
        b = Document(source="b.pdf", content="x")
        assert a.id != b.id

    def test_created_at_auto_set(self):
        doc = Document(source="a.pdf", content="x")
        assert doc.created_at is not None

    def test_metadata_defaults_to_empty_dict(self):
        doc = Document(source="a.pdf", content="x")
        assert doc.metadata == {}

    def test_custom_metadata(self):
        doc = Document(source="a.pdf", content="x", metadata={"page": 3})
        assert doc.metadata["page"] == 3

    def test_frozen(self):
        doc = Document(source="a.pdf", content="x")
        with pytest.raises(ValidationError):
            doc.content = "mutated"  # type: ignore[misc]

    def test_explicit_id_accepted(self):
        doc = Document(id="fixed-id", source="a.pdf", content="x")
        assert doc.id == "fixed-id"


# ── Chunk ──────────────────────────────────────────────────────────────────────


class TestChunk:
    def test_round_trip(self):
        chunk = Chunk(document_id="doc-1", text="some text")
        restored = Chunk.model_validate(chunk.model_dump())
        assert restored == chunk

    def test_embedding_defaults_to_none(self):
        chunk = Chunk(document_id="doc-1", text="text")
        assert chunk.embedding is None

    def test_sparse_vector_defaults_to_none(self):
        chunk = Chunk(document_id="doc-1", text="text")
        assert chunk.sparse_vector is None

    def test_add_embedding_via_model_copy(self):
        chunk = Chunk(document_id="doc-1", text="text")
        vector = [0.1, 0.2, 0.3]
        embedded = chunk.model_copy(update={"embedding": vector})
        assert embedded.embedding == vector
        assert chunk.embedding is None  # original unchanged

    def test_add_sparse_vector_via_model_copy(self):
        chunk = Chunk(document_id="doc-1", text="text")
        sparse = {101: 0.9, 202: 0.4}
        updated = chunk.model_copy(update={"sparse_vector": sparse})
        assert updated.sparse_vector == sparse

    def test_round_trip_with_vectors(self):
        chunk = Chunk(
            document_id="doc-1",
            text="text",
            embedding=[0.1, 0.2],
            sparse_vector={1: 0.5, 2: 0.3},
            metadata={"source": "manual.pdf", "page": 1},
        )
        restored = Chunk.model_validate(chunk.model_dump())
        assert restored == chunk

    def test_frozen(self):
        chunk = Chunk(document_id="doc-1", text="text")
        with pytest.raises(ValidationError):
            chunk.text = "mutated"  # type: ignore[misc]

    def test_unique_ids(self):
        a = Chunk(document_id="d", text="t")
        b = Chunk(document_id="d", text="t")
        assert a.id != b.id


# ── Query ──────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_round_trip(self):
        q = Query(text="What is IAM?")
        assert Query.model_validate(q.model_dump()) == q

    def test_expanded_texts_defaults_to_empty(self):
        q = Query(text="question")
        assert q.expanded_texts == []

    def test_embedding_defaults_to_none(self):
        q = Query(text="question")
        assert q.embedding is None

    def test_with_expansions(self):
        q = Query(text="q", expanded_texts=["q1", "q2"])
        assert len(q.expanded_texts) == 2

    def test_frozen(self):
        q = Query(text="q")
        with pytest.raises(ValidationError):
            q.text = "mutated"  # type: ignore[misc]

    def test_round_trip_full(self):
        q = Query(text="q", expanded_texts=["q1"], embedding=[0.1, 0.2, 0.3])
        assert Query.model_validate(q.model_dump()) == q


# ── Answer ─────────────────────────────────────────────────────────────────────


class TestAnswer:
    def test_round_trip(self):
        a = Answer(query_id="q-1", text="The answer is 42.")
        assert Answer.model_validate(a.model_dump()) == a

    def test_sources_defaults_to_empty(self):
        a = Answer(query_id="q-1", text="answer")
        assert a.sources == []

    def test_latency_and_token_defaults(self):
        a = Answer(query_id="q-1", text="answer")
        assert a.latency_ms == 0.0
        assert a.token_count == 0

    def test_full_construction(self):
        a = Answer(
            query_id="q-1",
            text="answer",
            sources=["chunk-1", "chunk-2"],
            latency_ms=123.4,
            token_count=512,
        )
        assert a.sources == ["chunk-1", "chunk-2"]
        assert a.latency_ms == pytest.approx(123.4)
        assert a.token_count == 512

    def test_frozen(self):
        a = Answer(query_id="q-1", text="answer")
        with pytest.raises(ValidationError):
            a.text = "mutated"  # type: ignore[misc]


# ── EvalSample ─────────────────────────────────────────────────────────────────


class TestEvalSample:
    def test_round_trip(self):
        s = EvalSample(question="q?", expected_answer="a")
        assert EvalSample.model_validate(s.model_dump()) == s

    def test_defaults(self):
        s = EvalSample(question="q?", expected_answer="a")
        assert s.retrieved_chunks == []
        assert s.generated_answer == ""
        assert s.scores == {}

    def test_full_construction(self):
        s = EvalSample(
            question="q?",
            expected_answer="expected",
            retrieved_chunks=["c1", "c2"],
            generated_answer="generated",
            scores={"faithfulness": 0.9, "relevance": 0.85},
        )
        restored = EvalSample.model_validate(s.model_dump())
        assert restored == s
        assert restored.scores["faithfulness"] == pytest.approx(0.9)

    def test_frozen(self):
        s = EvalSample(question="q?", expected_answer="a")
        with pytest.raises(ValidationError):
            s.question = "mutated"  # type: ignore[misc]


# ── No circular imports ────────────────────────────────────────────────────────


class TestNoCircularImports:
    def test_all_importable_from_package(self):
        from src.domain.entities import Answer, Chunk, Document, EvalSample, Query  # noqa: F401

    def test_importable_individually(self):
        from src.domain.entities.answer import Answer  # noqa: F401
        from src.domain.entities.chunk import Chunk  # noqa: F401
        from src.domain.entities.document import Document  # noqa: F401
        from src.domain.entities.evaluation import EvalSample  # noqa: F401
        from src.domain.entities.query import Query  # noqa: F401
