"""T-011 — chunking strategy tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.constants import CHUNK_INDEX_KEY, CHUNK_PARENT_ID_KEY, CHUNK_SOURCE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking import Chunker, get_chunker
from src.rag.chunking.parent_child_chunker import ParentChildChunker
from src.rag.chunking.recursive_chunker import RecursiveChunker
from src.rag.chunking.semantic_chunker import SemanticChunker

# ── helpers ────────────────────────────────────────────────────────────────────

_PARA = "word " * 120  # ~120 tokens each paragraph


def _ones_encoder(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 2))


def _doc(content: str, source: str = "test.md") -> Document:
    return Document(source=source, content=content)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _max_tokens(chunks: list[Chunk]) -> int:
    return max(_approx_tokens(c.text) for c in chunks)


# ── token approximation utility ───────────────────────────────────────────────


class TestApproxTokens:
    def test_empty_string_returns_one(self):
        assert _approx_tokens("") == 1

    def test_proportional_to_length(self):
        assert _approx_tokens("a" * 400) == 100


# ── RecursiveChunker ───────────────────────────────────────────────────────────


class TestRecursiveChunker:
    def test_returns_list_of_chunks(self):
        doc = _doc("Hello world. " * 10)
        result = RecursiveChunker().chunk(doc)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_document_id_set(self):
        doc = _doc("text " * 50)
        chunks = RecursiveChunker().chunk(doc)
        assert all(c.document_id == doc.id for c in chunks)

    def test_no_chunk_exceeds_max_tokens(self):
        long_text = (_PARA + "\n\n") * 10
        chunks = RecursiveChunker(chunk_size=200).chunk(_doc(long_text))
        assert _max_tokens(chunks) <= 200

    def test_short_text_produces_single_chunk(self):
        chunks = RecursiveChunker(chunk_size=500).chunk(_doc("Short text."))
        assert len(chunks) == 1
        assert "Short text" in chunks[0].text

    def test_long_text_produces_multiple_chunks(self):
        text = (_PARA + "\n\n") * 6  # ~720 tokens
        chunks = RecursiveChunker(chunk_size=200).chunk(_doc(text))
        assert len(chunks) > 1

    def test_overlap_overlap_error(self):
        with pytest.raises(ValueError, match="overlap"):
            RecursiveChunker(chunk_size=100, overlap=100)

    def test_source_in_metadata(self):
        chunks = RecursiveChunker().chunk(_doc("text", source="my/doc.pdf"))
        assert all(c.metadata[CHUNK_SOURCE_KEY] == "my/doc.pdf" for c in chunks)

    def test_index_in_metadata(self):
        text = (_PARA + "\n\n") * 6
        chunks = RecursiveChunker(chunk_size=200).chunk(_doc(text))
        indices = [c.metadata[CHUNK_INDEX_KEY] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_content_covers_full_document(self):
        # All words from the doc should appear somewhere in the chunks
        word = "UNIQUE_WORD_XYZ"
        text = f"intro text. {word}. more text. " + "filler " * 200
        chunks = RecursiveChunker(chunk_size=100).chunk(_doc(text))
        combined = " ".join(c.text for c in chunks)
        assert word in combined

    def test_splits_on_paragraph_boundary_first(self):
        # With two paragraphs each fitting in chunk_size, we should get ≤ 2 chunks
        para = "word " * 60  # ~60 tokens
        text = para + "\n\n" + para
        chunks = RecursiveChunker(chunk_size=200).chunk(_doc(text))
        assert len(chunks) <= 2

    def test_get_chunker_returns_recursive(self):
        chunker = get_chunker("recursive", chunk_size=300)
        assert isinstance(chunker, RecursiveChunker)


# ── SemanticChunker ────────────────────────────────────────────────────────────


def _fixed_encoder(texts: list[str]):
    """Returns embeddings that make pairs 0-1 similar and 1-2 dissimilar."""
    rng = np.random.default_rng(seed=42)
    # All sentences in the same segment get similar embeddings.
    embeddings = []
    for i, _ in enumerate(texts):
        # Alternate between two very different embedding directions.
        base = np.array([1.0, 0.0]) if i < len(texts) // 2 else np.array([0.0, 1.0])
        noise = rng.normal(0, 0.01, 2)
        embeddings.append(base + noise)
    return np.array(embeddings)


class TestSemanticChunker:
    def test_returns_list_of_chunks(self):
        doc = _doc("First sentence. Second sentence. Third sentence.")
        chunks = SemanticChunker(encode=lambda t: np.eye(len(t))).chunk(doc)
        assert isinstance(chunks, list)

    def test_document_id_set(self):
        doc = _doc("Sentence one. Sentence two.")
        chunks = SemanticChunker(encode=_ones_encoder).chunk(doc)
        assert all(c.document_id == doc.id for c in chunks)

    def test_empty_document_returns_empty(self):
        doc = _doc("")
        assert SemanticChunker(encode=_ones_encoder).chunk(doc) == []

    def test_splits_at_topic_boundary(self):
        # Two blocks of 5 sentences each; encoder makes block 0 vs. 1 dissimilar.
        sentences_a = [f"Topic A sentence {i}." for i in range(5)]
        sentences_b = [f"Topic B sentence {i}." for i in range(5)]
        text = " ".join(sentences_a + sentences_b)
        doc = _doc(text)
        chunks = SemanticChunker(
            similarity_threshold=0.3,
            encode=_fixed_encoder,
        ).chunk(doc)
        assert len(chunks) >= 2

    def test_no_chunk_exceeds_max_tokens(self):
        text = ". ".join(["word " * 40] * 10)
        chunks = SemanticChunker(max_tokens=100, encode=_ones_encoder).chunk(_doc(text))
        assert _max_tokens(chunks) <= 100

    def test_source_in_metadata(self):
        doc = _doc("One sentence. Two sentence.", source="s.md")
        chunks = SemanticChunker(encode=_ones_encoder).chunk(doc)
        assert all(c.metadata[CHUNK_SOURCE_KEY] == "s.md" for c in chunks)

    def test_get_chunker_returns_semantic(self):
        chunker = get_chunker("semantic", encode=_ones_encoder)
        assert isinstance(chunker, SemanticChunker)


# ── ParentChildChunker ─────────────────────────────────────────────────────────


class TestParentChildChunker:
    @staticmethod
    def _chunker() -> ParentChildChunker:
        return ParentChildChunker(parent_chunk_size=300, child_chunk_size=80, overlap=10)

    @staticmethod
    def _long_doc() -> Document:
        return _doc((_PARA + "\n\n") * 8)  # ~960 tokens

    def test_returns_list_of_chunks(self):
        assert isinstance(self._chunker().chunk(self._long_doc()), list)

    def test_produces_parents_and_children(self):
        chunks = self._chunker().chunk(self._long_doc())
        parents = [c for c in chunks if CHUNK_PARENT_ID_KEY not in c.metadata]
        children = [c for c in chunks if CHUNK_PARENT_ID_KEY in c.metadata]
        assert len(parents) >= 1
        assert len(children) >= 1

    def test_children_reference_valid_parents(self):
        chunks = self._chunker().chunk(self._long_doc())
        parent_ids = {c.id for c in chunks if CHUNK_PARENT_ID_KEY not in c.metadata}
        children = [c for c in chunks if CHUNK_PARENT_ID_KEY in c.metadata]
        for child in children:
            assert child.metadata[CHUNK_PARENT_ID_KEY] in parent_ids

    def test_children_smaller_than_parents(self):
        chunks = self._chunker().chunk(self._long_doc())
        parents = [c for c in chunks if CHUNK_PARENT_ID_KEY not in c.metadata]
        children = [c for c in chunks if CHUNK_PARENT_ID_KEY in c.metadata]
        avg_parent = sum(_approx_tokens(p.text) for p in parents) / len(parents)
        avg_child = sum(_approx_tokens(c.text) for c in children) / len(children)
        assert avg_child < avg_parent

    def test_no_child_exceeds_child_chunk_size(self):
        chunker = ParentChildChunker(parent_chunk_size=400, child_chunk_size=100, overlap=10)
        children = [
            c for c in chunker.chunk(self._long_doc())
            if CHUNK_PARENT_ID_KEY in c.metadata
        ]
        assert _max_tokens(children) <= 100

    def test_document_id_set_on_all_chunks(self):
        doc = self._long_doc()
        for c in self._chunker().chunk(doc):
            assert c.document_id == doc.id

    def test_child_size_must_be_smaller_than_parent(self):
        with pytest.raises(ValueError, match="child_chunk_size"):
            ParentChildChunker(parent_chunk_size=100, child_chunk_size=200)

    def test_get_chunker_returns_parent_child(self):
        chunker = get_chunker("parent_child", parent_chunk_size=400, child_chunk_size=100)
        assert isinstance(chunker, ParentChildChunker)


# ── Chunker protocol ───────────────────────────────────────────────────────────


class TestChunkerProtocol:
    def test_recursive_satisfies_protocol(self):
        chunker: Chunker = RecursiveChunker()
        assert callable(chunker.chunk)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_chunker("unknown_strategy")
