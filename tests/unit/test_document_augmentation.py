"""T-121 — document augmentation (synthetic questions) tests."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.core.constants import (
    CHUNK_RAW_TEXT_KEY,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_SYNTHETIC,
    SOURCE_CHUNK_ID_KEY,
)
from src.domain.entities.chunk import Chunk
from src.rag.enrichment.document_augmentation import (
    DocumentAugmentor,
    generate_questions,
    is_synthetic_question,
    load_question_template,
    make_question_chunk,
    resolve_synthetic_questions,
)


def _source_chunk(text: str = "Revenue grew 12% year over year.") -> Chunk:
    return Chunk(
        id="source-1",
        document_id="doc-1",
        text=f"[Document: report.pdf]\n{text}",
        metadata={CHUNK_RAW_TEXT_KEY: text, "section": "Revenue"},
    )


class TestGenerateQuestions:
    def test_parses_json_string_list(self):
        llm = MagicMock()
        llm.generate.return_value = '["What was revenue growth?", "How much did revenue grow?"]'
        questions = generate_questions(_source_chunk(), llm, n=3)
        assert questions == ["What was revenue growth?", "How much did revenue grow?"]

    def test_parses_json_object_list(self):
        llm = MagicMock()
        llm.generate.return_value = '[{"question": "What drove growth?"}]'
        questions = generate_questions(_source_chunk(), llm, n=1)
        assert questions == ["What drove growth?"]

    def test_uses_raw_text_not_header(self):
        llm = MagicMock()
        llm.generate.return_value = '["Q1"]'
        generate_questions(_source_chunk("Body only."), llm, n=1)
        prompt = llm.generate.call_args.kwargs.get("prompt") or llm.generate.call_args.args[0]
        assert "Body only." in prompt
        assert "[Document:" not in prompt

    def test_respects_max_count(self):
        llm = MagicMock()
        llm.generate.return_value = '["Q1", "Q2", "Q3", "Q4"]'
        questions = generate_questions(_source_chunk(), llm, n=2)
        assert questions == ["Q1", "Q2"]

    def test_returns_empty_on_unparseable_response(self, caplog):
        llm = MagicMock()
        llm.generate.return_value = "Sorry, I cannot generate questions."
        with caplog.at_level(logging.WARNING):
            assert generate_questions(_source_chunk(), llm, n=3) == []
        assert "Could not parse synthetic questions" in caplog.text

    def test_returns_empty_on_empty_or_whitespace_response(self):
        llm = MagicMock()
        llm.generate.return_value = ""
        assert generate_questions(_source_chunk(), llm, n=3) == []

        llm.generate.return_value = "   "
        assert generate_questions(_source_chunk(), llm, n=3) == []

    def test_returns_empty_on_invalid_json(self):
        llm = MagicMock()
        llm.generate.return_value = "[not valid json"
        assert generate_questions(_source_chunk(), llm, n=3) == []

    def test_returns_empty_on_empty_json_list(self):
        llm = MagicMock()
        llm.generate.return_value = "[]"
        assert generate_questions(_source_chunk(), llm, n=3) == []

    def test_returns_empty_on_json_object_instead_of_list(self):
        llm = MagicMock()
        llm.generate.return_value = '{"question": "Q?"}'
        assert generate_questions(_source_chunk(), llm, n=1) == []

    def test_skips_blank_strings_in_json_list(self):
        llm = MagicMock()
        llm.generate.return_value = '["", "  ", "Valid?"]'
        assert generate_questions(_source_chunk(), llm, n=3) == ["Valid?"]

    def test_extracts_json_array_embedded_in_prose(self):
        llm = MagicMock()
        llm.generate.return_value = 'Here are questions:\n["Embedded question?"]\nThanks.'
        questions = generate_questions(_source_chunk(), llm, n=1)
        assert questions == ["Embedded question?"]

    def test_loads_default_template_from_disk(self):
        template = load_question_template()
        assert "$n" in template.template
        assert "$passage" in template.template


class TestMakeQuestionChunk:
    def test_metadata_links_to_source(self):
        source = _source_chunk()
        question = make_question_chunk(source, "What was revenue growth?")
        assert question.metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_SYNTHETIC
        assert question.metadata[SOURCE_CHUNK_ID_KEY] == source.id
        assert question.document_id == source.document_id
        assert question.text == "What was revenue growth?"
        assert CHUNK_RAW_TEXT_KEY not in question.metadata


class TestDocumentAugmentor:
    def test_disabled_path_not_applicable(self):
        """Augmentor is only constructed when enabled; this tests the augment API."""
        llm = MagicMock()
        llm.generate.return_value = '["Question one?"]'
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1, 0.2]], [{1: 0.9}])
        augmentor = DocumentAugmentor(llm=llm, embedder=embedder, n_questions=2)
        result = augmentor.augment([_source_chunk()])
        assert len(result) == 1
        assert result[0].embedding == [0.1, 0.2]
        assert is_synthetic_question(result[0])

    def test_failure_on_one_chunk_continues(self, caplog):
        llm = MagicMock()
        llm.generate.side_effect = [RuntimeError("LLM down"), '["Question two?"]']
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1]], [{1: 0.9}])
        augmentor = DocumentAugmentor(llm=llm, embedder=embedder, n_questions=1)
        with caplog.at_level(logging.WARNING):
            result = augmentor.augment([_source_chunk("A"), _source_chunk("B")])
        assert len(result) == 1
        assert "Augmentation failed" in caplog.text

    def test_returns_empty_when_no_questions_generated(self):
        llm = MagicMock()
        llm.generate.return_value = "not json"
        embedder = MagicMock()
        augmentor = DocumentAugmentor(llm=llm, embedder=embedder, n_questions=2)
        assert augmentor.augment([_source_chunk()]) == []
        embedder.embed_both.assert_not_called()

    def test_returns_empty_when_embedding_fails(self, caplog):
        llm = MagicMock()
        llm.generate.return_value = '["Question one?"]'
        embedder = MagicMock()
        embedder.embed_both.side_effect = RuntimeError("embed failed")
        augmentor = DocumentAugmentor(llm=llm, embedder=embedder, n_questions=1)
        with caplog.at_level(logging.WARNING):
            assert augmentor.augment([_source_chunk()]) == []
        assert "Embedding augmented questions failed" in caplog.text

    def test_skips_whitespace_only_questions(self):
        llm = MagicMock()
        llm.generate.return_value = '["", "  ", "Real question?"]'
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1]], [{1: 0.9}])
        augmentor = DocumentAugmentor(llm=llm, embedder=embedder, n_questions=3)
        result = augmentor.augment([_source_chunk()])
        assert len(result) == 1
        assert result[0].text == "Real question?"


class TestResolveSyntheticQuestions:
    def test_maps_question_to_source(self):
        source = Chunk(id="source-1", document_id="doc", text="Body text.")
        question = make_question_chunk(source, "What is body text?")
        lookup = MagicMock(return_value=source)
        resolved = resolve_synthetic_questions([(question, 0.9)], lookup)
        assert len(resolved) == 1
        assert resolved[0][0].id == "source-1"
        assert resolved[0][1] == pytest.approx(0.9)

    def test_passes_through_source_chunks(self):
        source = Chunk(id="source-1", document_id="doc", text="Body text.")
        resolved = resolve_synthetic_questions([(source, 0.5)], lambda _id: None)
        assert resolved[0][0].id == "source-1"

    def test_skips_unresolved_questions(self):
        source = Chunk(id="source-1", document_id="doc", text="Body text.")
        question = make_question_chunk(source, "Missing source?")
        resolved = resolve_synthetic_questions([(question, 0.9)], lambda _id: None)
        assert resolved == []

    def test_skips_synthetic_question_without_source_id(self, caplog):
        question = Chunk(
            id="q-bad",
            document_id="doc",
            text="Orphan question?",
            metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_SYNTHETIC},
        )
        with caplog.at_level(logging.DEBUG):
            resolved = resolve_synthetic_questions([(question, 0.8)], lambda _id: None)
        assert resolved == []
        assert "missing source_chunk_id" in caplog.text

    def test_skips_synthetic_question_with_non_string_source_id(self, caplog):
        question = Chunk(
            id="q-bad",
            document_id="doc",
            text="Bad link?",
            metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_SYNTHETIC, SOURCE_CHUNK_ID_KEY: 123},
        )
        with caplog.at_level(logging.DEBUG):
            resolved = resolve_synthetic_questions([(question, 0.8)], lambda _id: None)
        assert resolved == []
        assert "missing source_chunk_id" in caplog.text

    def test_logs_when_source_lookup_fails(self, caplog):
        source = Chunk(id="source-1", document_id="doc", text="Body text.")
        question = make_question_chunk(source, "Where is source?")
        with caplog.at_level(logging.DEBUG):
            resolved = resolve_synthetic_questions([(question, 0.7)], lambda _id: None)
        assert resolved == []
        assert "Source chunk source-1 not found" in caplog.text

    def test_deduplicates_multiple_synthetics_for_same_source(self):
        source = Chunk(id="source-1", document_id="doc", text="Body text.")
        q1 = make_question_chunk(source, "Question one?")
        q2 = make_question_chunk(source, "Question two?")
        lookup = MagicMock(return_value=source)
        resolved = resolve_synthetic_questions([(q1, 0.7), (q2, 0.95)], lookup)
        assert len(resolved) == 1
        assert resolved[0][0].id == "source-1"
        assert resolved[0][1] == pytest.approx(0.95)

    def test_merges_direct_source_hit_with_synthetic(self):
        source = Chunk(id="source-1", document_id="doc", text="Body text.")
        question = make_question_chunk(source, "Question?")
        lookup = MagicMock(return_value=source)
        resolved = resolve_synthetic_questions([(source, 0.6), (question, 0.9)], lookup)
        assert len(resolved) == 1
        assert resolved[0][1] == pytest.approx(0.9)
