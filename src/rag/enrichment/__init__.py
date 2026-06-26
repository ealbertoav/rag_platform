"""Synthetic question generation and retrieval resolution (T-121)."""

from src.rag.enrichment.document_augmentation import (
    DocumentAugmentor,
    generate_questions,
    is_synthetic_question,
    make_question_chunk,
    resolve_synthetic_questions,
)

__all__ = [
    "DocumentAugmentor",
    "generate_questions",
    "is_synthetic_question",
    "make_question_chunk",
    "resolve_synthetic_questions",
]
