"""Runtime quality gates for retrieval (Reliable RAG, Self-RAG, CRAG)."""

from src.rag.quality.crag import (
    ContextResolution,
    CRAGAction,
    CRAGDecision,
    RetrievalQualityScore,
    crag_fallback_without_web,
    determine_crag_action,
    eval_contexts_for_resolution,
    explainable_chunks_for_resolution,
    refine_knowledge,
    score_retrieval_quality,
)
from src.rag.quality.explainable_retrieval import ChunkExplanation, explain_chunks
from src.rag.quality.reliable_rag import ChunkRelevance, grade_relevance
from src.rag.quality.self_rag import (
    RetrievalDecision,
    SupportCheck,
    UtilityAction,
    UtilityScore,
    check_support,
    decide_retrieval,
    score_utility,
)

__all__ = [
    "CRAGAction",
    "CRAGDecision",
    "ChunkExplanation",
    "ChunkRelevance",
    "ContextResolution",
    "RetrievalDecision",
    "RetrievalQualityScore",
    "SupportCheck",
    "UtilityAction",
    "UtilityScore",
    "check_support",
    "crag_fallback_without_web",
    "decide_retrieval",
    "determine_crag_action",
    "eval_contexts_for_resolution",
    "explain_chunks",
    "explainable_chunks_for_resolution",
    "grade_relevance",
    "refine_knowledge",
    "score_retrieval_quality",
    "score_utility",
]
