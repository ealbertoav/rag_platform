from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── Qdrant ─────────────────────────────────────────────────────────────────────
DEFAULT_COLLECTION_NAME = "rag_documents"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# ── Chunk metadata keys ────────────────────────────────────────────────────────
CHUNK_DOCUMENT_ID_KEY = "document_id"
CHUNK_SOURCE_KEY = "source"
CHUNK_PAGE_KEY = "page"
CHUNK_SECTION_KEY = "section"
CHUNK_PARENT_ID_KEY = "parent_id"
CHUNK_HASH_KEY = "content_hash"
CHUNK_INDEX_KEY = "chunk_index"
PROPOSITION_INDEX_KEY = "proposition_index"
CHUNK_RAW_TEXT_KEY = "raw_text"
CHUNK_TYPE_KEY = "type"
CHUNK_TYPE_PROPOSITION = "proposition"
CHUNK_TYPE_SYNTHETIC = "synthetic_question"
CHUNK_TYPE_HYPE = "hype_question"
CHUNK_TYPE_SUMMARY = "summary"
CHUNK_TYPE_DETAIL = "detail"
CHUNK_TYPE_TABLE = "table"
CHUNK_TYPE_CAPTION = "caption"
CHUNK_TYPE_FIGURE = "figure"
CHUNK_TYPE_PAGE = "page"
TABLE_ID_KEY = "table_id"
FIGURE_ID_KEY = "figure_id"
BBOX_KEY = "bbox"
ASSET_PATH_KEY = "asset_path"
# First-class modality labels (T-210). Align with CHUNK_TYPE_* where applicable;
# MODALITY_IMAGE covers raw figure assets / CLIP paths without a text chunk type.
MODALITY_TEXT = "text"
MODALITY_TABLE = "table"
MODALITY_FIGURE = "figure"
MODALITY_CAPTION = "caption"
MODALITY_PAGE = "page"
MODALITY_IMAGE = "image"
KNOWN_MODALITIES: frozenset[str] = frozenset(
    {
        MODALITY_TEXT,
        MODALITY_TABLE,
        MODALITY_FIGURE,
        MODALITY_CAPTION,
        MODALITY_PAGE,
        MODALITY_IMAGE,
    }
)
CHUNK_TYPE_TO_MODALITY: dict[str, str] = {
    CHUNK_TYPE_TABLE: MODALITY_TABLE,
    CHUNK_TYPE_FIGURE: MODALITY_FIGURE,
    CHUNK_TYPE_CAPTION: MODALITY_CAPTION,
    CHUNK_TYPE_PAGE: MODALITY_PAGE,
}
# Document-level layout/outline metadata; excluded from per-chunk spreads.
# Per-chunk section labels use CHUNK_SECTION_KEY (promoted by chunk_metadata).
LAYOUT_DOCUMENT_METADATA_KEYS: frozenset[str] = frozenset(
    {"tables", "figures", "sections", "headings"}
)
# Multimodal parse/chunk metadata reuses CHUNK_PAGE_KEY and CHUNK_SECTION_KEY above.
SOURCE_CHUNK_ID_KEY = "source_chunk_id"
MERGED_CHUNK_IDS_KEY = "merged_chunk_ids"
RSE_MERGED_KEY = "rse_merged"
PARENT_CONTEXT_TEXT_KEY = "parent_context_text"
FEEDBACK_SCORE_KEY = "feedback_score"
FEEDBACK_REVISION_KEY = "feedback_revision"

# ── Supported document types ───────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".htm",
        ".md",
        ".markdown",
    }
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
EXPORTS_DIR = DATA_DIR / "exports"
BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.json"
BM25_LEGACY_PICKLE_PATH = PROCESSED_DIR / "bm25_index.pkl"
BM25_DISK_PATH = PROCESSED_DIR / "bm25_disk"
METADATA_DB_PATH = PROCESSED_DIR / "metadata.db"

MODELS_DIR = ROOT / "models"
DATASETS_DIR = ROOT / "datasets"
PROMPTS_DIR = ROOT / "src" / "prompts"
CVE_ALLOWLIST_PATH = ROOT / "configs" / "cve-allowlist.yaml"

# ── Retrieval ──────────────────────────────────────────────────────────────────
RRF_K = 60  # constant in Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)

# ── Embedding providers ─────────────────────────────────────────────────────────
API_EMBEDDING_PROVIDERS: frozenset[str] = frozenset({"openai", "voyage", "cohere", "gemini"})

SELF_HOSTED_EMBEDDING_MODEL_PATHS: dict[str, str] = {
    "bge_m3": "models/embeddings/bge-m3",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "qwen_embedding": "Qwen/Qwen3-Embedding-0.6B",
}

SELF_HOSTED_EMBEDDING_DEFAULT_DIMS: dict[str, int] = {
    "bge_m3": 1024,
    "nomic": 768,
    "qwen_embedding": 1024,
}
