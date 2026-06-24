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

# ── Supported document types ───────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
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
BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"

MODELS_DIR = ROOT / "models"
DATASETS_DIR = ROOT / "datasets"
PROMPTS_DIR = ROOT / "src" / "prompts"

# ── Retrieval ──────────────────────────────────────────────────────────────────
RRF_K = 60  # constant in Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)

# ── Embedding providers ─────────────────────────────────────────────────────────
API_EMBEDDING_PROVIDERS: frozenset[str] = frozenset({"openai", "voyage", "cohere", "gemini"})
