# RAG Implementation

Retrieval-augmented generation pipeline: embeds and indexes a document corpus, retrieves and reranks chunks against a user query, and generates an answer with an LLM.

## Language

**Hot path**:
The query-time sequence — embedding the user's query (`embed_query`) and reranking retrieved chunks — that runs synchronously on every user request and directly drives perceived response latency.
_Avoid_: Query path, live path

**Indexing**:
The offline batch process that embeds documents (`embed_passage`) into the vector store ahead of any query. Latency here does not affect user-perceived response time, and is optimized separately from the hot path.
_Avoid_: Embedding pipeline, bulk embedding, ingestion

**Self-hosted provider**:
An embedding or reranker provider that loads model weights and runs inference in-process (e.g. on `mps`/`cuda`), with no network round trip per call.
_Avoid_: Local model, on-prem provider

**API provider**:
An embedding or reranker provider that delegates inference to an external hosted endpoint over the network (e.g. NVIDIA NIM, OpenAI, Cohere, Voyage, Gemini).
_Avoid_: Remote provider, hosted model
