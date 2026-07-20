# RAG Platform Corpus

## Vector database

The platform uses Qdrant as its vector database. Qdrant stores dense embeddings behind an HNSW (Hierarchical Navigable Small World) index, which gives approximate nearest-neighbor search with sub-linear query time even as the collection grows into the millions of points. Each point in the collection carries a dense vector, an optional sparse vector for lexical hybrid search, and a payload with chunk text and metadata such as document ID, section, and page number. The default collection name is `rag_documents`, configurable via `qdrant.collection` in settings, and the dense vector size is sized automatically from whichever embedding provider is active so switching providers does not silently corrupt an existing collection.

## Hybrid search fusion

Hybrid search combines a dense leg (Qdrant cosine similarity over embeddings) with a sparse lexical leg (BM25) so that queries relying on exact keyword matches are not lost to purely semantic retrieval. The two ranked lists are fused with Reciprocal Rank Fusion (RRF) by default, controlled by `retrieval.hybrid_fusion` in `configs/retrieval.yaml`. An alternative weighted-linear fusion mode blends normalized scores using `retrieval.hybrid_alpha`, where 1.0 means pure dense and 0.0 means pure BM25. Additional optional legs — graph traversal, HyDE, HyPE, hierarchical summaries, and image-dense search — plug into the same RRF fusion step, each with its own per-leg weight multiplier so operators can tune how much any single retrieval strategy influences the final ranking.

## Embedding provider

BGE-M3 is the default embedding provider for both dense and sparse vectors, running self-hosted on Apple Silicon via the `mps` device. It produces 1024-dimensional dense vectors alongside a 30522-dimension sparse lexical head, so a single model call yields both representations needed by the hybrid retriever. Query embeddings are never cached, because caching helps only repeated indexing-time calls, not one-off user queries, and because task-type mismatches between query and passage embeddings would make a naive cache actively harmful for API providers like Cohere or Voyage. Alternative embedding providers — OpenAI, Voyage, Cohere, Gemini, and NVIDIA NIM — can be swapped in with a single config change without touching retrieval or generation code.

## Local LLM runtime

Local generation runs through `llama.cpp` via `LlamaCppProvider`, which loads a GGUF-format model file directly from disk with no network round trip per call. This keeps the full request path self-hosted end to end when no API keys are configured. Model selection is controlled by a per-model YAML file under `configs/llm/`, so switching between a smaller fast model and a larger accurate one is a one-line settings change rather than a code change. When an API-based LLM provider such as NVIDIA NIM is configured instead, the same `LLMRepository` interface is used, so retrieval, compression, and generation code paths are provider-agnostic and do not need to know which backend actually serves a given request.

## Cross-encoder reranker

The cross-encoder reranker re-scores the fused hybrid candidates by jointly encoding the query and each candidate passage, which produces much higher precision than the independent dense and BM25 similarity scores used during initial retrieval. The default reranker is a self-hosted BGE reranker model, with `qwen_reranker` and an NVIDIA NIM-hosted reranker available as alternatives. Reranking runs in batches sized by `reranker.batch_size` and keeps only the top `reranker.top_k` candidates. If the reranker call fails for any reason, the pipeline degrades gracefully by falling back to the original fused ranking order rather than failing the whole request, trading precision for availability.

## Document ingestion

Documents are ingested by running `make ingest SOURCE=path` or `scripts/ingest.py --source path`, which loads the source file, chunks it according to the configured chunking strategy, embeds each chunk, and writes the resulting vectors and payloads into both the Qdrant collection and the BM25 index. Ingestion is additive: re-running it against the same source document replaces only that document's chunks (matched by stable content hashing), leaving every other previously ingested document untouched. This lets a corpus grow incrementally from many separate ingestion runs without needing a full index rebuild each time.

## Chunking strategies

The platform supports several chunking strategies selectable via `chunking.strategy`: recursive (the default, splitting on paragraph and sentence boundaries up to a token budget), semantic (splitting where embedding similarity between adjacent sentences drops below a threshold), proposition (one atomic factual statement per chunk, LLM-extracted), parent-child (large parent chunks paired with smaller indexed child chunks), section, and page-boundary chunking for paginated sources like PDFs. The default recursive chunker targets a 500-token chunk size with 50 tokens of overlap between consecutive chunks, approximating token count as roughly one token per four characters to stay dependency-free.

## HyDE retrieval

HyDE, short for Hypothetical Document Embedding, generates a hypothetical answer document from the query using an LLM before embedding that hypothetical answer and searching with it instead of (or alongside) the raw query embedding. Because a hypothetical answer is written in the same style and vocabulary as the documents it is meant to retrieve, it often sits closer in embedding space to the true relevant passages than a short question does. HyDE is disabled by default because it adds one extra LLM call per query, and its benefit varies by query type — it tends to help analytical questions more than short factual lookups.

## Multi-query expansion

Multi-query expansion generates several paraphrased variants of the original query using an LLM, retrieves candidates for each variant independently, and fuses all of the resulting ranked lists together with reciprocal rank fusion. This widens recall by covering different phrasings a user might not have typed themselves, at the cost of running the retrieval pipeline once per variant. Query expansion is enabled by default with three variants, configurable via `query_expansion.n_variants` in `configs/retrieval.yaml`. A related technique, step-back prompting, generates one broader background query instead of paraphrases, which pairs well with the analytical adaptive-retrieval strategy.

## Contextual compression

Contextual compression reduces the token footprint of retrieved passages before they reach the generator by extracting only the sentences or spans that are actually relevant to the query, discarding the rest of each chunk's text. This keeps the generation prompt smaller and more focused, which both lowers cost and reduces the chance that an irrelevant sentence in an otherwise-relevant chunk distracts the model into an incorrect answer. Compression is enabled by default, capped at `compression.max_tokens` per compressed passage, and if the compression call fails for a given chunk it falls back to using that chunk's original, uncompressed text so a transient failure never drops content from the answer.

## Reliable RAG relevancy grading

Reliable RAG grades each retrieved passage against the user's query using an LLM judge after reranking and parent-context expansion, then drops any passage scoring below `quality.reliable_rag.min_score`. This acts as a quality gate that filters out chunks which survived the earlier ranking stages by lexical or embedding similarity alone but are not actually relevant to answering the specific question asked. If every retrieved passage fails the grade, generation returns a fixed "I don't have information about this" response rather than guessing from irrelevant context. The feature is disabled by default because it adds one LLM call per retrieved passage on every request.

## Self-RAG decision loop

The Self-RAG decision loop replaces the standard agent decision logic on the agent chat endpoints with a sequence of explicit LLM-driven gates: first deciding whether retrieval is even needed for this query, then checking whether the draft answer is actually supported by the retrieved context, and finally scoring the utility of the answer before accepting it, re-retrieving with a different strategy, or refusing to answer. Because each gate is its own LLM call, a single Self-RAG-gated request can involve several extra model round trips compared to the standard pipeline. It pairs well with Reliable RAG's passage-level grading inside the retrieval path.

## Retrieval feedback loop

The retrieval feedback loop lets clients submit relevance votes through `POST /feedback`, marking a previously retrieved chunk as relevant or not relevant to a given query. These votes accumulate into a per-chunk feedback score stored in chunk metadata, which then additively boosts that chunk's rank in future hybrid retrieval results after RRF fusion and again after cross-encoder reranking. Over time this lets the system learn from real usage which chunks are actually useful for which kinds of queries, without requiring any model retraining. The feature is disabled by default and, when enabled, can widen the candidate pool during fusion so that a heavily-boosted chunk has room to move up in rank.

## Multi-replica feedback backends

Multi-replica deployments running more than one API instance need feedback score updates to be atomic across replicas, since two requests voting on the same chunk at the same time must not silently drop one vote. The feedback loop supports three backends for this: Qdrant compare-and-set point updates, Redis using `HINCRBYFLOAT` for atomic increments, and Postgres with atomic SQL updates against a dedicated feedback table. Qdrant is the default backend and requires no additional infrastructure beyond the vector store already in use, while Redis and Postgres are better suited to very high feedback-write throughput.

## Evaluation metrics

The `POST /evals/run` endpoint reports five metrics computed against the golden QA dataset: Recall@5 for retrieval quality, Faithfulness and Relevance from Ragas for generation quality, Context Precision for how much of the retrieved context was actually used, and a Hallucination score from DeepEval measuring whether the generated answer is grounded in the retrieved passages. Each metric has a configurable threshold in `configs/evals.yaml`; a full benchmark run reports pass or fail against every threshold rather than a single aggregate score, so a regression in one dimension cannot be masked by strength in another.

## Golden dataset generation

The golden QA dataset is generated by running `make evals`, which invokes `SyntheticDatasetBuilder` over chunks already present in the BM25 index. For each sampled chunk, an LLM is prompted to write question-answer pairs whose answer is grounded in that chunk's text, and each generated pair is deduplicated against previously generated questions by embedding similarity so near-identical questions are not double-counted. Generation continues, sampling additional chunks as needed, until at least `evals.min_qa_pairs` real pairs have been produced. The resulting QA pairs are also synced into a parallel retrieval-only golden dataset that records just the query and its ground-truth relevant chunk IDs.

## MMR diversity retrieval

Maximal Marginal Relevance (MMR) diversity re-ranking runs after the cross-encoder and rebalances the final passage list between relevance to the query and dissimilarity to passages already chosen, controlled by a `lambda` parameter where 1.0 is pure relevance and 0.0 is maximum diversity. Without it, a reranked list can end up dominated by several near-duplicate chunks that all restate the same sentence from slightly different chunk boundaries, wasting context budget that could have gone toward covering additional distinct facts. MMR is disabled by default and, when enabled, is one of the cheaper optional techniques since it only requires vector similarity math rather than any additional model call.

## Parent context enrichment

Parent context enrichment resolves each retrieved child chunk back to its parent chunk's full text before that content is sent to the generator, so the generator sees broader surrounding context than the narrow indexed child chunk alone would provide. This pairs naturally with the parent-child chunking strategy, where small child chunks are indexed for precise retrieval matching while the paired larger parent chunk supplies the context actually shown to the LLM. The feature is disabled by default and, because it only requires substituting already-available parent text rather than any additional model call, is inexpensive to enable once parent-child chunking is in use.

## Explainable retrieval

Passing `explain=true` to `POST /chat/full` adds a per-source explanation to the response describing why each cited chunk was retrieved and how its content relates to the user's question, generated by an LLM call after the main answer has already been produced. When highlighting is also requested in the same call, a single combined LLM call handles both explanation and highlighting together first, falling back to separate dedicated calls only for whichever side that combined call did not successfully cover. The feature defaults to off and adds no extra LLM calls when not explicitly requested, so normal chat requests pay none of its cost.

## Source highlighting

Source highlighting, requested via `highlights=true` on `POST /chat/full`, extracts verbatim supporting spans from each cited chunk's LLM-facing context text and returns them alongside the answer. Every returned highlight is validated to be an exact verbatim substring of the passage text the generator actually saw, after normalizing whitespace, so a highlight can never accidentally point at text the model never had access to. This is important because parent-context expansion and contextual headers can both change what text counts as the LLM-facing passage compared to the chunk's raw stored text, and the highlighter must match against whichever text the generator actually consumed.

## CI regression gates

A dedicated CI job guards retrieval quality against regressions by running the retrieval benchmark test suite and then a regression gate script against the committed golden datasets before a pull request can merge. The gate validates that the golden dataset still meets its minimum sample count, that the retrieval golden file is in sync with the QA golden file, and that live retrieval against the configured pipeline still clears the committed Recall@5 baseline. The job is configured to skip gracefully rather than fail outright when the infrastructure it needs, such as a reachable vector database or downloaded model weights, is not available in the current environment.

## Technique benchmarking

Running `make benchmark-techniques` compares several optional RAG techniques side by side against the same golden QA dataset without requiring any code changes between runs: a baseline configuration with every optional technique disabled, multi-query expansion, HyDE, contextual compression, Reliable RAG, the Self-RAG agent decision loop, and an A/B comparison of the retrieval feedback loop's ranking boost turned on versus off. Each technique is toggled through isolated environment variable overrides for a single benchmark run, and results are exported as a timestamped JSON file so different runs can be compared over time as the underlying models or corpus change.
