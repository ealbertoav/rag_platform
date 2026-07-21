# Type Safety (T-170)

Audit date: 2026-07-08 · Branch: `feat/t-170-type-ignore-audit`

## Summary

| Metric | Before | After |
|--------|--------|-------|
| `# type: ignore` in `src/` | 56 | **0** |
| `uv run mypy src` | pass | pass |
| `warn_unused_ignores` | `false` | **`true`** |

All former inline suppressions were removed by applying proper types, `typing.cast()` at third-party boundaries, or `ParamSpec`-based decorators. Test files may still use `# type: ignore` for mock assertions — those are out of scope for the production audit.

## Remaining inline suppressions

**None in `src/`.** Every previous `# type: ignore` has a documented replacement below.

## Mypy module overrides (`pyproject.toml`)

These modules lack complete stubs or have union types mypy cannot resolve under `strict = true`. Errors are suppressed at the module level instead of with scattered inline comments:

| Module pattern | Reason | Removal plan |
|----------------|--------|--------------|
| `deepeval.*` | Optional eval dependency; lazy imports inside metric runners | Upstream stubs or narrow per-function overrides when DeepEval ships a typed API |
| `FlagEmbedding.*` | BGE-M3 encode return type is `Any` in stubs | Cast at call site (done in `bge_m3.py`); drop override when FlagEmbedding types encode() |
| `llama_cpp.*` | Chat completion stream unions not expressible in stubs | Cast at call site (done in `llama_cpp_provider.py`); drop override when llama-cpp-python stubs improve |
| `rank_bm25.*` | Legacy BM25 library without stubs | Wrap in typed adapter or contribute stubs |
| `redis`, `redis.*` | Partial typing; connection errors are a wide union | Typed `Redis` import under `TYPE_CHECKING` (done in `cached_embedding_provider.py`) |
| `openai.*`, `voyageai.*`, `cohere.*`, `google.*` | SDK kwargs and response shapes vary by version | Cast at API boundaries (done in provider modules) |
| `src.infrastructure.llm.llama_cpp_provider` | Historical blanket override; kept until llama-cpp stubs stabilize | Re-evaluate after llama-cpp-python ≥ stub refresh |
| `src.evals.generation.hallucination` | DeepEval metric objects are dynamically imported | Remove when DeepEval exports typed metric classes |

## Replacement patterns (former ignores → fix)

| Area | Former issue | Fix |
|------|--------------|-----|
| `ingestion_pipeline.py`, `retrieval_pipeline.py` | Optional enrichers typed as `object` | Concrete `TYPE_CHECKING` imports (`GraphIndexer`, `HyPEIndexer`, …) and `EmbeddingRepository` |
| `hybrid_retriever` wiring | Retriever args ignored | Factory helpers return typed retriever instances |
| `hype_retriever.py` | `chunk_lookup.get_by_id` attr-defined | `ChunkLookup` protocol; `BM25Retriever.get_by_id` → `Chunk \| None` |
| `chunking/__init__.py` | `**kwargs` to chunker ctors | `cast(Any, kwargs)` |
| `observability/tracing.py` | Decorator misc/return-value | `ParamSpec` + `TypeVar` + `cast` on async wrapper |
| `qdrant.py` | Qdrant client query/point ID variance | `cast()` on query vectors and point ID lists |
| `neo4j_graph.py` | Driver typed as `object` | `AsyncDriver` under `TYPE_CHECKING` |
| `llama_cpp_provider.py` | Chat message / stream chunk types | `cast(dict[str, Any], …)` on completion payloads |
| Embedding providers | SDK `Any` returns | `cast()` on encode/API responses |
| Eval dataclasses `to_dict()` | `asdict` return-value | `cast(dict[str, object], dataclasses.asdict(self))` |
| Benchmark pipelines | `ChatPipeline` vs `BenchmarkPipeline` | Shared `BenchmarkPipeline` protocol in `benchmark_samples.py` |
| `generation_service.py` | Async generator misc | Return type `AsyncGenerator[str, None]` |

## Verification

```bash
make lint                  # must exit 0 (matches CI)
uv run python scripts/check_lint_gate.py  # config alignment + mypy
rg 'type:\s*ignore' src/   # expect 0 matches
make test                  # 100% line coverage on src/
```

## CI gate (T-171)

- `.github/workflows/ci.yml` lint job fails on any `mypy src` error (no `continue-on-error`, no CLI `--ignore-missing-imports`)
- `make lint` and CI run identical commands in order: ruff check → ruff format --check → mypy → basedpyright
- basedpyright uses `typeCheckingMode = "standard"` and `failOnWarnings = false` — CI fails only on errors (`--level error`), not standard-mode warnings mypy already covers
- Pre-commit mypy hook targets `^src/` and relies on `pyproject.toml` mypy settings
- `scripts/check_lint_gate.py` validates configuration drift before merge

## Basedpyright burn-down (T-173)

Audit date: 2026-07-08 · Branch: `feat/t-173-basedpyright-burn-down`

### Mode progression

| Stage | `typeCheckingMode` | CI gate | Status |
|-------|-------------------|---------|--------|
| T-171 baseline | `basic` | errors only | superseded |
| **T-173 (current)** | **`standard`** | errors only | **done** |
| Future | `recommended` | flip `failOnWarnings` when debt cleared | planned |

### Initial inventory (recommended mode, pre-fix)

Captured with all actionable rules enabled (excluding stub/`Any` noise):

| Rule | Count before | Count after |
|------|-------------|-------------|
| `reportUnannotatedClassAttribute` | 250 | **0** |
| `reportImplicitOverride` | 83 | **0** |
| `reportUnusedCallResult` | 47 | **0** |
| `reportImplicitStringConcatenation` | 20 | **0** |
| `reportCallInDefaultInitializer` | 11 | suppressed (FastAPI `Depends`, mirrors ruff B008) |
| Quick wins (`reportUnusedImport`, `reportUnreachable`, `reportUnnecessary*`, `reportUnusedParameter`) | 8 | **0** |
| **Total actionable** | **421** | **0** (at `--level warning`, standard mode) |

Out of scope (unchanged from T-170/T-171): `reportMissingTypeStubs`, `reportAny`, `reportExplicitAny`, `reportUnknown*Type`.

### Enabled / suppressed rules (`pyproject.toml`)

| Rule | Setting | Rationale |
|------|---------|-----------|
| `reportCallInDefaultInitializer` | `false` | FastAPI `Depends()` defaults — intentional (ruff B008) |
| `reportMissingImports` | `false` | Optional runtime deps |
| `reportMissingTypeStubs` | `false` | Third-party stub gaps (neo4j, ragas, …) |
| `reportAny` / `reportUnknown*Type` | `false` | Aligned with mypy overrides (T-170) |

### `failOnWarnings` flip criteria

Keep `failOnWarnings = false` until:

1. `typeCheckingMode = "recommended"` is enabled and `--level warning` reports **0** warnings (excluding permanently suppressed rules above).
2. Linux CI and macOS both pass `uv run basedpyright --level warning src`.

Then enable `failOnWarnings = true` in a follow-up PR.

### Verification

```bash
make lint                              # exit 0 (standard mode, error-level basedpyright)
uv run basedpyright --level warning src  # 0 warnings at standard mode
uv run mypy src                        # strict, exit 0
make test                              # 100% line coverage on src/
```
