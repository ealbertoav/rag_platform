# Security Advisories

Formal risk acceptance and compensating controls for accepted dependency CVEs.

## CVE-2025-69872 — diskcache insecure deserialization (T-162)

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2025-69872](https://nvd.nist.gov/vuln/detail/CVE-2025-69872) |
| **CVSS v3.1** | 9.8 (Critical) — `CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H` |
| **Affected package** | `diskcache` ≤ 5.6.3 (also exposed via `diskcache-weave` fork line ≤ 5.6.3.post1 until upstream fix) |
| **Exposure path** | Transitive via `llama-cpp-python` → `llama_cpp.llama_cache.LlamaDiskCache` (pickle-based disk prompt cache) |
| **Review schedule** | Quarterly — next review **2026-09-01** (aligned with `configs/cve-allowlist.yaml`) |

### Impact assessment

`diskcache` serializes cached llama.cpp prompt states with Python `pickle`. An attacker with **write access** to the cache directory can plant malicious pickle payloads that execute when the application reads the cache. In this platform:

- **Default deployment:** `LlamaCppProvider` does **not** enable `LlamaDiskCache`; prompt caching uses in-memory `LlamaRAMCache` only, avoiding disk-backed pickle reads.
- **Risk surface:** Environments that manually call `Llama.set_cache(LlamaDiskCache(...))`, or future llama-cpp-python defaults that enable disk cache, inherit the advisory exposure.
- **Production blast radius:** RCE in the API process context when disk cache is active and the cache directory is writable by untrusted principals.

### Compensating controls (in order)

1. **Dependency override:** `pyproject.toml` redirects transitive `diskcache` requirements to the patched fork `diskcache-weave>=5.6.3.post1` (see T-161 allowlist).
2. **RAM-only prompt cache:** Application code never configures `LlamaDiskCache`.
3. **Emergency kill switch:** `LLM__DISABLE_DISK_CACHE=true` disables all llama.cpp prompt caching via `settings.llm.disable_disk_cache` (see `src/infrastructure/llm/llama_cpp_provider.py`).
4. **Automated upstream monitoring:** `./scripts/check_diskcache_cve.sh` (exit 0 while no PyPI fix exists; exit 2 when a patched upstream release is available but not applied).
5. **Dependabot:** weekly `llama-cpp-python` update PRs (`.github/dependabot.yml`) to pick up upstream cache-policy changes quickly.

### Operator actions

```bash
# Monitor for an upstream fix (run in CI or locally)
./scripts/check_diskcache_cve.sh

# Disable llama.cpp prompt caching if disk-cache exploitation becomes active
export LLM__DISABLE_DISK_CACHE=true
```

When upstream `diskcache` publishes a fixed release above 5.6.3 (including post-releases such as `5.6.3.post1`), upgrade the direct override in `pyproject.toml`, remove or renew the T-161 allowlist entry, and re-run `make audit-deps`.

## transformers RCE advisories — unreachable in this platform's usage

Four `transformers` CVEs are pinned into the compatible window (`transformers>=4.56,<5` —
see `pyproject.toml`) required for BGE-M3/BGE-Reranker loading (`src/infrastructure/embeddings/bge_m3.py`,
`src/infrastructure/rerankers/bge_reranker.py`). Two (`CVE-2026-1839`, `CVE-2026-4372`) are
only fixed starting at `transformers>=5.0`/`>=5.3`; the other two (`CVE-2025-14929`,
`CVE-2026-5241`) have no fixed release at all yet per the OSV advisory. Moving to
`transformers>=5` requires FlagEmbedding (currently pinned at the latest available 1.4.0)
to drop its `tokenizer.prepare_for_model()` call, which 5.x removed — no such release
exists yet upstream.

All four share the same reason none apply to how this platform actually uses
`transformers`: it only performs **local inference** with two fixed, pre-downloaded model
directories (`models/embeddings/bge-m3`, `models/rerankers/bge-reranker-v2-m3`), never
`transformers.Trainer`, X-CLIP, or LightGlue, and never passes an attacker-controlled or
remote Hub repo ID to `from_pretrained()`.

| CVE | CVSS v3 | Affected component | Why unreachable here |
|-----|---------|---------------------|------------------------|
| [CVE-2025-14929](https://nvd.nist.gov/vuln/detail/CVE-2025-14929) | 7.7 (High) | X-CLIP checkpoint-conversion deserialization | No X-CLIP models or checkpoint-conversion scripts are used. |
| [CVE-2026-5241](https://nvd.nist.gov/vuln/detail/CVE-2026-5241) | 9.6 (Critical) | LightGlue model loading, `trust_remote_code` bypass | No LightGlue models are loaded; both providers pass `trust_remote_code=False` against local checkpoints only. |
| [CVE-2026-1839](https://nvd.nist.gov/vuln/detail/CVE-2026-1839) | 7.7 (High) | `transformers.Trainer._load_rng_state()` unsafe `torch.load()` | This platform is inference-only; `transformers.Trainer` is never imported or instantiated. |
| [CVE-2026-4372](https://nvd.nist.gov/vuln/detail/CVE-2026-4372) | 7.7 (High) | Malicious `config.json` `_attn_implementation_internal` RCE via `from_pretrained()` | `from_pretrained()` is only ever called with a local, pinned model directory path — never a remote or attacker-supplied Hub repo ID. |

**Review schedule:** Quarterly — next review **2026-10-20** (aligned with `configs/cve-allowlist.yaml`; `transformers` sits on the live embedding/reranking request path, so it gets the more frequent "active production exposure" cadence rather than the six-month transitive-issue default).

**Compensating controls:**

1. Both `BGEM3EmbeddingProvider` and `BGERerankerProvider` load exclusively from local, version-controlled-path model directories — never a Hub repo ID resolved at request time.
2. Neither provider ever sets `trust_remote_code=True`.
3. `transformers.Trainer` has zero references in this codebase (`grep -r "Trainer" src/` — inference-only usage).

**Upstream fix path:** once FlagEmbedding ships a release compatible with `transformers>=5.3` (tracking the removed `prepare_for_model()` API), bump the `transformers` override in `pyproject.toml` to `>=5.3`, drop these four allowlist entries, and re-run `make audit-deps`.
