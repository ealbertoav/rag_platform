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
