# Dependency Vulnerability Policy (T-161)

This project scans Python dependencies on every pull request and via `make audit-deps`.
The scan uses [pip-audit](https://pypi.org/project/pip-audit/) against the active virtual
environment created from `uv.lock`.

## Severity gate

- **Blocking:** CVSS v3 base score **≥ 7.0** (high and critical)
- **Non-blocking:** medium and low findings are reported but do not fail CI
- **Unknown severity:** treated as blocking (fail-safe)

Severity is resolved from the [OSV API](https://google.github.io/osv.dev/) using each
vulnerability ID reported by pip-audit.

## Allowlist process

Known unfixable or accepted risks are recorded in `configs/cve-allowlist.yaml`.

Each entry must include:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | CVE or advisory ID (e.g. `CVE-2025-69872`) |
| `packages` | recommended | Package names the entry applies to (empty = any package) |
| `reason` | yes | Why the risk is accepted and what compensating controls exist |
| `review_date` | yes | ISO date (`YYYY-MM-DD`) when the entry must be re-reviewed |

### Adding an allowlist entry

1. Open a PR that updates `configs/cve-allowlist.yaml` alongside the dependency change.
2. Document impact, exposure path, and compensating controls in the PR description.
3. Set `review_date` no more than **six months** ahead for transitive issues; **quarterly**
   for active production exposures.
4. Link a follow-up task (e.g. T-162) when a permanent fix is expected upstream.

Expired entries are **ignored** automatically — the scan will fail until the CVE is fixed
or the entry is renewed with a new review date.

## Local usage

```bash
make audit-deps
# or
./scripts/check_dependencies.sh
```

Ensure dependencies are installed first (`uv sync --group dev`) so pip-audit audits the
same resolved graph CI uses.

## CI integration

The `dependency-scan` job in `.github/workflows/ci.yml` runs after `uv sync --group dev`
and executes `./scripts/check_dependencies.sh`. The job must complete in under two minutes.
