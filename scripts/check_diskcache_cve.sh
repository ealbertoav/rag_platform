#!/usr/bin/env bash
# Upstream diskcache CVE monitor (T-162) — exit 0 while no PyPI fix exists,
# exit 2 when a patched release is available but not applied.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python scripts/check_diskcache_cve.py "$@"
