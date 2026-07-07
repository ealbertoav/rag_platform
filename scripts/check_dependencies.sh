#!/usr/bin/env bash
# Dependency vulnerability scan (T-161) — same entrypoint as CI `dependency-scan` job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python scripts/check_dependencies.py "$@"
