# Feedback Loop — Multi-Replica Deployment Guide (T-146)

This document describes safe deployment modes for the retrieval feedback loop (`POST /feedback`) when running multiple API replicas behind Kubernetes HPA.

## Architecture Summary

| Component | Role |
|---|---|
| **Qdrant** | Vector store; default feedback write source of truth (CAS accumulation) |
| **Redis** (optional) | Atomic `HINCRBYFLOAT` backend under heavy same-chunk contention |
| **SQL store** (optional) | Atomic `UPDATE score = score + delta` via `backend=postgres` (SQLite file for local dev) |
| **BM25 disk index** | Lexical search only — **not** updated on the feedback path |

Feedback scores affect retrieval ranking globally (not per user/session). Boost reads live scores at query time via `get_feedback_scores()`.

## Safe Deployment Modes

### 1. Local / single replica (default)

- Docker Compose with one `api` container, or `uvicorn --workers 1`
- `quality.feedback_loop.backend: qdrant` (default)
- No rate limiting required for internal use
- **Safe for all feedback volumes**

### 2. Production — single API replica

- Helm `replicaCount.api: 1`, HPA disabled or `minReplicas: 1`
- Qdrant CAS handles concurrent requests from multiple clients
- Enable `api.rate_limit.enabled: true` if the API is public-facing
- **Safe for normal human feedback volume**

### 3. Production — HPA ≥ 2 replicas

Required before business-critical feedback-driven ranking under horizontal scale:

1. **Feedback backend:** keep `backend: qdrant` (CAS retries) or switch to `backend: redis` for extreme same-chunk contention
2. **Rate limiting:** set `api.rate_limit.enabled: true` — `/feedback` is included in protected routes (T-160)
3. **BM25 PVC:** shared volume uses dirty-tracking — unchanged indexes skip shutdown save (last-writer-wins mitigation)
4. **Load test:** run `tests/benchmarks/test_feedback_concurrency.py` (or T-172 scenario 5) before enabling HPA in prod

```yaml
# configs/retrieval.yaml
quality:
  feedback_loop:
    enabled: true
    backend: redis          # optional — for high-contention prod
    boost_multiplier: 0.05

# configs/app.yaml
api:
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst: 10
```

## Backend Selection

| Backend | When to use | Multi-replica |
|---|---|---|
| `qdrant` | Default; scores stored in chunk payload metadata | CAS retries (20 attempts) |
| `redis` | Heavy concurrent votes on the same chunk | `HINCRBYFLOAT` — truly atomic |
| `postgres` | SQL atomic increment (local SQLite file or future Postgres DSN) | SQLite file: single-node only; use Redis for pods |

Environment overrides:

```bash
QUALITY__FEEDBACK_LOOP__BACKEND=redis
REDIS__URL=redis://redis:6379
API__RATE_LIMIT__ENABLED=true
```

## Gap Tracker Status (T-146)

| Gap | Status |
|---|---|
| BM25 disk write on feedback path | **Fixed** (T-145) |
| Non-atomic feedback accumulation | **Fixed** — Qdrant CAS |
| Per-pod BM25 metadata drift | **Fixed** — Qdrant-only writes |
| `/feedback` rate limiting | **Fixed** — T-160 middleware includes `/feedback` |
| Shared BM25 PVC last-writer-wins | **Mitigated** — skip save when unchanged |
| CAS retry under extreme contention | Use `backend: redis` |
| Multi-pod load test | `tests/benchmarks/test_feedback_concurrency.py` |

## Do Not Deploy Without

- Public API + HPA `minReplicas ≥ 2` + business-critical feedback ranking **without** rate limiting enabled
- Redis feedback backend in prod without a managed Redis instance (scores live in Redis hash `rag:feedback:scores`)

## Related Tasks

- **T-145** — Feedback API and retrieval boost
- **T-160** — API rate limiting middleware
- **T-172** — Infra benchmark scenario 5 (concurrent feedback across simulated pods)
