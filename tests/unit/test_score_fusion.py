"""T-022 — score_fusion tests with mock data."""

from __future__ import annotations

from src.domain.entities.chunk import Chunk
from src.rag.ranking.score_fusion import rrf_fuse, weighted_linear_fuse

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=f"chunk {i}")


def _r(i: int, score: float = 1.0) -> tuple[Chunk, float]:
    return _chunk(i), score


# ── rrf_fuse ──────────────────────────────────────────────────────────────────


class TestRrfFuse:
    def test_empty_lists_return_empty(self):
        assert rrf_fuse([], [], top_k=5) == []

    def test_single_list_preserves_order(self):
        results = rrf_fuse([_r(0, 0.9), _r(1, 0.5), _r(2, 0.1)], top_k=3)
        assert [c.id for c, _ in results] == ["c0", "c1", "c2"]

    def test_chunk_in_both_lists_scores_higher(self):
        shared = _chunk(0)
        other = _chunk(1)
        dense = [(shared, 0.5), (other, 0.9)]
        sparse = [(shared, 0.8)]
        fused = rrf_fuse(dense, sparse, top_k=2)
        assert fused[0][0].id == shared.id

    def test_top_k_limits_output(self):
        lists = [_r(i) for i in range(10)]
        assert len(rrf_fuse(lists, top_k=3)) == 3

    def test_deduplicates_by_chunk_id(self):
        chunk = _chunk(0)
        result = rrf_fuse([(chunk, 0.9)], [(chunk, 0.8)], top_k=5)
        assert sum(1 for c, _ in result if c.id == chunk.id) == 1

    def test_scores_are_floats(self):
        results = rrf_fuse([_r(0), _r(1)], top_k=2)
        assert all(isinstance(s, float) for _, s in results)

    def test_scores_sorted_descending(self):
        dense = [_r(0), _r(1), _r(2)]
        sparse = [_r(1), _r(0), _r(2)]
        results = rrf_fuse(dense, sparse, top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_three_lists(self):
        a = [_r(0), _r(1)]
        b = [_r(1), _r(2)]
        c = [_r(0), _r(2)]
        results = rrf_fuse(a, b, c, top_k=3)
        ids = {chunk.id for chunk, _ in results}
        assert ids == {"c0", "c1", "c2"}

    def test_custom_k_parameter(self):
        # Lower k → bigger score boost for top-ranked items
        high_k = rrf_fuse([_r(0)], top_k=1, k=120)[0][1]
        low_k = rrf_fuse([_r(0)], top_k=1, k=10)[0][1]
        assert low_k > high_k

    def test_chunk_instance_from_first_occurrence(self):
        c_a = Chunk(id="same", document_id="d", text="version A")
        c_b = Chunk(id="same", document_id="d", text="version B")
        results = rrf_fuse([(c_a, 0.9)], [(c_b, 0.8)], top_k=1)
        assert results[0][0].text == "version A"

    def test_top_k_larger_than_results_returns_all(self):
        results = rrf_fuse([_r(0), _r(1)], top_k=10)
        assert len(results) == 2


# ── weighted_linear_fuse ──────────────────────────────────────────────────────


class TestWeightedLinearFuse:
    def test_alpha_1_uses_only_dense(self):
        dense = [_r(0, 1.0), _r(1, 0.5)]
        sparse = [_r(2, 0.9)]
        results = weighted_linear_fuse(dense, sparse, alpha=1.0, top_k=3)
        ids = [c.id for c, _ in results]
        assert "c0" in ids
        assert "c1" in ids

    def test_alpha_0_uses_only_sparse(self):
        dense = [_r(0, 1.0)]
        sparse = [_r(1, 1.0), _r(2, 0.5)]
        results = weighted_linear_fuse(dense, sparse, alpha=0.0, top_k=3)
        top_id = results[0][0].id
        assert top_id in ("c1", "c2")

    def test_empty_inputs_return_empty(self):
        assert weighted_linear_fuse([], [], alpha=0.7, top_k=5) == []

    def test_top_k_respected(self):
        dense = [_r(i) for i in range(5)]
        sparse = [_r(i) for i in range(5)]
        assert len(weighted_linear_fuse(dense, sparse, alpha=0.7, top_k=3)) == 3

    def test_scores_in_range(self):
        dense = [_r(0, 0.9), _r(1, 0.1)]
        sparse = [_r(0, 0.8), _r(1, 0.3)]
        results = weighted_linear_fuse(dense, sparse, alpha=0.7, top_k=2)
        assert all(0.0 <= s <= 1.0 for _, s in results)

    def test_deduplicates_by_id(self):
        chunk = _chunk(0)
        results = weighted_linear_fuse([(chunk, 0.9)], [(chunk, 0.8)], alpha=0.7, top_k=5)
        assert sum(1 for c, _ in results if c.id == chunk.id) == 1
