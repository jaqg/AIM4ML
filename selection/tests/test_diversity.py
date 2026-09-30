"""test_diversity.py — unit tests for lib/diversity.py."""

import numpy as np
import pytest

from selection.lib import diversity as div


class TestTanimoto:
    def test_identical(self):
        assert div.tanimoto_sim([0, 1, 5], [0, 1, 5]) == 1.0

    def test_disjoint(self):
        assert div.tanimoto_sim([0, 1], [2, 3]) == 0.0

    def test_partial(self):
        assert div.tanimoto_sim([0, 1, 2], [2, 3]) == pytest.approx(1 / 4)

    def test_both_empty(self):
        assert div.tanimoto_sim([], []) == 0.0


class TestMatrixMaxmin:
    def test_matrix_shape_and_self_sim(self):
        M = div.onbits_to_matrix([[0, 1], [2, 3], [0, 2]], nbits=8)
        assert M.shape == (3, 8)
        counts = M.sum(axis=1).astype(np.float32)
        sim = div.sim_to_reference(M, counts, 0)
        assert sim[0] == pytest.approx(1.0)

    def test_centroid_seed_in_range(self):
        idx = div.centroid_seed(np.eye(4, dtype=np.float32))
        assert 0 <= idx < 4

    def test_maxmin_k_geq_n(self):
        assert div.maxmin_indices(np.eye(5, dtype=np.float32), 10, 0) == [0, 1, 2, 3, 4]

    def test_maxmin_distinct_and_seeded(self):
        sel = div.maxmin_indices(np.eye(6, dtype=np.float32), 3, 2)
        assert len(sel) == 3
        assert len(set(sel)) == 3
        assert sel[0] == 2

    def test_maxmin_no_duplicates_near_full(self):
        # regression: only the last pick was masked → re-selection
        sel = div.maxmin_indices(np.eye(6, dtype=np.float32), 5, 2)
        assert len(sel) == 5
        assert len(set(sel)) == 5

    def test_maxmin_masked_respects_pool(self):
        M = np.eye(6, dtype=np.float32)
        mask = np.array([True, True, True, True, False, False])
        counts = M.sum(axis=1).astype(np.float32)
        # k=3 < m=4 → ≥2 greedy iterations; regression: non-pool rows leaked
        # back in after the np.minimum update.
        sel = div.maxmin_masked(M, counts, mask, 3, 0)
        assert len(sel) == 3
        assert set(sel) <= {0, 1, 2, 3}
        assert len(set(sel)) == 3

    def test_maxmin_masked_scores(self):
        M = np.eye(6, dtype=np.float32)
        mask = np.array([True, True, True, True, False, False])
        counts = M.sum(axis=1).astype(np.float32)
        scores = []
        div.maxmin_masked(M, counts, mask, 3, 0, scores=scores)
        assert len(scores) == 3
        assert scores[0] is None  # seed has no reference
