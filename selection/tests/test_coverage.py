"""test_coverage.py — unit tests for lib/coverage.py."""

import numpy as np
import pytest

from selection.lib import coverage as cov
from selection.lib import diversity as div


class TestEnvFrequency:
    def test_counts(self):
        atom_envs = [
            [[10, 20], [30]],   # mol 0: envs 10,20,30
            [[10], [40]],       # mol 1: envs 10,40
            [[30], [50]],       # mol 2: envs 30,50
        ]
        elements = [["C", "O"], ["C", "N"], ["O", "S"]]
        env_elem, env_natoms, env_nmols = cov.env_frequency(atom_envs, elements)
        assert env_natoms[10] == 2
        assert env_nmols[10] == 2       # mols 0 and 1
        assert env_nmols[20] == 1       # mol 0 only
        assert env_elem[30] == "O"

    def test_env_to_mols(self):
        d = cov.env_to_mols([[[1, 2]], [[1, 3]]])
        assert d[1] == [0, 1]
        assert d[2] == [0]
        assert d[3] == [1]


class TestFloorPass:
    def test_forced_pick(self):
        atom_envs = [[[1]], [[1, 2]]]
        env_elem, env_natoms, env_nmols = cov.env_frequency(
            atom_envs, [["C"], ["C", "C"]],
        )
        fp = div.onbits_to_matrix([[0], [0]], nbits=8)
        sel, covered = cov.floor_pass(atom_envs, env_nmols, env_natoms, fp, 2, 0.5)
        assert sel == [1]
        assert covered == {1, 2}

    def test_choice_prefers_diverse(self):
        atom_envs = [[[1, 2]], [[2, 3]], [[1, 3]]]
        env_elem, env_natoms, env_nmols = cov.env_frequency(
            atom_envs, [["C"], ["C"], ["C"]],
        )
        fp = div.onbits_to_matrix([[0, 1, 2, 3], [0, 1, 2, 4], [5, 6, 7]], nbits=8)
        sel, covered = cov.floor_pass(atom_envs, env_nmols, env_natoms, fp, 2, 0.5)
        assert covered == {1, 2, 3}
        assert sel[0] == 0
        assert sel[1] == 2

    def test_respects_budget(self):
        atom_envs = [[[1]], [[1, 2]]]
        env_elem, env_natoms, env_nmols = cov.env_frequency(
            atom_envs, [["C"], ["C", "C"]],
        )
        fp = div.onbits_to_matrix([[0], [0]], nbits=8)
        sel, _ = cov.floor_pass(atom_envs, env_nmols, env_natoms, fp, 1, 0.5)
        assert len(sel) <= 1

    def test_forced_capped_by_budget(self):
        # 3 single-carrier envs, budget 2 → only 2 forced picks enter
        atom_envs = [[[1]], [[2]], [[3]]]
        env_elem, env_natoms, env_nmols = cov.env_frequency(
            atom_envs, [["C"], ["C"], ["C"]],
        )
        fp = div.onbits_to_matrix([[0], [0], [0]], nbits=8)
        sel, _ = cov.floor_pass(atom_envs, env_nmols, env_natoms, fp, 2, 0.5)
        assert len(sel) == 2

    def test_scores_recorded(self):
        atom_envs = [[[1]], [[1, 2]]]
        env_elem, env_natoms, env_nmols = cov.env_frequency(
            atom_envs, [["C"], ["C", "C"]],
        )
        fp = div.onbits_to_matrix([[0], [0]], nbits=8)
        scores = []
        cov.floor_pass(atom_envs, env_nmols, env_natoms, fp, 2, 0.5, scores=scores)
        assert len(scores) == 1
        assert scores[0] is None  # forced pick → no weighted score


class TestCostCurve:
    def test_monotonic(self):
        atom_envs = [[[1]], [[1, 2]], [[3]]]
        env_elem, env_natoms, env_nmols = cov.env_frequency(
            atom_envs, [["C"], ["C", "C"], ["C"]],
        )
        xs, ys = cov.greedy_set_cover_curve(atom_envs, env_nmols, env_natoms)
        assert xs == sorted(xs)
        assert ys == sorted(ys)
        assert ys[-1] == len(env_nmols)  # all envs covered


class TestJSDivergence:
    def test_identical(self):
        a = np.random.default_rng(0).normal(size=500)
        assert cov.js_divergence(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_disjoint(self):
        assert cov.js_divergence(np.zeros(200), np.ones(200)) > 0.99
