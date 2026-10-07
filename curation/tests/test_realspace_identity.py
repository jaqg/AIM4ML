"""test_realspace_identity.py — Unit tests for curation/realspace/identity.py.

Part 1 (P3): geometry_key. Part 2 (P4): conformer_rmsd + near-dup.
"""

import numpy as np
import pytest

from curation.realspace.identity import (
    NEAR_DUP_ANGSTROM,
    conformer_rmsd,
    geometry_key,
    is_near_dup,
)

# HeH+ fixture: He at origin, H on x at 0.772 A, charge=+1, multiplicity=1.
# Payload: "1:0.772:0.000:0.000;2:0.000:0.000:0.000|1|1" (Z-sort: H first).
HEH_SYMBOLS = ["He", "H"]
HEH_COORDS = np.array([[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]])
HEH_KEY = "98e3caa9339f0b4e832717e0921ca7d1"  # golden value — identity definition


class TestGeometryKeyGolden:
    def test_golden_value(self):
        """Acceptance criterion 2: the serialization IS the identity.

        This exact hexdigest pins the byte-level definition in the module
        docstring (round 1e-3, sort (Z,x,y,z), %.3f, |charge|mult, MD5).
        Any change to the definition breaks this pin BY DESIGN.
        """
        assert geometry_key(HEH_SYMBOLS, HEH_COORDS, 1, 1) == HEH_KEY

    def test_permuted_atom_order_same_key(self):
        got = geometry_key(HEH_SYMBOLS[::-1], HEH_COORDS[::-1], 1, 1)
        assert got == HEH_KEY


class TestGeometryKeySensitivity:
    def test_charge_sensitivity(self):
        assert geometry_key(HEH_SYMBOLS, HEH_COORDS, 0, 1) != HEH_KEY

    def test_multiplicity_sensitivity(self):
        assert geometry_key(HEH_SYMBOLS, HEH_COORDS, 1, 2) != HEH_KEY

    def test_element_sensitivity(self):
        swapped = np.array([[0.772, 0.0, 0.0], [0.0, 0.0, 0.0]])  # H at origin, He at .772
        assert geometry_key(["He", "H"], swapped, 1, 1) != HEH_KEY

    def test_translated_copy_changes_key(self):
        # geometry_key is absolute-position identity (registry), not
        # translation-invariant — translation handling is RMSD's job (P4).
        moved = HEH_COORDS + np.array([1.0, 2.0, 3.0])
        assert geometry_key(HEH_SYMBOLS, moved, 1, 1) != HEH_KEY

    def test_float_charge_coerced(self):
        # charge/multiplicity coerced via int(): 1.0 == 1.
        assert geometry_key(HEH_SYMBOLS, HEH_COORDS, 1.0, 1) == HEH_KEY


class TestGeometryKeyRoundingBoundary:
    """Pin exact rounding behavior at the 1e-3 A boundary (plan P3 tests).

    Base H-x = 0.772. CPython binary representations (verified):
      0.772 + 0.0005 -> 0.7725       (floats: 0.7725000000000001) -> round3 0.772
      0.772 - 0.0005 -> 0.7715000000000001                    -> round3 0.772
      literal 0.7715 (0.7714999999999999)                    -> round3 0.771
    So +/-0.0005 offsets KEEP the key at this coordinate; 0.001 flips it.
    """

    def _key_at_hx(self, x):
        coords = HEH_COORDS.copy()
        coords[1, 0] = x
        return geometry_key(HEH_SYMBOLS, coords, 1, 1)

    def test_offset_below_boundary_same_key(self):
        assert self._key_at_hx(0.772 + 0.00049) == HEH_KEY
        assert self._key_at_hx(0.772 - 0.00049) == HEH_KEY

    def test_exact_half_offset_keeps_key_at_0772(self):
        # +0.0005 -> 0.7725000000000001 -> rounds to 0.772 -> SAME key.
        # -0.0005 -> 0.7715000000000001 -> rounds to 0.772 -> SAME key.
        assert self._key_at_hx(0.772 + 0.0005) == HEH_KEY
        assert self._key_at_hx(0.772 - 0.0005) == HEH_KEY

    def test_literal_07715_rounds_down(self):
        # Binary literal 0.7715 < midpoint -> 0.771 -> key flips. This is
        # the "0.0005 side may differ" case pinned to exact behavior.
        assert self._key_at_hx(0.7715) != HEH_KEY

    def test_one_mill_flips_key(self):
        assert self._key_at_hx(0.772 + 0.001) != HEH_KEY
        assert self._key_at_hx(0.772 - 0.001) != HEH_KEY


class TestGeometryKeyValidation:
    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="unknown element symbol"):
            geometry_key(["Xx", "H"], HEH_COORDS, 1, 1)

    def test_bad_coords_shape_raises(self):
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            geometry_key(["He", "H"], np.zeros((2, 2)), 1, 1)

    def test_symbol_coord_mismatch_raises(self):
        with pytest.raises(ValueError, match="symbols vs"):
            geometry_key(["He", "H"], np.zeros((3, 3)), 1, 1)

    def test_non_finite_coord_raises(self):
        coords = HEH_COORDS.copy()
        coords[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            geometry_key(HEH_SYMBOLS, coords, 1, 1)

    def test_empty_frame_key_deterministic(self):
        # Zero-atom frame: payload "|0|1" — deterministic, no crash.
        assert geometry_key([], np.zeros((0, 3)), 0, 1) == "9b0fd3b850698317bef57fa6e71a38ba"


# ---------------------------------------------------------------------------
# Part 2 (P4): conformer_rmsd + near-dup decision.
# ---------------------------------------------------------------------------

ROT_Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about z

# Bent water-like triatomic: O heavy, two H (exercises same-element matching).
BENT_SYMBOLS = ["O", "H", "H"]
BENT_COORDS = np.array([[0.96, 0.0, 0.0], [0.0, 0.0, 0.0], [1.4, 0.85, 0.0]])

# Chiral Li4 cluster (no mirror symmetry): mirror image NOT superimposable
# by proper rotations — the proper-rotations-only regression guard.
LI4_SYMBOLS = ["Li", "Li", "Li", "Li"]
LI4_COORDS = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.5, 0.3], [0.8, 0.6, 2.7]])
LI4_CONFORMER2 = LI4_COORDS.copy()
LI4_CONFORMER2[3] += [1.5, 1.0, 0.6]  # one atom moved ~1.95 A -> RMSD 0.479

# Deterministic "arbitrary" rotation (QR of seeded RNG, det=+1).
_RNG = np.random.default_rng(7)
_Q, _ = np.linalg.qr(_RNG.normal(size=(3, 3)))
if np.linalg.det(_Q) < 0:
    _Q[:, 0] *= -1
ARB_ROT = _Q

# Isomer pair: same Hill formula (B2H2), genuinely different arrangement.
ISO_SYMBOLS = ["B", "H", "B", "H"]
ISO_A = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.7, 0.0, 0.0], [2.7, 0.0, 0.0]])
ISO_B = np.array([[0.0, 0.0, 0.0], [1.2, 0.9, 0.0], [3.0, 0.0, 0.0], [1.8, -0.9, 0.0]])


class TestConformerRmsdRigidMotions:
    def test_identical_is_zero(self):
        assert conformer_rmsd(HEH_SYMBOLS, HEH_COORDS, HEH_SYMBOLS, HEH_COORDS) == 0.0

    def test_translated_copy_near_zero(self):
        moved = HEH_COORDS + np.array([1.0, 2.0, 3.0])
        r = conformer_rmsd(HEH_SYMBOLS, HEH_COORDS, HEH_SYMBOLS, moved)
        assert r < 1e-9
        assert is_near_dup(HEH_SYMBOLS, HEH_COORDS, HEH_SYMBOLS, moved)

    def test_rotated_copy_near_zero(self):
        rotated = HEH_COORDS @ ROT_Z.T
        r = conformer_rmsd(HEH_SYMBOLS, HEH_COORDS, HEH_SYMBOLS, rotated)
        assert r < 1e-9
        assert is_near_dup(HEH_SYMBOLS, HEH_COORDS, HEH_SYMBOLS, rotated)

    def test_rotated_translated_bent_near_zero(self):
        moved = BENT_COORDS @ ROT_Z.T + np.array([5.0, 5.0, 5.0])
        r = conformer_rmsd(BENT_SYMBOLS, BENT_COORDS, BENT_SYMBOLS, moved)
        assert r < 1e-9

    def test_arbitrary_rotation_near_zero(self):
        moved = BENT_COORDS @ ARB_ROT.T + np.array([3.0, -1.0, 2.0])
        r = conformer_rmsd(BENT_SYMBOLS, BENT_COORDS, BENT_SYMBOLS, moved)
        assert r < 1e-9

    def test_permuted_atom_order_still_caught(self):
        # Same species, atoms delivered in different file order.
        perm = LI4_COORDS[[2, 0, 3, 1]]
        r = conformer_rmsd(LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, perm)
        assert r < 1e-9
        assert is_near_dup(LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, perm)

    def test_rotation_plus_permutation_near_zero(self):
        moved = LI4_COORDS[[2, 0, 3, 1]] @ ARB_ROT.T + np.array([1.0, 1.0, 1.0])
        r = conformer_rmsd(LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, moved)
        assert r < 1e-9

    def test_symmetry_metric_is_symmetric(self):
        r_ab = conformer_rmsd(LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, LI4_CONFORMER2)
        r_ba = conformer_rmsd(LI4_SYMBOLS, LI4_CONFORMER2, LI4_SYMBOLS, LI4_COORDS)
        assert r_ab == pytest.approx(r_ba, abs=1e-12)

    def test_deterministic(self):
        args = (LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, LI4_CONFORMER2)
        assert conformer_rmsd(*args) == conformer_rmsd(*args)


class TestConformerRmsdDecisions:
    def test_distinct_conformer_not_near_dup(self):
        r = conformer_rmsd(LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, LI4_CONFORMER2)
        assert r == pytest.approx(0.4793, abs=1e-3)
        assert r >= NEAR_DUP_ANGSTROM
        assert not is_near_dup(LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, LI4_CONFORMER2)

    def test_mirror_image_not_near_dup(self):
        # Proper rotations only (det=+1): chiral cluster vs its mirror keeps
        # positive RMSD. Regression guard for the enantiomer policy.
        mirror = LI4_COORDS * np.array([1.0, 1.0, -1.0])
        r = conformer_rmsd(LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, mirror)
        assert r == pytest.approx(1.8564, abs=1e-3)
        assert not is_near_dup(LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, mirror)

    def test_isomer_pair_not_near_dup(self):
        # Same formula (B2H2), genuinely different arrangement -> kept.
        r = conformer_rmsd(ISO_SYMBOLS, ISO_A, ISO_SYMBOLS, ISO_B)
        assert r > NEAR_DUP_ANGSTROM
        assert not is_near_dup(ISO_SYMBOLS, ISO_A, ISO_SYMBOLS, ISO_B)

    def test_species_mismatch_inf(self):
        r = conformer_rmsd(HEH_SYMBOLS, HEH_COORDS, ["Li", "H"], HEH_COORDS)
        assert r == float("inf")
        assert not is_near_dup(HEH_SYMBOLS, HEH_COORDS, ["Li", "H"], HEH_COORDS)

    def test_element_count_mismatch_inf(self):
        coords3 = np.vstack([HEH_COORDS, [2.0, 2.0, 2.0]])
        r = conformer_rmsd(HEH_SYMBOLS, HEH_COORDS, ["He", "H", "H"], coords3)
        assert r == float("inf")

    def test_empty_vs_empty_zero(self):
        assert conformer_rmsd([], np.zeros((0, 3)), [], np.zeros((0, 3))) == 0.0

    def test_single_atom_always_zero(self):
        # One atom: any placement superimposes by translation.
        r = conformer_rmsd(["He"], np.zeros((1, 3)), ["He"], np.array([[5.0, 5.0, 5.0]]))
        assert r < 1e-12

    def test_custom_threshold(self):
        r_pair = (LI4_SYMBOLS, LI4_COORDS, LI4_SYMBOLS, LI4_CONFORMER2)
        assert is_near_dup(*r_pair, threshold=1.0)  # 0.479 < 1.0
        assert not is_near_dup(*r_pair, threshold=0.3)  # 0.479 > 0.3


class TestConformerRmsdValidation:
    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="\(N, 3\)"):
            conformer_rmsd(["He", "H"], np.zeros((2, 2)), ["He", "H"], np.zeros((2, 3)))

    def test_symbol_coord_mismatch_raises(self):
        with pytest.raises(ValueError, match="symbols vs"):
            conformer_rmsd(["He", "H"], np.zeros((3, 3)), ["He", "H"], np.zeros((3, 3)))

    def test_non_finite_raises(self):
        bad = HEH_COORDS.copy()
        bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            conformer_rmsd(HEH_SYMBOLS, HEH_COORDS, HEH_SYMBOLS, bad)

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="unknown element symbol"):
            conformer_rmsd(["Xx", "H"], HEH_COORDS, HEH_SYMBOLS, HEH_COORDS)
