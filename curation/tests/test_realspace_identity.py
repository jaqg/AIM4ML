"""test_realspace_identity.py — Unit tests for curation/realspace/identity.py.

Part 1 (P3): geometry_key. Part 2 (P4): conformer_rmsd + near-dup.
"""

import numpy as np
import pytest

from curation.realspace.identity import geometry_key

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
