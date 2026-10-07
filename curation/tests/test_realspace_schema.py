"""test_realspace_schema.py — Unit tests for curation/realspace/schema.py (P2)."""

import warnings

import numpy as np
import pytest
from realspace_fixtures import XYZ_HEH_EXTXYZ, XYZ_HEH_PERATOM_CHARGE, write_xyz_text

from curation.realspace.schema import (
    REALSPACE_COLUMNS,
    charges_from_text,
    charges_to_text,
    coords_from_text,
    coords_to_text,
    frame_to_row,
    header_from_json,
    header_to_json,
    hill_formula,
    row_to_frame,
    symbols_from_text,
    symbols_to_text,
    write_realspace_batch,
)
from lib.parquet_io import read_batch, write_batch
from lib.xyz_io import read_xyz_frames


def _heh_row(tmp_path, text=XYZ_HEH_EXTXYZ, **overrides):
    frame = read_xyz_frames(write_xyz_text(tmp_path, text))[0]
    kwargs = dict(
        geometry_id="a" * 32,
        charge=1,
        multiplicity=1,
        source_id="heh.xyz:0",
        source_file="heh.xyz",
        source_index=0,
    )
    kwargs.update(overrides)
    return frame_to_row(frame, **kwargs)


class TestCodecs:
    def test_coords_round_trip_exact(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]])
        back = coords_from_text(coords_to_text(coords))
        # Value-exact for %.6f-representable input (delivery convention class).
        np.testing.assert_array_equal(back, coords)

    def test_coords_precision_at_1e3(self):
        # Arbitrary float: round-trips within registry identity tolerance 1e-3 A.
        coords = np.array([[0.123456789, -1.987654321, 3.141592653589]])
        back = coords_from_text(coords_to_text(coords))
        assert np.all(np.abs(back - coords) < 1e-3)

    def test_coords_lines_per_atom(self):
        text = coords_to_text(np.zeros((3, 3)))
        assert text.count("\n") == 2  # one line per atom, no trailing newline

    def test_coords_empty(self):
        assert coords_from_text("").shape == (0, 3)

    def test_coords_bad_shape_raises(self):
        with pytest.raises(ValueError, match="x y z"):
            coords_from_text("0.0 0.0")

    def test_symbols_round_trip(self):
        assert symbols_from_text(symbols_to_text(["He", "H", "Li"])) == ["He", "H", "Li"]

    def test_charges_round_trip(self):
        charges = np.array([0.3, -0.3])
        back = charges_from_text(charges_to_text(charges))
        np.testing.assert_array_equal(back, charges)

    def test_charges_none_round_trip(self):
        assert charges_to_text(None) is None
        assert charges_from_text(None) is None
        assert charges_from_text("") is None

    def test_header_round_trip_sorted_keys(self):
        header = {"charge": "1", "Properties": "species:S:1:pos:R:3", "multiplicity": "1"}
        text = header_to_json(header)
        assert text.index("Properties") < text.index("charge")  # deterministic order
        assert header_from_json(text) == header

    def test_header_empty(self):
        assert header_to_json({}) == "{}"
        assert header_from_json(None) == {}
        assert header_from_json("") == {}


class TestHillFormula:
    def test_with_carbon_ch_first(self):
        assert hill_formula(["C", "C", "O", "H", "H", "H", "H", "H", "H"]) == "C2H6O"

    def test_without_carbon_alphabetical(self):
        # HeH+ -> "HHe": no carbon, full alphabetical (H before He).
        assert hill_formula(["He", "H"]) == "HHe"

    def test_count_one_omitted(self):
        assert hill_formula(["Li", "Li", "Li", "Li"]) == "Li4"

    def test_mixed_boron_hydride(self):
        assert hill_formula(["B", "B", "H", "H", "H", "H", "H", "H"]) == "B2H6"

    def test_empty(self):
        assert hill_formula([]) == ""

    def test_matches_symbols_counts(self):
        # Consistency property: formula re-parses to the symbol multiset.
        symbols = ["C", "H", "H", "H", "H", "H", "H", "O"]
        formula = hill_formula(symbols)
        assert formula == "CH6O"  # 1 C, 6 H, 1 O — count-one elided
        assert symbols == ["C", "H", "H", "H", "H", "H", "H", "O"]  # input untouched


class TestFrameRow:
    def test_row_has_all_t2_columns(self, tmp_path):
        row = _heh_row(tmp_path)
        expected = {
            "geometry_id",
            "symbols",
            "n_atoms",
            "coords",
            "charges",
            "formula",
            "charge",
            "multiplicity",
            "source_id",
            "source_file",
            "source_index",
            "header",
        }
        assert set(row) == expected

    def test_n_atoms_consistent(self, tmp_path):
        row = _heh_row(tmp_path)
        assert row["n_atoms"] == len(symbols_from_text(row["symbols"]))
        assert row["n_atoms"] == len([ln for ln in row["coords"].splitlines() if ln.strip()])

    def test_formula_matches_symbols(self, tmp_path):
        row = _heh_row(tmp_path)
        assert row["formula"] == hill_formula(symbols_from_text(row["symbols"]))

    def test_row_to_frame_round_trip(self, tmp_path):
        row = _heh_row(tmp_path, XYZ_HEH_PERATOM_CHARGE)
        frame = row_to_frame(row)
        assert frame.symbols == ["He", "H"]
        np.testing.assert_array_equal(frame.coords, [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]])
        assert frame.charges is not None
        np.testing.assert_array_equal(frame.charges, [0.3, -0.3])
        assert frame.header["charge"] == "1"

    def test_shape_mismatch_raises(self, tmp_path):
        frame = read_xyz_frames(write_xyz_text(tmp_path, XYZ_HEH_EXTXYZ))[0]
        frame.coords = np.zeros((3, 3))  # corrupt: 3 rows vs 2 symbols
        with pytest.raises(ValueError, match="coords shape"):
            frame_to_row(
                frame,
                geometry_id="x",
                charge=0,
                multiplicity=1,
                source_id="s",
                source_file="f",
                source_index=0,
            )


class TestParquetRoundTrip:
    """Through the frozen lib/parquet_io path, with scoped warnings."""

    def test_batch_round_trip(self, tmp_path):
        rows = [_heh_row(tmp_path), _heh_row(tmp_path, XYZ_HEH_PERATOM_CHARGE, source_index=1)]
        path = str(tmp_path / "batch.parquet")
        write_realspace_batch(path, rows)
        back = read_batch(path)

        assert len(back) == 2
        a, b = back[0], back[1]
        # coords text parses back to identical ndarray (10^-3 A class, exact here).
        np.testing.assert_array_equal(
            a["coords"].split("\n"), ["0.000000 0.000000 0.000000", "0.772000 0.000000 0.000000"]
        )
        np.testing.assert_array_equal(
            coords_from_text(a["coords"]), [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]]
        )
        # symbols text <-> list.
        assert symbols_from_text(a["symbols"]) == ["He", "H"]
        # per-atom charges nullable: first row None, second round-trips.
        assert a["charges"] is None
        np.testing.assert_array_equal(charges_from_text(b["charges"]), [0.3, -0.3])
        # JSON header round-trips.
        assert header_from_json(a["header"])["multiplicity"] == "1"
        # identity columns survive.
        assert a["geometry_id"] == "a" * 32
        assert a["source_id"] == "heh.xyz:0"
        assert a["n_atoms"] == 2
        assert a["formula"] == "HHe"

    def test_write_emits_no_warning(self, tmp_path):
        path = str(tmp_path / "batch.parquet")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_realspace_batch(path, [_heh_row(tmp_path)])
        assert caught == []

    def test_unexpected_warning_loud(self, tmp_path, monkeypatch):
        # A non-registry warning during a scoped write must RAISE, not vanish.
        def _warn_then_write(path, rows):
            warnings.warn("boom", UserWarning, stacklevel=2)
            write_batch(path, rows)

        monkeypatch.setattr("curation.realspace.schema.write_batch", _warn_then_write)
        with pytest.raises(UserWarning, match="boom"):
            write_realspace_batch(str(tmp_path / "x.parquet"), [_heh_row(tmp_path)])

    def test_registry_lists_all_columns(self):
        assert set(REALSPACE_COLUMNS) >= {
            "geometry_id",
            "symbols",
            "n_atoms",
            "coords",
            "charges",
            "formula",
            "charge",
            "multiplicity",
            "source_id",
            "source_file",
            "source_index",
            "header",
            "dedup_status",
            "cluster_id",
        }
