"""test_xyz_io.py — Unit tests for lib/xyz_io.py (realspace track, T1)."""

import numpy as np
import pytest
from realspace_fixtures import (
    XYZ_HEH_EXTXYZ,
    XYZ_HEH_PERATOM_CHARGE,
    XYZ_HEH_PLAIN,
    XYZ_MALFORMED_EOF,
    XYZ_TWO_FRAMES,
    write_xyz_text,
)

from lib.xyz_io import XyzFormatError, XyzFrame, read_xyz_frames, write_xyz_frames


class TestReadExtxyz:
    def test_reads_symbols_and_coords(self, tmp_path):
        frames = read_xyz_frames(write_xyz_text(tmp_path, XYZ_HEH_EXTXYZ))
        assert len(frames) == 1
        frame = frames[0]
        assert frame.symbols == ["He", "H"]
        np.testing.assert_array_equal(frame.coords, [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]])

    def test_reads_header_keys(self, tmp_path):
        frame = read_xyz_frames(write_xyz_text(tmp_path, XYZ_HEH_EXTXYZ))[0]
        assert frame.header["charge"] == "1"
        assert frame.header["multiplicity"] == "1"
        assert "Properties" in frame.header

    def test_no_per_atom_charges_stays_none(self, tmp_path):
        frame = read_xyz_frames(write_xyz_text(tmp_path, XYZ_HEH_EXTXYZ))[0]
        assert frame.charges is None

    def test_missing_charge_multiplicity_stays_missing(self, tmp_path):
        # Reader invents NO defaults: absent keys stay absent.
        text = XYZ_HEH_EXTXYZ.replace(" charge=1 multiplicity=1", "")
        frame = read_xyz_frames(write_xyz_text(tmp_path, text))[0]
        assert "charge" not in frame.header
        assert "multiplicity" not in frame.header

    def test_reads_per_atom_charges(self, tmp_path):
        frame = read_xyz_frames(write_xyz_text(tmp_path, XYZ_HEH_PERATOM_CHARGE))[0]
        assert frame.charges is not None
        np.testing.assert_array_equal(frame.charges, [0.3, -0.3])

    def test_reads_multi_frame_file(self, tmp_path):
        frames = read_xyz_frames(write_xyz_text(tmp_path, XYZ_TWO_FRAMES))
        assert len(frames) == 2
        assert frames[0].header["charge"] == "1"
        assert frames[1].header == {"title": "HeH+ cation"}


class TestReadPlainXyz:
    def test_title_fallback(self, tmp_path):
        frame = read_xyz_frames(write_xyz_text(tmp_path, XYZ_HEH_PLAIN))[0]
        assert frame.header == {"title": "HeH+ cation"}
        assert frame.symbols == ["He", "H"]
        assert frame.charges is None

    def test_empty_comment_line(self, tmp_path):
        text = XYZ_HEH_PLAIN.replace("HeH+ cation\n", "\n")
        frame = read_xyz_frames(write_xyz_text(tmp_path, text))[0]
        assert frame.header == {}


class TestRoundTrip:
    def test_extxyz_round_trip_exact(self, tmp_path):
        src = write_xyz_text(tmp_path, XYZ_HEH_EXTXYZ, "src.xyz")
        out = tmp_path / "out.xyz"
        original = read_xyz_frames(src)
        write_xyz_frames(out, original)
        reread = read_xyz_frames(out)
        assert len(reread) == len(original) == 1
        assert reread[0].symbols == original[0].symbols
        # Value-identical float64 (bit-identical ndarray; %.6f-representable).
        np.testing.assert_array_equal(reread[0].coords, original[0].coords)
        assert reread[0].header == original[0].header
        assert reread[0].charges is None

    def test_per_atom_charges_round_trip(self, tmp_path):
        src = write_xyz_text(tmp_path, XYZ_HEH_PERATOM_CHARGE, "src.xyz")
        out = tmp_path / "out.xyz"
        original = read_xyz_frames(src)
        write_xyz_frames(out, original)
        reread = read_xyz_frames(out)
        assert reread[0].charges is not None
        np.testing.assert_array_equal(reread[0].charges, original[0].charges)
        np.testing.assert_array_equal(reread[0].coords, original[0].coords)

    def test_plain_xyz_round_trip_preserves_title(self, tmp_path):
        src = write_xyz_text(tmp_path, XYZ_HEH_PLAIN, "src.xyz")
        out = tmp_path / "out.xyz"
        original = read_xyz_frames(src)
        write_xyz_frames(out, original)
        reread = read_xyz_frames(out)
        assert reread[0].header == {"title": "HeH+ cation"}
        np.testing.assert_array_equal(reread[0].coords, original[0].coords)

    def test_multi_frame_round_trip(self, tmp_path):
        src = write_xyz_text(tmp_path, XYZ_TWO_FRAMES, "src.xyz")
        out = tmp_path / "out.xyz"
        original = read_xyz_frames(src)
        write_xyz_frames(out, original)
        reread = read_xyz_frames(out)
        assert len(reread) == 2
        for got, want in zip(reread, original):
            np.testing.assert_array_equal(got.coords, want.coords)
            assert got.symbols == want.symbols
            assert got.header == want.header

    def test_header_properties_derived_from_data(self, tmp_path):
        # Writer must not trust stale Properties in the header dict.
        frame = XyzFrame(
            symbols=["He", "H"],
            coords=np.array([[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]]),
            charges=np.array([0.3, -0.3]),
            header={"Properties": "species:S:1:pos:R:3", "charge": "1"},
        )
        out = tmp_path / "out.xyz"
        write_xyz_frames(out, [frame])
        reread = read_xyz_frames(out)[0]
        assert reread.charges is not None  # derived, not the stale spec


class TestMalformed:
    def test_eof_mid_frame_raises(self, tmp_path):
        with pytest.raises(XyzFormatError, match="EOF"):
            read_xyz_frames(write_xyz_text(tmp_path, XYZ_MALFORMED_EOF))

    def test_non_integer_atom_count_raises(self, tmp_path):
        text = "two\ncomment\nHe 0.0 0.0 0.0\n"
        with pytest.raises(XyzFormatError, match="not an integer"):
            read_xyz_frames(write_xyz_text(tmp_path, text))

    def test_non_numeric_coordinate_raises(self, tmp_path):
        text = "2\ncomment\nHe x 0.0 0.0\nH 0.772 0.0 0.0\n"
        with pytest.raises(XyzFormatError, match="non-numeric"):
            read_xyz_frames(write_xyz_text(tmp_path, text))

    def test_wrong_atom_line_columns_raises(self, tmp_path):
        text = "2\ncomment\nHe 0.0 0.0\nH 0.772 0.0 0.0\n"
        with pytest.raises(XyzFormatError, match="columns"):
            read_xyz_frames(write_xyz_text(tmp_path, text))

    def test_missing_comment_line_raises(self, tmp_path):
        # Atom count line then EOF: no comment line at all.
        with pytest.raises(XyzFormatError, match="missing comment line"):
            read_xyz_frames(write_xyz_text(tmp_path, "2\n"))

    def test_comment_line_consumed_then_eof_raises(self, tmp_path):
        # Line 2 is always taken as the comment line: "He 0.0 0.0 0.0" here
        # becomes free-text title, then the atom block runs dry.
        text = "2\nHe 0.0 0.0 0.0\nH 0.772 0.0 0.0\n"
        with pytest.raises(XyzFormatError, match="EOF"):
            read_xyz_frames(write_xyz_text(tmp_path, text))

    def test_bad_properties_spec_raises(self, tmp_path):
        text = "2\nProperties=species:S:1:pos\nHe 0.0 0.0 0.0\nH 0.772 0.0 0.0\n"
        with pytest.raises(XyzFormatError, match="triplets"):
            read_xyz_frames(write_xyz_text(tmp_path, text))

    def test_non_finite_coordinate_raises(self, tmp_path):
        text = "2\ncomment\nHe nan 0.0 0.0\nH 0.772 0.0 0.0\n"
        with pytest.raises(XyzFormatError, match="non-finite"):
            read_xyz_frames(write_xyz_text(tmp_path, text))
