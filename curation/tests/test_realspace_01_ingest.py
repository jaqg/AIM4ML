"""test_realspace_01_ingest.py — Integration tests for realspace 01_ingest.

Runs the stage as a subprocess (repo convention for digit-named stages).
"""

import json
import os
import subprocess
import sys

from realspace_fixtures import (
    XYZ_B2H6_EXTXYZ,
    XYZ_BAD_CHARGE_HEADER,
    XYZ_HEH_EXTXYZ,
    XYZ_HEH_PERATOM_CHARGE,
    XYZ_HEH_PLAIN,
    XYZ_LI4_EXTXYZ,
    XYZ_MALFORMED_EOF,
    write_xyz_text,
)

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "realspace",
    "01_ingest.py",
)

# Golden P3 identity for the HeH+ extxyz fixture (charge=1, multiplicity=1).
GOLDEN_HEH_EXTXYZ_ID = "98e3caa9339f0b4e832717e0921ca7d1"


def run_ingest(input_dir, output_dir, batch_size=5000, rejects_dir=None, pattern="*.xyz"):
    cmd = [
        sys.executable,
        SCRIPT,
        input_dir,
        "-o",
        output_dir,
        "--pattern",
        pattern,
        "--batch-size",
        str(batch_size),
    ]
    if rejects_dir is not None:
        cmd += ["--rejects-dir", rejects_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def read_rows(output_dir):
    from lib.parquet_io import read_batch

    files = sorted(f for f in os.listdir(output_dir) if f.endswith(".parquet"))
    rows = []
    for f in files:
        rows.extend(read_batch(os.path.join(output_dir, f)))
    return files, rows


class TestIngestColumns:
    def test_all_t2_columns_written(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "heh.xyz")
        out = str(tmp_path / "out")
        rc, stdout, _ = run_ingest(input_dir, out)
        assert rc == 0
        files, rows = read_rows(out)
        assert files == ["batch_0000.parquet"]
        r = rows[0]
        for col in (
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
        ):
            assert col in r, col

    def test_geometry_id_is_golden_key(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "heh.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        assert rows[0]["geometry_id"] == GOLDEN_HEH_EXTXYZ_ID

    def test_symbols_coords_formula(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "heh.xyz")
        write_xyz_text(tmp_path / "in", XYZ_B2H6_EXTXYZ, "b2h6.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        by_src = {r["source_file"]: r for r in rows}
        heh = by_src["heh.xyz"]
        assert heh["symbols"] == "He H"
        assert heh["n_atoms"] == 2
        assert len(heh["coords"].splitlines()) == 2
        assert heh["formula"] == "HHe"
        b2h6 = by_src["b2h6.xyz"]
        assert b2h6["formula"] == "B2H6"
        assert b2h6["n_atoms"] == 8

    def test_charge_multiplicity_from_header(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "heh.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        assert rows[0]["charge"] == 1
        assert rows[0]["multiplicity"] == 1

    def test_charge_multiplicity_defaults_filled(self, tmp_path):
        # Plain xyz (no header keys) -> ingest fills 0/1 explicitly.
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_PLAIN, "plain.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        assert rows[0]["charge"] == 0
        assert rows[0]["multiplicity"] == 1
        assert rows[0]["geometry_id"] != GOLDEN_HEH_EXTXYZ_ID  # charge differs

    def test_per_atom_charges_nullable(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_PERATOM_CHARGE, "q.xyz")
        write_xyz_text(tmp_path / "in", XYZ_HEH_PLAIN, "plain.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        by_src = {r["source_file"]: r for r in rows}
        assert by_src["q.xyz"]["charges"] == "0.300000 -0.300000"
        assert by_src["plain.xyz"]["charges"] is None

    def test_header_json_column(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "heh.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        header = json.loads(rows[0]["header"])
        assert header["charge"] == "1"
        assert header["multiplicity"] == "1"
        assert "Properties" in header


class TestIngestProvenance:
    def test_source_id_format(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "heh.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        assert rows[0]["source_id"] == "heh.xyz:0"
        assert rows[0]["source_file"] == "heh.xyz"
        assert rows[0]["source_index"] == 0

    def test_multi_frame_indices(self, tmp_path):
        two_frames = XYZ_HEH_EXTXYZ + "\n" + XYZ_HEH_PLAIN
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", two_frames, "multi.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        assert [r["source_index"] for r in rows] == [0, 1]
        assert [r["source_id"] for r in rows] == ["multi.xyz:0", "multi.xyz:1"]

    def test_deterministic_file_order(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        # Written out of alphabetical order; ingest sorts by name.
        write_xyz_text(tmp_path / "in", XYZ_LI4_EXTXYZ, "b_li4.xyz")
        write_xyz_text(tmp_path / "in", XYZ_B2H6_EXTXYZ, "a_b2h6.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        _, rows = read_rows(out)
        assert [r["source_file"] for r in rows] == ["a_b2h6.xyz", "b_li4.xyz"]

    def test_provenance_recorded(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "heh.xyz")
        base = str(tmp_path / "base")
        out = os.path.join(base, "batches")
        rc, _, _ = run_ingest(input_dir, out)
        assert rc == 0
        prov_path = os.path.join(base, "provenance.json")
        assert os.path.exists(prov_path)
        with open(prov_path) as f:
            prov = json.load(f)
        stages = [s["stage"] for s in prov["stages"]]
        assert "01_ingest" in stages


class TestIngestRejects:
    def test_malformed_file_rejected(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "good.xyz")
        write_xyz_text(tmp_path / "in", XYZ_MALFORMED_EOF, "bad.xyz")
        out = str(tmp_path / "out")
        rejects = str(tmp_path / "rejects")
        rc, stdout, _ = run_ingest(input_dir, out, rejects_dir=rejects)
        assert rc == 0
        _, rows = read_rows(out)
        assert [r["source_file"] for r in rows] == ["good.xyz"]  # good file survives
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "ingest_rejected.parquet"))
        assert len(rej) == 1
        assert rej[0]["source_file"] == "bad.xyz"
        assert "EOF" in rej[0]["reason"] or "expected" in rej[0]["reason"]
        assert os.path.exists(os.path.join(rejects, ".REJECTED"))

    def test_bad_header_int_rejected(self, tmp_path):
        # Frame-level semantic reject: charge header not parseable as int.
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "good.xyz")
        write_xyz_text(tmp_path / "in", XYZ_BAD_CHARGE_HEADER, "badcharge.xyz")
        out = str(tmp_path / "out")
        rejects = str(tmp_path / "rejects")
        rc, _, _ = run_ingest(input_dir, out, rejects_dir=rejects)
        assert rc == 0
        _, rows = read_rows(out)
        assert [r["source_file"] for r in rows] == ["good.xyz"]
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "ingest_rejected.parquet"))
        assert len(rej) == 1
        assert rej[0]["source_id"] == "badcharge.xyz:0"
        assert "charge" in rej[0]["reason"]

    def test_all_rejected_no_batches(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_MALFORMED_EOF, "bad.xyz")
        out = str(tmp_path / "out")
        rejects = str(tmp_path / "rejects")
        rc, _, _ = run_ingest(input_dir, out, rejects_dir=rejects)
        assert rc == 0
        assert not any(f.endswith(".parquet") for f in os.listdir(out))
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "ingest_rejected.parquet"))
        assert len(rej) == 1


class TestIngestBatching:
    def test_batch_splitting(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        three = XYZ_HEH_EXTXYZ + "\n" + XYZ_HEH_PLAIN + "\n" + XYZ_LI4_EXTXYZ
        write_xyz_text(tmp_path / "in", three, "three.xyz")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out, batch_size=2)
        assert rc == 0
        files, rows = read_rows(out)
        assert files == ["batch_0000.parquet", "batch_0001.parquet"]
        assert len(rows) == 3
        from lib.parquet_io import read_batch

        assert len(read_batch(os.path.join(out, files[0]))) == 2
        assert len(read_batch(os.path.join(out, files[1]))) == 1

    def test_pattern_filter(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        write_xyz_text(tmp_path / "in", XYZ_HEH_EXTXYZ, "keep.xyz")
        write_xyz_text(tmp_path / "in", XYZ_LI4_EXTXYZ, "skip.txt")
        out = str(tmp_path / "out")
        rc, _, _ = run_ingest(input_dir, out, pattern="*.xyz")
        assert rc == 0
        _, rows = read_rows(out)
        assert [r["source_file"] for r in rows] == ["keep.xyz"]

    def test_empty_dir_warns_but_succeeds(self, tmp_path):
        input_dir = str(tmp_path / "in")
        os.makedirs(input_dir)
        out = str(tmp_path / "out")
        rc, _, stderr = run_ingest(input_dir, out)
        assert rc == 0
        assert "no files matched" in stderr
