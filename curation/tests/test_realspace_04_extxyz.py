"""Tests for curation/realspace/04_extxyz.py — delivery files, neutral tags."""

import json
import os
import subprocess
import sys

import numpy as np
from conftest import (
    XYZ_HEH_EXTXYZ,
    XYZ_HEH_PERATOM_CHARGE,
    XYZ_LI4_EXTXYZ,
    write_xyz_text,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_EXTXYZ = os.path.join(ROOT, "curation", "realspace", "04_extxyz.py")


def run_extxyz(input_dir, output_dir, *extra):
    cmd = [sys.executable, SCRIPT_EXTXYZ, "-i", input_dir, "-o", output_dir, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return proc.returncode, proc.stdout, proc.stderr


def make_row(
    symbols,
    coords,
    charge=0,
    multiplicity=1,
    source_id="x:0",
    source_file="x.xyz",
    charges=None,
    geometry_id="g_test",
    filter_status=None,
):
    from curation.realspace.schema import frame_to_row
    from lib.xyz_io import XyzFrame

    frame = XyzFrame(
        symbols=symbols,
        coords=np.asarray(coords, dtype=np.float64),
        charges=None if charges is None else np.asarray(charges, dtype=np.float64),
    )
    row = frame_to_row(
        frame,
        geometry_id=geometry_id,
        charge=charge,
        multiplicity=multiplicity,
        source_id=source_id,
        source_file=source_file,
        source_index=0,
    )
    if filter_status is not None:
        row["filter_status"] = filter_status
    return row


def craft_batches(tmp_path, rows_by_batch):
    """rows_by_batch: list of (batch_name, [rows]). Returns batch dir path."""
    from curation.realspace.schema import write_realspace_batch

    batches = tmp_path / "batches"
    for name, rows in rows_by_batch:
        write_realspace_batch(str(batches / name), rows)
    return str(batches)


def read_frames(batch_dir):
    """All delivered frames across all .extxyz files in dir, sorted by name."""
    from lib.xyz_io import read_xyz_frames

    frames = []
    for fname in sorted(os.listdir(batch_dir)):
        if fname.endswith(".extxyz"):
            frames.extend(read_xyz_frames(os.path.join(batch_dir, fname)))
    return frames


def comment_line(extxyz_path):
    with open(extxyz_path) as f:
        f.readline()
        return f.readline().strip()


HEH_ROW = make_row(
    ["He", "H"],
    [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]],
    charge=1,
    source_id="heh.xyz:0",
    source_file="heh.xyz",
    geometry_id="abc123",
)
LI2_ROW = make_row(
    ["Li", "Li"],
    [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    source_id="li2.xyz:0",
    source_file="li2.xyz",
    geometry_id="def456",
)


class TestWriteAndContent:
    def test_one_file_per_batch(self, tmp_path):
        batches = craft_batches(
            tmp_path, [("batch_0000.parquet", [HEH_ROW]), ("batch_0001.parquet", [LI2_ROW])]
        )
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        files = sorted(f for f in os.listdir(out) if f.endswith(".extxyz"))
        assert files == ["batch_0000.extxyz", "batch_0001.extxyz"]

    def test_frame_content_matches_row(self, tmp_path):
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [HEH_ROW])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        frames = read_frames(out)
        assert len(frames) == 1
        frame = frames[0]
        assert frame.symbols == ["He", "H"]
        np.testing.assert_allclose(frame.coords, [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]], atol=1e-6)

    def test_nat_line_and_atom_lines(self, tmp_path):
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [HEH_ROW])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        with open(os.path.join(out, "batch_0000.extxyz")) as f:
            lines = f.read().splitlines()
        assert lines[0] == "2"
        assert lines[2].startswith("He ")
        assert lines[3].startswith("H ")

    def test_no_charge_column_when_absent(self, tmp_path):
        # Rows without per-atom charges -> plain species:pos Properties.
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [HEH_ROW])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        frames = read_frames(out)
        assert frames[0].charges is None
        comment = comment_line(os.path.join(out, "batch_0000.extxyz"))
        assert "Properties=species:S:1:pos:R:3" in comment
        assert "charge:R:1" not in comment

    def test_per_atom_charges_delivered(self, tmp_path):
        # Chain through ingest so per-atom charges survive to delivery.
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        write_xyz_text(input_dir, XYZ_HEH_PERATOM_CHARGE, "heh.xyz")
        batches = str(tmp_path / "batches")
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "curation", "realspace", "01_ingest.py"),
                str(input_dir),
                "-o",
                batches,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        frames = read_frames(out)
        assert len(frames) == 1
        np.testing.assert_allclose(frames[0].charges, [0.3, -0.3], atol=1e-6)
        comment = comment_line(os.path.join(out, "batch_0000.extxyz"))
        assert "charge:R:1" in comment


class TestTags:
    def test_neutral_tags_present(self, tmp_path):
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [HEH_ROW])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        comment = comment_line(os.path.join(out, "batch_0000.extxyz"))
        for tag in [
            "geometry_id=abc123",
            "charge=1",
            "multiplicity=1",
            "elements=HHe",
            "source_id=heh.xyz:0",
            "source_file=heh.xyz",
            "source_index=0",
        ]:
            assert tag in comment, f"missing {tag} in: {comment}"

    def test_tags_survive_read_xyz_frames(self, tmp_path):
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [HEH_ROW])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        frames = read_frames(out)
        header = frames[0].header
        assert header["geometry_id"] == "abc123"
        assert header["charge"] == "1"
        assert header["multiplicity"] == "1"
        assert header["elements"] == "HHe"  # Hill: no carbon -> alphabetical
        assert header["source_id"] == "heh.xyz:0"
        assert header["source_file"] == "heh.xyz"
        assert header["source_index"] == "0"

    def test_stored_header_not_forwarded(self, tmp_path):
        # Delivery emits ONLY the neutral tag set — original header keys dropped.
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [HEH_ROW])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        header = read_frames(out)[0].header
        assert set(header) - {"Properties"} == {
            "geometry_id",
            "charge",
            "multiplicity",
            "elements",
            "source_id",
            "source_file",
            "source_index",
        }


class TestExcludeAndFailures:
    def test_exclude_rule(self, tmp_path):
        rejected = dict(LI2_ROW, filter_status="rejected")
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [HEH_ROW, rejected])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out, "--exclude", "filter_status=rejected")
        assert rc == 0, out_txt + err
        frames = read_frames(out)
        assert [f.header["source_file"] for f in frames] == ["heh.xyz"]
        assert "Excluded:     1" in out_txt

    def test_fully_excluded_batch_not_written(self, tmp_path):
        rejected = dict(LI2_ROW, filter_status="rejected")
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [rejected])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out, "--exclude", "filter_status=rejected")
        assert rc == 0, out_txt + err
        assert not any(f.endswith(".extxyz") for f in os.listdir(out))
        assert "Frames:       0" in out_txt

    def test_decode_failed_row_skipped_no_crash(self, tmp_path):
        bad = dict(HEH_ROW, coords="garbage not coords", source_id="bad:0")
        batches = craft_batches(tmp_path, [("batch_0000.parquet", [bad, LI2_ROW])])
        out = str(tmp_path / "extxyz")
        rc, out_txt, err = run_extxyz(batches, out)
        assert rc == 0, out_txt + err
        frames = read_frames(out)
        assert [f.header["source_file"] for f in frames] == ["li2.xyz"]
        assert "Failed:       1" in out_txt

    def test_empty_input_dir_exits_1(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc, out_txt, _ = run_extxyz(str(empty), str(tmp_path / "extxyz"))
        assert rc == 1
        assert "No .parquet files found" in out_txt

    def test_missing_input_dir_exits_1(self, tmp_path):
        rc, out_txt, _ = run_extxyz(str(tmp_path / "nope"), str(tmp_path / "extxyz"))
        assert rc == 1
        assert "Input dir not found" in out_txt


class TestChainSmoke:
    def test_full_chain_ingest_identity_filter_extxyz(self, tmp_path):
        """ingest -> identity -> filter -> extxyz; provenance per stage output."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        write_xyz_text(input_dir, XYZ_HEH_EXTXYZ, "a_heh.xyz")  # charge 1
        write_xyz_text(input_dir, XYZ_LI4_EXTXYZ, "b_li4.xyz")  # neutral
        base = str(tmp_path)
        batches = os.path.join(base, "batches")
        deduped = os.path.join(base, "deduped_batches")
        filtered = os.path.join(base, "filtered_batches")
        out = os.path.join(base, "extxyz")

        stages = [
            (
                [
                    sys.executable,
                    os.path.join(ROOT, "curation", "realspace", "01_ingest.py"),
                    input_dir,
                    "-o",
                    batches,
                ],
                batches,
            ),
            (
                [
                    sys.executable,
                    os.path.join(ROOT, "curation", "realspace", "02_identity.py"),
                    "-i",
                    batches,
                    "-o",
                    deduped,
                    "--rejects-dir",
                    os.path.join(base, "rejects", "02_identity"),
                ],
                deduped,
            ),
            (
                [
                    sys.executable,
                    os.path.join(ROOT, "curation", "realspace", "03_filter.py"),
                    "-i",
                    deduped,
                    "-o",
                    filtered,
                    "--preset",
                    "none",
                    "--rejects-dir",
                    os.path.join(base, "rejects", "03_filter"),
                ],
                filtered,
            ),
            (
                [
                    sys.executable,
                    os.path.join(ROOT, "curation", "realspace", "04_extxyz.py"),
                    "-i",
                    filtered,
                    "-o",
                    out,
                ],
                out,
            ),
        ]
        for cmd, expect_dir in stages:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            assert proc.returncode == 0, " ".join(cmd) + "\n" + proc.stdout + proc.stderr
            assert os.path.isdir(expect_dir), expect_dir

        # Delivery: both geometries survive preset none, frames parse, tags present.
        frames = read_frames(out)
        assert len(frames) == 2
        by_file = {f.header["source_file"]: f for f in frames}
        assert set(by_file) == {"a_heh.xyz", "b_li4.xyz"}
        assert by_file["a_heh.xyz"].header["charge"] == "1"
        assert by_file["a_heh.xyz"].header["multiplicity"] == "1"
        assert by_file["b_li4.xyz"].header["charge"] == "0"
        assert by_file["a_heh.xyz"].header["elements"] == "HHe"
        assert by_file["b_li4.xyz"].header["elements"] == "Li4"
        # geometry_id tag = recomputed key (P3 golden digest for HeH+).
        assert by_file["a_heh.xyz"].header["geometry_id"] == "98e3caa9339f0b4e832717e0921ca7d1"
        np.testing.assert_allclose(
            by_file["a_heh.xyz"].coords, [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]], atol=1e-6
        )

        # Provenance: one append-only file at the common parent, all 4 stages.
        prov_path = os.path.join(base, "provenance.json")
        assert os.path.exists(prov_path)
        with open(prov_path) as f:
            prov = json.load(f)
        recorded = [s["stage"] for s in prov["stages"]]
        assert recorded == ["01_ingest", "02_identity", "03_filter", "04_extxyz"]

        # Round trip: delivery files are re-ingestable by the chain's own reader.
        from lib.xyz_io import read_xyz_frames

        reread = read_xyz_frames(os.path.join(out, "batch_0000.extxyz"))
        assert len(reread) == 2
        assert reread[0].symbols == ["He", "H"]
        assert reread[0].header["source_id"] == "a_heh.xyz:0"
        np.testing.assert_allclose(
            reread[0].coords, [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]], atol=1e-6
        )
