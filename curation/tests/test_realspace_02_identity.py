"""test_realspace_02_identity.py — Integration tests for realspace 02_identity.

Runs stages as subprocesses (repo convention for digit-named stages).
"""

import json
import os
import subprocess
import sys

from realspace_fixtures import (
    XYZ_B2H6_EXTXYZ,
    XYZ_HEH_EXTXYZ,
    XYZ_HEH_EXTXYZ_TRANSLATED,
    XYZ_ISO_A_EXTXYZ,
    XYZ_ISO_B_EXTXYZ,
    XYZ_LI4_CONFORMER2_EXTXYZ,
    XYZ_LI4_EXTXYZ,
    write_xyz_text,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_INGEST = os.path.join(ROOT, "curation", "realspace", "01_ingest.py")
SCRIPT_IDENTITY = os.path.join(ROOT, "curation", "realspace", "02_identity.py")


def run_ingest(input_dir, output_dir, batch_size=5000):
    cmd = [
        sys.executable,
        SCRIPT_INGEST,
        input_dir,
        "-o",
        output_dir,
        "--batch-size",
        str(batch_size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout


def run_identity(input_dir, output_dir, rejects_dir=None, threshold=None, force_keep=False):
    cmd = [sys.executable, SCRIPT_IDENTITY, "-i", input_dir, "-o", output_dir]
    if rejects_dir is not None:
        cmd += ["--rejects-dir", rejects_dir]
    if threshold is not None:
        cmd += ["--near-dup-threshold", str(threshold)]
    if force_keep:
        cmd += ["--force-keep-rejected"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def read_rows(batch_dir):
    from lib.parquet_io import read_batch

    files = sorted(f for f in os.listdir(batch_dir) if f.endswith(".parquet"))
    rows = []
    for f in files:
        rows.extend(read_batch(os.path.join(batch_dir, f)))
    return files, rows


def make_input_dir(tmp_path, fixtures):
    """fixtures: list of (name, text). Returns the input dir path."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for name, text in fixtures:
        write_xyz_text(input_dir, text, name)
    return str(input_dir)


def chain(tmp_path, fixtures, batch_size=5000, **identity_kwargs):
    """01_ingest then 02_identity on a fixture dir. Returns (rc, rows, rejects_rows, stdout)."""
    input_dir = make_input_dir(tmp_path, fixtures)
    batches = str(tmp_path / "batches")
    deduped = str(tmp_path / "deduped")
    rejects = str(tmp_path / "rejects" / "02_identity")
    rc_i, out_i = run_ingest(input_dir, batches, batch_size=batch_size)
    assert rc_i == 0, out_i
    rc_d, out_d, err_d = run_identity(batches, deduped, rejects_dir=rejects, **identity_kwargs)
    assert rc_d == 0, out_d + err_d
    _, rows = read_rows(deduped)
    from lib.parquet_io import read_batch

    rej_rows = []
    if os.path.isdir(rejects):
        for f in sorted(os.listdir(rejects)):
            if f.endswith(".parquet"):
                rej_rows.extend(read_batch(os.path.join(rejects, f)))
    return rc_d, rows, rej_rows, out_d, out_i, rejects


HEH_FIXTURES = [
    ("a_heh.xyz", XYZ_HEH_EXTXYZ),
    ("b_heh_copy.xyz", XYZ_HEH_EXTXYZ),  # bit-identical re-delivery
    ("c_heh_moved.xyz", XYZ_HEH_EXTXYZ_TRANSLATED),  # near-dup (RMSD 0)
    ("d_li4.xyz", XYZ_LI4_EXTXYZ),
    ("e_li4_conf2.xyz", XYZ_LI4_CONFORMER2_EXTXYZ),  # distinct conformer (0.479 A)
]


class TestExactDedup:
    def test_bit_identical_redelivery_dropped_keep_first(self, tmp_path):
        _, rows, rej, out, _, _ = chain(tmp_path, HEH_FIXTURES[:2] + HEH_FIXTURES[3:])
        kept = [r for r in rows if r["source_file"] == "a_heh.xyz"]
        assert len(kept) == 1
        dropped = [r for r in rej if r["source_file"] == "b_heh_copy.xyz"]
        assert len(dropped) == 1
        assert dropped[0]["dedup_status"] == "exact_dup"
        assert "duplicate geometry_id" in dropped[0]["reason"]
        # keep-first: the surviving row is the FIRST file alphabetically
        assert kept[0]["dedup_status"] == "kept"

    def test_exact_dup_inherits_representative_cluster_id(self, tmp_path):
        _, rows, rej, _, _, _ = chain(tmp_path, HEH_FIXTURES[:2] + HEH_FIXTURES[3:])
        kept_heh = next(r for r in rows if r["source_file"] == "a_heh.xyz")
        dup = next(r for r in rej if r["source_file"] == "b_heh_copy.xyz")
        assert dup["cluster_id"] == kept_heh["cluster_id"]

    def test_rejected_sentinel_written(self, tmp_path):
        _, _, _, _, _, rejects = chain(tmp_path, HEH_FIXTURES[:2] + HEH_FIXTURES[3:])
        assert os.path.exists(os.path.join(rejects, ".REJECTED"))

    def test_same_composition_different_multiplicity_not_deduped(self, tmp_path):
        # Different multiplicity -> different geometry_key AND different
        # composition group -> both kept even though geometry matches.
        heh_m2 = XYZ_HEH_EXTXYZ.replace("multiplicity=1", "multiplicity=2")
        fixtures = [("a_heh.xyz", XYZ_HEH_EXTXYZ), ("b_heh_m2.xyz", heh_m2)]
        _, rows, rej, _, _, _ = chain(tmp_path, fixtures)
        assert len(rows) == 2
        assert rej == []
        assert {r["multiplicity"] for r in rows} == {1, 2}


class TestNearDupClustering:
    def test_translated_copy_dropped_near_dup(self, tmp_path):
        _, rows, rej, _, _, _ = chain(tmp_path, HEH_FIXTURES)
        kept = {r["source_file"] for r in rows}
        assert kept == {"a_heh.xyz", "d_li4.xyz", "e_li4_conf2.xyz"}
        moved = [r for r in rej if r["source_file"] == "c_heh_moved.xyz"]
        assert len(moved) == 1
        assert moved[0]["dedup_status"] == "near_dup"
        assert "RMSD" in moved[0]["reason"]
        assert "0.25" in moved[0]["reason"]

    def test_removed_sentinel_written(self, tmp_path):
        _, _, _, _, _, rejects = chain(tmp_path, HEH_FIXTURES)
        assert os.path.exists(os.path.join(rejects, ".REMOVED"))
        with open(os.path.join(rejects, ".REMOVED")) as f:
            assert "1 frames removed" in f.read()

    def test_distinct_conformers_kept_with_own_cluster_ids(self, tmp_path):
        _, rows, _, _, _, _ = chain(tmp_path, HEH_FIXTURES)
        li4 = next(r for r in rows if r["source_file"] == "d_li4.xyz")
        conf2 = next(r for r in rows if r["source_file"] == "e_li4_conf2.xyz")
        assert li4["dedup_status"] == "kept"
        assert conf2["dedup_status"] == "kept"
        assert li4["cluster_id"] != conf2["cluster_id"]

    def test_isomer_pair_both_kept(self, tmp_path):
        fixtures = [("a_iso_a.xyz", XYZ_ISO_A_EXTXYZ), ("b_iso_b.xyz", XYZ_ISO_B_EXTXYZ)]
        _, rows, rej, _, _, _ = chain(tmp_path, fixtures)
        assert len(rows) == 2
        assert rej == []
        assert all(r["dedup_status"] == "kept" for r in rows)
        assert rows[0]["cluster_id"] != rows[1]["cluster_id"]

    def test_cluster_id_semantics_near_dup_carries_rep_id(self, tmp_path):
        _, rows, rej, _, _, _ = chain(tmp_path, HEH_FIXTURES)
        kept_heh = next(r for r in rows if r["source_file"] == "a_heh.xyz")
        moved = next(r for r in rej if r["source_file"] == "c_heh_moved.xyz")
        assert moved["cluster_id"] == kept_heh["cluster_id"]

    def test_custom_threshold_catches_conformer(self, tmp_path):
        # Li4 conformer2 RMSD 0.479 A: kept at 0.25, dropped at 1.0.
        _, rows, rej, _, _, _ = chain(tmp_path, HEH_FIXTURES, threshold=1.0)
        assert {r["source_file"] for r in rows} == {"a_heh.xyz", "d_li4.xyz"}
        dropped = {r["source_file"] for r in rej if r["dedup_status"] == "near_dup"}
        assert dropped == {"c_heh_moved.xyz", "e_li4_conf2.xyz"}

    def test_threshold_range_guard(self, tmp_path):
        input_dir = make_input_dir(tmp_path, [("a.xyz", XYZ_HEH_EXTXYZ)])
        batches = str(tmp_path / "batches")
        rc, _ = run_ingest(input_dir, batches)
        assert rc == 0
        for bad in (0.05, 5.1):
            rc, _, err = run_identity(batches, str(tmp_path / f"out_{bad}"), threshold=bad)
            assert rc == 1
            assert "--near-dup-threshold" in err

    def test_out_of_group_never_compared(self, tmp_path):
        # B2H6 vs Li4: different composition groups, no RMSD between them;
        # both kept regardless of geometry proximity.
        fixtures = [("a_b2h6.xyz", XYZ_B2H6_EXTXYZ), ("b_li4.xyz", XYZ_LI4_EXTXYZ)]
        _, rows, rej, _, _, _ = chain(tmp_path, fixtures)
        assert len(rows) == 2
        assert rej == []


class TestForceKeep:
    def test_force_keep_rejected_keeps_all(self, tmp_path):
        _, rows, rej, _, _, _ = chain(tmp_path, HEH_FIXTURES, force_keep=True)
        assert len(rows) == 5
        statuses = {r["source_file"]: r["dedup_status"] for r in rows}
        assert statuses["b_heh_copy.xyz"] == "exact_dup"
        assert statuses["c_heh_moved.xyz"] == "near_dup"
        assert statuses["a_heh.xyz"] == "kept"
        # rejects parquet + sentinels still written (audit trail)
        assert len(rej) == 2

    def test_cluster_id_recorded_when_force_kept(self, tmp_path):
        _, rows, _, _, _, _ = chain(tmp_path, HEH_FIXTURES, force_keep=True)
        kept_heh = next(r for r in rows if r["source_file"] == "a_heh.xyz")
        dup = next(r for r in rows if r["source_file"] == "b_heh_copy.xyz")
        moved = next(r for r in rows if r["source_file"] == "c_heh_moved.xyz")
        assert dup["cluster_id"] == kept_heh["cluster_id"]
        assert moved["cluster_id"] == kept_heh["cluster_id"]


class TestOutputContract:
    def test_stage_columns_present(self, tmp_path):
        _, rows, _, _, _, _ = chain(tmp_path, HEH_FIXTURES)
        for r in rows:
            assert r["dedup_status"] == "kept"
            assert isinstance(r["cluster_id"], int)
            assert r["geometry_id"]  # canonical value rewritten at load

    def test_geometry_id_recomputed_matches_p3(self, tmp_path):
        from curation.realspace.identity import geometry_key

        _, rows, _, _, _, _ = chain(tmp_path, [("a.xyz", XYZ_HEH_EXTXYZ)])
        r = rows[0]
        expected = geometry_key(r["symbols"].split(), _coords(r), r["charge"], r["multiplicity"])
        assert r["geometry_id"] == expected

    def test_error_row_rejected_not_crash(self, tmp_path):
        # Hand-craft a batch with one corrupt row + one good row.
        import numpy as np

        from curation.realspace.schema import frame_to_row, write_realspace_batch
        from lib.xyz_io import XyzFrame

        good = frame_to_row(
            XyzFrame(
                symbols=["He", "H"],
                coords=np.array([[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]]),
                charges=None,
                header={},
            ),
            geometry_id="x",
            charge=1,
            multiplicity=1,
            source_id="good.xyz:0",
            source_file="good.xyz",
            source_index=0,
        )
        bad = dict(good, coords="garbage not coords", source_id="bad.xyz:0", source_file="bad.xyz")
        batches = str(tmp_path / "batches")
        write_realspace_batch(os.path.join(batches, "batch_0000.parquet"), [bad, good])
        rejects = str(tmp_path / "rejects" / "02_identity")
        rc, out, err = run_identity(batches, str(tmp_path / "deduped"), rejects_dir=rejects)
        assert rc == 0, out + err
        _, rows = read_rows(str(tmp_path / "deduped"))
        assert [r["source_file"] for r in rows] == ["good.xyz"]
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "identity_rejected.parquet"))
        assert len(rej) == 1
        assert rej[0]["dedup_status"] == "error"
        assert "row decode failed" in rej[0]["reason"]

    def test_empty_input_dir_exits(self, tmp_path):
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        rc, _, _ = run_identity(empty, str(tmp_path / "out"))
        assert rc == 1

    def test_provenance_recorded(self, tmp_path):
        base = tmp_path / "base"
        input_dir = make_input_dir(tmp_path, HEH_FIXTURES)
        batches = str(base / "batches")
        deduped = str(base / "deduped")
        rc, _ = run_ingest(input_dir, batches)
        assert rc == 0
        rc, _, err = run_identity(
            batches, deduped, rejects_dir=str(base / "rejects" / "02_identity")
        )
        assert rc == 0, err
        with open(base / "provenance.json") as f:
            prov = json.load(f)
        stages = [s["stage"] for s in prov["stages"]]
        assert "01_ingest" in stages
        assert "02_identity" in stages


class TestMiniChainSmoke:
    def test_ingest_then_identity_end_to_end(self, tmp_path):
        _, rows, rej, out, out_i, _ = chain(tmp_path, HEH_FIXTURES, batch_size=2)
        kept = sorted(r["source_file"] for r in rows)
        assert kept == ["a_heh.xyz", "d_li4.xyz", "e_li4_conf2.xyz"]
        assert sorted(r["source_file"] for r in rej) == ["b_heh_copy.xyz", "c_heh_moved.xyz"]
        assert "Written to output" in out  # identity report reached the end
        assert "3" in out.split("Written to output")[1][:20]

    def test_batch_size_preserved_through_chain(self, tmp_path):
        # 5 frames, batch_size=2 -> 3 ingest batches; identity keeps rows in
        # original batch files where non-empty.
        _, rows, _, _, out_i, _ = chain(tmp_path, HEH_FIXTURES, batch_size=2)
        assert "batch_0002.parquet" in out_i  # 3rd ingest batch was written
        assert len(rows) == 3


def _coords(row):
    import numpy as np

    from curation.realspace.schema import coords_from_text

    return np.asarray(coords_from_text(row["coords"]))
