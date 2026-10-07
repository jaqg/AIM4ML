"""test_realspace_03_filter.py — Integration tests for realspace 03_filter.

Runs stages as subprocesses (repo convention for digit-named stages).
Unit-level cases hand-craft input batches via write_realspace_batch
(filter is a scalar stage — no need to round-trip xyz for every case);
one full-chain smoke test covers ingest -> identity -> filter.
"""

import json
import os
import subprocess
import sys

import numpy as np
from realspace_fixtures import (
    XYZ_HEH_EXTXYZ,
    XYZ_LI4_EXTXYZ,
    write_xyz_text,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_FILTER = os.path.join(ROOT, "curation", "realspace", "03_filter.py")
SCRIPT_INGEST = os.path.join(ROOT, "curation", "realspace", "01_ingest.py")
SCRIPT_IDENTITY = os.path.join(ROOT, "curation", "realspace", "02_identity.py")


def run_filter(input_dir, output_dir, *extra, rejects_dir=None):
    cmd = [sys.executable, SCRIPT_FILTER, "-i", input_dir, "-o", output_dir, *extra]
    if rejects_dir is not None:
        cmd += ["--rejects-dir", rejects_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def make_row(symbols, coords, charge=0, multiplicity=1, source_id="x:0", source_file="x.xyz"):
    from curation.realspace.schema import frame_to_row
    from lib.xyz_io import XyzFrame

    return frame_to_row(
        XyzFrame(symbols=symbols, coords=np.asarray(coords, dtype=float), charges=None, header={}),
        geometry_id=f"g_{source_id}",
        charge=charge,
        multiplicity=multiplicity,
        source_id=source_id,
        source_file=source_file,
        source_index=0,
    )


HEH_ROW = make_row(
    ["He", "H"],
    [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]],
    charge=1,
    source_id="heh.xyz:0",
    source_file="heh.xyz",
)
HEH_M2_ROW = make_row(
    ["He", "H"],
    [[0.0, 0.0, 0.0], [0.772, 0.0, 0.0]],
    charge=0,
    multiplicity=2,
    source_id="hehm2.xyz:0",
    source_file="hehm2.xyz",
)
NEUTRAL_ROW = make_row(
    ["Li", "Li"],
    [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    source_id="li2.xyz:0",
    source_file="li2.xyz",
)
B2H6_ROW = make_row(
    ["B", "B", "H", "H", "H", "H", "H", "H"],
    np.zeros((8, 3)),
    source_id="b2h6.xyz:0",
    source_file="b2h6.xyz",
)


def craft_batches(tmp_path, rows, name="batches"):
    from curation.realspace.schema import write_realspace_batch

    batches = tmp_path / name
    write_realspace_batch(str(batches / "batch_0000.parquet"), rows)
    return str(batches)


def read_rows(batch_dir):
    from lib.parquet_io import read_batch

    files = sorted(f for f in os.listdir(batch_dir) if f.endswith(".parquet"))
    rows = []
    for f in files:
        rows.extend(read_batch(os.path.join(batch_dir, f)))
    return rows


class TestPassThrough:
    def test_preset_none_default_pass_through(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, NEUTRAL_ROW, B2H6_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(batches, str(tmp_path / "out"), rejects_dir=rejects)
        assert rc == 0, out + err
        assert "Preset: none" in out
        rows = read_rows(str(tmp_path / "out"))
        assert len(rows) == 3
        assert all(r["filter_status"] == "ok" for r in rows)
        assert all("filter_reason" not in r or r["filter_reason"] in (None, "") for r in rows)
        assert not os.path.exists(rejects)  # nothing rejected, no sentinel dir

    def test_explicit_cli_ranges_apply_without_preset(self, tmp_path):
        # no preset -> none, but explicit --charge-range still active
        batches = craft_batches(tmp_path, [HEH_ROW, NEUTRAL_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--charge-range", "0:0", rejects_dir=rejects
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["li2.xyz"]

    def test_empty_input_dir_exits(self, tmp_path):
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        rc, _, _ = run_filter(empty, str(tmp_path / "out"))
        assert rc == 1


class TestChecksIndividually:
    def test_charge_range_rejects(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, NEUTRAL_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--charge-range", "0:0", rejects_dir=rejects
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["li2.xyz"]
        assert os.path.exists(os.path.join(rejects, ".REJECTED"))
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        assert len(rej) == 1
        assert rej[0]["source_file"] == "heh.xyz"
        assert rej[0]["filter_reason"].startswith("charge=")
        assert rej[0]["filter_status"] == "rejected"

    def test_charge_range_accepts_within(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, NEUTRAL_ROW])
        rc, out, err = run_filter(batches, str(tmp_path / "out"), "--charge-range=-1:1")
        assert rc == 0, out + err
        assert len(read_rows(str(tmp_path / "out"))) == 2

    def test_multiplicity_range_rejects(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, HEH_M2_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--multiplicity-range", "1:1", rejects_dir=rejects
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["heh.xyz"]
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        assert rej[0]["filter_reason"] == "multiplicity=2>1"

    def test_min_atoms_rejects(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, B2H6_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--min-atoms", "4", rejects_dir=rejects
        )
        assert rc == 0, out + err
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        assert rej[0]["source_file"] == "heh.xyz"
        assert rej[0]["filter_reason"] == "n_atoms=2<4"

    def test_max_atoms_rejects(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, B2H6_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--max-atoms", "4", rejects_dir=rejects
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["heh.xyz"]

    def test_first_failing_reason_wins(self, tmp_path):
        # B2H6 fails BOTH elements (B not in whitelist) and max-atoms;
        # elements is checked first.
        batches = craft_batches(tmp_path, [B2H6_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches,
            str(tmp_path / "out"),
            "--elements",
            "H,Li",
            "--max-atoms",
            "4",
            rejects_dir=rejects,
        )
        assert rc == 0, out + err
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        assert rej[0]["filter_reason"] == "forbidden_elements:B"


class TestElementsWhitelist:
    def test_whitelist_rejects_foreign_element(self, tmp_path):
        # B2H6 contains B; whitelist He,H -> rejected with B named.
        batches = craft_batches(tmp_path, [B2H6_ROW, HEH_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--elements", "He,H", rejects_dir=rejects
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["heh.xyz"]
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        assert len(rej) == 1
        assert rej[0]["filter_reason"] == "forbidden_elements:B"

    def test_whitelist_is_exact_h_must_be_listed(self, tmp_path):
        # Divergence from graph 03_filter (implicit H): realspace whitelist
        # is exact — --elements Li rejects HeH because H and He are absent.
        batches = craft_batches(tmp_path, [HEH_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--elements", "Li", rejects_dir=rejects
        )
        assert rc == 0, out + err
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        assert rej[0]["filter_reason"] == "forbidden_elements:H,He"

    def test_whitelist_normalizes_case(self, tmp_path):
        # 'li,h' normalizes to 'H,Li' — neutral Li2 passes.
        batches = craft_batches(tmp_path, [NEUTRAL_ROW])
        rc, out, err = run_filter(batches, str(tmp_path / "out"), "--elements", "li,h")
        assert rc == 0, out + err
        assert len(read_rows(str(tmp_path / "out"))) == 1


class TestPresets:
    def test_preset_neutral_closed_shell(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, HEH_M2_ROW, NEUTRAL_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--preset", "neutral_closed_shell", rejects_dir=rejects
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["li2.xyz"]
        from lib.parquet_io import read_batch

        reasons = {
            r["source_file"]: r["filter_reason"]
            for r in read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        }
        assert reasons["heh.xyz"] == "charge=1>0"  # charge fails first
        assert reasons["hehm2.xyz"] == "multiplicity=2>1"

    def test_preset_neutral(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, NEUTRAL_ROW])
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--preset", "neutral", rejects_dir=str(tmp_path / "rej")
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["li2.xyz"]

    def test_preset_closed_shell(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, HEH_M2_ROW])
        rc, out, err = run_filter(
            batches,
            str(tmp_path / "out"),
            "--preset",
            "closed_shell",
            rejects_dir=str(tmp_path / "rej"),
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["heh.xyz"]

    def test_explicit_cli_overrides_preset(self, tmp_path):
        # preset neutral says charge 0:0; explicit --charge-range -1:1 wins
        # and lets the cation through. ('=' form: argparse rejects a bare
        # leading-dash value as a flag.)
        batches = craft_batches(tmp_path, [HEH_ROW])
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--preset", "neutral", "--charge-range=-1:1"
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert len(rows) == 1


class TestRejectsContract:
    def test_rejects_rows_carry_full_context(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches, str(tmp_path / "out"), "--max-atoms", "1", rejects_dir=rejects
        )
        assert rc == 0, out + err
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        r = rej[0]
        assert r["source_id"] == "heh.xyz:0"
        assert r["geometry_id"] == "g_heh.xyz:0"
        assert "0.772000" in r["coords"]
        assert r["filter_reason"] == "n_atoms=2>1"
        assert "frames rejected" in open(os.path.join(rejects, ".REJECTED")).read()

    def test_force_keep_rejected_keeps_rows_with_status(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW, NEUTRAL_ROW])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(
            batches,
            str(tmp_path / "out"),
            "--charge-range",
            "0:0",
            "--force-keep-rejected",
            rejects_dir=rejects,
        )
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert len(rows) == 2
        statuses = {r["source_file"]: r["filter_status"] for r in rows}
        assert statuses == {"heh.xyz": "rejected", "li2.xyz": "ok"}
        # audit trail still written
        from lib.parquet_io import read_batch

        assert len(read_batch(os.path.join(rejects, "filter_rejected.parquet"))) == 1

    def test_bad_range_specs_exit(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW])
        for flag, value in [
            ("--charge-range", "2:1"),
            ("--charge-range", "abc"),
            ("--charge-range", "1"),
            ("--multiplicity-range", "3:2"),
        ]:
            rc, _, err = run_filter(batches, str(tmp_path / "out"), flag, value)
            assert rc == 1, (flag, value)
            assert flag in err

    def test_bad_elements_spec_exits(self, tmp_path):
        batches = craft_batches(tmp_path, [HEH_ROW])
        rc, _, err = run_filter(batches, str(tmp_path / "out"), "--elements", "L1")
        assert rc == 1
        assert "--elements" in err
        rc, _, err = run_filter(batches, str(tmp_path / "out"), "--elements", ",")
        assert rc == 1

    def test_decode_failed_row_rejected_no_crash(self, tmp_path):
        from curation.realspace.schema import write_realspace_batch

        good = NEUTRAL_ROW
        bad = dict(good, coords="garbage not coords", source_id="bad:0", source_file="bad.xyz")
        batches = str(tmp_path / "batches")
        write_realspace_batch(os.path.join(batches, "batch_0000.parquet"), [bad, good])
        rejects = str(tmp_path / "rejects" / "03_filter")
        rc, out, err = run_filter(batches, str(tmp_path / "out"), rejects_dir=rejects)
        assert rc == 0, out + err
        rows = read_rows(str(tmp_path / "out"))
        assert [r["source_file"] for r in rows] == ["li2.xyz"]
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        assert rej[0]["filter_reason"].startswith("row_decode_failed:")
        assert "Decode failed:  1" in out


class TestChainSmoke:
    def test_ingest_identity_filter_end_to_end(self, tmp_path):
        # Fixture dir: HeH+ (cation) + neutral Li4. Chain ingest ->
        # identity -> filter --preset neutral: only Li4 survives.
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True)
        write_xyz_text(input_dir, XYZ_HEH_EXTXYZ, "a_heh.xyz")
        write_xyz_text(input_dir, XYZ_LI4_EXTXYZ, "b_li4.xyz")
        base = tmp_path / "base"
        batches = str(base / "batches")
        deduped = str(base / "deduped")
        filtered = str(base / "filtered")
        rejects = str(base / "rejects" / "03_filter")
        for script, out, extra in [
            (SCRIPT_INGEST, batches, []),
            (SCRIPT_IDENTITY, deduped, ["--rejects-dir", str(base / "rejects" / "02_identity")]),
        ]:
            cmd = [sys.executable, script]
            if script == SCRIPT_INGEST:
                cmd += [str(input_dir)]
            else:
                cmd += ["-i", batches]
            cmd += ["-o", out] + extra
            r = subprocess.run(cmd, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
        rc, out, err = run_filter(deduped, filtered, "--preset", "neutral", rejects_dir=rejects)
        assert rc == 0, out + err
        rows = read_rows(filtered)
        assert [r["source_file"] for r in rows] == ["b_li4.xyz"]
        assert rows[0]["filter_status"] == "ok"
        from lib.parquet_io import read_batch

        rej = read_batch(os.path.join(rejects, "filter_rejected.parquet"))
        assert rej[0]["source_file"] == "a_heh.xyz"
        assert rej[0]["filter_reason"] == "charge=1>0"
        with open(base / "provenance.json") as f:
            prov = json.load(f)
        stages = [s["stage"] for s in prov["stages"]]
        assert "01_ingest" in stages
        assert "02_identity" in stages
        assert "03_filter" in stages
