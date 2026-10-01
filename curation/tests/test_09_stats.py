"""test_09_stats.py — Tests for Stage 9 descriptors/diversity stats.

Covers compute_descriptors, fingerprint/Tanimoto helpers, exclude rules,
the funnel report, and the full CLI (CSV + plots + provenance).
"""

import csv
import importlib
import json
import math
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SCRIPT = os.path.join(_SCRIPT_DIR, "09_stats.py")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_stats = importlib.import_module("09_stats")

from lib.parquet_io import write_batch

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _run_stats(input_dir, output_dir, extra_args=None):
    cmd = [sys.executable, _SCRIPT, "-i", input_dir, "-o", output_dir]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def _row(smiles, cid, source_id, energy=-500.0, conformer_status="kept", **extra):
    row = {
        "CompoundID": cid,
        "CanonicalSMILES": smiles,
        "Energy_Ha": energy,
        "FormalCharge": 0,
        "Multiplicity": 1,
        "SourceID": source_id,
        "conformer_status": conformer_status,
        "num_atoms": 3,
    }
    row.update(extra)
    return row


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [dict(zip(headers, r)) for r in reader]
    return headers, rows


# -----------------------------------------------------------------------
# Unit tests: compute_descriptors
# -----------------------------------------------------------------------


class TestComputeDescriptors:
    def test_ethanol_known_values(self):
        """CCO → MolWt≈46.07, TPSA≈20.23, nrot=0, 3 heavy atoms."""
        mw, tpsa, logp, nrot, nat = _stats.compute_descriptors("CCO")
        assert math.isclose(mw, 46.07, abs_tol=0.05)
        assert math.isclose(tpsa, 20.23, abs_tol=0.05)
        assert -1.0 < logp < 1.0  # ethanol near zero; exact Crippen value not assumed
        assert nrot == 0
        assert nat == 3

    def test_benzene(self):
        """c1ccccc1 → TPSA 0, no rotatable bonds, 6 atoms."""
        mw, tpsa, logp, nrot, nat = _stats.compute_descriptors("c1ccccc1")
        assert tpsa == 0.0
        assert nrot == 0
        assert nat == 6
        assert math.isclose(mw, 78.11, abs_tol=0.05)

    def test_invalid_smiles_returns_none_tuple(self):
        """Unparseable SMILES → 5×None, no exception."""
        assert _stats.compute_descriptors("C1CC") == (None, None, None, None, None)
        assert _stats.compute_descriptors("") == (None, None, None, None, None)

    def test_zwitterion_descriptors_despite_unsanitized_parse(self):
        """N valence-4 zwitterion: sanitize=False path must still yield
        descriptors (the documented reason for skipping sanitization)."""
        mw, tpsa, logp, nrot, nat = _stats.compute_descriptors("C[NH2+]CC(=O)[O-]")
        assert mw is not None
        assert nat == 6
        assert tpsa > 0  # amide + carboxylate polar surface


# -----------------------------------------------------------------------
# Unit tests: fingerprints + Tanimoto
# -----------------------------------------------------------------------


class TestFingerprintsAndTanimoto:
    def test_fingerprints_preserve_invalid_slots(self):
        fps = _stats.compute_fingerprints(["CCO", "C1CC", "c1ccccc1"])
        assert fps[0][0] is not None
        assert fps[1] == (None, "C1CC")  # slot kept so mapping stays aligned
        assert fps[2][0] is not None

    def test_identical_mols_give_equal_fps(self):
        fps = _stats.compute_fingerprints(["CCO", "CCO"])
        assert fps[0][0] == fps[1][0]

    def test_fewer_than_two_valid_fps_all_none(self):
        """<2 valid fps → every molecule maps to None (no crash)."""
        res = _stats.nearest_neighbour_tanimoto([(None, "C1CC"), (None, "C1CCC")])
        assert res == {"C1CC": None, "C1CCC": None}
        res1 = _stats.nearest_neighbour_tanimoto(_stats.compute_fingerprints(["CCO"]))
        assert res1 == {"CCO": None}

    def test_duplicate_pair_tanimoto_one(self):
        res = _stats.nearest_neighbour_tanimoto(_stats.compute_fingerprints(["CCO", "CCO"]))
        assert res == {"CCO": 1.0}

    def test_distinct_mols_below_one(self):
        res = _stats.nearest_neighbour_tanimoto(_stats.compute_fingerprints(["CCO", "c1ccccc1"]))
        # tiny ethanol vs benzene share no Morgan bits → 0.0 is legitimate;
        # the point is they are not identical (sim < 1.0)
        assert res["CCO"] < 1.0
        assert res["c1ccccc1"] < 1.0

    def test_invalid_smiles_maps_to_none(self):
        fps = _stats.compute_fingerprints(["CCO", "CCC", "C1CC"])
        res = _stats.nearest_neighbour_tanimoto(fps)
        assert res["C1CC"] is None
        assert res["CCO"] is not None

    def test_workers_2_matches_serial(self):
        """Multiprocessing path (imap_unordered + reconstruction) must give
        the same map as the serial loop."""
        smiles = ["CCO", "CCO", "CCC", "c1ccccc1"]
        fps = _stats.compute_fingerprints(smiles)
        serial = _stats.nearest_neighbour_tanimoto(fps, workers=1)
        parallel = _stats.nearest_neighbour_tanimoto(fps, workers=2)
        assert serial == parallel
        assert parallel["CCO"] == 1.0


# -----------------------------------------------------------------------
# Unit tests: exclude rules + funnel
# -----------------------------------------------------------------------


class TestHelpers:
    def test_should_exclude(self):
        row = {"filter_status": "rejected"}
        assert _stats._should_exclude(row, {"filter_status": "rejected"}) is True
        assert _stats._should_exclude(row, {"filter_status": "ok"}) is False
        assert _stats._should_exclude(row, {"missing_col": "x"}) is False

    def test_funnel_cumulative_counts(self, capsys):
        rows = [
            {"energy_status": "flagged_ols"},
            {"energy_status": "ok", "filter_status": "rejected"},
            {"energy_status": "ok", "filter_status": "ok"},
            {"energy_status": "ok", "filter_status": "ok", "stereo_status": "removed_enantiomer"},
        ]
        _stats._print_funnel(rows)
        out = capsys.readouterr().out
        assert "Curation funnel" in out
        final = next(line for line in out.splitlines() if "Final curated" in line)
        assert final.split()[2] == "1"  # 4 input → 1 after 3 rejections


# -----------------------------------------------------------------------
# CLI tests
# -----------------------------------------------------------------------


class TestStatsCLI:
    def test_end_to_end_outputs(self, tmp_path):
        """CSV with descriptor columns, 4 histograms, provenance, funnel."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "stats"
        in_dir.mkdir()
        write_batch(
            str(in_dir / "batch_000.parquet"),
            [_row("CCO", "C1", "M1"), _row("c1ccccc1", "C2", "M2", energy=-499.0)],
        )

        rc, stdout, stderr = _run_stats(str(in_dir), str(out_dir))
        assert rc == 0, stderr

        headers, rows = _read_csv(str(out_dir / "stats_summary.csv"))
        for col in (
            "CompoundID",
            "SourceID",
            "MolWt",
            "TPSA",
            "logP",
            "nrot",
            "num_atoms",
            "max_tanimoto",
        ):
            assert col in headers
        assert len(rows) == 2
        assert all(r["MolWt"] for r in rows)
        assert all(r["max_tanimoto"] == "" for r in rows)  # --tanimoto not passed

        for name in ("hist_nat.pdf", "hist_molwt.pdf", "hist_tpsa.pdf", "hist_energy.pdf"):
            assert (out_dir / "plots" / name).exists()

        # record_run writes provenance.json to the PARENT of the output dir
        prov = json.loads((tmp_path / "provenance.json").read_text())
        assert "09_stats" in json.dumps(prov)

        assert "Curation funnel" in stdout
        assert "Final curated" in stdout

    def test_compoundid_dedup_keeps_conformer_kept(self, tmp_path):
        """Same CompoundID twice → CSV keeps the conformer_status=kept row."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "stats"
        in_dir.mkdir()
        rows = [
            _row("CCO", "CID1", "KEEP_A", energy=-500.1, conformer_status="kept"),
            _row("CCO", "CID1", "KEEP_B", energy=-500.0, conformer_status="ok"),
            _row("CCC", "CID2", "KEEP_C"),
        ]
        write_batch(str(in_dir / "batch_000.parquet"), rows)

        rc, stdout, stderr = _run_stats(str(in_dir), str(out_dir))
        assert rc == 0, stderr

        _, data = _read_csv(str(out_dir / "stats_summary.csv"))
        assert len(data) == 2
        by_cid = {r["CompoundID"]: r for r in data}
        assert by_cid["CID1"]["SourceID"] == "KEEP_A"
        assert "Deduplicated" in stdout

    def test_exclude_rule(self, tmp_path):
        """--exclude filter_status=rejected drops the row before stats."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "stats"
        in_dir.mkdir()
        rows = [
            _row("CCO", "C1", "M1"),
            _row("CCC", "C2", "M2", filter_status="rejected"),
        ]
        write_batch(str(in_dir / "batch_000.parquet"), rows)

        rc, stdout, stderr = _run_stats(
            str(in_dir), str(out_dir), extra_args=["--exclude", "filter_status=rejected"]
        )
        assert rc == 0, stderr
        assert "Excluded" in stdout

        _, data = _read_csv(str(out_dir / "stats_summary.csv"))
        assert len(data) == 1
        assert data[0]["SourceID"] == "M1"

    def test_tanimoto_flag(self, tmp_path):
        """--tanimoto populates max_tanimoto; duplicate SMILES → 1.0."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "stats"
        in_dir.mkdir()
        rows = [
            _row("CCO", "C1", "M1"),
            _row("CCO", "C2", "M2"),  # same SMILES, different CID → survives dedup
            _row("c1ccccc1", "C3", "M3"),
        ]
        write_batch(str(in_dir / "batch_000.parquet"), rows)

        rc, stdout, stderr = _run_stats(str(in_dir), str(out_dir), extra_args=["--tanimoto"])
        assert rc == 0, stderr

        _, data = _read_csv(str(out_dir / "stats_summary.csv"))
        tanimoto = {r["SourceID"]: float(r["max_tanimoto"]) for r in data}
        assert tanimoto["M1"] == 1.0
        assert tanimoto["M2"] == 1.0
        assert tanimoto["M3"] < 1.0
        assert (out_dir / "plots" / "hist_tanimoto.pdf").exists()

    def test_unparseable_smiles_does_not_crash(self, tmp_path):
        """Bad SMILES row → None descriptors in CSV, failure counted, rc=0."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "stats"
        in_dir.mkdir()
        rows = [_row("CCO", "C1", "M1"), _row("C1CC", "C2", "M2")]
        write_batch(str(in_dir / "batch_000.parquet"), rows)

        rc, stdout, stderr = _run_stats(str(in_dir), str(out_dir))
        assert rc == 0, stderr
        assert "descriptor failures" in stdout

        _, data = _read_csv(str(out_dir / "stats_summary.csv"))
        by_src = {r["SourceID"]: r for r in data}
        assert by_src["M1"]["MolWt"] != ""
        assert by_src["M2"]["MolWt"] == ""  # None → empty cell

    def test_empty_input_dir_exits_1(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        rc, stdout, _ = _run_stats(str(in_dir), str(tmp_path / "stats"))
        assert rc == 1
        assert "No .parquet files" in stdout
