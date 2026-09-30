"""test_02_energy_prefilter.py — Integration tests for 02_energy_prefilter.py."""

import os
import sys
import subprocess
import tempfile
import pytest

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "curation", "02_energy_prefilter.py",
)


def run_prefilter(input_dir, output_dir, threshold=3.5, rejects_dir=None):
    cmd = [sys.executable, SCRIPT,
           "-i", input_dir, "-o", output_dir,
           "--threshold", str(threshold)]
    if rejects_dir:
        cmd.extend(["--rejects-dir", rejects_dir])
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout


class TestEnergyPrefilter:
    def test_produces_energy_status_column(self, valid_sdf):
        """Run split → prefilter on fixture data, verify energy_status exists."""
        # Build a minimal batch from the fixture
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered")
            rejects_dir = os.path.join(tmp, "rejects")

            # Run split first
            split_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                valid_sdf, "-o", batches_dir, "--batch-size", "10",
            ]
            subprocess.run(split_cmd, capture_output=True)

            # Run prefilter
            rc, stdout = run_prefilter(batches_dir, out_dir)
            assert rc == 0
            assert "Fitting OLS" in stdout

            # Verify output
            from lib.parquet_io import read_batch
            files = sorted(f for f in os.listdir(out_dir)
                           if f.endswith(".parquet"))
            assert len(files) > 0
            batch = read_batch(os.path.join(out_dir, files[0]))
            assert "energy_status" in batch[0]
            for row in batch:
                assert row["energy_status"] in ("ok", "flagged_ols", "energy_positive", "mol_corrupt")

    def test_total_rows_preserved(self, valid_sdf):
        """Prefilter should not drop rows — only tag them."""
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered")

            split_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                valid_sdf, "-o", batches_dir, "--batch-size", "10",
            ]
            subprocess.run(split_cmd, capture_output=True)

            run_prefilter(batches_dir, out_dir)

            from lib.parquet_io import read_batch
            total_in = 0
            for f in sorted(os.listdir(batches_dir)):
                if f.endswith(".parquet"):
                    total_in += len(read_batch(os.path.join(batches_dir, f)))

            total_out = 0
            for f in sorted(os.listdir(out_dir)):
                if f.endswith(".parquet"):
                    total_out += len(read_batch(os.path.join(out_dir, f)))

            assert total_in == total_out

    def test_high_threshold_flags_nothing(self, valid_sdf):
        """Threshold of 100 should flag zero molecules."""
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered")

            split_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                valid_sdf, "-o", batches_dir, "--batch-size", "10",
            ]
            subprocess.run(split_cmd, capture_output=True)

            rc, stdout = run_prefilter(batches_dir, out_dir, threshold=100.0)
            assert rc == 0
            assert "Flagged (OLS):      0" in stdout

    def test_rejects_sdf_not_written_when_none_flagged(self, valid_sdf):
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered")
            rejects_dir = os.path.join(tmp, "rejects")

            split_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                valid_sdf, "-o", batches_dir, "--batch-size", "10",
            ]
            subprocess.run(split_cmd, capture_output=True)

            rc, stdout = run_prefilter(
                batches_dir, out_dir, threshold=100.0,
                rejects_dir=os.path.join(rejects_dir, "02_energy_prefilter"),
            )
            assert rc == 0
            # No reject SDF should be created
            reject_path = os.path.join(
                rejects_dir, "02_energy_prefilter", "energy_flagged.sdf"
            )
            assert not os.path.exists(reject_path)

    def test_fit_ols_zero_mad(self):
        """MAD=0 (perfect fit) → z all zero, no inf/nan."""
        import importlib
        import numpy as np
        _prefilter = importlib.import_module("02_energy_prefilter")

        X = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
        y = np.array([-40.0, -40.0, -40.0])
        beta, residuals, z = _prefilter.fit_ols(X, y)
        assert np.all(np.isfinite(z))
        assert np.all(z == 0.0)

    def test_count_atoms_from_block(self):
        """Text parser matches RDKit counts for explicit-H mol_block."""
        import importlib
        from rdkit import Chem
        _prefilter = importlib.import_module("02_energy_prefilter")

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        block = Chem.MolToMolBlock(mol)
        counts = _prefilter._count_atoms_from_block(block, _prefilter.ATOM_TYPES)
        assert counts["C"] == 2
        assert counts["H"] == 6
        assert counts["O"] == 1
        assert counts["N"] == 0

        # corrupt block → None
        assert _prefilter._count_atoms_from_block("not a mol block", _prefilter.ATOM_TYPES) is None
