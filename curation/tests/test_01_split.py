"""test_01_split.py — Integration tests for 01_split.py."""

import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "common",
    "01_split.py",
)


def run_split(input_sdf, output_dir, batch_size=50):
    cmd = [sys.executable, SCRIPT, input_sdf, "-o", output_dir, "--batch-size", str(batch_size)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout


class TestSplit:
    def test_splits_valid_sdf(self, valid_sdf):
        with tempfile.TemporaryDirectory() as tmp:
            rc, stdout = run_split(valid_sdf, tmp)
            assert rc == 0
            assert "3 molecules loaded" in stdout

    def test_produces_one_parquet_file(self, valid_sdf):
        with tempfile.TemporaryDirectory() as tmp:
            rc, _ = run_split(valid_sdf, tmp, batch_size=50)
            files = sorted(f for f in os.listdir(tmp) if f.endswith(".parquet"))
            assert len(files) == 1
            assert files[0] == "batch_0000.parquet"

    def test_batch_size_respected(self, valid_sdf):
        with tempfile.TemporaryDirectory() as tmp:
            run_split(valid_sdf, tmp, batch_size=2)
            from lib.parquet_io import read_batch

            files = sorted(f for f in os.listdir(tmp) if f.endswith(".parquet"))
            # 3 molecules, batch_size=2 → 2 batches (2 + 1)
            assert len(files) == 2
            batch0 = read_batch(os.path.join(tmp, files[0]))
            assert len(batch0) == 2
            batch1 = read_batch(os.path.join(tmp, files[1]))
            assert len(batch1) == 1

    def test_metadata_types_correct(self, valid_sdf):
        with tempfile.TemporaryDirectory() as tmp:
            run_split(valid_sdf, tmp, batch_size=10)
            from lib.parquet_io import read_batch

            batch = read_batch(os.path.join(tmp, "batch_0000.parquet"))
            r = batch[0]
            assert isinstance(r["Energy_Ha"], float)
            assert isinstance(r["FormalCharge"], (int, type(None)))
            assert isinstance(r["Multiplicity"], (int, type(None)))
            assert isinstance(r["num_atoms"], (int, type(None)))
            assert isinstance(r["num_bonds"], (int, type(None)))

    def test_mol_block_reconstructible(self, valid_sdf):
        with tempfile.TemporaryDirectory() as tmp:
            run_split(valid_sdf, tmp, batch_size=10)
            from rdkit import Chem

            from lib.parquet_io import read_batch

            batch = read_batch(os.path.join(tmp, "batch_0000.parquet"))
            for row in batch:
                mol = Chem.MolFromMolBlock(row["mol_block"], sanitize=False)
                assert mol is not None
                assert mol.GetNumAtoms() == row["num_atoms"]
                assert mol.GetNumBonds() == row["num_bonds"]
