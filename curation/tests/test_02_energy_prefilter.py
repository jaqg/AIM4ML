"""test_02_energy_prefilter.py — Integration tests for 02_energy_prefilter.py."""

import importlib
import os
import subprocess
import sys
import tempfile

import numpy as np
from lib.parquet_io import read_batch, write_batch
from rdkit import Chem
from rdkit.Chem import AllChem

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "02_energy_prefilter.py",
)

_prefilter = importlib.import_module("02_energy_prefilter")


def run_prefilter(input_dir, output_dir, threshold=3.5, rejects_dir=None, extra_args=None):
    cmd = [sys.executable, SCRIPT, "-i", input_dir, "-o", output_dir, "--threshold", str(threshold)]
    if rejects_dir:
        cmd.extend(["--rejects-dir", rejects_dir])
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout


def _prow(smiles, source_id, energy):
    """Parquet row with 3D explicit-H mol_block (stage-1 style)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return {
        "mol_block": Chem.MolToMolBlock(mol),
        "num_atoms": mol.GetNumAtoms(),
        "Energy_Ha": energy,
        "SourceID": source_id,
    }


# OLS design probe (verified): 10 clean rows (5 CHON compositions, ±1 Ha
# noise) + 1 row with the CCO composition but bogus energy → clean |z| ≤ 1.4,
# outlier z ≈ -6 at threshold 3.5, using --atom-types C H O.
def _outlier_rows():
    comps = [
        ("CCO", -190.0),
        ("CCC", -160.0),
        ("CCOCC", -290.0),
        ("CCCCC", -260.0),
        ("CC(=O)O", -260.0),
    ]
    rows = []
    for i, (smi, e) in enumerate(comps):
        rows.append(_prow(smi, f"CLEAN_{i}a", e + 1.0))
        rows.append(_prow(smi, f"CLEAN_{i}b", e - 1.0))
    rows.append(_prow("CCO", "OUTLIER", -500.0))  # same comp as CCO, bogus E
    return rows


class TestEnergyPrefilter:
    def test_produces_energy_status_column(self, valid_sdf):
        """Run split → prefilter on fixture data, verify energy_status exists."""
        # Build a minimal batch from the fixture
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered")

            # Run split first
            split_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                valid_sdf,
                "-o",
                batches_dir,
                "--batch-size",
                "10",
            ]
            subprocess.run(split_cmd, capture_output=True)

            # Run prefilter
            rc, stdout = run_prefilter(batches_dir, out_dir)
            assert rc == 0
            assert "Fitting OLS" in stdout

            # Verify output
            from lib.parquet_io import read_batch

            files = sorted(f for f in os.listdir(out_dir) if f.endswith(".parquet"))
            assert len(files) > 0
            batch = read_batch(os.path.join(out_dir, files[0]))
            assert "energy_status" in batch[0]
            for row in batch:
                assert row["energy_status"] in (
                    "ok",
                    "flagged_ols",
                    "energy_positive",
                    "mol_corrupt",
                )

    def test_total_rows_preserved(self, valid_sdf):
        """Prefilter should not drop rows — only tag them."""
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered")

            split_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                valid_sdf,
                "-o",
                batches_dir,
                "--batch-size",
                "10",
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
                valid_sdf,
                "-o",
                batches_dir,
                "--batch-size",
                "10",
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
                valid_sdf,
                "-o",
                batches_dir,
                "--batch-size",
                "10",
            ]
            subprocess.run(split_cmd, capture_output=True)

            rc, stdout = run_prefilter(
                batches_dir,
                out_dir,
                threshold=100.0,
                rejects_dir=os.path.join(rejects_dir, "02_energy_prefilter"),
            )
            assert rc == 0
            # No reject SDF should be created
            reject_path = os.path.join(rejects_dir, "02_energy_prefilter", "energy_flagged.sdf")
            assert not os.path.exists(reject_path)

    def test_fit_ols_zero_mad(self):
        """MAD=0 (perfect fit) → z all zero, no inf/nan."""
        X = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
        y = np.array([-40.0, -40.0, -40.0])
        beta, residuals, z = _prefilter.fit_ols(X, y)
        assert np.all(np.isfinite(z))
        assert np.all(z == 0.0)

    def test_fit_ols_detects_outlier(self):
        """Synthetic design with one bogus-energy row → its |z| > 3.5,
        all clean rows below."""
        X = np.array(
            [
                [2, 6, 1],
                [2, 6, 1],
                [3, 8, 0],
                [3, 8, 0],
                [4, 10, 1],
                [4, 10, 1],
                [5, 12, 0],
                [5, 12, 0],
                [2, 4, 2],
                [2, 4, 2],
                [2, 6, 1],
            ],
            dtype=float,
        )
        y = np.array(
            [-191.0, -189.0, -161.0, -159.0, -291.0, -289.0, -261.0, -259.0, -261.0, -259.0, -500.0]
        )
        _, _, z = _prefilter.fit_ols(X, y)
        assert abs(z[-1]) > 3.5
        assert np.all(np.abs(z[:-1]) < 3.5)

    def test_count_atoms_from_block(self):
        """Text parser matches RDKit counts for explicit-H mol_block."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        block = Chem.MolToMolBlock(mol)
        counts = _prefilter._count_atoms_from_block(block, _prefilter.ATOM_TYPES)
        assert counts["C"] == 2
        assert counts["H"] == 6
        assert counts["O"] == 1
        assert counts["N"] == 0

        # corrupt block → None
        assert _prefilter._count_atoms_from_block("not a mol block", _prefilter.ATOM_TYPES) is None

    def test_count_atoms_corrupt_variants(self):
        """All corrupt-block branches of the text parser → None."""
        at = _prefilter.ATOM_TYPES
        count = _prefilter._count_atoms_from_block
        assert count("a\nb", at) is None  # too few lines
        assert count("a\nb\nc\nxxxx\n", at) is None  # natoms not an int
        assert count("a\nb\nc\n-1 0\n", at) is None  # negative natoms
        assert count("a\nb\nc\n99 0 0\n", at) is None  # natoms > available lines
        assert count("a\nb\nc\n1 0 0 0 0\nC\n", at) is None  # short atom line

    def test_count_atoms_ignores_out_of_set_elements(self):
        """Cl present in block but not in atom_types → silently ignored."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCl"))
        block = Chem.MolToMolBlock(mol)
        counts = _prefilter._count_atoms_from_block(block, ["C", "H", "O"])
        assert counts == {"C": 1, "H": 3, "O": 0}


class TestEnergyPrefilterHardening:
    """CLI paths beyond the happy path: skip/passthrough, energy-positive,
    OLS outlier, corrupt rows, force-keep, abort conditions."""

    def _write_batch_dir(self, tmp, rows, name="in"):
        in_dir = os.path.join(tmp, name)
        os.makedirs(in_dir, exist_ok=True)
        write_batch(os.path.join(in_dir, "batch_000.parquet"), rows)
        return in_dir

    def test_skip_passthrough(self, tmp_path):
        """--skip copies all rows tagged energy_status='skipped'."""
        rows = [_prow("CCO", "S1", -500.0), _prow("CCO", "S2", 50.0)]
        in_dir = self._write_batch_dir(str(tmp_path), rows)
        out_dir = str(tmp_path / "out")

        rc, stdout = run_prefilter(in_dir, out_dir, extra_args=["--skip"])
        assert rc == 0
        assert "Passthrough complete." in stdout

        batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
        assert len(batch) == 2
        assert all(r["energy_status"] == "skipped" for r in batch)

    def test_energy_positive_flagged_dropped_sentinel(self, tmp_path):
        """E>0 row → energy_positive, dropped from output, .FLAGGED +
        reject SDF written, percentages reported."""
        rows = _outlier_rows()  # 10 clean + outlier
        rows.append(_prow("CCO", "POSITIVE", 50.0))
        in_dir = self._write_batch_dir(str(tmp_path), rows)
        out_dir = str(tmp_path / "out")
        rejects = str(tmp_path / "rejects")

        rc, stdout = run_prefilter(
            in_dir, out_dir, rejects_dir=rejects, extra_args=["--atom-types", "C", "H", "O"]
        )
        assert rc == 0
        assert "Energy positive:    1" in stdout
        assert "Flagged (OLS):      1" in stdout
        assert "Dropped:" in stdout
        assert "(% rejected:" in stdout

        batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
        assert len(batch) == 10  # 11 clean-ish minus outlier minus positive
        assert all(r["energy_status"] == "ok" for r in batch)
        assert not any(r["SourceID"] in ("OUTLIER", "POSITIVE") for r in batch)

        flagged = os.path.join(rejects, ".FLAGGED")
        assert os.path.exists(flagged)
        assert "ols=1" in open(flagged).read()
        assert os.path.exists(os.path.join(rejects, "energy_flagged.sdf"))

    def test_ols_outlier_flagged_end_to_end(self, tmp_path):
        """Bogus-energy row (same composition as clean) → flagged_ols."""
        in_dir = self._write_batch_dir(str(tmp_path), _outlier_rows())
        out_dir = str(tmp_path / "out")

        rc, stdout = run_prefilter(in_dir, out_dir, extra_args=["--atom-types", "C", "H", "O"])
        assert rc == 0
        assert "Flagged (OLS):      1" in stdout

        batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
        assert len(batch) == 10
        assert not any(r["SourceID"] == "OUTLIER" for r in batch)

    def test_force_keep_rejected(self, tmp_path):
        """--force-keep-rejected keeps flagged rows in output with status."""
        in_dir = self._write_batch_dir(str(tmp_path), _outlier_rows())
        out_dir = str(tmp_path / "out")

        rc, _ = run_prefilter(
            in_dir,
            out_dir,
            extra_args=["--atom-types", "C", "H", "O", "--force-keep-rejected"],
        )
        assert rc == 0

        batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
        assert len(batch) == 11
        outlier = next(r for r in batch if r["SourceID"] == "OUTLIER")
        assert outlier["energy_status"] == "flagged_ols"

    def test_skip_energy_positive_check(self, tmp_path):
        """With the check off, E>0 rows enter the OLS fit (→ flagged_ols
        here, not energy_positive). 15 clean rows keep the single-row
        leverage low so the fit cannot absorb the outlier."""
        comps = [
            ("CCO", -190.0),
            ("CCC", -160.0),
            ("CCOCC", -290.0),
            ("CCCCC", -260.0),
            ("CC(=O)O", -260.0),
        ]
        rows = []
        for i, (smi, e) in enumerate(comps):
            for noise in (1.0, -1.0, 0.0):
                rows.append(_prow(smi, f"CLEAN_{i}_{noise}", e + noise))
        rows.append(_prow("CCO", "POS", 50.0))
        in_dir = self._write_batch_dir(str(tmp_path), rows)
        out_dir = str(tmp_path / "out")

        rc, stdout = run_prefilter(
            in_dir,
            out_dir,
            extra_args=[
                "--atom-types",
                "C",
                "H",
                "O",
                "--skip-energy-positive-check",
                "--force-keep-rejected",
            ],
        )
        assert rc == 0
        assert "Energy>0 check skipped" in stdout
        assert "Energy positive:    0" in stdout
        assert "Flagged (OLS):      1" in stdout

        batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
        statuses = {r["SourceID"]: r["energy_status"] for r in batch}
        assert statuses["POS"] == "flagged_ols"  # huge residual, not energy_positive

    def test_corrupt_row_marked_and_dropped(self, tmp_path):
        """Corrupt mol_block → mol_corrupt, dropped from output by default."""
        rows = [
            _prow("CCO", "S1", -190.5),
            _prow("CCC", "S2", -159.5),
            _prow("CCOCC", "S3", -290.5),
            _prow("CCCCC", "S4", -259.5),
            _prow("CC(=O)O", "S5", -260.5),
            {"mol_block": "garbage", "Energy_Ha": -500.0, "SourceID": "CORRUPT"},
        ]
        in_dir = self._write_batch_dir(str(tmp_path), rows)
        out_dir = str(tmp_path / "out")

        rc, stdout = run_prefilter(in_dir, out_dir)
        assert rc == 0
        assert "corrupt mols excluded" in stdout
        assert "Mol corrupt:        1" in stdout

        batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
        assert not any(r["SourceID"] == "CORRUPT" for r in batch)

    def test_all_corrupt_nothing_to_fit_aborts(self, tmp_path):
        """Zero fittable rows → 'Nothing to fit' + exit 1."""
        rows = [
            {"mol_block": "garbage1", "Energy_Ha": -500.0, "SourceID": "C1"},
            {"mol_block": "garbage2", "Energy_Ha": -400.0, "SourceID": "C2"},
        ]
        in_dir = self._write_batch_dir(str(tmp_path), rows)

        (
            rc,
            stdout,
        ) = run_prefilter(in_dir, str(tmp_path / "out"))[:2]
        assert rc == 1
        assert "Nothing to fit" in stdout

    def test_empty_input_dir_exits_1(self, tmp_path):
        in_dir = str(tmp_path / "empty")
        os.makedirs(in_dir)
        rc, stdout = run_prefilter(in_dir, str(tmp_path / "out"))
        assert rc == 1
        assert "No .parquet files" in stdout
