"""test_05_validate.py — Tests for Stage 5 integrity cross-checks.

Covers validate_row (all 5 per-molecule checks + kekulize-fail branch)
and the CLI (valid/invalid batches, CID collisions, reject SDF,
sentinel, --skip, provenance).
"""

import hashlib
import importlib
import json
import os
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "graph")
_SCRIPT = os.path.join(_SCRIPT_DIR, "05_validate.py")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_validate = importlib.import_module("05_validate")

from lib.parquet_io import write_batch

# Aromatic bond-type-4 5-ring: parses unsanitized, SanitizeMol raises
# KekulizeException → exercises the "kekulize failed" branch.
AROM4_BLOCK = (
    "\n     RDKit          2D\n\n"
    "  5  5  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    1.0000    0.0000 C   0  0\n"
    "    0.9511    0.3090    0.0000 C   0  0\n"
    "    0.5878   -0.8090    0.0000 C   0  0\n"
    "   -0.5878   -0.8090    0.0000 C   0  0\n"
    "   -0.9511    0.3090    0.0000 C   0  0\n"
    "  1  2  4  0\n  2  3  4  0\n  3  4  4  0\n  4  5  4  0\n  5  1  4  0\n"
    "M  END\n"
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _row(
    smiles="CCO",
    source_id="S1",
    energy=-500.0,
    atrop="",
    block=None,
    **extra,
):
    """Row with valid mol_block + consistent metadata.

    CanonicalSMILES and CompoundID are computed exactly the way
    validate_row recomputes them (round-trip safe by construction).
    """
    if block is None:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
        block = Chem.MolToMolBlock(mol)

    mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    canonical = ""
    if mol is not None:
        try:
            mol_san = Chem.Mol(mol)
            Chem.SanitizeMol(mol_san)
            canonical = Chem.MolToSmiles(mol_san)
        except Exception:
            pass  # unsanitizable block (kekulize test) → empty canonical

    cid_input = canonical + (f"|{atrop}" if atrop else "")
    row = {
        "mol_block": block,
        "num_atoms": mol.GetNumAtoms() if mol else 0,
        "num_bonds": mol.GetNumBonds() if mol else 0,
        "CanonicalSMILES": canonical,
        "Energy_Ha": energy,
        "CompoundID": hashlib.md5(cid_input.encode()).hexdigest(),
        "SourceID": source_id,
    }
    if atrop:
        row["AtropisomerKey"] = atrop
    row.update(extra)
    return row


def _run_validate(input_dir, extra_args=None):
    cmd = [sys.executable, _SCRIPT, "-i", input_dir]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


# -----------------------------------------------------------------------
# Unit tests: validate_row
# -----------------------------------------------------------------------


class TestValidateRow:
    def test_valid_row_ok(self):
        status, errors = _validate.validate_row(_row())
        assert status == "ok"
        assert errors == []

    def test_valid_row_with_atrop_key_ok(self):
        """CID recomputed WITH atrop suffix → consistent, ok."""
        status, errors = _validate.validate_row(_row(atrop="6:5"))
        assert status == "ok"
        assert errors == []

    def test_corrupt_molblock(self):
        status, errors = _validate.validate_row(_row(block="not a mol block"))
        assert status == "invalid"
        assert errors == ["mol_block: could not parse"]

    def test_num_atoms_mismatch(self):
        row = _row()
        row["num_atoms"] = 999
        status, errors = _validate.validate_row(row)
        assert status == "invalid"
        assert any("num_atoms mismatch" in e for e in errors)

    def test_num_bonds_mismatch(self):
        row = _row()
        row["num_bonds"] = 999
        status, errors = _validate.validate_row(row)
        assert status == "invalid"
        assert any("num_bonds mismatch" in e for e in errors)

    def test_canonical_smiles_mismatch(self):
        row = _row()
        row["CanonicalSMILES"] = "CCC"  # stored ≠ recomputed from mol_block
        status, errors = _validate.validate_row(row)
        assert status == "invalid"
        assert any("CanonicalSMILES mismatch" in e for e in errors)

    def test_energy_not_finite(self):
        for bad in (float("nan"), float("inf"), None):
            row = _row()
            row["Energy_Ha"] = bad
            status, errors = _validate.validate_row(row)
            assert status == "invalid", bad
            assert any("Energy_Ha: not finite" in e for e in errors)

    def test_compoundid_mismatch(self):
        row = _row()
        row["CompoundID"] = "0" * 32
        status, errors = _validate.validate_row(row)
        assert status == "invalid"
        assert any("CompoundID mismatch" in e for e in errors)

    def test_kekulize_failure_reported(self):
        """Aromatic bond-type-4 block parses but fails sanitization →
        'kekulize failed' error, not a crash."""
        row = _row(block=AROM4_BLOCK)
        row["CanonicalSMILES"] = "C1=CC=CC=C1"
        status, errors = _validate.validate_row(row)
        assert status == "invalid"
        assert any("kekulize failed" in e for e in errors)


# -----------------------------------------------------------------------
# CLI tests
# -----------------------------------------------------------------------


class TestValidateCLI:
    def test_valid_batch_passes(self, tmp_path):
        in_dir = tmp_path / "in"
        rejects = tmp_path / "rejects"
        in_dir.mkdir()
        write_batch(str(in_dir / "batch_000.parquet"), [_row(source_id="S1"), _row(source_id="S2")])

        rc, stdout, stderr = _run_validate(str(in_dir), ["--rejects-dir", str(rejects)])
        assert rc == 0, stderr
        assert "All checks passed." in stdout
        assert "OK:             2" in stdout
        assert not (rejects / "validate_invalid.sdf").exists()
        assert not (rejects / ".INVALID").exists()

        prov = json.loads((tmp_path / "provenance.json").read_text())
        assert "05_validate" in json.dumps(prov)

    def test_invalid_batch_fails_with_sentinel(self, tmp_path):
        in_dir = tmp_path / "in"
        rejects = tmp_path / "rejects"
        in_dir.mkdir()
        bad = _row(source_id="BAD")
        bad["num_atoms"] = 999
        write_batch(str(in_dir / "batch_000.parquet"), [_row(source_id="S1"), bad])

        rc, stdout, stderr = _run_validate(str(in_dir), ["--rejects-dir", str(rejects)])
        assert rc == 1, stderr
        assert "Invalid:        1" in stdout
        assert (rejects / "validate_invalid.sdf").exists()
        sentinel = rejects / ".INVALID"
        assert sentinel.exists()
        assert "1 invalid" in sentinel.read_text()

    def test_cid_collision_warns(self, tmp_path):
        """Two identical molecules (same CID, both valid) → collision
        warning; exit stays 0 (warning, not failure)."""
        in_dir = tmp_path / "in"
        rejects = tmp_path / "rejects"
        in_dir.mkdir()
        rows = [_row(source_id="S1"), _row(source_id="S2")]  # same SMILES → same CID
        write_batch(str(in_dir / "batch_000.parquet"), rows)

        rc, stdout, stderr = _run_validate(str(in_dir), ["--rejects-dir", str(rejects)])
        assert rc == 0, stderr
        assert "CID collisions: 1" in stdout
        assert "CompoundID collisions detected" in stdout

    def test_rejected_status_warning(self, tmp_path):
        in_dir = tmp_path / "in"
        rejects = tmp_path / "rejects"
        in_dir.mkdir()
        row = _row(source_id="S1", filter_status="rejected")
        write_batch(str(in_dir / "batch_000.parquet"), [row])

        rc, stdout, stderr = _run_validate(str(in_dir), ["--rejects-dir", str(rejects)])
        assert rc == 0, stderr
        assert "still present in batches" in stdout

    def test_skip_flag(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_batch(str(in_dir / "batch_000.parquet"), [_row()])

        rc, stdout, stderr = _run_validate(str(in_dir), ["--skip"])
        assert rc == 0, stderr
        assert "Validation skipped" in stdout

    def test_empty_input_dir_exits_1(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        rc, stdout, _ = _run_validate(str(in_dir))
        assert rc == 1
        assert "No .parquet files" in stdout

    def test_report_counts_mixed_batch(self, tmp_path):
        """1 ok + 2 invalid → report arithmetic correct."""
        in_dir = tmp_path / "in"
        rejects = tmp_path / "rejects"
        in_dir.mkdir()
        bad1 = _row(source_id="B1")
        bad1["num_atoms"] = 999
        bad2 = _row(source_id="B2", block="garbage")
        write_batch(
            str(in_dir / "batch_000.parquet"),
            [_row(source_id="S1"), bad1, bad2],
        )

        rc, stdout, stderr = _run_validate(str(in_dir), ["--rejects-dir", str(rejects)])
        assert rc == 1, stderr
        assert "Total:          3" in stdout
        assert "OK:             1" in stdout
        assert "Invalid:        2" in stdout
