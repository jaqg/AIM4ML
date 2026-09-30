"""test_04_dedup.py — Tests for Stage 4 dedup, focused on atropisomer
preservation (atrop-enantiomers must survive dedup with distinct
CompoundIDs so they reach Stage 6)."""

import os
import sys
import subprocess
import tempfile

import pytest
from rdkit import Chem

_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "04_dedup.py",
)
if os.path.dirname(_SCRIPT) not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCRIPT))

from lib.parquet_io import read_batch, write_batch


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_atrop_block(cw=True):
    """H-free biaryl mol_block with atropisomer bond stereo (bond 6)."""
    mol = Chem.MolFromSmiles("Clc1ccccc1-c1ccccc1C")
    stereo = (Chem.BondStereo.STEREOATROPCW if cw
              else Chem.BondStereo.STEREOATROPCCW)
    mol.GetBondWithIdx(6).SetStereo(stereo)
    return Chem.MolToMolBlock(mol)


def _make_simple_block(smiles):
    """H-free mol_block from a plain SMILES."""
    return Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))


def _make_input_row(mol_block, smiles, source_id, energy=-500.0):
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    return {
        "mol_block":       mol_block,
        "num_atoms":       mol.GetNumAtoms(),
        "num_bonds":       mol.GetNumBonds(),
        "Energy_Ha":       energy,
        "FormalCharge":    0,
        "Multiplicity":    1,
        "SMILES":          smiles,
        "SourceID":        source_id,
    }


def _run_dedup(input_dir, output_dir, rejects_dir=None, extra_args=None):
    cmd = [sys.executable, _SCRIPT, "-i", input_dir, "-o", output_dir]
    if rejects_dir:
        cmd.extend(["--rejects-dir", rejects_dir])
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestDedupAtropisomer:
    def test_atrop_enantiomers_survive_dedup(self):
        """Atrop-enantiomers (CW vs CCW) get DIFFERENT CompoundIDs →
        both survive dedup with status 'ok'."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_input_row(_make_atrop_block(True),
                                "Clc1ccccc1-c1ccccc1C", "CW", -500.0),
                _make_input_row(_make_atrop_block(False),
                                "Clc1ccccc1-c1ccccc1C", "CCW", -499.9),
            ]
            write_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_dedup(in_dir, out_dir)
            assert rc == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            assert all(r["dedup_status"] == "ok" for r in batch)
            assert batch[0]["CompoundID"] != batch[1]["CompoundID"]

            keys = {r["AtropisomerKey"] for r in batch}
            assert keys == {"6:5", "6:6"}

    def test_atrop_duplicate_deduped(self):
        """Two identical atrop configs → same CompoundID → second flagged
        as conformer_duplicate."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_input_row(_make_atrop_block(True),
                                "Clc1ccccc1-c1ccccc1C", "CW_1", -500.0),
                _make_input_row(_make_atrop_block(True),
                                "Clc1ccccc1-c1ccccc1C", "CW_2", -499.9),
            ]
            write_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_dedup(in_dir, out_dir)
            assert rc == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            statuses = {r["dedup_status"] for r in batch}
            assert statuses == {"ok", "conformer_duplicate"}
            # Same CompoundID for both
            assert batch[0]["CompoundID"] == batch[1]["CompoundID"]

    def test_no_atrop_empty_key(self):
        """Molecule without atropisomer bonds gets empty AtropisomerKey."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_input_row(_make_simple_block("CCO"), "CCO", "ETOH",
                                -500.0),
            ]
            write_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_dedup(in_dir, out_dir)
            assert rc == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            assert batch[0]["AtropisomerKey"] == ""
            assert batch[0]["dedup_status"] == "ok"

    def test_formula_hill_order(self):
        """Formula uses Hill order (CH3Br, not BrCH3)."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            mol = Chem.AddHs(Chem.MolFromSmiles("CBr"))
            block = Chem.MolToMolBlock(mol)
            row = _make_input_row(block, "CBr", "BROMO", -500.0)
            write_batch(os.path.join(in_dir, "batch_000.parquet"), [row])

            rc, stdout, stderr = _run_dedup(in_dir, out_dir)
            assert rc == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            assert batch[0]["Formula"] == "CH3Br"


# -----------------------------------------------------------------------
# Hardening (defense-in-depth): invalid mol_blocks must not pass silently
# -----------------------------------------------------------------------

def _make_bad_zwitterion_block(smiles):
    """Explicit-H single-bond neutral mol_block with N neutral + 4 bonds
    (secondary ammonium) — invalid valence. Simulates broken upstream output."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for b in mol.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
    for a in mol.GetAtoms():
        a.SetFormalCharge(0)
    return Chem.MolToMolBlock(mol)


class TestDedupHardening:
    def test_bad_molblock_recovers_via_template(self):
        """N neutral valence-4 mol_block + SMILES → template fallback recovers
        the N+ charge instead of silently writing a broken mol_block."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            smi = "C[NH2+]CC(=O)[O-]"
            row = _make_input_row(_make_bad_zwitterion_block(smi), smi,
                                  "ZWIBUG", -500.0)
            write_batch(os.path.join(in_dir, "batch_000.parquet"), [row])

            rc, stdout, stderr = _run_dedup(in_dir, out_dir)
            assert rc == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            assert batch[0]["dedup_status"] == "ok"
            # output mol_block must re-parse with sanitize (stage-6 condition)
            m = Chem.MolFromMolBlock(batch[0]["mol_block"])
            assert m is not None
            charges = {(a.GetSymbol(), a.GetFormalCharge())
                       for a in m.GetAtoms() if a.GetFormalCharge() != 0}
            assert ("N", 1) in charges
            assert ("O", -1) in charges

    def test_bad_molblock_no_smiles_fails_loudly(self):
        """N neutral valence-4 mol_block WITHOUT SMILES → bond_assignment_failed
        (not silently 'ok')."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            smi = "C[NH2+]CC(=O)[O-]"
            row = _make_input_row(_make_bad_zwitterion_block(smi), None,
                                  "ZWIBUG", -500.0)
            write_batch(os.path.join(in_dir, "batch_000.parquet"), [row])

            rc, stdout, stderr = _run_dedup(
                in_dir, out_dir, extra_args=["--force-keep-rejected"]
            )
            assert rc == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            assert batch[0]["dedup_status"] == "bond_assignment_failed"

    def test_bad_rows_dropped_by_default(self):
        """Default: bond_assignment_failed + mol_corrupt dropped from output;
        conformer_duplicate kept (needed by Stage 8 for RMSD selection)."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            bad = _make_input_row(
                _make_bad_zwitterion_block("C[NH2+]CC(=O)[O-]"), None,
                "BAD", -500.0)
            corrupt = {
                "mol_block": "not a valid mol block",
                "num_atoms": 0, "num_bonds": 0,
                "Energy_Ha": -500.0, "FormalCharge": 0, "Multiplicity": 1,
                "SMILES": None, "SourceID": "CORRUPT",
            }
            good1 = _make_input_row(_make_simple_block("CCO"), "CCO",
                                    "E1", -500.0)
            good2 = _make_input_row(_make_simple_block("CCO"), "CCO",
                                    "E2", -499.9)

            write_batch(os.path.join(in_dir, "batch_000.parquet"),
                        [bad, corrupt, good1, good2])

            rc, stdout, stderr = _run_dedup(in_dir, out_dir)
            assert rc == 0
            assert "Bond assignment fail" in stdout
            assert "Mol corrupt" in stdout
            assert "Dropped from output" in stdout

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            statuses = {r["dedup_status"] for r in batch}
            assert statuses == {"ok", "conformer_duplicate"}
            assert {r["SourceID"] for r in batch} == {"E1", "E2"}
