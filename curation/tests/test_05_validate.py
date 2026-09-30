"""test_05_validate.py — Regression tests for Stage 5 integrity checks."""

import hashlib
import importlib
import os
import sys

from rdkit import Chem

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_validate = importlib.import_module("05_validate")
_dedup = importlib.import_module("04_dedup")


def _stage4_row(smiles):
    """Build a stage-4-style row (aromatic canonical + CompoundID) from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    block = Chem.MolToMolBlock(mol)  # kekulized, explicit H
    can, new_block, status, reason = _dedup.canonicalize_and_assign(block, smiles)
    assert status == "ok", reason
    mol2 = Chem.MolFromMolBlock(new_block, sanitize=False, removeHs=False)
    return {
        "mol_block": new_block,
        "num_atoms": mol2.GetNumAtoms(),
        "num_bonds": mol2.GetNumBonds(),
        "CanonicalSMILES": can,
        "CompoundID": hashlib.md5(can.encode()).hexdigest(),
        "AtropisomerKey": "",
        "Energy_Ha": -500.0,
        "SourceID": "TEST",
    }


class TestValidateCanonical:
    def test_aromatic_canonical_roundtrip_ok(self):
        """Stage 4 emits AROMATIC canonical SMILES; Stage 5 sanitize round-trip
        must match. Regression: unsanitized MolToSmiles (kekule) caused
        122,949 false 'CanonicalSMILES mismatch' invalidations."""
        row = _stage4_row("c1ccccc1")
        assert "c1" in row["CanonicalSMILES"]  # aromatic, not kekule
        status, errors = _validate.validate_row(row)
        assert status == "ok", errors

    def test_compoundid_with_atrop_key_ok(self):
        """CompoundID = MD5(canonical | atrop_key); validate must reproduce it
        (Stage 4 appends atrop_key, old Stage 5 ignored it)."""
        row = _stage4_row("CCO")
        row["AtropisomerKey"] = "6:5"
        row["CompoundID"] = hashlib.md5((row["CanonicalSMILES"] + "|6:5").encode()).hexdigest()
        status, errors = _validate.validate_row(row)
        assert status == "ok", errors
