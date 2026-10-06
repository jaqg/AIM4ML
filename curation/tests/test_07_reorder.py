"""test_07_reorder.py — Regression test for Stage 7 RDKit reorder."""

import importlib
import os
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import rdGeometry

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "graph",
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_reorder = importlib.import_module("07_reorder")
_dedup = importlib.import_module("04_dedup")


def _flat_block(smiles):
    """mol_block with explicit H and FLAT (z=0) coords, via the stage-4 path."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, rdGeometry.Point3D(p.x, p.y, 0.0))
    block = Chem.MolToMolBlock(mol)
    can, new_block, status, reason = _dedup.canonicalize_and_assign(block, smiles)
    assert status == "ok", reason
    return new_block


class TestReorder:
    def test_flat_nitrile_imine_no_crash(self):
        """Flat (z=0) coords + nitrile/imine N crashed CanonicalRankAtoms with
        'getNumImplicitHs() without calcImplicitValence()'. UpdatePropertyCache
        fixes it."""
        block = _flat_block("N#CN=c1cc(Cl)[nH]c2ccsc21")
        out = _reorder.reorder_rdkit(block)
        assert out is not None
        m2 = Chem.MolFromMolBlock(out, sanitize=False, removeHs=False)
        assert m2 is not None
        assert m2.GetNumAtoms() == 17  # 13 heavy + 4 explicit H preserved

    def test_normal_molecule_still_reorders(self):
        """Non-degenerate 3D molecule reorders fine (no regression)."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        block = Chem.MolToMolBlock(mol)
        can, new_block, status, reason = _dedup.canonicalize_and_assign(
            block,
            "c1ccccc1",
        )
        assert status == "ok", reason
        out = _reorder.reorder_rdkit(new_block)
        assert out is not None
        m2 = Chem.MolFromMolBlock(out, sanitize=False, removeHs=False)
        assert m2.GetNumAtoms() == 12  # 6 C + 6 H
