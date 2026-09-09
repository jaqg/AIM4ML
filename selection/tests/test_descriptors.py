"""test_descriptors.py — unit tests for lib/descriptors.py."""

import pytest
from rdkit import Chem

from selection.lib import descriptors as dsc


class TestParseMol:
    def test_valid_block(self):
        mol = Chem.MolFromSmiles("CCO")
        mb = Chem.MolToMolBlock(mol)
        parsed = dsc.parse_mol(mb)
        assert parsed is not None
        assert parsed.GetNumAtoms() == 3

    def test_removes_hydrogens(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        mb = Chem.MolToMolBlock(mol)
        parsed = dsc.parse_mol(mb)
        assert parsed.GetNumAtoms() == 3  # heavy atoms only

    def test_none_and_garbage(self):
        assert dsc.parse_mol(None) is None
        assert dsc.parse_mol("") is None
        assert dsc.parse_mol("this is not a mol block") is None


class TestMorganOnbits:
    def test_sorted_unique_ints(self):
        mol = Chem.MolFromSmiles("CCO")
        ob = dsc.morgan_onbits(mol)
        assert ob == sorted(ob)
        assert len(set(ob)) == len(ob)
        assert all(0 <= i < dsc.NBITS for i in ob)
        assert len(ob) > 0

    def test_deterministic(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert dsc.morgan_onbits(mol) == dsc.morgan_onbits(mol)


class TestAtomEnvs:
    def test_one_list_per_atom(self):
        mol = Chem.MolFromSmiles("CCO")
        envs = dsc.atom_envs(mol)
        assert len(envs) == 3
        for e in envs:
            assert e == sorted(e)
            assert all(isinstance(x, int) for x in e)

    def test_distinct_contexts_have_distinct_envs(self):
        mol = Chem.MolFromSmiles("CC(=O)O")
        envs = dsc.atom_envs(mol)
        elements = dsc.heavy_elements(mol)
        c_idx = [i for i, e in enumerate(elements) if e == "C"]
        assert len(c_idx) == 2
        assert set(envs[c_idx[0]]) != set(envs[c_idx[1]])


class TestScaffold:
    def test_ring_molecule(self):
        s = dsc.scaffold_smiles(Chem.MolFromSmiles("Cc1ccccc1"))
        assert s is not None
        assert "c1ccccc1" in s

    def test_acyclic_is_none(self):
        assert dsc.scaffold_smiles(Chem.MolFromSmiles("CCCO")) is None


class TestBondStrings:
    def test_orders(self):
        bonds = dsc.bond_strings(Chem.MolFromSmiles("CC(=O)O"))
        assert "C-C:1" in bonds
        assert "C-O:2" in bonds
        assert "C-O:1" in bonds
        assert bonds == sorted(bonds)

    def test_aromatic(self):
        bonds = dsc.bond_strings(Chem.MolFromSmiles("c1ccccc1"))
        assert any(b.endswith(":ar") for b in bonds)


class TestMolwtTpsa:
    def test_positive(self):
        mol = Chem.MolFromSmiles("CCO")
        mw, tpsa = dsc.molwt_tpsa(mol)
        assert mw > 0
        assert tpsa >= 0
