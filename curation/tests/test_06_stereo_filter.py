"""test_06_stereo_filter.py — TDD tests for Stage 6 enantiomer filter.

Verifies:
  - Default: enantiomers detected and KEPT (D09-overturn)
  - --remove-enantiomers: one per pair removed (legacy behavior)
  - --force-keep-rejected: removed entirely (script rejects flag)
  - smiles_unparseable handling removed
  - Double bond stereochemistry inversion in enantiomer detection
  - Complex (multi-fragment) molecules skipped
  - --robust-stereo path works
"""

import os
import sys
import subprocess
import tempfile
import re

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "06_stereo_filter.py",
)
if os.path.dirname(_SCRIPT) not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCRIPT))

from lib.parquet_io import read_batch, write_batch
from lib.schema import compound_id


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_mol_block(smiles, source_id, energy=-500.0):
    """Create a minimal valid SDF mol_block from a SMILES string.

    Uses 2D coordinates so tetrahedral + E/Z stereo round-trips through
    MolToMolBlock → MolFromMolBlock (Stage 6 reads mol_block now).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    mol.SetProp("_Name", source_id)
    mol.SetProp("Energy_Ha", str(energy))
    mol.SetProp("FormalCharge", "0")
    mol.SetProp("Multiplicity", "1")
    mol.SetProp("SMILES", smiles)
    mol.SetProp("SourceID", source_id)
    return Chem.MolToMolBlock(mol)


def _canonical_isomeric(smiles):
    """Canonical isomeric SMILES via RDKit."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return smiles
    Chem.SanitizeMol(mol, catchErrors=True)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _make_row(smiles, source_id, energy=-500.0):
    """Build a single Parquet row for the stereo filter."""
    can_smi = _canonical_isomeric(smiles)
    cid = compound_id(can_smi)
    mol_block = _make_mol_block(smiles, source_id, energy)
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    return {
        "mol_block":        mol_block,
        "num_atoms":        mol.GetNumAtoms(),
        "num_bonds":        mol.GetNumBonds(),
        "Energy_Ha":        energy,
        "FormalCharge":     0,
        "Multiplicity":     1,
        "SMILES":           smiles,
        "SourceID":         source_id,
        "CompoundID":       cid,
        "CanonicalSMILES":  can_smi,
    }


def _make_atrop_row(smiles, source_id, atrop_bond_idx, atrop_cw=True,
                    energy=-500.0):
    """Build a row whose mol_block carries atropisomer bond stereo.

    Uses AddHs + Compute2DCoords so tet chirality round-trips; atrop bond
    stereo is set directly on the bond (SMILES cannot encode it).
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    stereo = (Chem.BondStereo.STEREOATROPCW if atrop_cw
              else Chem.BondStereo.STEREOATROPCCW)
    mol.GetBondWithIdx(atrop_bond_idx).SetStereo(stereo)
    mol.SetProp("_Name", source_id)
    mol.SetProp("Energy_Ha", str(energy))
    mol.SetProp("FormalCharge", "0")
    mol.SetProp("Multiplicity", "1")
    mol.SetProp("SMILES", smiles)
    mol.SetProp("SourceID", source_id)
    block = Chem.MolToMolBlock(mol)
    can_smi = _canonical_isomeric(smiles)
    cid = compound_id(can_smi)
    mol2 = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    return {
        "mol_block":        block,
        "num_atoms":        mol2.GetNumAtoms(),
        "num_bonds":        mol2.GetNumBonds(),
        "Energy_Ha":        energy,
        "FormalCharge":     0,
        "Multiplicity":     1,
        "SMILES":           smiles,
        "SourceID":         source_id,
        "CompoundID":       cid,
        "CanonicalSMILES":  can_smi,
        "AtropisomerKey":   f"{atrop_bond_idx}:{5 if atrop_cw else 6}",
    }


def _make_batch(path, rows):
    write_batch(path, rows)


def _run_filter(input_dir, output_dir, remove_enantiomers=False,
                robust_stereo=False, rejects_dir=None):
    cmd = [
        sys.executable, _SCRIPT,
        "-i", input_dir, "-o", output_dir,
    ]
    if remove_enantiomers:
        cmd.append("--remove-enantiomers")
    if robust_stereo:
        cmd.append("--robust-stereo")
    if rejects_dir:
        cmd.extend(["--rejects-dir", rejects_dir])
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def _report_count(stdout, label):
    """Extract integer count from report line matching label."""
    pattern = re.escape(label) + r"\s+(\d+)"
    m = re.search(pattern, stdout)
    if m:
        return int(m.group(1))
    return None


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def alanine_enantiomers():
    """L-alanine and D-alanine — single tetrahedral stereocenter."""
    return {
        "l_ala":  _make_row("C[C@H](N)C(=O)O", "ALA_L", energy=-500.0),
        "d_ala":  _make_row("C[C@@H](N)C(=O)O", "ALA_D", energy=-499.9),
    }


@pytest.fixture
def enantiomer_with_double_bond():
    """(E,R) vs (E,S) — mirror flips tet only, E stays E → enantiomers."""
    return {
        "er": _make_row("C[C@H](O)/C=C/C", "EN_E_R", energy=-500.0),
        "es": _make_row("C[C@@H](O)/C=C/C", "EN_E_S", energy=-499.9),
    }


@pytest.fixture
def diastereomer_pair():
    """True diastereomers: same tet, opposite double bond (E vs Z).
    Mirror keeps E/Z, so these must NOT be flagged as enantiomers."""
    return {
        "er": _make_row("C[C@H](O)/C=C/C", "DIA_E_R", energy=-500.0),
        "zr": _make_row("C[C@H](O)/C=C\\C", "DIA_Z_R", energy=-499.9),
    }


@pytest.fixture
def diastereomer_both_flipped():
    """(E,R) vs (Z,S) — tet AND double bond differ → diastereomers.
    Regression guard: old code flipped double bonds and wrongly removed these."""
    return {
        "er": _make_row("C[C@H](O)/C=C/C", "DIA2_E_R", energy=-500.0),
        "zs": _make_row("C[C@@H](O)/C=C\\C", "DIA2_Z_S", energy=-499.9),
    }


@pytest.fixture
def complex_molecule():
    """Multi-fragment complex (salt) — should be skipped by stereo filter."""
    return {
        "complex": _make_row("[Na+].[Cl-]", "SALT_1", energy=-500.0),
    }


@pytest.fixture
def no_stereo_molecules():
    """Two molecules with different flat SMILES — no enantiomer relation."""
    return {
        "ethanol": _make_row("CCO", "ETOH", energy=-500.0),
        "propane": _make_row("CCC", "PROP", energy=-499.9),
    }


@pytest.fixture
def spiro_enantiomers():
    """Chiral spiro carbon enantiomers (RDKit bug #9391 territory).
    
    Atom 1 (spiro carbon) sits in two SSSR rings with a chiral tag,
    triggering the robust-path auto-fallback.
    """
    return {
        "r": _make_row("C[C@]12CCCCC1CCCCC2", "SPIRO_R", energy=-500.0),
        "s": _make_row("C[C@@]12CCCCC1CCCCC2", "SPIRO_S", energy=-499.9),
    }


@pytest.fixture
def atrop_enantiomers():
    """Biaryl atropisomer enantiomers (axial chirality CW vs CCW)."""
    return {
        "cw": _make_atrop_row("Clc1ccccc1-c1ccccc1C", "ATROP_CW", 6, True,
                              -500.0),
        "ccw": _make_atrop_row("Clc1ccccc1-c1ccccc1C", "ATROP_CCW", 6, False,
                               -499.9),
    }


@pytest.fixture
def atrop_diastereomers():
    """Tet + atrop: (R,CW) vs (R,CCW) — same tet, opposite atrop.
    Diastereomers (mirror of (R,CW) is (S,CCW))."""
    return {
        "r_cw": _make_atrop_row("C[C@H](O)c1ccccc1-c1ccccc1Cl", "DIA_R_CW",
                                8, True, -500.0),
        "r_ccw": _make_atrop_row("C[C@H](O)c1ccccc1-c1ccccc1Cl", "DIA_R_CCW",
                                 8, False, -499.9),
    }


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestStereoFilter:

    def test_remove_enantiomers_flag(self, alanine_enantiomers):
        """--remove-enantiomers: first enantiomer kept, second removed. Reject SDF written."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            rejects_dir = os.path.join(tmp, "rejects")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [alanine_enantiomers["l_ala"], alanine_enantiomers["d_ala"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir, remove_enantiomers=True, rejects_dir=rejects_dir)
            assert rc == 0
            assert _report_count(stdout, "Removed enantiomers:") == 1

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            assert batch[0]["stereo_status"] == "kept"

            # Reject SDF exists
            reject_sdf = os.path.join(rejects_dir, "stereo_removed.sdf")
            assert os.path.exists(reject_sdf)

    def test_default_keeps_enantiomers(self, alanine_enantiomers):
        """Default (D09-overturn): all molecules kept, no reject SDF,
        stdout reports detected pair count."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            rejects_dir = os.path.join(tmp, "rejects")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [alanine_enantiomers["l_ala"], alanine_enantiomers["d_ala"]]
            )

            rc, stdout, stderr = _run_filter(
                in_dir, out_dir, rejects_dir=rejects_dir
            )
            assert rc == 0
            assert _report_count(stdout, "Enantiomer pairs detected:") == 1
            assert _report_count(stdout, "Removed enantiomers:") == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            for row in batch:
                assert row["stereo_status"] == "kept"

            # No reject SDF
            reject_sdf = os.path.join(rejects_dir, "stereo_removed.sdf")
            assert not os.path.exists(reject_sdf)

    def test_force_keep_rejected_removed(self, alanine_enantiomers):
        """--force-keep-rejected no longer accepted in Stage 6."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [alanine_enantiomers["l_ala"]]
            )

            cmd = [
                sys.executable, _SCRIPT,
                "-i", in_dir, "-o", out_dir,
                "--force-keep-rejected",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode != 0
            assert "unrecognized arguments: --force-keep-rejected" in result.stderr

    def test_enantiomers_with_double_bond(self, enantiomer_with_double_bond):
        """(E,R) vs (E,S) — enantiomers detected despite double bond present."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [enantiomer_with_double_bond["er"], enantiomer_with_double_bond["es"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir, remove_enantiomers=True)
            assert rc == 0
            assert _report_count(stdout, "Removed enantiomers:") == 1

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1

    def test_diastereomers_not_matched(self, diastereomer_pair):
        """(E,R) vs (Z,R) — same tet, opposite double bond → diastereomers.
        Both kept (different energies)."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [diastereomer_pair["er"], diastereomer_pair["zr"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0
            assert _report_count(stdout, "Removed enantiomers:") == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            for row in batch:
                assert row["stereo_status"] == "kept"

    def test_diastereomer_both_flipped_not_matched(self, diastereomer_both_flipped):
        """(E,R) vs (Z,S) — tet AND double bond differ → diastereomers.
        Regression guard: old code (double-bond flip) wrongly removed one."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [diastereomer_both_flipped["er"], diastereomer_both_flipped["zs"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0
            assert _report_count(stdout, "Removed enantiomers:") == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            for row in batch:
                assert row["stereo_status"] == "kept"

    def test_atrop_enantiomers_removed(self, atrop_enantiomers):
        """Atropisomer enantiomers (axial CW vs CCW) detected and removed."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [atrop_enantiomers["cw"], atrop_enantiomers["ccw"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir, remove_enantiomers=True)
            assert rc == 0
            assert _report_count(stdout, "Removed enantiomers:") == 1
            assert _report_count(stdout, "Atropisomer pairs detected:") == 1

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1

    def test_atrop_diastereomers_kept(self, atrop_diastereomers):
        """(R,CW) vs (R,CCW) — diastereomers, both kept."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [atrop_diastereomers["r_cw"], atrop_diastereomers["r_ccw"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0
            assert _report_count(stdout, "Removed enantiomers:") == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            for row in batch:
                assert row["stereo_status"] == "kept"

    def test_mol_block_roundtrip_preserves_stereo(self):
        """mol_block (2D coords + explicit H) round-trips tet/E/Z/atrop.
        Guards the explicit-H caveat: MolFromMolBlock(sanitize=True)
        must reproduce the same stereo as the original SMILES."""
        for smi in ["C[C@H](N)C(=O)O", "C/C=C/[C@H](C)O"]:
            block = _make_mol_block(smi, "RT")
            rt = Chem.MolFromMolBlock(block)  # sanitize=True, removeHs=True
            assert Chem.MolToSmiles(rt, isomericSmiles=True) == \
                   _canonical_isomeric(smi)
        # atropisomer bond stereo
        mol = Chem.MolFromSmiles("Clc1ccccc1-c1ccccc1C")
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        mol.GetBondWithIdx(6).SetStereo(Chem.BondStereo.STEREOATROPCW)
        block = Chem.MolToMolBlock(mol)
        rt = Chem.MolFromMolBlock(block)
        assert rt.GetBondWithIdx(6).GetStereo() == \
               Chem.BondStereo.STEREOATROPCW

    def test_complex_skipped(self, complex_molecule):
        """Multi-fragment molecules get stereo_status=complex, not processed."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [complex_molecule["complex"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0
            assert _report_count(stdout, "Complexes (skipped):") == 1

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            assert batch[0]["stereo_status"] == "complex"

    def test_no_stereo_groups(self, no_stereo_molecules):
        """Molecules with no stereochemistry — all kept, no groups."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [no_stereo_molecules["ethanol"], no_stereo_molecules["propane"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0
            assert _report_count(stdout, "Stereo groups (>1 member):") == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            for row in batch:
                assert row["stereo_status"] == "kept"

    def test_smiles_unparseable_removed(self, alanine_enantiomers):
        """smiles_unparseable counter and message removed from output.
        All valid SMILES molecules process normally."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [alanine_enantiomers["l_ala"], alanine_enantiomers["d_ala"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0
            assert "SMILES unparseable" not in stdout

    def test_robust_stereo_flag(self, alanine_enantiomers):
        """--robust-stereo forces substructure-based comparison.
        Alanine enantiomers detected via robust path."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [alanine_enantiomers["l_ala"], alanine_enantiomers["d_ala"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir, remove_enantiomers=True, robust_stereo=True)
            assert rc == 0
            assert _report_count(stdout, "Removed enantiomers:") == 1

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1

    def test_spiro_auto_fallback(self, spiro_enantiomers):
        """Spiro chiral center triggers robust-path auto-fallback.
        Enantiomers detected, 'robust_fallback' logged."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            _make_batch(
                os.path.join(in_dir, "batch_000.parquet"),
                [spiro_enantiomers["r"], spiro_enantiomers["s"]]
            )

            rc, stdout, stderr = _run_filter(in_dir, out_dir, remove_enantiomers=True)
            assert rc == 0
            assert _report_count(stdout, "Removed enantiomers:") == 1
            assert "Robust fallback groups:" in stdout

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1

    def test_four_stereoisomers_two_pairs(self):
        """4 stereoisomers (2 chiral centers, no symmetry) → 2 enantiomer
        pairs removed, 2 kept. Exercises O(n) hashing with n=4."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_row("C[C@H](O)[C@H](Cl)C", "RR", energy=-500.0),
                _make_row("C[C@@H](O)[C@@H](Cl)C", "SS", energy=-499.9),
                _make_row("C[C@H](O)[C@@H](Cl)C", "RS", energy=-499.8),
                _make_row("C[C@@H](O)[C@H](Cl)C", "SR", energy=-499.7),
            ]
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_filter(in_dir, out_dir, remove_enantiomers=True)
            assert rc == 0
            assert _report_count(stdout, "Enantiomer pairs detected:") == 2
            assert _report_count(stdout, "Removed enantiomers:") == 2

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            for row in batch:
                assert row["stereo_status"] == "kept"

    def test_pair_plus_lone_diastereomer(self):
        """(RR)+(SS) enantiomer pair + lone (RS) diastereomer (mirror absent)
        → 1 removed (pair collapsed), 2 kept."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_row("C[C@H](O)[C@H](Cl)C", "RR", energy=-500.0),
                _make_row("C[C@@H](O)[C@@H](Cl)C", "SS", energy=-499.9),
                _make_row("C[C@H](O)[C@@H](Cl)C", "RS", energy=-499.8),
            ]
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_filter(in_dir, out_dir, remove_enantiomers=True)
            assert rc == 0
            assert _report_count(stdout, "Enantiomer pairs detected:") == 1
            assert _report_count(stdout, "Removed enantiomers:") == 1

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            for row in batch:
                assert row["stereo_status"] == "kept"

    def test_meso_plus_pair(self):
        """Meso + enantiomer pair in one group (3 members, same flat SMILES).
        Meso kept (bucket of 1); chiral pair shares a bucket → 1 removed."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_row("C[C@H](O)[C@H](O)C", "MESO", energy=-500.0),
                _make_row("C[C@H](O)[C@@H](O)C", "CHIR_R", energy=-499.9),
                _make_row("C[C@@H](O)[C@H](O)C", "CHIR_S", energy=-499.8),
            ]
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_filter(in_dir, out_dir, remove_enantiomers=True)
            assert rc == 0
            assert _report_count(stdout, "Enantiomer pairs detected:") == 1
            assert _report_count(stdout, "Removed enantiomers:") == 1

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            kept_statuses = sorted(r["stereo_status"] for r in batch)
            assert kept_statuses == ["kept", "kept"]
            # meso must be among the kept rows
            kept_smiles = {r["CanonicalSMILES"] for r in batch}
            assert _canonical_isomeric("C[C@H](O)[C@H](O)C") in kept_smiles

    def test_conformer_duplicate_not_removed_as_enantiomer(self):
        """Conformer duplicate + enantiomer in same flat group → only the
        true enantiomer removed (dup left for stage 8); 1 pair detected.

        Regression: O(n) hashing bucketed conformer dups (identical canonical
        SMILES) as a second enantiomer → 2 removals from a size-3 bucket.
        """
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_row("C[C@H](N)C(=O)O", "ALA_L", energy=-500.0),
                _make_row("C[C@H](N)C(=O)O", "ALA_L_DUP", energy=-500.1),
                _make_row("C[C@@H](N)C(=O)O", "ALA_D", energy=-499.9),
            ]
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_filter(
                in_dir, out_dir, remove_enantiomers=True
            )
            assert rc == 0
            assert _report_count(stdout, "Enantiomer pairs detected:") == 1
            assert _report_count(stdout, "Removed enantiomers:") == 1

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            kept_ids = {r["SourceID"] for r in batch}
            # conformer duplicate survives (left for stage 8); only true
            # enantiomer removed
            assert "ALA_L" in kept_ids
            assert "ALA_L_DUP" in kept_ids

    def test_dup_of_removed_enantiomer_also_removed(self):
        """Order [Y, X1, X2] (enantiomer first, its conformer dup last) →
        removing the enantiomer also removes its dup. Exactly one stereo
        survives.

        Regression: the seen-guard skipped X2 (kept it), leaving both
        enantiomer stereos alive. Two-pass SMILES-collapse removes all
        conformers of the losing stereo.
        """
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_row("C[C@@H](N)C(=O)O", "ALA_D", energy=-500.0),
                _make_row("C[C@H](N)C(=O)O", "ALA_L", energy=-499.9),
                _make_row("C[C@H](N)C(=O)O", "ALA_L_DUP", energy=-499.8),
            ]
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_filter(
                in_dir, out_dir, remove_enantiomers=True
            )
            assert rc == 0
            assert _report_count(stdout, "Enantiomer pairs detected:") == 1
            assert _report_count(stdout, "Removed enantiomers:") == 2

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            assert batch[0]["SourceID"] == "ALA_D"

    def test_default_keeps_hashing(self):
        """Default (D09-overturn) with 4 stereoisomers → 2 pairs detected,
        0 removed, all 4 kept."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)

            rows = [
                _make_row("C[C@H](O)[C@H](Cl)C", "RR", energy=-500.0),
                _make_row("C[C@@H](O)[C@@H](Cl)C", "SS", energy=-499.9),
                _make_row("C[C@H](O)[C@@H](Cl)C", "RS", energy=-499.8),
                _make_row("C[C@@H](O)[C@H](Cl)C", "SR", energy=-499.7),
            ]
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), rows)

            rc, stdout, stderr = _run_filter(
                in_dir, out_dir
            )
            assert rc == 0
            assert _report_count(stdout, "Enantiomer pairs detected:") == 2
            assert _report_count(stdout, "Removed enantiomers:") == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 4
            for row in batch:
                assert row["stereo_status"] == "kept"
