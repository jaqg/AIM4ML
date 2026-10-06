"""test_10_extxyz.py — Tests for Stage 10 extXYZ delivery format.

Covers mol_block_to_extxyz (frame structure, metadata, CNSO, formula
fallback) and the full CLI (multi-batch conversion, exclude rules,
corrupt-row handling, provenance, atomic writes).
"""

import importlib
import json
import os
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SCRIPT = os.path.join(_SCRIPT_DIR, "10_extxyz.py")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_extxyz = importlib.import_module("10_extxyz")

from lib.parquet_io import write_batch

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _block_3d(smiles):
    """3D mol_block (explicit Hs, embedded coordinates)."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return Chem.MolToMolBlock(mol)


def _row(smiles, source_id, energy=-500.123, **extra):
    block = _block_3d(smiles)
    mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    formula = rdMolDescriptors.CalcMolFormula(Chem.AddHs(Chem.MolFromSmiles(smiles)))
    row = {
        "mol_block": block,
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "Energy_Ha": energy,
        "FormalCharge": 0,
        "Multiplicity": 1,
        "CanonicalSMILES": smiles,
        "SourceID": source_id,
        "CompoundID": f"CID_{source_id}",
        "Formula": formula,
        "ICONF": 7,
    }
    row.update(extra)
    return row


def _parse_meta(meta_line):
    """Parse 'k,v,k,v,...' metadata line into a dict."""
    parts = meta_line.split(",")
    assert len(parts) % 2 == 0, f"odd metadata field count: {meta_line}"
    return dict(zip(parts[0::2], parts[1::2]))


def _parse_extxyz(text):
    """Split extXYZ text into list of (nat, meta_dict, atom_lines)."""
    lines = text.splitlines()
    frames = []
    i = 0
    while i < len(lines):
        nat = int(lines[i])
        meta = _parse_meta(lines[i + 1])
        atoms = lines[i + 2 : i + 2 + nat]
        assert len(atoms) == nat, "atom lines do not match nat"
        frames.append((nat, meta, atoms))
        i += 2 + nat
    return frames


def _run_extxyz(input_dir, output_dir, extra_args=None):
    cmd = [sys.executable, _SCRIPT, "-i", input_dir, "-o", output_dir]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


# -----------------------------------------------------------------------
# Unit tests: mol_block_to_extxyz
# -----------------------------------------------------------------------


class TestMolBlockToExtxyz:
    def test_frame_structure_and_metadata(self):
        """nat line, comma-paired metadata with correct values, atom lines."""
        row = _row("CCO", "S1")
        frame = _extxyz.mol_block_to_extxyz(row["mol_block"], row, family="QM40")
        lines = frame.splitlines()

        nat = int(lines[0])
        assert nat == row["num_atoms"] == 9  # 3 heavy + 6 H

        meta = _parse_meta(lines[1])
        assert meta["SourceID"] == "S1"
        assert meta["CompoundID"] == "CID_S1"
        assert meta["Formula"] == "C2H6O"
        assert meta["Nat"] == "9"
        assert meta["CNSO"] == "3"  # 2 C + 1 O (H not counted)
        assert meta["chrg"] == "0"
        assert meta["mult"] == "1"
        assert meta["E"] == "-500.123"
        assert meta["Family"] == "QM40"
        assert meta["smiles"] == "CCO"
        assert meta["iconf"] == "7"
        assert meta["nfrag"] == "1"

        atoms = lines[2:]
        assert len(atoms) == nat
        for line in atoms:
            sym, x, y, z = line.split()
            assert sym in {"C", "O", "H"}
            float(x), float(y), float(z)  # coordinates parse as floats

    def test_nfrag_reads_n_fragments(self):
        """nfrag metadata comes from row['n_fragments'] (complexes
        export nfrag,2); legacy rows without the column default to 1."""
        row2 = _row("CCO.O", "DIM", n_fragments=2)
        frame = _extxyz.mol_block_to_extxyz(row2["mol_block"], row2, family="X")
        meta = _parse_meta(frame.splitlines()[1])
        assert meta["nfrag"] == "2"

    def test_cnso_counts_cnso_elements_only(self):
        """CNSO = count of C+N+S+O atoms: chlorobenzene → 6 (all C);
        acetic acid → 4 (2 C + 2 O)."""
        row_cl = _row("Clc1ccccc1", "S2")
        frame = _extxyz.mol_block_to_extxyz(row_cl["mol_block"], row_cl)
        meta = _parse_meta(frame.splitlines()[1])
        assert meta["CNSO"] == "6"
        assert meta["Formula"] == "C6H5Cl"

        row_ac = _row("CC(=O)O", "S3")
        frame = _extxyz.mol_block_to_extxyz(row_ac["mol_block"], row_ac)
        meta = _parse_meta(frame.splitlines()[1])
        assert meta["CNSO"] == "4"  # 2 C + 2 O

    def test_family_override(self):
        row = _row("CCO", "S4")
        frame = _extxyz.mol_block_to_extxyz(row["mol_block"], row, family="QMugs")
        assert _parse_meta(frame.splitlines()[1])["Family"] == "QMugs"

    def test_formula_fallback_when_missing(self):
        """No Formula in row → CalcMolFormula fills it from the mol block."""
        row = _row("CCO", "S5")
        del row["Formula"]
        frame = _extxyz.mol_block_to_extxyz(row["mol_block"], row)
        assert _parse_meta(frame.splitlines()[1])["Formula"] == "C2H6O"

    def test_descriptors_from_smiles(self):
        """tpsa/logp/nrot computed from CanonicalSMILES."""
        row = _row("CCO", "S6")
        frame = _extxyz.mol_block_to_extxyz(row["mol_block"], row)
        meta = _parse_meta(frame.splitlines()[1])
        assert float(meta["tpsa"]) > 0
        assert meta["logp"] != ""
        assert meta["nrot"] == "0"

    def test_empty_smiles_leaves_descriptors_empty(self):
        row = _row("CCO", "S7")
        row["CanonicalSMILES"] = ""
        frame = _extxyz.mol_block_to_extxyz(row["mol_block"], row)
        meta = _parse_meta(frame.splitlines()[1])
        assert meta["tpsa"] == ""
        assert meta["logp"] == ""
        assert meta["nrot"] == ""

    def test_corrupt_molblock_returns_none(self):
        assert _extxyz.mol_block_to_extxyz("not a mol block", {}) is None


# -----------------------------------------------------------------------
# CLI tests
# -----------------------------------------------------------------------


class TestExtxyzCLI:
    def test_end_to_end_frames(self, tmp_path):
        """2 rows → 2 parseable frames + provenance recorded."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "extxyz"
        in_dir.mkdir()
        write_batch(
            str(in_dir / "batch_000.parquet"),
            [_row("CCO", "S1"), _row("CCC", "S2", energy=-499.0)],
        )

        rc, stdout, stderr = _run_extxyz(str(in_dir), str(out_dir))
        assert rc == 0, stderr

        out_file = out_dir / "batch_000.extxyz"
        assert out_file.exists()
        frames = _parse_extxyz(out_file.read_text())
        assert len(frames) == 2
        nat1, meta1, _ = frames[0]
        assert meta1["SourceID"] == "S1"
        assert meta1["E"] == "-500.123"

        assert "Written:      2" in stdout
        prov = json.loads((tmp_path / "provenance.json").read_text())
        assert "10_extxyz" in json.dumps(prov)

    def test_multiple_batches_one_file_each(self, tmp_path):
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "extxyz"
        in_dir.mkdir()
        write_batch(str(in_dir / "batch_000.parquet"), [_row("CCO", "S1")])
        write_batch(str(in_dir / "batch_001.parquet"), [_row("CCC", "S2")])

        rc, _, stderr = _run_extxyz(str(in_dir), str(out_dir))
        assert rc == 0, stderr
        assert (out_dir / "batch_000.extxyz").exists()
        assert (out_dir / "batch_001.extxyz").exists()

    def test_exclude_and_corrupt_row_mix(self, tmp_path):
        """Excluded row skipped, corrupt row counted as failed, good row
        still written; rc stays 0."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "extxyz"
        in_dir.mkdir()
        good = _row("CCO", "S1")
        rejected = _row("CCC", "S2", filter_status="rejected")
        corrupt = _row("CCO", "S3")
        corrupt["mol_block"] = "garbage mol block"
        write_batch(str(in_dir / "batch_000.parquet"), [good, rejected, corrupt])

        rc, stdout, stderr = _run_extxyz(
            str(in_dir), str(out_dir), extra_args=["--exclude", "filter_status=rejected"]
        )
        assert rc == 0, stderr
        assert "Written:      1" in stdout
        assert "Excluded:     1" in stdout
        assert "Failed:       1" in stdout

        frames = _parse_extxyz((out_dir / "batch_000.extxyz").read_text())
        assert len(frames) == 1
        assert frames[0][1]["SourceID"] == "S1"

    def test_atomic_write_no_tmp_residue(self, tmp_path):
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "extxyz"
        in_dir.mkdir()
        write_batch(str(in_dir / "batch_000.parquet"), [_row("CCO", "S1")])

        rc, _, stderr = _run_extxyz(str(in_dir), str(out_dir))
        assert rc == 0, stderr
        assert not any(".tmp." in f for f in os.listdir(out_dir))

    def test_empty_input_dir_exits_1(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        rc, stdout, _ = _run_extxyz(str(in_dir), str(tmp_path / "extxyz"))
        assert rc == 1
        assert "No .parquet files" in stdout
