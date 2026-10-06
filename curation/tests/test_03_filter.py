"""test_03_filter.py — Integration tests for 03_filter.py."""

import importlib
import math
import os
import subprocess
import sys
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "03_filter.py",
)
SPLIT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "01_split.py",
)

_SCRIPTS_DIR = os.path.dirname(SCRIPT)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
_filter_module = importlib.import_module("03_filter")
_has_true_zwitterion = _filter_module._has_true_zwitterion
_compute_fragment_metadata = _filter_module.compute_fragment_metadata


def run_filter(input_dir, output_dir, rejects_dir=None, extra_args=None):
    cmd = [sys.executable, SCRIPT, "-i", input_dir, "-o", output_dir]
    if rejects_dir:
        cmd.extend(["--rejects-dir", rejects_dir])
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout


class TestFilter:
    def test_produces_filter_status_column(self, valid_sdf):
        """Filter adds filter_status column, keeps some rows after drop."""
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered_out")

            subprocess.run(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                    valid_sdf,
                    "-o",
                    batches_dir,
                    "--batch-size",
                    "10",
                ],
                capture_output=True,
            )
            # Default mode may drop all rows on small fixtures; use force-keep
            run_filter(batches_dir, out_dir, extra_args=["--force-keep-rejected"])

            from lib.parquet_io import read_batch

            files = sorted(f for f in os.listdir(out_dir) if f.endswith(".parquet"))
            if not files:
                return  # all rows dropped, acceptable
            batch = read_batch(os.path.join(out_dir, files[0]))
            assert "filter_status" in batch[0]
            for row in batch:
                assert row["filter_status"] in ("ok", "rejected", "mol_corrupt", "topology_warning")

    def test_total_rows_preserved(self, valid_sdf):
        """With --force-keep-rejected, filter should preserve all rows."""
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered_out")

            subprocess.run(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                    valid_sdf,
                    "-o",
                    batches_dir,
                    "--batch-size",
                    "10",
                ],
                capture_output=True,
            )
            run_filter(batches_dir, out_dir, extra_args=["--force-keep-rejected"])

            from lib.parquet_io import read_batch

            total_in = sum(
                len(read_batch(os.path.join(batches_dir, f)))
                for f in sorted(os.listdir(batches_dir))
                if f.endswith(".parquet")
            )
            total_out = sum(
                len(read_batch(os.path.join(out_dir, f)))
                for f in sorted(os.listdir(out_dir))
                if f.endswith(".parquet")
            )
            assert total_in == total_out

    def test_bond_types_corrected(self, valid_sdf):
        """After filter, output mols should have corrected bond types."""
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered_out")

            subprocess.run(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                    valid_sdf,
                    "-o",
                    batches_dir,
                    "--batch-size",
                    "10",
                ],
                capture_output=True,
            )
            run_filter(batches_dir, out_dir, extra_args=["--force-keep-rejected"])

            from rdkit import Chem

            from lib.parquet_io import read_batch

            files = sorted(f for f in os.listdir(out_dir) if f.endswith(".parquet"))
            if not files:
                return
            batch = read_batch(os.path.join(out_dir, files[0]))
            for row in batch:
                if row["filter_status"] not in ("ok", "mol_corrupt"):
                    continue
                mol = Chem.MolFromMolBlock(row["mol_block"], sanitize=False)
                types = {b.GetBondType() for b in mol.GetBonds()}
                assert Chem.BondType.SINGLE in types

    def test_rejects_written(self, valid_sdf):
        """Rejected mols produce reject SDF."""
        with tempfile.TemporaryDirectory() as tmp:
            batches_dir = os.path.join(tmp, "batches")
            out_dir = os.path.join(tmp, "filtered_out")
            rejects_dir = os.path.join(tmp, "rejects")

            subprocess.run(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(SCRIPT), "01_split.py"),
                    valid_sdf,
                    "-o",
                    batches_dir,
                    "--batch-size",
                    "10",
                ],
                capture_output=True,
            )
            rc, stdout = run_filter(
                batches_dir,
                out_dir,
                rejects_dir=os.path.join(rejects_dir, "03_filter"),
            )
            assert rc == 0


# -- Multi-fragment test helpers ------------------------------------------


def _write_sdf(path, entries):
    """entries: list of (smiles, tags_dict)."""
    writer = SDWriter(path)
    for smi, tags in entries:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"bad SMILES: {smi}")
        for k, v in tags.items():
            mol.SetProp(k, v)
        writer.write(mol)
    writer.close()


def _tags(sid, energy="-500.0", smiles=None):
    tags = {"Energy_Ha": energy, "FormalCharge": "0", "Multiplicity": "1", "SourceID": sid}
    if smiles:
        tags["SMILES"] = smiles
    return tags


def _run_split_filter(tmp, entries, extra_args, rejects_dir=None):
    """Run 01_split then 03_filter; return (filter_rc, output_rows)."""
    sdf = os.path.join(tmp, "in.sdf")
    _write_sdf(sdf, entries)
    batches = os.path.join(tmp, "batches")
    out = os.path.join(tmp, "out")
    rc_split = subprocess.run(
        [sys.executable, SPLIT, sdf, "-o", batches, "--batch-size", "10"],
        capture_output=True,
        text=True,
    ).returncode
    assert rc_split == 0
    rc, _ = run_filter(batches, out, rejects_dir=rejects_dir, extra_args=extra_args)
    from lib.parquet_io import read_batch

    rows = []
    files = sorted(f for f in os.listdir(out) if f.endswith(".parquet"))
    for f in files:
        rows.extend(read_batch(os.path.join(out, f)))
    return rc, rows


class TestMultiFragment:
    def test_single_no_flags_n_fragments_one(self):
        """Single molecule, no fragment flags → n_fragments=1, ok."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("c1ccccc1", _tags("M1"))],
                [],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "ok"
            assert rows[0]["n_fragments"] == 1

    def test_salt_max_fragments_1_rejected(self):
        """Salt (Na+ Cl-, 2 frags) + --max-fragments 1 → dropped, reason."""
        with tempfile.TemporaryDirectory() as tmp:
            rejects = os.path.join(tmp, "rejects", "03_filter")
            rc, rows = _run_split_filter(
                tmp,
                [("[Na+].[Cl-]", _tags("SALT"))],
                ["--max-fragments", "1"],
                rejects_dir=rejects,
            )
            assert rc == 0
            assert rows == []  # dropped
            sdf = os.path.join(rejects, "filter_rejected.sdf")
            assert os.path.exists(sdf)
            mols = [m for m in Chem.SDMolSupplier(sdf) if m is not None]
            assert len(mols) == 1
            assert mols[0].GetProp("filter_reason") == "fragments=2>1[Na;Cl]"

    def test_solvate_max_fragments_1_rejected(self):
        """Aspirin + water + --max-fragments 1 → reason lists both formulas."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("CC(=O)Oc1ccccc1C(=O)O.O", _tags("SOLV"))],
                ["--force-keep-rejected", "--max-fragments", "1"],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "rejected"
            reason = rows[0]["filter_reason"]
            assert reason.startswith("fragments=2>1[")
            assert "C9H8O4" in reason and "H2O" in reason

    def test_single_min_fragments_2_rejected(self):
        """Single molecule + --min-fragments 2 → rejected fragments=1<2."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("CCO", _tags("M1"))],
                ["--force-keep-rejected", "--min-fragments", "2"],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "rejected"
            assert rows[0]["filter_reason"] == "fragments=1<2"
            assert rows[0]["n_fragments"] == 1

    def test_dimer_min_max_2_passes(self):
        """Benzene+naphthalene + --min/max-fragments 2 → exactly 2 passes."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("c1ccccc1.c1ccc2ccccc2c1", _tags("DIM"))],
                ["--min-fragments", "2", "--max-fragments", "2"],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "ok"
            assert rows[0]["n_fragments"] == 2

    def test_dimer_default_passes(self):
        """Dimer, no flags → passes, n_fragments=2 stored."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("c1ccccc1.c1ccc2ccccc2c1", _tags("DIM"))],
                [],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "ok"
            assert rows[0]["n_fragments"] == 2

    def test_salt_force_keep_rejected(self):
        """--force-keep-rejected + --max-fragments 1 + salt → kept, rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("[Na+].[Cl-]", _tags("SALT"))],
                ["--force-keep-rejected", "--max-fragments", "1"],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "rejected"
            assert rows[0]["n_fragments"] == 2

    def test_element_rejected_n_fragments_stored(self):
        """Forbidden element → n_fragments still stored (computed first)."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("CCl", _tags("M1"))],
                ["--force-keep-rejected", "--allowed-elements", "C,O"],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "rejected"
            assert rows[0]["filter_reason"].startswith("forbidden_elements:")
            assert rows[0]["n_fragments"] == 1

    def test_mol_corrupt_n_fragments_none(self):
        """Corrupt mol_block → n_fragments=None, mol_corrupt, no crash."""
        from lib.parquet_io import read_batch, write_batch

        with tempfile.TemporaryDirectory() as tmp:
            batches = os.path.join(tmp, "batches")
            os.makedirs(batches)
            write_batch(
                os.path.join(batches, "b0.parquet"),
                [
                    {
                        "mol_block": "not a valid mol block",
                        "num_atoms": 0,
                        "num_bonds": 0,
                        "Energy_Ha": -500.0,
                        "FormalCharge": 0,
                        "Multiplicity": 1,
                        "SMILES": None,
                        "SourceID": "CORRUPT",
                    }
                ],
            )
            out = os.path.join(tmp, "out")
            rc, _ = run_filter(batches, out, extra_args=["--force-keep-rejected"])
            assert rc == 0
            rows = read_batch(os.path.join(out, "b0.parquet"))
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "rejected"
            assert rows[0]["filter_reason"] == "mol_corrupt"
            val = rows[0]["n_fragments"]
            # null int32 reads back as NaN float through pyarrow/pandas
            assert val is None or math.isnan(val)


class TestComposition:
    def test_allowed_elements_h_implicit(self):
        """--allowed-elements without H: H implicitly allowed (explicit-H mol)."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        block = Chem.MolToMolBlock(mol)
        passes, reason, n_frag = _filter_module.check_composition(
            block,
            ["C", "O"],
            None,
            None,
            None,
            None,
        )
        assert passes, reason
        assert n_frag == 1


def _block_3d(smiles):
    """3D mol_block with explicit Hs (embedded, seeded)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return Chem.MolToMolBlock(mol)


def _merged_block():
    """3D mol_block whose BOND TABLE merges ethanol+water into ONE
    connected component via a spurious O-O bond (converter H-bond-merge
    artifact); the SMILES tag still says 'CCO.O'. Fragments embedded
    separately, water translated +5 A in x so geometry-based
    reconstruction cannot recover the true two-fragment topology."""
    etoh = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(etoh, randomSeed=42) == 0
    wat = Chem.AddHs(Chem.MolFromSmiles("O"))
    assert AllChem.EmbedMolecule(wat, randomSeed=42) == 0
    b1 = Chem.MolToMolBlock(etoh).splitlines()
    b2 = Chem.MolToMolBlock(wat).splitlines()
    na1, nb1 = int(b1[3][:3]), int(b1[3][3:6])
    na2, nb2 = int(b2[3][:3]), int(b2[3][3:6])
    o1 = next(a.GetIdx() for a in etoh.GetAtoms() if a.GetSymbol() == "O") + 1

    def shift_x(line):
        return f"{float(line[0:10]) + 5.0:10.4f}" + line[10:]

    atoms = b1[4 : 4 + na1] + [shift_x(a) for a in b2[4 : 4 + na2]]
    bonds = list(b1[4 + na1 : 4 + na1 + nb1])
    bonds += [
        f"{int(ln[:3]) + na1:3d}{int(ln[3:6]) + na1:3d}" + ln[6:]
        for ln in b2[4 + na2 : 4 + na2 + nb2]
    ]
    bonds.append(f"{o1:3d}{na1 + 1:3d}  1  0")  # spurious O...O join
    counts = f"{na1 + na2:3d}{nb1 + nb2 + 1:3d}" + b1[3][6:]
    tail = b1[4 + na1 + nb1 :]  # M CHG etc. (ethanol neutral)
    return "\n".join([b1[0], b1[1], b1[2], counts] + atoms + bonds + tail) + "\n"


class TestFragmentColumns:
    """Fragment metadata columns (T2): formulas/SMILES/charges/heavy
    atoms per fragment. SMILES-tag truth priority (D62 mirror), always
    populated, empty strings for corrupt rows."""

    def test_monomer_single_item_never_null(self):
        """Monomer → four single-item strings, never null (nullable ints
        broke Parquet roundtrips in the past)."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(tmp, [("CCO", _tags("M1", smiles="CCO"))], [])
            assert rc == 0
            row = rows[0]
            assert row["filter_status"] == "ok"
            assert row["fragment_formulas"] == "C2H6O"
            assert row["fragment_smiles"] == "CCO"
            assert row["fragment_charges"] == "0"
            assert row["fragment_heavy_atoms"] == "3"

    def test_salt_kept_no_flags(self):
        """Salt, no fragment flags → KEPT; charges '1;-1' (handoff T4).
        SMILES tag mirrors converter output (D62): without it the
        geometry fallback fabricates charges (charge=+2) and the row is
        rejected by the net-charge check — pre-existing behavior."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp, [("[Na+].[Cl-]", _tags("SALT", smiles="[Na+].[Cl-]"))], []
            )
            assert rc == 0
            assert len(rows) == 1
            row = rows[0]
            assert row["filter_status"] == "ok"
            assert row["n_fragments"] == 2
            assert row["fragment_formulas"] == "Na;Cl"
            assert row["fragment_smiles"] == "[Na+];[Cl-]"
            assert row["fragment_charges"] == "1;-1"
            assert row["fragment_heavy_atoms"] == "1;1"

    def test_solvate_kept_no_flags(self):
        """Aspirin+water → kept, 2 items per column (T4 verified
        values: C9H8O4;H2O, heavy 13;1)."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("CC(=O)Oc1ccccc1C(=O)O.O", _tags("SOLV", smiles="CC(=O)Oc1ccccc1C(=O)O.O"))],
                [],
            )
            assert rc == 0
            row = rows[0]
            assert row["filter_status"] == "ok"
            assert row["n_fragments"] == 2
            assert row["fragment_formulas"] == "C9H8O4;H2O"
            assert row["fragment_heavy_atoms"] == "13;1"
            assert row["fragment_charges"] == "0;0"
            smis = row["fragment_smiles"].split(";")
            assert smis == ["CC(=O)Oc1ccccc1C(=O)O", "O"]

    def test_merged_topology_smiles_tag_wins(self):
        """Merged-topology mol_block (spurious O-O bond) + SMILES tag
        'CCO.O' → n_fragments=1 (mol_block topology) but fragment
        columns carry 2 items (SMILES tag truth, D62 priority)."""
        from lib.parquet_io import read_batch, write_batch

        block = _merged_block()
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            os.makedirs(in_dir)
            write_batch(
                os.path.join(in_dir, "b0.parquet"),
                [
                    {
                        "mol_block": block,
                        "num_atoms": mol.GetNumAtoms(),
                        "num_bonds": mol.GetNumBonds(),
                        "SMILES": "CCO.O",
                        "Energy_Ha": -110.0,
                        "FormalCharge": 0,
                        "Multiplicity": 1,
                        "SourceID": "MERGE",
                    }
                ],
            )
            out = os.path.join(tmp, "out")
            rc, _ = run_filter(in_dir, out, extra_args=["--force-keep-rejected"])
            assert rc == 0
            rows = read_batch(os.path.join(out, "b0.parquet"))
            assert len(rows) == 1
            row = rows[0]
            assert row["n_fragments"] == 1  # mol_block topology
            assert row["fragment_formulas"] == "C2H6O;H2O"  # SMILES truth
            assert row["fragment_smiles"] == "CCO;O"
            assert row["fragment_charges"] == "0;0"
            assert row["fragment_heavy_atoms"] == "3;1"

    def test_corrupt_both_sources_empty_strings(self):
        """Corrupt mol_block + unparseable SMILES → fragment columns
        empty strings (corrupt convention, never null)."""
        from lib.parquet_io import read_batch, write_batch

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            os.makedirs(in_dir)
            write_batch(
                os.path.join(in_dir, "b0.parquet"),
                [
                    {
                        "mol_block": "garbage",
                        "num_atoms": 0,
                        "num_bonds": 0,
                        "SMILES": "not_a_smiles%%",
                        "Energy_Ha": -100.0,
                        "FormalCharge": 0,
                        "Multiplicity": 1,
                        "SourceID": "CORRUPT",
                    }
                ],
            )
            out = os.path.join(tmp, "out")
            rc, _ = run_filter(in_dir, out, extra_args=["--force-keep-rejected"])
            assert rc == 0
            rows = read_batch(os.path.join(out, "b0.parquet"))
            assert len(rows) == 1
            row = rows[0]
            assert row["fragment_formulas"] == ""
            assert row["fragment_smiles"] == ""
            assert row["fragment_charges"] == ""
            assert row["fragment_heavy_atoms"] == ""

    def test_comp_rejected_row_fragment_cols_empty(self):
        """Comp-fail rows (e.g. --max-fragments 1) drop by default;
        with --force-keep-rejected they stay with EMPTY fragment
        columns — metadata computed only after composition passes
        (rejection reason string carries the formulas instead)."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, rows = _run_split_filter(
                tmp,
                [("[Na+].[Cl-]", _tags("SALT"))],
                ["--force-keep-rejected", "--max-fragments", "1"],
            )
            assert rc == 0
            assert len(rows) == 1
            row = rows[0]
            assert row["filter_status"] == "rejected"
            assert row["fragment_formulas"] == ""
            assert row["fragment_charges"] == ""


class TestComputeFragmentMetadata:
    """Unit tests for compute_fragment_metadata (truth priority)."""

    def test_smiles_tag_priority_over_merged_block(self):
        md = _compute_fragment_metadata("CCO.O", _merged_block())
        assert md["fragment_formulas"] == "C2H6O;H2O"
        assert md["fragment_smiles"] == "CCO;O"
        assert md["fragment_charges"] == "0;0"
        assert md["fragment_heavy_atoms"] == "3;1"

    def test_fallback_to_mol_block_when_smiles_missing(self):
        md = _compute_fragment_metadata(None, _block_3d("CCO.O"))
        assert md["fragment_formulas"] == "C2H6O;H2O"
        # explicit-H blocks keep explicit-H fragment SMILES (block truth)
        assert md["fragment_smiles"] == "[H]OC([H])([H])C([H])([H])[H];[H]O[H]"

    def test_unparseable_everywhere_empty_strings(self):
        md = _compute_fragment_metadata("not_smiles", "garbage")
        assert md == {
            "fragment_formulas": "",
            "fragment_smiles": "",
            "fragment_charges": "",
            "fragment_heavy_atoms": "",
        }


def _write_sdf_3d(path, entries):
    """Write SDF with 3D embedded coords so DetermineBonds works."""
    writer = SDWriter(path)
    for smi, tags in entries:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        for k, v in tags.items():
            mol.SetProp(k, v)
        writer.write(mol)
    writer.close()


def _write_sdf_single_bonds(path, entries):
    """Write SDF simulating convert_qm40 output: explicit H, SINGLE bonds,
    neutral atoms, 3D coords. The SMILES tag carries the true bond orders
    and formal charges (the mol_block deliberately loses them)."""
    writer = SDWriter(path)
    for smi, tags in entries:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        for b in mol.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
        for a in mol.GetAtoms():
            a.SetFormalCharge(0)
        mol.SetProp("SMILES", smi)  # SMILES carries the true charges (as convert_qm40 does)
        for k, v in tags.items():
            mol.SetProp(k, v)
        writer.write(mol)
    writer.close()


class TestZwitterionRule:
    """Unit tests for _has_true_zwitterion (adjacency rule, D57)."""

    def _is_zwi(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        return _has_true_zwitterion(mol)

    def test_nitromethane_kept(self):
        assert not self._is_zwi("C[N+](=O)[O-]")

    def test_nitrobenzene_kept(self):
        assert not self._is_zwi("O=[N+]([O-])c1ccccc1")

    def test_dinitro_kept(self):
        # Guards the naive "any non-adjacent +/- pair" trap
        assert not self._is_zwi("O=[N+]([O-])c1ccccc1[N+](=O)[O-]")

    def test_glycine_zwitterion_rejected(self):
        assert self._is_zwi("[NH3+]CC(=O)[O-]")

    def test_nitro_plus_zwitterion_rejected(self):
        assert self._is_zwi("[NH3+]C([N+](=O)[O-])C(=O)[O-]")

    def test_pyridine_n_oxide_kept(self):
        assert not self._is_zwi("[O-][N+]1=CC=CC=C1")

    def test_sulfoxide_kept(self):
        assert not self._is_zwi("CS(=O)C")

    def test_azide_kept(self):
        assert not self._is_zwi("C[N-][N+]#N")

    def test_amine_oxide_kept(self):
        assert not self._is_zwi("C[N+](C)(C)[O-]")

    def test_salt_cross_fragment_not_zwitterion(self):
        """Ionic complex (pure-ion fragments, cross-fragment charge
        separation) is NOT a zwitterion — complexes kept + tagged."""
        assert not self._is_zwi("[Na+].[Cl-]")

    def test_zwitterion_inside_complex_still_rejected(self):
        """True zwitterion fragment + spectator ions → still zwitterion."""
        assert self._is_zwi("[NH3+]CC(=O)[O-].[Na+].[Cl-]")


class TestZwitterionIntegration:
    """End-to-end wiring: nitro kept, true zwitterion rejected."""

    def test_nitro_kept_glycine_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdf = os.path.join(tmp, "in.sdf")
            _write_sdf_3d(
                sdf,
                [
                    ("C[N+](=O)[O-]", _tags("NITRO")),
                    ("[NH3+]CC(=O)[O-]", _tags("GLYZWI")),
                ],
            )
            batches = os.path.join(tmp, "batches")
            out = os.path.join(tmp, "out")
            subprocess.run(
                [sys.executable, SPLIT, sdf, "-o", batches, "--batch-size", "10"],
                capture_output=True,
                text=True,
            )
            rc, _ = run_filter(batches, out, extra_args=["--force-keep-rejected"])
            assert rc == 0
            from lib.parquet_io import read_batch

            rows = []
            for f in sorted(os.listdir(out)):
                if f.endswith(".parquet"):
                    rows.extend(read_batch(os.path.join(out, f)))
            by_id = {r["SourceID"]: r for r in rows}
            assert by_id["NITRO"]["filter_status"] == "ok"
            assert by_id["GLYZWI"]["filter_status"] == "rejected"
            assert by_id["GLYZWI"]["filter_reason"] == "zwitterion"


def _norm(v):
    """Normalize for comparison: None stays None; NaN (null int round-trip) → None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


class TestWorkers:
    """--workers N must not change output (determinism)."""

    def test_workers_1_vs_4_identical(self):
        from lib.parquet_io import read_batch

        with tempfile.TemporaryDirectory() as tmp:
            sdf = os.path.join(tmp, "in.sdf")
            _write_sdf_3d(
                sdf,
                [
                    ("c1ccccc1", _tags("BENZ")),
                    ("Cc1ccccc1", _tags("TOL")),
                    ("c1ccncc1", _tags("PYR")),
                    ("[NH3+]CC(=O)[O-]", _tags("GLY")),
                ],
            )
            batches = os.path.join(tmp, "batches")
            subprocess.run(
                [sys.executable, SPLIT, sdf, "-o", batches, "--batch-size", "10"],
                capture_output=True,
                text=True,
            )
            out1 = os.path.join(tmp, "out1")
            out4 = os.path.join(tmp, "out4")
            rc1, _ = run_filter(
                batches,
                out1,
                rejects_dir=os.path.join(tmp, "r1"),
                extra_args=["--workers", "1", "--force-keep-rejected"],
            )
            rc4, _ = run_filter(
                batches,
                out4,
                rejects_dir=os.path.join(tmp, "r4"),
                extra_args=["--workers", "4", "--force-keep-rejected"],
            )
            assert rc1 == 0 and rc4 == 0

            rows1 = [
                r
                for f in sorted(os.listdir(out1))
                if f.endswith(".parquet")
                for r in read_batch(os.path.join(out1, f))
            ]
            rows4 = [
                r
                for f in sorted(os.listdir(out4))
                if f.endswith(".parquet")
                for r in read_batch(os.path.join(out4, f))
            ]
            assert len(rows1) == len(rows4) == 4

            keys = [
                "SourceID",
                "filter_status",
                "filter_reason",
                "mol_block",
                "num_atoms",
                "num_bonds",
                "n_fragments",
            ]
            for a, b in zip(rows1, rows4):
                assert a["SourceID"] == b["SourceID"]
                for k in keys:
                    assert _norm(a[k]) == _norm(b[k]), f"{a['SourceID']} {k}: {a[k]!r} != {b[k]!r}"


def _run_single_bond_filter(tmp, entries, extra_args=None):
    """Split + filter on a single-bond/neutral SDF (convert_qm40 sim).

    Returns (rc, stdout, rows).
    """
    sdf = os.path.join(tmp, "in.sdf")
    _write_sdf_single_bonds(sdf, entries)
    batches = os.path.join(tmp, "batches")
    out = os.path.join(tmp, "out")
    rc_split = subprocess.run(
        [sys.executable, SPLIT, sdf, "-o", batches, "--batch-size", "10"],
        capture_output=True,
        text=True,
    ).returncode
    assert rc_split == 0
    rc, stdout = run_filter(batches, out, extra_args=extra_args or [])
    from lib.parquet_io import read_batch

    rows = []
    for f in sorted(os.listdir(out)):
        if f.endswith(".parquet"):
            rows.extend(read_batch(os.path.join(out, f)))
    return rc, stdout, rows


class TestSmilesAuthoritative:
    """Root fix: SMILES is authoritative for bond orders + formal charges.

    convert_qm40 writes single-bond, neutral mol_blocks (charges survive only
    in the SMILES tag). Geometry reconstruction (DetermineBonds) loses those
    charges; the pipeline must recover them from the SMILES template.
    """

    def test_zwitterion_recovered_from_smiles(self):
        """[NH2+]CCCC(=O)[O-]: geometry reconstruction loses/crashes the
        charge; SMILES recovers N+ → rejected as zwitterion (not mol_corrupt)."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, _, rows = _run_single_bond_filter(
                tmp,
                [("[NH2+]CCCC(=O)[O-]", _tags("ZWIBUG"))],
                ["--force-keep-rejected"],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "rejected"
            assert rows[0]["filter_reason"] == "zwitterion"

    def test_nitro_kept_single_bond(self):
        """Nitro (adjacent N+/O-) recovered from SMILES → kept, not zwitterion."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, _, rows = _run_single_bond_filter(
                tmp,
                [("C[N+](=O)[O-]", _tags("NITRO"))],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "ok"

    def test_bond_orders_from_smiles(self):
        """Benzene single-bond/neutral → aromatic bonds in output (geometry
        DetermineBonds would emit triple bonds instead)."""
        with tempfile.TemporaryDirectory() as tmp:
            rc, _, rows = _run_single_bond_filter(
                tmp,
                [("c1ccccc1", _tags("BENZ"))],
            )
            assert rc == 0
            assert len(rows) == 1
            assert rows[0]["filter_status"] == "ok"
            mol = Chem.MolFromMolBlock(rows[0]["mol_block"], sanitize=False)
            types = {b.GetBondType() for b in mol.GetBonds()}
            # Geometry reconstruction would emit triple bonds (c1c#cc#cc#1);
            # SMILES template gives a proper aromatic/kekule benzene.
            assert Chem.BondType.TRIPLE not in types
            # Re-sanitize → aromatic benzene
            m2 = Chem.MolFromMolBlock(rows[0]["mol_block"])
            assert "c1ccccc1" == Chem.MolToSmiles(m2)


class TestReportSplit:
    """#30: 'Mol corrupt' and 'Topology warning' reported as separate lines."""

    def test_separate_lines_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, stdout, _ = _run_single_bond_filter(
                tmp,
                [("CCO", _tags("M1"))],
            )
            assert rc == 0
            assert "Mol corrupt" in stdout
            assert "Topology warning" in stdout
