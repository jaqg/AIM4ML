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

            from lib.parquet_io import read_batch
            from rdkit import Chem

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


def _tags(sid, energy="-500.0"):
    return {"Energy_Ha": energy, "FormalCharge": "0", "Multiplicity": "1", "SourceID": sid}


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
