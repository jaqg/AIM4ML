"""test_08_conformer_filter.py — TDD tests for conformer deduplication.

Verifies pipeline removes duplicate conformers (same geometry, artifacts)
while keeping genuinely different conformers (MaxMin clustering).
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "08_conformer_filter.py",
)
if os.path.dirname(_SCRIPT) not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCRIPT))

import importlib

from rdkit import Chem

from curation.graph.schema import compound_id
from lib.parquet_io import read_batch, write_batch

_conf = importlib.import_module("08_conformer_filter")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_mol_block(smiles, coords, energy_ha, source_id):
    """Create a valid SDF V2000 mol_block with 3D coordinates.

    coords: list of (x, y, z) tuples, one per atom (including H).
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf)
    mol.SetProp("_Name", source_id)
    mol.SetProp("Energy_Ha", str(energy_ha))
    mol.SetProp("FormalCharge", "0")
    mol.SetProp("Multiplicity", "1")
    mol.SetProp("SMILES", smiles)
    mol.SetProp("SourceID", source_id)
    return Chem.MolToMolBlock(mol, includeStereo=False)


def _make_batch(path, compounds):
    """Write a Parquet batch from a list of compound dicts.

    compounds: list of dicts with keys:
        'smiles'     — canonical SMILES
        'compound_id' — pre-computed CompoundID
        'conformers'  — list of {'coords': [...], 'energy': float, 'source_id': str}
    """
    rows = []
    for comp in compounds:
        for conf in comp["conformers"]:
            mol_block = _make_mol_block(
                comp["smiles"], conf["coords"], conf["energy"], conf["source_id"]
            )
            mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
            rows.append(
                {
                    "mol_block": mol_block,
                    "num_atoms": mol.GetNumAtoms(),
                    "num_bonds": mol.GetNumBonds(),
                    "Energy_Ha": conf["energy"],
                    "FormalCharge": 0,
                    "Multiplicity": 1,
                    "CompoundID": comp["compound_id"],
                    "SMILES": comp["smiles"],
                    "SourceID": conf["source_id"],
                }
            )
    write_batch(path, rows)


def _run_filter(input_dir, output_dir, threshold=1.0, rejects_dir=None, workers=1):
    """Run stage 8 via subprocess, return (rc, stdout)."""
    cmd = [
        sys.executable,
        _SCRIPT,
        "-i",
        input_dir,
        "-o",
        output_dir,
        "--rmsd-threshold",
        str(threshold),
        "--workers",
        str(workers),
    ]
    if rejects_dir:
        cmd.extend(["--rejects-dir", rejects_dir])
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

# Ethane (C2H6) fixtures — 2 heavy atoms needed for meaningful heavy-atom RMSD.
# Methane would give RMSD=0 for any internal deformation (only 1 heavy atom).

# Helper: 8-atom ethane coords (2 C + 6 H) — H positions arbitrary, don't affect test
_ETHANE_BASE = [
    (0.0, 0.0, 0.0),  # C1 (heavy)
    (1.0, 0.0, 0.0),  # C2 (heavy)
    (0.5, 1.0, 0.0),  # H
    (0.5, -1.0, 0.0),  # H
    (0.5, 0.0, 1.0),  # H
    (1.5, 1.0, 0.0),  # H
    (1.5, -1.0, 0.0),  # H
    (1.5, 0.0, 1.0),  # H
]

# C-C distance = 1.0 Å in base; moved to 4.0 Å in different variant.
# Heavy-atom RMSD after Kabsch = 1.5 Å > 1.0 threshold.


@pytest.fixture
def ethane_two_identical():
    """Ethane: two IDENTICAL conformers → one removed."""
    smiles = "CC"
    cid = compound_id(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True))
    return {
        "smiles": smiles,
        "compound_id": cid,
        "conformers": [
            {"coords": _ETHANE_BASE, "energy": -500.0, "source_id": "MOL_A_1"},
            {"coords": _ETHANE_BASE, "energy": -499.9, "source_id": "MOL_A_2"},
        ],
    }


@pytest.fixture
def ethane_two_different():
    """Ethane: two DIFFERENT conformers (C2 moved to 4.0 Å → RMSD=1.5 Å)."""
    smiles = "CC"
    cid = compound_id(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True))
    coords1 = list(_ETHANE_BASE)
    # C2 moved from (1.0,0,0) to (4.0,0,0); attached H atoms follow
    coords2 = [
        (0.0, 0.0, 0.0),  # C1 (same)
        (4.0, 0.0, 0.0),  # C2 (moved!)
        (0.5, 1.0, 0.0),  # H on C1 (same)
        (0.5, -1.0, 0.0),  # H on C1 (same)
        (0.5, 0.0, 1.0),  # H on C1 (same)
        (4.5, 1.0, 0.0),  # H on C2 (follows C2)
        (4.5, -1.0, 0.0),  # H on C2 (follows C2)
        (4.5, 0.0, 1.0),  # H on C2 (follows C2)
    ]
    return {
        "smiles": smiles,
        "compound_id": cid,
        "conformers": [
            {"coords": coords1, "energy": -500.0, "source_id": "MOL_B_1"},
            {"coords": coords2, "energy": -499.9, "source_id": "MOL_B_2"},
        ],
    }


@pytest.fixture
def ethane_single():
    """Ethane: single conformer."""
    smiles = "CC"
    cid = compound_id(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True))
    return {
        "smiles": smiles,
        "compound_id": cid,
        "conformers": [
            {"coords": _ETHANE_BASE, "energy": -500.0, "source_id": "MOL_C_1"},
        ],
    }


@pytest.fixture
def two_groups_multi():
    """Two molecules with 3 + 2 conformers (exercises parallel flatten + reconstruct)."""
    smiles1 = "CC"
    cid1 = compound_id(Chem.MolToSmiles(Chem.MolFromSmiles(smiles1), canonical=True))
    smiles2 = "C=C"
    cid2 = compound_id(Chem.MolToSmiles(Chem.MolFromSmiles(smiles2), canonical=True))

    def ethane_coords(c2x):
        # Only C1, C2 matter for heavy-atom RMSD; H arbitrary
        return [
            (0.0, 0.0, 0.0),
            (c2x, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.5, -1.0, 0.0),
            (0.5, 0.0, 1.0),
            (c2x + 0.5, 1.0, 0.0),
            (c2x + 0.5, -1.0, 0.0),
            (c2x + 0.5, 0.0, 1.0),
        ]

    def ethylene_coords(c2x):
        return [
            (0.0, 0.0, 0.0),
            (c2x, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (c2x, 1.0, 0.0),
            (c2x, -1.0, 0.0),
        ]

    return [
        {
            "smiles": smiles1,
            "compound_id": cid1,
            "conformers": [
                {"coords": ethane_coords(1.0), "energy": -500.0, "source_id": "A_1"},
                {"coords": ethane_coords(4.0), "energy": -499.9, "source_id": "A_2"},
                {"coords": ethane_coords(7.0), "energy": -499.8, "source_id": "A_3"},
            ],
        },
        {
            "smiles": smiles2,
            "compound_id": cid2,
            "conformers": [
                {"coords": ethylene_coords(1.3), "energy": -400.0, "source_id": "B_1"},
                {"coords": ethylene_coords(3.0), "energy": -399.9, "source_id": "B_2"},
            ],
        },
    ]


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestConformerFilter:
    def test_single_conformer_kept(self, ethane_single):
        """One conformer per CompoundID → always kept."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_single])

            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0
            assert "Kept:                 1" in stdout

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            assert batch[0]["conformer_status"] == "kept"
            assert batch[0]["cluster_id"] == 1

    def test_duplicate_conformers_removed(self, ethane_two_identical):
        """Two identical conformers → one removed, one kept."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_identical])

            # Test default (maxmin)
            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0
            assert "Kept:                 1" in stdout
            assert "Removed conformers:   1" in stdout

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 1
            kept = batch[0]
            assert kept["conformer_status"] == "kept"
            assert kept["SourceID"] == "MOL_A_1"  # lowest energy

    def test_different_conformers_kept(self, ethane_two_different):
        """Two conformers with RMSD=1.5 Å → both kept (above 1.0 Å threshold)."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_different])

            rc, stdout, stderr = _run_filter(in_dir, out_dir, threshold=1.0)
            assert rc == 0
            assert "Kept:                 2" in stdout
            assert "Removed conformers:   0" in stdout

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            statuses = {r["conformer_status"] for r in batch}
            assert statuses == {"kept"}

    def test_threshold_boundary(self, ethane_two_different):
        """RMSD=1.5 Å between conformers.
        threshold=0.5 → 1.5 >= 0.5 → different → both kept.
        threshold=1.5 → 1.5 < 1.5 is False → different → both kept.
        threshold=2.5 → 1.5 < 2.5 → same → one removed."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_different])

            # Both kept (below threshold)
            for thresh in [0.5, 1.5]:
                rc, stdout, stderr = _run_filter(in_dir, out_dir, threshold=thresh)
                assert rc == 0
                assert "Kept:                 2" in stdout, (
                    f"Expected both kept at threshold {thresh}"
                )
                assert "Removed conformers:   0" in stdout

            # One removed (within threshold)
            rc, stdout, stderr = _run_filter(in_dir, out_dir, threshold=2.5)
            assert rc == 0
            assert "Kept:                 1" in stdout
            assert "Removed conformers:   1" in stdout

    def test_invalid_threshold_rejected(self):
        """Threshold outside [0.1, 5.0] → script exits with error."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            # Need at least one batch so argparse succeeds before range check
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [])

            for bad_thresh in [0.05, 10.0, -1.0]:
                rc, stdout, stderr = _run_filter(in_dir, out_dir, threshold=bad_thresh)
                assert rc == 1, f"Expected exit 1 for threshold {bad_thresh}"
                assert "must be in range" in stdout or "must be in range" in stderr

    def test_reject_sdf_written(self, ethane_two_identical):
        """Removed conformers written to reject SDF."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            rejects_dir = os.path.join(tmp, "rejects")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_identical])

            rc, stdout, stderr = _run_filter(in_dir, out_dir, rejects_dir=rejects_dir)
            assert rc == 0

            reject_sdf = os.path.join(rejects_dir, "conformer_removed.sdf")
            assert os.path.exists(reject_sdf)

            # Should contain the removed conformer
            mols = list(Chem.SDMolSupplier(reject_sdf, sanitize=False, removeHs=False))
            assert len(mols) == 1

    def test_force_keep_rejected(self, ethane_two_identical):
        """--force-keep-rejected retains removed conformers in output."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_identical])

            cmd = [
                sys.executable,
                _SCRIPT,
                "-i",
                in_dir,
                "-o",
                out_dir,
                "--force-keep-rejected",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            statuses = {r["conformer_status"] for r in batch}
            assert statuses == {"kept", "removed_conformer"}

    def test_maxmin_keeps_lowest_energy_first(self, ethane_two_different):
        """MaxMin seeds with lowest-energy conformer (MOL_B_1 at -500.0)."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_different])

            rc, stdout, stderr = _run_filter(in_dir, out_dir)
            assert rc == 0

            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2
            kept_ids = {r["SourceID"] for r in batch if r["conformer_status"] == "kept"}
            assert "MOL_B_1" in kept_ids  # lowest energy, -500.0

    # -- New flag tests -------------------------------------------------

    def test_no_energy_aware(self, ethane_two_different):
        """--no-energy-aware keeps input order; MOL_B_2 may be seed."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_different])

            cmd = [
                sys.executable,
                _SCRIPT,
                "-i",
                in_dir,
                "-o",
                out_dir,
                "--no-energy-aware",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0

            # Both kept since RMSD=1.5 > 1.0 threshold
            batch = read_batch(os.path.join(out_dir, "batch_000.parquet"))
            assert len(batch) == 2

    def test_normalized_mode_runs(self, ethane_two_identical):
        """--rmsd-mode=normalized uses default threshold 0.15."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_identical])

            cmd = [
                sys.executable,
                _SCRIPT,
                "-i",
                in_dir,
                "-o",
                out_dir,
                "--rmsd-mode",
                "normalized",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0
            assert "mode: normalized" in result.stdout
            assert "Kept:                 1" in result.stdout
            assert "Removed conformers:   1" in result.stdout

    def test_normalized_threshold_range(self):
        """Normalized mode rejects threshold outside [0.05, 0.50]."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [])

            for bad in [0.01, 0.6]:
                cmd = [
                    sys.executable,
                    _SCRIPT,
                    "-i",
                    in_dir,
                    "-o",
                    out_dir,
                    "--rmsd-mode",
                    "normalized",
                    "--rmsd-threshold",
                    str(bad),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                assert result.returncode == 1, f"Expected fail for {bad}"
                assert "must be in range" in result.stdout or "must be in range" in result.stderr

    def test_coverage_guarantee(self, ethane_two_different):
        """--coverage-guarantee with threshold=2.5: RMSD=1.5 < 2.5,
        so first conformer covers second → only 1 kept."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_different])

            cmd = [
                sys.executable,
                _SCRIPT,
                "-i",
                in_dir,
                "-o",
                out_dir,
                "--rmsd-threshold",
                "2.5",
                "--coverage-guarantee",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0
            assert "coverage-guarantee" in result.stdout
            assert "Kept:                 1" in result.stdout
            assert "Removed conformers:   1" in result.stdout

    def test_coverage_guarantee_incomplete(self, ethane_two_different):
        """--coverage-guarantee with threshold=1.0: RMSD=1.5 > 1.0,
        first covers nothing → need second → both kept."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), [ethane_two_different])

            cmd = [
                sys.executable,
                _SCRIPT,
                "-i",
                in_dir,
                "-o",
                out_dir,
                "--rmsd-threshold",
                "1.0",
                "--coverage-guarantee",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0
            assert "Kept:                 2" in result.stdout
            assert "Removed conformers:   0" in result.stdout

    # -- Parallelism (determinism) --------------------------------------

    def test_workers_determinism(self, two_groups_multi):
        """workers=1 vs workers=4 → identical output rows (status + cluster_id)."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out1 = os.path.join(tmp, "out1")
            out4 = os.path.join(tmp, "out4")
            os.makedirs(in_dir)
            _make_batch(os.path.join(in_dir, "batch_000.parquet"), two_groups_multi)

            rc1, so1, se1 = _run_filter(in_dir, out1, threshold=2.0, workers=1)
            rc4, so4, se4 = _run_filter(in_dir, out4, threshold=2.0, workers=4)
            assert rc1 == 0, se1
            assert rc4 == 0, se4
            assert "Kept:                 3" in so1
            assert "Removed conformers:   2" in so1

            b1 = read_batch(os.path.join(out1, "batch_000.parquet"))
            b4 = read_batch(os.path.join(out4, "batch_000.parquet"))
            assert len(b1) == len(b4) == 3
            for r1, r4 in zip(b1, b4):
                assert r1["SourceID"] == r4["SourceID"]
                assert r1["conformer_status"] == r4["conformer_status"]
                assert r1["cluster_id"] == r4["cluster_id"]


class TestKabschCoords:
    """Unit tests for the coordinates-only RMSD refactor."""

    def test_kabsch_known_distance(self):
        """Ethane C-C 1.0 vs 4.0 Å → heavy-atom RMSD 1.5 Å."""
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        d = _conf._kabsch_rmsd(a, b)
        assert abs(d - 1.5) < 1e-9

    def test_extract_heavy_coords_excludes_h(self):
        """Extracts heavy atoms only; None for corrupt mol."""
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, (1.0, 0.0, 0.0))
        for i in range(2, mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (9.0, 9.0, 9.0))
        mol.AddConformer(conf)

        coords = _conf._extract_heavy_coords(mol)
        assert coords.shape == (2, 3)
        assert coords.dtype == np.float64
        np.testing.assert_allclose(coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert _conf._extract_heavy_coords(None) is None
