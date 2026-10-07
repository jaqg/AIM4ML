"""
conftest.py — Shared test fixtures for the AIM4ML curation pipeline.
"""

import os
import sys
import tempfile

import pytest

# Make repo-root lib/ importable from the tests/ directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rdkit import Chem
from rdkit.Chem import SDWriter

# -----------------------------------------------------------------------
# Helper: write a small SDF from a list of (tags_dict) dicts
# -----------------------------------------------------------------------


def _make_sdf(path, molecules):
    """
    molecules: list of dicts, each with keys:
        'tags'  — dict of SDF property tags
        'smiles' (optional) — SMILES string for the molecule
    """
    writer = SDWriter(path)
    for entry in molecules:
        tags = entry.get("tags", {})
        smi = entry.get("smiles", "C")  # methane by default
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp("_Name", tags.get("SourceID", f"mol_{len(tags)}"))
        for k, v in tags.items():
            mol.SetProp(k, v)
        writer.write(mol)
    writer.close()


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def valid_sdf():
    """SDF with 3 perfectly valid molecules (all tags present)."""
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        path = f.name
    base_tags = {
        "Energy_Ha": "-500.0",
        "FormalCharge": "0",
        "Multiplicity": "1",
        "SMILES": "CCO",
        "SourceID": "MOL01",
        "HOMO_Ha": "-0.23",
        "LUMO_Ha": "-0.02",
        "HL_Gap_Ha": "0.21",
        "PartialCharges": "0.0 0.0",
    }
    _make_sdf(
        path,
        [
            {"tags": base_tags, "smiles": "CCO"},
            {"tags": {**base_tags, "SourceID": "MOL02", "Energy_Ha": "-600.0"}, "smiles": "CCN"},
            {"tags": {**base_tags, "SourceID": "MOL03", "Energy_Ha": "-700.0"}, "smiles": "CCCO"},
        ],
    )
    yield path
    os.unlink(path)


@pytest.fixture
def broken_missing_required():
    """SDF with 2 molecules: one missing Energy_Ha, one missing Multiplicity."""
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        path = f.name
    _make_sdf(
        path,
        [
            {
                "tags": {
                    "FormalCharge": "0",
                    "Multiplicity": "1",
                    "SMILES": "CC",
                    "SourceID": "MOL01",
                }
            },  # missing Energy_Ha
            {
                "tags": {"Energy_Ha": "-500.0", "FormalCharge": "0"},
                "smiles": "CCO",
            },  # missing Multiplicity
        ],
    )
    yield path
    os.unlink(path)


@pytest.fixture
def broken_bad_types():
    """SDF with 2 molecules: bad Energy_Ha type, bad Multiplicity type."""
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        path = f.name
    _make_sdf(
        path,
        [
            {
                "tags": {
                    "Energy_Ha": "not_a_number",
                    "FormalCharge": "0",
                    "Multiplicity": "1",
                    "SMILES": "CC",
                    "SourceID": "MOL01",
                }
            },
            {
                "tags": {
                    "Energy_Ha": "-500.0",
                    "FormalCharge": "0",
                    "Multiplicity": "two",
                    "SMILES": "CCO",
                    "SourceID": "MOL02",
                }
            },
        ],
    )
    yield path
    os.unlink(path)


@pytest.fixture
def broken_mult_zero():
    """SDF with a molecule where Multiplicity = 0 (must be > 0)."""
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        path = f.name
    _make_sdf(
        path,
        [
            {
                "tags": {
                    "Energy_Ha": "-500.0",
                    "FormalCharge": "0",
                    "Multiplicity": "0",
                    "SMILES": "CC",
                    "SourceID": "MOL01",
                }
            },
        ],
    )
    yield path
    os.unlink(path)


@pytest.fixture
def missing_recommended():
    """SDF with 1 molecule missing SMILES and SourceID (recommended)."""
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        path = f.name
    _make_sdf(
        path,
        [
            {
                "tags": {
                    "Energy_Ha": "-500.0",
                    "FormalCharge": "0",
                    "Multiplicity": "1",
                    "HOMO_Ha": "-0.23",
                    "LUMO_Ha": "-0.02",
                    "HL_Gap_Ha": "0.21",
                    "PartialCharges": "0.0",
                }
            },
        ],
    )
    yield path
    os.unlink(path)


@pytest.fixture
def missing_optional():
    """SDF with 1 molecule missing optional tags (HOMO, LUMO, etc.)."""
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        path = f.name
    _make_sdf(
        path,
        [
            {
                "tags": {
                    "Energy_Ha": "-500.0",
                    "FormalCharge": "0",
                    "Multiplicity": "1",
                    "SMILES": "CCO",
                    "SourceID": "MOL01",
                }
            },
        ],
    )
    yield path
    os.unlink(path)


# -----------------------------------------------------------------------
# Realspace xyz fixture texts (semantic cases, reviewable constants).
# Coordinates are %.6f-representable so xyz_io round-trips are value-exact.
# -----------------------------------------------------------------------

XYZ_HEH_EXTXYZ = """2
Properties=species:S:1:pos:R:3 charge=1 multiplicity=1
He 0.000000 0.000000 0.000000
H 0.772000 0.000000 0.000000
"""

XYZ_HEH_PERATOM_CHARGE = """2
Properties=species:S:1:pos:R:3:charge:R:1 charge=1 multiplicity=1
He 0.000000 0.000000 0.000000 0.300000
H 0.772000 0.000000 0.000000 -0.300000
"""

XYZ_HEH_PLAIN = """2
HeH+ cation
He 0.000000 0.000000 0.000000
H 0.772000 0.000000 0.000000
"""

XYZ_TWO_FRAMES = XYZ_HEH_EXTXYZ + "\n" + XYZ_HEH_PLAIN

XYZ_MALFORMED_EOF = """3
Properties=species:S:1:pos:R:3
He 0.000000 0.000000 0.000000
H 0.772000 0.000000 0.000000
"""

XYZ_LI4_EXTXYZ = """4
Properties=species:S:1:pos:R:3 charge=0 multiplicity=1
Li 0.000000 0.000000 0.000000
Li 3.000000 0.000000 0.000000
Li 1.000000 2.500000 0.300000
Li 0.800000 0.600000 2.700000
"""

XYZ_B2H6_EXTXYZ = """8
Properties=species:S:1:pos:R:3 charge=0 multiplicity=1
B 0.000000 0.000000 1.190000
B 0.000000 0.000000 -1.190000
H 0.000000 1.200000 0.000000
H 0.000000 -1.200000 0.000000
H 1.030000 0.000000 1.830000
H -1.030000 0.000000 1.830000
H 1.030000 0.000000 -1.830000
H -1.030000 0.000000 -1.830000
"""

XYZ_BAD_CHARGE_HEADER = """2
Properties=species:S:1:pos:R:3 charge=notanumber multiplicity=1
He 0.000000 0.000000 0.000000
H 0.772000 0.000000 0.000000
"""


def write_xyz_text(tmp_path, text, name="input.xyz"):
    """Write xyz fixture text to tmp_path/name; returns the Path."""
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture
def empty_sdf():
    """Creates an empty SDF (actually, SDWriter with 0 mols — RDKit may not
    write at all. Return path to a non-existent file for integration testing."""
    # SDF with 0 molecules: no molecules written → no file or empty file.
    # Use a file with just a header but no mol blocks.
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        pass  # empty file
    yield f.name
    os.unlink(f.name)
