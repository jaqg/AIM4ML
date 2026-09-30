"""
descriptors.py — RDKit descriptor computation for the selection pipeline.

Deterministic, pure functions (no I/O).  All descriptors are computed from a
mol_block → RDKit Mol (NOT re-parsed from SMILES — the explicit-H canonical
SMILES re-sanitizes ~0.5% of QM40 (nitro / N-oxide / quaternary-N valence),
whereas the curated mol_block carries the resolved topology).

Conventions
-----------
- Morgan fingerprint (diversity): ECFP4 (radius=2), 2048 bits, chirality ON.
  Stored as a sorted list of on-bit indices (compact; rebuild a float32
  matrix when a distance matrix is needed).
- Atom environments (coverage): unfolded 32-bit Morgan environment ids,
  radius=2, chirality OFF (topology-level).  One list per heavy atom.
  Exact identity (no 2048-bit folding collision) — required for the
  "exactly ONE carrier molecule" forced-pick logic.
- Scaffold: Bemis-Murcko ring scaffold.  Acyclic molecules → None.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput
from rdkit.Chem.Scaffolds import MurckoScaffold

RADIUS = 2
NBITS = 2048

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=NBITS,
    includeChirality=True,
)
_ENV_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=NBITS,
    includeChirality=False,
)

_BOND_ORDER = {
    Chem.BondType.SINGLE: "1",
    Chem.BondType.DOUBLE: "2",
    Chem.BondType.TRIPLE: "3",
    Chem.BondType.AROMATIC: "ar",
}


def parse_mol(mol_block):
    """Parse an SDF V2000 mol block → sanitized implicit-H mol, or None.

    Full sanitization is required for correct ECFP invariants (ring
    membership, aromaticity, H counts).  Molecules that cannot fully
    sanitize (e.g. neutral N with valence 4) return None and are logged
    by the caller as ``n_skipped``.
    """
    if not mol_block:
        return None
    try:
        mol = Chem.MolFromMolBlock(mol_block, sanitize=True, removeHs=True)
        return mol if mol is not None and mol.GetNumAtoms() > 0 else None
    except Exception:
        return None


def morgan_onbits(mol, nbits=NBITS):
    """ECFP4 fingerprint as a sorted list of on-bit indices."""
    fp = _MORGAN_GEN.GetFingerprintAsNumPy(mol)
    return [int(i) for i in np.flatnonzero(fp)]


def atom_envs(mol):
    """Unfolded Morgan env ids per heavy atom (radius=2, chirality OFF).

    Returns list of sorted env-id tuples; index == atom index.
    """
    ao = AdditionalOutput()
    ao.AllocateAtomToBits()
    _ENV_GEN.GetSparseCountFingerprint(mol, additionalOutput=ao)
    atb = ao.GetAtomToBits()  # tuple of tuples; index == atom index
    return [sorted(t) for t in atb]


def heavy_elements(mol):
    """Element symbol per heavy atom."""
    return [a.GetSymbol() for a in mol.GetAtoms()]


def scaffold_smiles(mol):
    """Bemis-Murcko ring scaffold SMILES; None for acyclic molecules."""
    s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return s if s else None


def bond_strings(mol):
    """Per-bond 'A-B:order' strings (sorted element pair).

    order ∈ {1, 2, 3, ar}.  Sorted list; e.g. ['C-C:1', 'C-O:2', 'C-O:ar'].
    """
    out = []
    for b in mol.GetBonds():
        a = b.GetBeginAtom().GetSymbol()
        c = b.GetEndAtom().GetSymbol()
        pair = tuple(sorted((a, c)))
        order = _BOND_ORDER.get(b.GetBondType(), str(b.GetBondType()))
        out.append(f"{pair[0]}-{pair[1]}:{order}")
    return sorted(out)


def molwt_tpsa(mol):
    """(MolWt, TPSA) from a sanitized implicit-H mol."""
    return Descriptors.MolWt(mol), rdMolDescriptors.CalcTPSA(mol)
