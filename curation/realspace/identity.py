"""
identity.py — Realspace-track identity semantics (geometry_key, RMSD).

geometry_key = MD5 of a canonical serialization of the registry identity:
element symbols + coordinates rounded to 1e-3 A + total charge +
multiplicity. Two frames share a geometry_key iff they are (to registry
precision) the same geometry with the same electronic state — the
"bit-identical registry entry" the generic stages dedup on.

Identity definition (acceptance criterion 2 — the serialization IS the
identity; changing any detail below changes every geometry_key):

1. Validate: symbols standard-case element symbols ("He", not "HE" or "he");
   coords (N, 3) float64, one symbol per row; unknown symbol or non-finite
   coordinate -> ValueError.
2. Round every coordinate to 1e-3 A (round() — banker's rounding on the
   binary float64, deterministic) and normalize -0.0 to +0.0.
3. Sort atoms by (Z, x, y, z) on the ROUNDED values.
4. Serialize each atom "Z:x:y:z" (%.3f fixed decimal — never float repr);
   atoms joined by ";".
5. Append "|charge|multiplicity" (coerced via int()).
6. geometry_key = MD5(payload.encode("utf-8")).hexdigest() — 32-char hex.

Example: HeH+ (He at origin, H at 0.772, charge=1, multiplicity=1)
serializes as "1:0.772:0.000:0.000;2:0.000:0.000:0.000|1|1" — Z=1 is H,
Z=2 is He, sorted by (Z, x, y, z) puts H (x=0.772) first.

Rounding boundary behavior (pinned in tests): offsets below 0.0005 A keep
the key; whether exactly ±0.0005 A flips the key depends on the binary
representation of the coordinate (CPython %.3f), and tests pin the exact
per-value behavior for the shared fixtures.

Inversion vs graph track (documented, not a bug): graph 04 keeps exact
dups (same SMILES = distinct conformers worth keeping); realspace drops
them (same geometry_key = re-delivery of the same registry entry).

This module owns identity + RMSD semantics only; column storage lives in
curation/realspace/schema.py.
"""

import hashlib

import numpy as np
from rdkit import Chem

_PT = Chem.GetPeriodicTable()


def _atomic_number(symbol: str) -> int:
    """Element symbol -> atomic number Z; unknown symbol -> ValueError."""
    try:
        return int(_PT.GetAtomicNumber(symbol))
    except Exception as exc:
        raise ValueError(f"unknown element symbol: {symbol!r}") from exc


def _round3(value: float) -> float:
    """Round a coordinate to 1e-3 A; normalize -0.0 to +0.0."""
    return round(value, 3) + 0.0


def geometry_key(symbols, coords, charge, multiplicity) -> str:
    """Canonical registry identity: MD5 of sorted (Z, rounded coords) + state.

    Parameters
    ----------
    symbols : list[str]
        Element symbols, one per atom (standard case, e.g. "He").
    coords : array-like, shape (N, 3)
        Cartesian coordinates in Angstrom.
    charge : int
        Total molecular charge (coerced via int()).
    multiplicity : int
        Spin multiplicity (coerced via int()).

    Returns
    -------
    str
        32-char MD5 hexdigest. See module docstring for the exact
        byte-level serialization — it IS the identity definition.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got shape {coords.shape}")
    if len(symbols) != coords.shape[0]:
        raise ValueError(f"{len(symbols)} symbols vs {coords.shape[0]} coord rows")
    if coords.size and not np.isfinite(coords).all():
        raise ValueError("coords contain non-finite values")
    atoms = [
        (_atomic_number(sym), _round3(float(x)), _round3(float(y)), _round3(float(zc)))
        for sym, (x, y, zc) in zip(symbols, coords)
    ]
    atoms.sort()  # (Z, x, y, z) on rounded values — deterministic order
    payload = ";".join(f"{zn}:{x:.3f}:{y:.3f}:{zc:.3f}" for zn, x, y, zc in atoms)
    payload += f"|{int(charge)}|{int(multiplicity)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
