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

Conformer RMSD (near-dup heuristic — acceptance criterion 3)
------------------------------------------------------------
conformer_rmsd(a, b) returns the raw (not normalized) Kabsch-aligned RMSD
in Angstrom between two atom sets, or inf when their element multisets
differ (fast exit — different composition can never be a near-dup).
Pipeline:

1. Element-multiset check (Counter) -> inf on mismatch.
2. Canonical order per structure: sort by (Z, distance-profile, x, y, z).
   The distance profile (per element, sorted interatomic distances) makes
   the order invariant under rotation/translation/permutation, so rigid-
   motion copies pair atom-for-atom on the FIRST Kabsch already. A raw
   (Z, x, y, z) sort is NOT rotation-invariant and empirically left
   rotated copies stuck at 0.14-0.77 A (re-match oscillating between
   crossed pairings) — the profile sort is a deliberate deviation from the
   worker plan's step 2, chosen so the plan's own acceptance tests
   ("rotated copy -> ~0") hold; it stays fully deterministic.
3. Initial pairing = index pairing of the two canonical orders.
4. Kabsch on matched pairs — proper rotations ONLY (det(R) = +1 enforced
   by the det<0 column flip, core mirrors
   curation/graph/08_conformer_filter._kabsch_rmsd). Mirror images of a
   chiral set therefore keep a positive RMSD and are NOT near-dups,
   consistent with graph-track enantiomer policy. NOTE: the superposition
   here applies cb @ rot (the minimizer of ||ca - cb R||); graph 08's
   _kabsch_rmsd applies cb @ rot.T, which coincides for its pre-aligned
   ethane fixtures but overestimates RMSD for large relative rotations
   (verified: rotated water 1.42 A vs true 0.0). Graph is untouched.
5. Re-match: greedy per-element nearest-neighbour pairing under the
   superposed frame (candidates sorted by (distance, i, j) for
determinism), re-Kabsch; capped at 5 re-matches, with a seen-set guard
   that stops oscillation. Best RMSD across iterations is returned.

Limits (documented, graceful degradation):
- Greedy matching is NOT the optimal assignment (Hungarian); the general
  case is NP-hard. Returned RMSD is an upper bound on the true permutation
  RMSD — wrong pairings can only inflate it, never deflate it, so a value
  below threshold still proves near-dup at that pairing.
- Raw RMSD only; a normalized mode (division by sqrt(N)) is a documented
  extension, not implemented here.
- Representative selection (keep-first under deterministic sort; energy-
  aware choice is a documented extension — xyz headers do not guarantee
  energies) happens in 02_identity, not here.
- Near-dup decision = rmsd < NEAR_DUP_ANGSTROM (0.25 default from the
  worker plan); threshold range-guarding (0.1-5.0 A) is a CLI concern
  handled by 02_identity.
"""

import hashlib
from collections import Counter

import numpy as np
from rdkit import Chem

_PT = Chem.GetPeriodicTable()

# Near-dup decision threshold in Angstrom (worker plan default, T3).
NEAR_DUP_ANGSTROM = 0.25

# Maximum Kabsch/re-match refinements after the initial pairing.
_MAX_REMATCHES = 5


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


def _canonical_order(symbols, coords):
    """Deterministic rotation-invariant atom order -> (z_list, coords_ordered).

    Sort key per atom: (Z, distance-profile, x, y, z). The distance profile
    (for each distinct element present, the sorted tuple of interatomic
    distances to atoms of that element, self excluded) is invariant under
    rigid motions and atom permutations, so copies of the same structure
    canonicalize to the same atom sequence regardless of orientation.
    """
    n = len(symbols)
    zs = [_atomic_number(s) for s in symbols]
    profiles = []
    for i in range(n):
        per_elem = {}
        for e in sorted(set(zs)):
            per_elem[e] = tuple(
                sorted(
                    float(np.linalg.norm(coords[i] - coords[j]))
                    for j in range(n)
                    if zs[j] == e and j != i
                )
            )
        profiles.append(per_elem)
    order = sorted(
        range(n),
        key=lambda i: (
            zs[i],
            tuple(profiles[i][e] for e in sorted(profiles[i])),
            float(coords[i][0]),
            float(coords[i][1]),
            float(coords[i][2]),
        ),
    )
    ordered = np.asarray([coords[i] for i in order], dtype=np.float64)
    if ordered.size == 0:
        ordered = np.zeros((0, 3), dtype=np.float64)
    return [zs[i] for i in order], ordered


def _greedy_match(zs_a, coords_a, zs_b, coords_b):
    """Greedy per-element nearest-neighbour pairing (deterministic).

    Candidates sorted by (distance, i, j); each candidate takes both atoms
    if still unmatched. Not optimal assignment (documented limit).
    """
    n = len(zs_a)
    candidates = sorted(
        (float(np.linalg.norm(coords_a[i] - coords_b[j])), i, j)
        for i in range(n)
        for j in range(n)
        if zs_a[i] == zs_b[j]
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _dist, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return pairs


def _kabsch_pair_rmsd(coords_a, coords_b, pairs):
    """Proper-rotation Kabsch RMSD on matched pairs + superposed copy of b.

    Core mirrors curation/graph/08_conformer_filter._kabsch_rmsd (det<0 ->
    flip last column of u, enforcing proper rotations), except the
    superposition applies cb @ rot (the true minimizer; see module docstring).
    Returns (rmsd, superposed_b_full) — b translated/rotated so that the
    matched pairs align with a, for the re-match step.
    """
    ia = [i for i, _ in pairs]
    ib = [j for _, j in pairs]
    x = coords_a[ia]
    y = coords_b[ib]
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    cov = yc.T @ xc
    u, _, vh = np.linalg.svd(cov)
    det = np.linalg.det(u @ vh)
    if det < 0:
        u[:, -1] *= -1
    rot = u @ vh
    diff = xc - yc @ rot
    rmsd = float(np.sqrt((diff * diff).sum() / len(ia))) if len(ia) else 0.0
    superposed = (coords_b - y.mean(axis=0)) @ rot + x.mean(axis=0)
    return rmsd, superposed


def conformer_rmsd(symbols_a, coords_a, symbols_b, coords_b):
    """Raw Kabsch-aligned RMSD (Angstrom) between two atom sets.

    Proper rotations only; greedy per-element matching with capped re-match
    refinement (see module docstring for pipeline + documented limits).
    inf when element multisets differ. Empty vs empty -> 0.0.
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    for tag, syms, xyz in (("a", symbols_a, coords_a), ("b", symbols_b, coords_b)):
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"coords_{tag} must be (N, 3), got shape {xyz.shape}")
        if len(syms) != xyz.shape[0]:
            raise ValueError(f"{len(syms)} symbols vs {xyz.shape[0]} coord rows in {tag}")
        if xyz.size and not np.isfinite(xyz).all():
            raise ValueError(f"coords_{tag} contain non-finite values")
    if Counter(_atomic_number(s) for s in symbols_a) != Counter(
        _atomic_number(s) for s in symbols_b
    ):
        return float("inf")
    n = len(symbols_a)
    if n == 0:
        return 0.0
    zs_a, a = _canonical_order(symbols_a, coords_a)
    zs_b, b = _canonical_order(symbols_b, coords_b)

    pairs = [(i, i) for i in range(n)]  # descriptor pairing
    current = b
    best = float("inf")
    seen: set[tuple] = set()
    for _ in range(1 + _MAX_REMATCHES):
        key = tuple(sorted(pairs))
        if key in seen:
            break  # oscillation guard (graceful degradation)
        seen.add(key)
        rmsd, current = _kabsch_pair_rmsd(a, current, pairs)
        best = min(best, rmsd)
        rematched = _greedy_match(zs_a, a, zs_b, current)
        if rematched == pairs:
            break
        pairs = rematched
    return best


def is_near_dup(symbols_a, coords_a, symbols_b, coords_b, threshold=NEAR_DUP_ANGSTROM):
    """Near-dup decision: conformer_rmsd < threshold (default 0.25 A)."""
    return conformer_rmsd(symbols_a, coords_a, symbols_b, coords_b) < threshold
