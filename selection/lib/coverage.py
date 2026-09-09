"""
coverage.py — atomic-environment coverage + selection-floor primitives.

Pure, deterministic functions.  The floor pass (weighted greedy) lives here
because it operates on atom environments; the diversity term it needs comes
from ``lib/diversity.sim_to_reference``.
"""

from collections import Counter, defaultdict

import numpy as np

from .diversity import sim_to_reference


def env_frequency(atom_envs, elements):
    """Aggregate atom environments across molecules.

    Parameters
    ----------
    atom_envs : list[list[list[int]]] — per molecule, per heavy atom, env ids.
    elements  : list[list[str]]      — parallel to atom_envs (per-atom symbol).

    Returns
    -------
    env_elem  : dict env_id → element symbol (root atom)
    env_natoms: dict env_id → number of atom instances carrying it
    env_nmols : dict env_id → number of distinct molecules carrying it
    """
    env_elem = {}
    env_natoms = Counter()
    env_nmols = Counter()
    for mol_envs, mol_elems in zip(atom_envs, elements):
        distinct = set()
        for atom_envs_i, elem in zip(mol_envs, mol_elems):
            for eid in atom_envs_i:
                env_natoms[eid] += 1
                env_elem[eid] = elem
                distinct.add(eid)
        for eid in distinct:
            env_nmols[eid] += 1
    return env_elem, env_natoms, env_nmols


def env_to_mols(atom_envs):
    """env_id → sorted list of molecule indices carrying it."""
    d = defaultdict(set)
    for mi, mol_envs in enumerate(atom_envs):
        for eid in set(e for atom_envs_i in mol_envs for e in atom_envs_i):
            d[eid].add(mi)
    return {eid: sorted(mols) for eid, mols in d.items()}


def floor_pass(atom_envs, env_nmols, env_natoms, fp_matrix, n_budget, w,
               target_envs=None, max_sim=None, counts=None, scores=None):
    """Coverage-constrained diversity (weighted greedy) — handoff §3.3.

    Forced picks first (env with exactly ONE carrier molecule), then choice
    picks scored by::

        score(m) = w * (# uncovered rare envs carried by m)
                   + (1 - w) * (1 - max Tanimoto to already-selected)

    "Rare envs" are ordered by rarity (fewest carriers first); forced picks
    (n=1) come first and choice picks weight uncovered-count.

    Parameters
    ----------
    atom_envs   : per-molecule atom env lists (used to build carrier index)
    env_nmols   : env_id → carrier-molecule count (from env_frequency)
    env_natoms  : env_id → atom-instance count (tie-break for rarity)
    fp_matrix   : (N, nbits) float32 fingerprint matrix for the diversity term
    n_budget    : maximum number of molecules to select
    w           : coverage weight in [0,1]; (1-w) on diversity
    target_envs : optional env ids to restrict coverage to (None = all envs)
    max_sim     : optional (N,) float32 array updated in place (max Tanimoto
                  to every selected so far) — reused by the depth-pass guard
    counts      : optional (N,) row bit-counts (avoids recompute per pass)
    scores      : optional list appended in place with each pick's score
                  (forced picks → None; choice picks → weighted score)

    Returns (selected_indices, covered_envs).
    """
    n = len(atom_envs)
    if n == 0:
        return [], set()

    env2mols = env_to_mols(atom_envs)
    if counts is None:
        counts = fp_matrix.sum(axis=1).astype(np.float32)
    if max_sim is None:
        max_sim = np.zeros(n, dtype=np.float32)

    if target_envs is None:
        target_envs = set(env_nmols.keys())
    else:
        target_envs = set(target_envs)
    rare_order = sorted(
        target_envs,
        key=lambda e: (env_nmols[e], env_natoms.get(e, 0), e),
    )
    rare_set = target_envs

    selected = []
    covered = set()
    in_selected = np.zeros(n, dtype=bool)

    def add(mi, score=None):
        in_selected[mi] = True
        selected.append(mi)
        if scores is not None:
            scores.append(score)
        for atom_envs_i in atom_envs[mi]:
            for e2 in atom_envs_i:
                if e2 in rare_set:
                    covered.add(e2)
        sim = sim_to_reference(fp_matrix, counts, mi)
        np.maximum(max_sim, sim, out=max_sim)

    # 1) Forced picks: env with exactly one carrier molecule.
    #    Mandatory, but still capped by n_budget (scattered-rare-env risk: if
    #    there are more single-carrier envs than budget, only the first
    #    n_budget enter — the rest surface in the cost curve / thin report).
    for eid in rare_order:
        if len(selected) >= n_budget:
            break
        if env_nmols[eid] != 1:
            continue
        mols = env2mols[eid]
        if len(mols) == 1 and not in_selected[mols[0]]:
            add(mols[0])

    # 2) Choice picks: weighted greedy until budget or all rare envs covered.
    while len(selected) < n_budget and not rare_set.issubset(covered):
        best = None
        best_score = -1.0
        seen = set()
        for eid in rare_order:
            if eid in covered:
                continue
            for mi in env2mols[eid]:
                if in_selected[mi] or mi in seen:
                    continue
                seen.add(mi)
                unc = sum(1 for atom_envs_i in atom_envs[mi]
                          for e2 in atom_envs_i
                          if e2 in rare_set and e2 not in covered)
                div = 1.0 - float(max_sim[mi])
                score = w * unc + (1.0 - w) * div
                # tie-break: more coverage, then lower index (deterministic)
                if score > best_score:
                    best, best_score = mi, score
        if best is None:
            break
        add(best, best_score)

    return selected, covered


def greedy_set_cover_curve(atom_envs, env_nmols, env_natoms):
    """Cost curve for env_analysis: (molecules picked, rare envs covered).

    Pure-greedy (w=1) floor pass over all envs, forced singles first.
    Returns two parallel lists: xs (molecules), ys (cumulative envs covered).
    """
    n = len(atom_envs)
    if n == 0:
        return [], []
    env2mols = env_to_mols(atom_envs)
    rare_order = sorted(
        env_nmols.keys(),
        key=lambda e: (env_nmols[e], env_natoms.get(e, 0), e),
    )
    rare_set = set(rare_order)
    selected = set()
    covered = set()
    xs, ys = [], []
    for eid in rare_order:
        if env_nmols[eid] != 1:
            continue
        mols = env2mols[eid]
        if len(mols) == 1:
            mi = mols[0]
            if mi not in selected:
                selected.add(mi)
                for atom_envs_i in atom_envs[mi]:
                    for e2 in atom_envs_i:
                        if e2 in rare_set:
                            covered.add(e2)
                xs.append(len(selected))
                ys.append(len(covered))
    while not rare_set.issubset(covered):
        best = None
        best_unc = -1
        for mi in range(n):
            if mi in selected:
                continue
            unc = sum(1 for atom_envs_i in atom_envs[mi]
                      for e2 in atom_envs_i
                      if e2 in rare_set and e2 not in covered)
            if unc > best_unc:
                best, best_unc = mi, unc
        if best is None or best_unc == 0:
            break
        selected.add(best)
        for atom_envs_i in atom_envs[best]:
            for e2 in atom_envs_i:
                if e2 in rare_set:
                    covered.add(e2)
        xs.append(len(selected))
        ys.append(len(covered))
    return xs, ys


def js_divergence(a, b, bins=50):
    """Jensen-Shannon divergence (log2, [0,1]) between two 1-D arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa.astype(float) / pa.sum()
    pb = pb.astype(float) / pb.sum()
    eps = 1e-12
    pa = np.where(pa > 0, pa, eps)
    pb = np.where(pb > 0, pb, eps)
    m = 0.5 * (pa + pb)

    def kl(x, y):
        return float(np.sum(x * np.log2(x / y)))

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
