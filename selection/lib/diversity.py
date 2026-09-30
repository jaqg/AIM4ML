"""
diversity.py — fingerprint distance + MaxMin selection primitives.

Pure, deterministic functions.  Morgan fingerprint matrices are float32
(N, nbits) for BLAS-speed Tanimoto similarity; on-bit index lists are the
storage format (compact).
"""

import numpy as np


def tanimoto_sim(a, b):
    """Tanimoto similarity of two sorted on-bit index lists."""
    i = j = 0
    inter = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        if a[i] == b[j]:
            inter += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    union = na + nb - inter
    return inter / union if union else 0.0


def onbits_to_matrix(onbits, nbits, dtype=np.float32):
    """(N, nbits) binary matrix from a list of on-bit lists."""
    M = np.zeros((len(onbits), nbits), dtype=dtype)
    for i, ob in enumerate(onbits):
        if len(ob):  # works for Python lists and numpy arrays
            M[i, ob] = 1.0
    return M


def sim_to_reference(matrix, counts, ref_idx):
    """Tanimoto similarity of every row to row `ref_idx`.

    matrix: (N, nbits) float32; counts: (N,) row bit-counts.
    """
    inter = matrix @ matrix[ref_idx]
    union = counts + counts[ref_idx] - inter
    sim = np.where(union > 0.0, inter / union, 0.0)
    return sim.astype(np.float32)


def centroid_seed(matrix):
    """Index of the molecule closest (cosine) to the mean fingerprint."""
    mean = matrix.mean(axis=0).astype(np.float64)
    dots = matrix.astype(np.float64) @ mean
    norms = np.sqrt(matrix.sum(axis=1).astype(np.float64))
    mean_norm = float(np.linalg.norm(mean))
    cos = dots / (norms * mean_norm + 1e-12)
    return int(np.argmax(cos))


def centroid_seed_masked(matrix, pool_mask):
    """Index of the pool molecule closest (cosine) to the pool mean FP."""
    sub = matrix[pool_mask]
    mean = sub.mean(axis=0).astype(np.float64)
    dots = sub.astype(np.float64) @ mean
    norms = np.sqrt(sub.sum(axis=1).astype(np.float64))
    cos = dots / (norms * float(np.linalg.norm(mean)) + 1e-12)
    local = int(np.argmax(cos))
    return int(np.flatnonzero(pool_mask)[local])


def maxmin_indices(matrix, k, seed_idx):
    """Greedy farthest-point (MaxMin) selection on a full matrix.

    Returns list of row indices (length min(k, n)); order is meaningful
    (index 0 = seed, subsequent = farthest from all previously selected).
    """
    n = matrix.shape[0]
    if k >= n:
        return list(range(n))
    counts = matrix.sum(axis=1).astype(np.float32)
    selected = [int(seed_idx)]
    picked = np.zeros(n, dtype=bool)
    picked[seed_idx] = True
    min_sim = sim_to_reference(matrix, counts, seed_idx)
    min_sim[picked] = np.inf
    for _ in range(k - 1):
        nxt = int(np.argmin(min_sim))
        selected.append(nxt)
        picked[nxt] = True
        sim = sim_to_reference(matrix, counts, nxt)
        np.minimum(min_sim, sim, out=min_sim)
        min_sim[picked] = np.inf
    return selected


def maxmin_masked(matrix, counts, pool_mask, k, seed_idx, max_sim=None, scores=None):
    """MaxMin restricted to a boolean subset of the full matrix.

    No submatrix copy: distances are computed on the full matrix and
    non-pool rows are masked to +inf so they are never picked.  Optionally
    maintains `max_sim` (max Tanimoto to all selected so far) in place —
    used by the depth-pass nearest-neighbour guard.  Optionally appends each
    pick's min-similarity (the MaxMin diversity score; lower = more diverse)
    to `scores` in place (seed → None).

    Returns list of global row indices (pool members only).
    """
    n = matrix.shape[0]
    m = int(pool_mask.sum())
    if k >= m:
        idx = list(np.flatnonzero(pool_mask))
        idx.remove(int(seed_idx))
        if scores is not None:
            scores.extend([None] * m)
        return [int(seed_idx)] + idx
    selected = [int(seed_idx)]
    if scores is not None:
        scores.append(None)
    picked = np.zeros(n, dtype=bool)
    picked[seed_idx] = True
    not_pool = ~pool_mask
    min_sim = sim_to_reference(matrix, counts, seed_idx)
    if max_sim is not None:
        np.maximum(max_sim, min_sim, out=max_sim)
    min_sim = min_sim.copy()
    min_sim[not_pool] = np.inf
    min_sim[picked] = np.inf
    for _ in range(k - 1):
        nxt = int(np.argmin(min_sim))
        selected.append(nxt)
        if scores is not None:
            scores.append(float(min_sim[nxt]))
        picked[nxt] = True
        sim = sim_to_reference(matrix, counts, nxt)
        if max_sim is not None:
            np.maximum(max_sim, sim, out=max_sim)
        np.minimum(min_sim, sim, out=min_sim)
        # re-mask BOTH non-pool rows and every already-picked row: the
        # np.minimum above reintroduces finite sims for both.
        min_sim[not_pool] = np.inf
        min_sim[picked] = np.inf
    return selected
