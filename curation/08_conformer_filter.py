#!/usr/bin/env python3
"""08_conformer_filter.py — Stage 8: Conformer deduplication via heavy-atom RMSD.

Groups molecules by CompoundID (same canonical SMILES = same molecule),
then clusters conformers by heavy-atom RMSD after Kabsch alignment
(MaxMin clustering).  Lowest-energy molecule per cluster is kept;
near-identical conformers (RMSD < threshold) are removed.

Flags:
  --energy-aware (default) — lowest-energy conformer always kept first.
  --rmsd-mode raw (default) | normalized — normalized divides by sqrt(N_heavy).
  --coverage-guarantee — stop when ≥95% of pool is within threshold.

Canonical atom ordering (from stage 7) is required.

Community standards:
  SPICE (Eastman 2023): MaxMin RMSD conformer selection.
  MARCEL (Zhu 2023): 0.5 Å (rigid), 1.0 Å (medium), 2.0 Å (flexible).
  ∇²DFT (Khrabrov 2024): Butina clustering at 95% coverage.

Usage:
    python3 08_conformer_filter.py -i reordered_batches/ -o conformer_batches/
"""

import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from lib.parallel import parallel_map
from lib.parquet_io import read_batch, write_batch
from lib.sdf_io import write_reject_sdf

# -- Heavy-atom RMSD ------------------------------------------------------


def _kabsch_rmsd(coords_a, coords_b):
    """
    Kabsch-aligned RMSD between two (n, 3) coordinate arrays.

    Assumes identical atom ordering (canonical reordering guarantees this
    for identical CompoundIDs).  Returns RMSD in Å.
    """
    n = coords_a.shape[0]
    if n == 0:
        return 0.0

    # Translate to origin
    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)

    # Kabsch: optimal rotation via SVD of covariance matrix
    cov = cb.T @ ca
    u, _, vh = np.linalg.svd(cov)
    det = np.linalg.det(u @ vh)
    if det < 0:
        u[:, -1] *= -1
    rot = u @ vh

    cb_rot = cb @ rot.T
    diff = ca - cb_rot
    return float(np.sqrt((diff * diff).sum() / n))


def _extract_heavy_coords(mol):
    """
    Heavy-atom (non-H) coordinates of a mol as an (n_heavy, 3) float64 array.

    Explicit x/y/z conversion guarantees float64 dtype regardless of RDKit
    version (Point3D may yield object dtype under np.array in some builds).
    Returns None for a corrupt (None) mol.
    """
    if mol is None:
        return None
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            p = conf.GetAtomPosition(atom.GetIdx())
            coords.append([p.x, p.y, p.z])
    return np.array(coords, dtype=np.float64).reshape(-1, 3)


# -- MaxMin clustering ----------------------------------------------------

# Worker globals (set before Pool creation; inherited via fork copy-on-write).
_COORDS = None
_RMSD_MODE = "raw"


def _rmsd_pair_worker(pair):
    """Worker: heavy-atom RMSD for one (i, j) pair of the flat coords list."""
    i, j = pair
    ca = _COORDS[i]
    cb = _COORDS[j]
    if ca is None or cb is None:
        return float("inf")
    try:
        d = _kabsch_rmsd(ca, cb)
    except Exception:
        return float("inf")
    if _RMSD_MODE == "normalized" and ca.shape[0] > 0:
        d = d / np.sqrt(ca.shape[0])
    return d


def _sort_group(rows, energy_aware=True):
    """Sort a group by Energy_Ha (lowest first, NaN/missing last)."""
    rows = list(rows)
    if energy_aware:
        rows = sorted(
            rows,
            key=lambda r: (
                r.get("Energy_Ha") is None,
                r.get("Energy_Ha", float("inf")),
            ),
        )
    return rows


def _parse_mol(row):
    """Parse a row's mol_block (unsanitized, explicit H preserved)."""
    return Chem.MolFromMolBlock(row["mol_block"], sanitize=False, removeHs=False)


def _compute_rmsd_matrix_serial(coords, rmsd_mode):
    """All-pair heavy-atom RMSD matrix (upper triangle), serial."""
    n = len(coords)
    rmsd_raw = np.full((n, n), np.inf)
    np.fill_diagonal(rmsd_raw, 0.0)
    for i in range(n):
        if coords[i] is None:
            continue
        for j in range(i + 1, n):
            if coords[j] is None:
                continue
            try:
                d = _kabsch_rmsd(coords[i], coords[j])
            except Exception:
                d = float("inf")
            if rmsd_mode == "normalized" and coords[i].shape[0] > 0:
                d = d / np.sqrt(coords[i].shape[0])
            rmsd_raw[i, j] = d
            rmsd_raw[j, i] = d
    return rmsd_raw


def _maxmin_select(rows, coords, rmsd_matrix, rmsd_threshold, coverage_guarantee=False):
    """MaxMin selection over a precomputed RMSD matrix (sequential).

    rows: energy-sorted rows (mutated in-place with conformer_status/cluster_id).
    coords: heavy-atom coordinate arrays (None = corrupt) aligned with rows.
    rmsd_matrix: (n, n) heavy-atom RMSD matrix (inf = corrupt/uncomputed).
    """
    n = len(rows)

    # Find first non-corrupt mol as seed
    seed_idx = None
    for i in range(n):
        if coords[i] is not None:
            seed_idx = i
            break

    if seed_idx is None:
        for row in rows:
            row["conformer_status"] = "mol_corrupt"
        return rows, 0, 0

    # MaxMin selection
    selected = [seed_idx]
    remaining = set(i for i in range(n) if i != seed_idx and coords[i] is not None)

    # Track min distance from each remaining to selected set
    min_dist = {}
    for j in remaining:
        min_dist[j] = rmsd_matrix[j][seed_idx]

    def _coverage():
        """Fraction of original pool within threshold of any selected."""
        if n <= 1:
            return 1.0
        covered = 0
        for i in range(n):
            if coords[i] is None:
                continue
            if i in selected:
                covered += 1
                continue
            dmin = min(rmsd_matrix[i][s] for s in selected)
            if dmin < rmsd_threshold:
                covered += 1
        valid = sum(1 for c in coords if c is not None)
        return covered / valid if valid else 1.0

    while remaining:
        # Find remaining conformer with max min-distance
        best_j = max(remaining, key=lambda j: min_dist.get(j, 0.0))

        if coverage_guarantee:
            # Stop if coverage already ≥ 95%
            if _coverage() >= 0.95:
                break
            selected.append(best_j)
            remaining.remove(best_j)
        else:
            # Threshold-based stopping
            if min_dist[best_j] < rmsd_threshold:
                break  # all remaining too close to selected
            selected.append(best_j)
            remaining.remove(best_j)

        # Update min distances for remaining
        for j in remaining:
            if rmsd_matrix[j][best_j] < min_dist.get(j, float("inf")):
                min_dist[j] = rmsd_matrix[j][best_j]

    # Assign status and cluster_id
    selected_set = set(selected)
    n_kept = 0
    n_removed = 0

    # Kept: assign cluster_id in selection order
    for cid, idx in enumerate(selected, start=1):
        rows[idx]["conformer_status"] = "kept"
        rows[idx]["cluster_id"] = cid
        n_kept += 1

    # Remaining: removed_conformer, assign to nearest kept
    for idx in range(n):
        if idx in selected_set:
            continue
        if coords[idx] is None:
            rows[idx]["conformer_status"] = "mol_corrupt"
            continue
        rows[idx]["conformer_status"] = "removed_conformer"
        best_dist = float("inf")
        best_cid = -1
        for s in selected:
            if rmsd_matrix[idx][s] < best_dist:
                best_dist = rmsd_matrix[idx][s]
                best_cid = rows[s]["cluster_id"]
        rows[idx]["cluster_id"] = best_cid
        n_removed += 1

    return rows, n_kept, n_removed


def _cluster_maxmin(
    grouped_rows,
    rmsd_threshold,
    energy_aware=True,
    rmsd_mode="raw",
    coverage_guarantee=False,
    coords=None,
    rmsd_matrix=None,
):
    """
    MaxMin RMSD clustering on a list of rows sharing the same CompoundID.

    Serial path: extracts heavy-atom coords, computes the all-pair RMSD
    matrix in-process, then runs MaxMin selection.  Callers may pass
    pre-extracted `coords` and a precomputed `rmsd_matrix` (e.g. from the
    parallel distance-matrix pass) to skip the O(n^2) matrix here.

    With energy_aware=True, seeds with the lowest-energy conformer.
    With coverage_guarantee=True, stops when ≥95% of pool is within
    threshold of any selected conformer (threshold becomes coverage radius).

    Order-independent — result does not depend on input ordering
    (beyond seed selection).

    Removed conformers are assigned the cluster_id of their nearest kept
    conformer for traceability.

    Returns (updated_rows, n_kept, n_removed).
    """
    rows = _sort_group(grouped_rows, energy_aware)
    if coords is None:
        coords = [_extract_heavy_coords(_parse_mol(row)) for row in rows]
    if rmsd_matrix is None:
        rmsd_matrix = _compute_rmsd_matrix_serial(coords, rmsd_mode)
    return _maxmin_select(rows, coords, rmsd_matrix, rmsd_threshold, coverage_guarantee)


# -- Main ----------------------------------------------------------------

RMSD_THRESHOLD_MIN = 0.1
RMSD_THRESHOLD_MAX = 5.0
RMSD_NORM_MIN = 0.05
RMSD_NORM_MAX = 0.50


def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 8 — Conformer deduplication (heavy-atom RMSD)."
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="reordered_batches",
        help="Input Parquet batch directory (default: reordered_batches/).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="conformer_batches",
        help="Output directory (default: conformer_batches/).",
    )
    p.add_argument(
        "--rejects-dir",
        type=str,
        default="rejects/08_conformer_filter",
        help="Rejected molecules SDF (default: rejects/08_conformer_filter/).",
    )
    p.add_argument(
        "--rmsd-threshold",
        type=float,
        default=None,
        help="Heavy-atom RMSD threshold (default: 1.0 Å for raw, 0.15 for normalized).",
    )
    p.add_argument(
        "--rmsd-mode",
        type=str,
        default="raw",
        choices=["raw", "normalized"],
        help="RMSD scaling: raw (default) or normalized by sqrt(N_heavy). EXPERIMENTAL.",
    )
    p.add_argument(
        "--energy-aware",
        dest="energy_aware",
        action="store_true",
        default=True,
        help="Pre-seed clustering with lowest-energy conformer (default).",
    )
    p.add_argument(
        "--no-energy-aware",
        dest="energy_aware",
        action="store_false",
        help="Disable energy-aware seeding.",
    )
    p.add_argument(
        "--coverage-guarantee",
        action="store_true",
        help="Stop when ≥95%% of pool is within threshold (EXPERIMENTAL).",
    )
    p.add_argument(
        "--force-keep-rejected",
        action="store_true",
        help="Keep removed conformers in output (default: drop them).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for the MaxMin distance matrix (default: 1).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from lib.provenance import record_run

    record_run(args.output_dir, "08_conformer_filter")

    # Set default threshold based on mode
    if args.rmsd_threshold is None:
        args.rmsd_threshold = 0.15 if args.rmsd_mode == "normalized" else 1.0
        print(
            f"Note: using default threshold {args.rmsd_threshold} for --rmsd-mode={args.rmsd_mode}"
        )

    # Validate threshold range
    if args.rmsd_mode == "normalized":
        lo, hi = RMSD_NORM_MIN, RMSD_NORM_MAX
    else:
        lo, hi = RMSD_THRESHOLD_MIN, RMSD_THRESHOLD_MAX

    if not (lo <= args.rmsd_threshold <= hi):
        print(
            f"Error: --rmsd-threshold must be in range [{lo}, {hi}] "
            f"for --rmsd-mode={args.rmsd_mode}, got {args.rmsd_threshold}"
        )
        sys.exit(1)

    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    mode_str = f"  mode: {args.rmsd_mode}"
    cov_str = "  coverage-guarantee" if args.coverage_guarantee else ""
    print(f"RMSD threshold: {args.rmsd_threshold} Å{mode_str}{cov_str}")
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    # -- Load all rows ----------------------------------------------------
    all_rows = []
    for fname in batch_files:
        path = os.path.join(args.input_dir, fname)
        batch = read_batch(path)
        for row in batch:
            row["_batch_file"] = fname
            all_rows.append(row)

    total = len(all_rows)
    print(f"  {total} molecules loaded")

    # -- Group by CompoundID ----------------------------------------------
    groups = {}
    for idx, row in enumerate(all_rows):
        cid = row.get("CompoundID", "")
        if not cid:
            row["conformer_status"] = "no_compound_id"
            continue
        groups.setdefault(cid, []).append(row)

    n_single = sum(1 for g in groups.values() if len(g) == 1)
    n_multi = sum(1 for g in groups.values() if len(g) > 1)

    print(f"  CompoundIDs: {len(groups)} ({n_single} single, {n_multi} multi-conformer)")

    # -- Cluster each multi-conformer group -------------------------------
    total_kept = 0
    total_removed = 0
    removed_rows = []
    n_dropped = 0

    multi_groups = {}
    for cid, group in groups.items():
        if len(group) == 1:
            group[0]["conformer_status"] = "kept"
            group[0]["cluster_id"] = 1
            total_kept += 1
        else:
            multi_groups[cid] = group

    if args.workers > 1 and multi_groups:
        # Parallel distance matrix: flatten all pairs across groups, compute
        # RMSD in ONE Pool, then reconstruct matrices and run MaxMin (sequential).
        flat_coords = []
        all_pairs = []
        group_meta = {}  # cid -> (rows, coords, pair_indices, local_pairs)

        for cid, group in multi_groups.items():
            rows = _sort_group(group, args.energy_aware)
            coords = [_extract_heavy_coords(_parse_mol(row)) for row in rows]
            start = len(flat_coords)
            flat_coords.extend(coords)
            pair_indices = []
            local_pairs = []
            for a in range(len(coords)):
                for b in range(a + 1, len(coords)):
                    if coords[a] is not None and coords[b] is not None:
                        all_pairs.append((start + a, start + b))
                        pair_indices.append(len(all_pairs) - 1)
                        local_pairs.append((a, b))
            group_meta[cid] = (rows, coords, pair_indices, local_pairs)

        global _COORDS, _RMSD_MODE
        _COORDS = flat_coords
        _RMSD_MODE = args.rmsd_mode
        dists = parallel_map(_rmsd_pair_worker, all_pairs, args.workers)

        for cid, (rows, coords, pair_indices, local_pairs) in group_meta.items():
            n = len(coords)
            mat = np.full((n, n), np.inf)
            np.fill_diagonal(mat, 0.0)
            for k, (a, b) in zip(pair_indices, local_pairs):
                d = dists[k]
                mat[a, b] = d
                mat[b, a] = d
            updated, n_k, n_r = _maxmin_select(
                rows,
                coords,
                mat,
                args.rmsd_threshold,
                coverage_guarantee=args.coverage_guarantee,
            )
            total_kept += n_k
            total_removed += n_r
            for row in updated:
                if row.get("conformer_status") == "removed_conformer":
                    removed_rows.append(row)
    else:
        # Serial path (workers == 1)
        for cid, group in multi_groups.items():
            updated, n_k, n_r = _cluster_maxmin(
                group,
                args.rmsd_threshold,
                energy_aware=args.energy_aware,
                rmsd_mode=args.rmsd_mode,
                coverage_guarantee=args.coverage_guarantee,
            )
            # Update rows in-place (they reference all_rows)
            total_kept += n_k
            total_removed += n_r
            for row in updated:
                if row.get("conformer_status") == "removed_conformer":
                    removed_rows.append(row)

    # Any row not in a group: mark as no_compound_id (already done above)

    # -- Write output batches ---------------------------------------------
    batch_groups = {}
    for row in all_rows:
        fname = row.pop("_batch_file")
        batch_groups.setdefault(fname, []).append(row)

    for fname, rows in batch_groups.items():
        out_path = os.path.join(args.output_dir, fname)
        if not args.force_keep_rejected:
            n_before = len(rows)
            rows = [r for r in rows if r.get("conformer_status") != "removed_conformer"]
            n_dropped += n_before - len(rows)
        write_batch(out_path, rows)

    # -- Reject SDF -------------------------------------------------------
    if removed_rows:
        reject_path = os.path.join(args.rejects_dir, "conformer_removed.sdf")
        write_reject_sdf(reject_path, removed_rows, reject_reason="removed_conformer")
        print(f"  {len(removed_rows)} conformers → {reject_path}")

    # -- Report ------------------------------------------------------------
    print("\nReport")
    print(f"  Total molecules:      {total}")
    print(f"  Kept:                 {total_kept}")
    print(f"  Removed conformers:   {total_removed}")
    if n_dropped:
        print(
            f"  Dropped:              {n_dropped} removed_conformer "
            f"(use --force-keep-rejected to keep)"
        )
    if total - total_kept - total_removed > 0:
        print(f"  Other (no ID):        {total - total_kept - total_removed}")

    if total_removed:
        sentinel = os.path.join(args.rejects_dir, ".REMOVED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{total_removed} conformers removed\n")


if __name__ == "__main__":
    main()
