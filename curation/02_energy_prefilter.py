#!/usr/bin/env python3
"""
02_energy_prefilter.py — Stage 2: Energy outlier detection.

Two complementary filters:
  1. Energy sanity:  energy_ha > 0  → physically impossible for bound
     ground-state molecules (SCF divergence / geometry collapse).
     Flagged as "energy_positive".  Can be disabled with
     --skip-energy-positive-check.
  2. OLS atom-type regression:  E_total ~ sum(n_i * e_i)  (no intercept —
     physically motivated decomposition into atomic contributions).
     MAD-based robust z-score flags statistical outliers ("flagged_ols").

Energy-positive molecules are excluded from the OLS fit to avoid
contaminating the regression.

Two-pass streaming: pass 1 reads every batch once, extracts per-element atom
counts (parsed from the mol_block text — no RDKit mol reconstruction) and
energies, and fits the global OLS model.  Pass 2 re-reads batches and applies
flags.  Memory scales with the scalar arrays (O(n) floats), not O(n) mol
objects — required for QMugs (~665k) and AQM (~3M) datasets.

References:
  Khrabrov et al., "∇²DFT: A Comprehensive Quantum Chemistry Dataset",
    NeurIPS 2024 — energy>0 sanity check.
  Eriksen et al., "Decomposing Chemical Space: Applications to the Machine
    Learning of Atomic Energies", J. Chem. Phys. 158, 194102 (2023).
    doi:10.1063/5.0145696 — atomic energy decomposition in QM datasets.
  Iglewicz & Hoaglin, "How to Detect and Handle Outliers" (1993).
    ASQC Quality Press — MAD-based robust z-score (k=0.6745).

Flagged molecules are written as human-readable SDF to
rejects/02_energy_prefilter/.

Usage:
    python3 02_energy_prefilter.py -i batches/ -o filtered_batches/ --threshold 3.5
    python3 02_energy_prefilter.py --skip  # skip energy prefilter entirely
    python3 02_energy_prefilter.py --skip-energy-positive-check
"""

import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np

from lib.parquet_io import read_batch, write_batch
from lib.sdf_io import write_reject_sdf


def _passthrough(input_dir, output_dir):
    """Copy all batches unchanged (passthrough mode)."""
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".parquet"):
            in_path  = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            rows = read_batch(in_path)
            for row in rows:
                row["energy_status"] = "skipped"
            write_batch(out_path, rows)
            print(f"  {fname} → {out_path} ({len(rows)} mols, skipped)")
    print("Passthrough complete.")

# Default atom types for the OLS regression.
ATOM_TYPES = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "P", "I"]


# -- Atom counting --------------------------------------------------------

def _count_atoms_from_block(mol_block, atom_types):
    """Count atoms per element from mol_block atom lines (no RDKit).

    V2000 layout: line index 3 is the counts line (natoms = first
    whitespace-separated field); atom lines follow at indices
    4..4+natoms-1, with the element symbol as the 4th whitespace-separated
    field (isotopes collapse to the base symbol, matching GetSymbol()).

    Returns a {symbol: count} dict, or None for a corrupt block.
    """
    lines = mol_block.split("\n")
    if len(lines) < 4:
        return None
    counts_fields = lines[3].split()
    if not counts_fields:
        return None
    try:
        natoms = int(counts_fields[0])
    except ValueError:
        return None
    if natoms < 0 or 4 + natoms > len(lines):
        return None
    counts = {sym: 0 for sym in atom_types}
    for i in range(4, 4 + natoms):
        fields = lines[i].split()
        if len(fields) < 4:
            return None
        sym = fields[3]
        if sym in counts:
            counts[sym] += 1
    return counts


# -- OLS fitting ----------------------------------------------------------

def fit_ols(X, y):
    """Fit OLS  E_total ~ sum(n_i * e_i)  (no intercept).

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Atom-type count matrix (columns indexed by atom_types order).
    y : np.ndarray, shape (n,)
        Total energies (Ha).

    Returns
    -------
    beta : np.ndarray, shape (k,)
        Fitted atomic energies.
    residuals : np.ndarray, shape (n,)
        Residuals (E_actual - E_fitted).
    z : np.ndarray, shape (n,)
        MAD-based robust z-score.
    """
    n = X.shape[0]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Residuals and MAD-based robust z-score (k=0.6745)
    residuals = y - X @ beta
    res_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - res_median)))
    if mad < 1e-12:
        # Zero MAD (perfect fit): no robust scale → no outliers detectable.
        z = np.zeros(n)
    else:
        z = 0.6745 * (residuals - res_median) / mad

    return beta, residuals, z


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 2 — Energy outlier detection (OLS atom-type)."
    )
    p.add_argument("-i", "--input-dir", type=str, default="batches",
                   help="Directory of Parquet batch files (default: batches/).")
    p.add_argument("-o", "--output-dir", type=str, default="filtered_batches",
                   help="Output directory for Parquet batches (default: filtered_batches/).")
    p.add_argument("--rejects-dir", type=str, default="rejects/02_energy_prefilter",
                   help="Rejected molecules SDF directory (default: rejects/02_energy_prefilter/).")
    p.add_argument("--threshold", type=float, default=3.5,
                   help="MAD-based robust z-score threshold (default: 3.5).")
    p.add_argument("--skip-energy-positive-check", action="store_true",
                   default=False,
                   help="Skip the energy>0 sanity check (default: check is on).")
    p.add_argument("--atom-types", type=str, nargs="*",
                   default=ATOM_TYPES,
                   help="Atom types for OLS regression (default: C H N O S F Cl Br P I).")
    p.add_argument("--skip", action="store_true",
                   help="Skip energy prefilter entirely (pass-through all molecules).")
    p.add_argument("--force-keep-rejected", action="store_true",
                   help="Keep flagged molecules in output (default: drop them).")
    return p.parse_args()


def main():
    args = parse_args()
    from lib.provenance import record_run
    record_run(args.output_dir, "02_energy_prefilter")

    if args.skip:
        print("Skipping energy prefilter (--skip). Copying batches as-is.")
        _passthrough(args.input_dir, args.output_dir)
        return
    atom_types = args.atom_types

    # -- Discover batches -------------------------------------------------
    batch_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".parquet")
    )
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    # -- Phase 1: stream batches, collect scalars, fit --------------------
    total = 0
    n_corrupt = 0
    n_energy_positive = 0
    fit_counts = []     # per-element counts for the OLS fit subset
    fit_energies = []

    for fname in batch_files:
        batch = read_batch(os.path.join(args.input_dir, fname))
        for row in batch:
            total += 1
            counts = _count_atoms_from_block(row["mol_block"], atom_types)
            if counts is None:
                n_corrupt += 1
                continue
            energy = row["Energy_Ha"]
            if not args.skip_energy_positive_check and energy > 0:
                n_energy_positive += 1
                continue
            fit_counts.append([counts[sym] for sym in atom_types])
            fit_energies.append(energy)

    print(f"  {total} molecules loaded")

    if n_corrupt:
        print(f"  {n_corrupt} corrupt mols excluded (will be marked 'mol_corrupt')")

    if not args.skip_energy_positive_check:
        if n_energy_positive:
            print(f"  {n_energy_positive} energy-positive molecules flagged "
                  f"(energy_ha > 0, excluded from OLS fit)")
    else:
        print("  Energy>0 check skipped (--skip-energy-positive-check)")

    n_fit = len(fit_energies)
    if n_fit == 0:
        print("Nothing to fit — aborting.")
        sys.exit(1)

    print(f"  {n_fit} molecules for OLS fit")

    X = np.array(fit_counts, dtype=np.float64)
    y = np.array(fit_energies, dtype=np.float64)

    # -- Fit --------------------------------------------------------------
    print(f"\nFitting OLS  E~sum(n_i*e_i)  ({len(atom_types)} atom types: {', '.join(atom_types)}) ...")
    beta, residuals, z = fit_ols(X, y)

    print("  Fitted atomic energies (Ha):")
    for sym, e in zip(atom_types, beta):
        print(f"    e_{sym:2s} = {e:12.6f}")

    res_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - res_median)))
    print(f"  Residual median: {res_median:.4g} Ha  MAD: {mad:.4g} Ha")

    # -- Phase 2: re-read batches, apply flags, write ---------------------
    os.makedirs(args.output_dir, exist_ok=True)
    n_ok = 0
    n_flagged_ols = 0
    n_corrupt_flag = 0
    n_energy_positive_flag = 0
    rejected_rows = []
    n_dropped = 0
    fit_idx = 0

    for fname in batch_files:
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        batch = read_batch(in_path)
        out_rows = []

        for row in batch:
            counts = _count_atoms_from_block(row["mol_block"], atom_types)
            if counts is None:
                row["energy_status"] = "mol_corrupt"
                n_corrupt_flag += 1
            elif not args.skip_energy_positive_check and row["Energy_Ha"] > 0:
                row["energy_status"] = "energy_positive"
                n_energy_positive_flag += 1
                rejected_rows.append(row)
            else:
                z_val = z[fit_idx]
                fit_idx += 1
                if abs(z_val) > args.threshold:
                    row["energy_status"] = "flagged_ols"
                    n_flagged_ols += 1
                    rejected_rows.append(row)
                else:
                    row["energy_status"] = "ok"
                    n_ok += 1

            out_rows.append(row)

        if not args.force_keep_rejected:
            n_before = len(out_rows)
            out_rows = [r for r in out_rows if r.get("energy_status") == "ok"]
            n_dropped += n_before - len(out_rows)

        write_batch(out_path, out_rows)

    # -- Reject SDF -------------------------------------------------------
    if rejected_rows:
        reject_path = os.path.join(args.rejects_dir, "energy_flagged.sdf")
        write_reject_sdf(reject_path, rejected_rows,
                         reject_reason="energy_filter")
        print(f"\n  {len(rejected_rows)} rejected → {reject_path}")

    # -- Report ------------------------------------------------------------
    n_total_rejected = n_energy_positive_flag + n_flagged_ols
    print(f"\nReport")
    print(f"  Total:              {total}")
    print(f"  OK:                 {n_ok}")
    print(f"  Flagged (OLS):      {n_flagged_ols}")
    print(f"  Energy positive:    {n_energy_positive_flag}")
    print(f"  Mol corrupt:        {n_corrupt_flag}")
    if n_dropped:
        print(f"  Dropped:            {n_dropped} rejected (use --force-keep-rejected to keep)")

    if n_total_rejected:
        pct = 100 * n_total_rejected / total
        print(f"  (% rejected:        {pct:.2f}%)")

    # Sentinel for Makefile
    if n_total_rejected:
        sentinel = os.path.join(args.rejects_dir, ".FLAGGED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{n_total_rejected} molecules rejected "
                    f"(ols={n_flagged_ols}, energy_positive={n_energy_positive_flag})\n")


if __name__ == "__main__":
    main()
