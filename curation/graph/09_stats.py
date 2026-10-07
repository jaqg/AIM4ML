#!/usr/bin/env python3
"""
09_stats.py — Stage 9: Compute molecular descriptors and diversity statistics.

Reads curated Parquet batches and produces:
  - stats_summary.csv   — per-molecule descriptors (MolWt, TPSA, logP, nrot)
  - Histogram plots     — NAT, MolWt, TPSA, Energy_Ha
  - Tanimoto similarity — nearest-neighbour (optional, --tanimoto)

Uses CanonicalSMILES from dedup step for descriptor computation.

Usage:
    python3 08_stats.py -i reordered_batches/ -o stats/
    python3 08_stats.py -i reordered_batches/ -o stats/ --tanimoto --workers 8
"""

import argparse
import os
import sys

from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from lib.parquet_io import read_batch

# Shared fingerprint list for multiprocessing workers.
_GLOBAL_FP_LIST = None


def _should_exclude(row, rules):
    """Return True if row matches any exclude rule."""
    for col, val in rules.items():
        if str(row.get(col, "")) == val:
            return True
    return False


# -- Descriptors ----------------------------------------------------------


def compute_descriptors(smiles):
    """Return (MolWt, TPSA, logP, nrot, num_atoms) from canonical SMILES.

    No sanitization — explicit-H SMILES from dedup may have N valence 4
    that triggers AtomValenceException. Descriptors compute correctly
    on unsanitized mols.
    """
    if not smiles:
        return None, None, None, None, None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, None, None, None, None
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)  # initialize ring info without sanitization
    try:
        return (
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumRotatableBonds(mol),
            mol.GetNumAtoms(),
        )
    except Exception:
        return None, None, None, None, None


# -- Fingerprints (Tanimoto) ----------------------------------------------


def compute_fingerprints(smiles_list, n_bits=2048, radius=2):
    """Morgan FP for a list of canonical SMILES. Returns list of fp objects."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is not None:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            fps.append((gen.GetFingerprint(mol), smi))
        else:
            fps.append((None, smi))
    return fps


def nearest_neighbour_tanimoto(fps, workers=1):
    """Compute max Tanimoto to nearest neighbour for each molecule.

    Uses multiprocessing when workers > 1.
    """
    valid = [(fp, smi) for fp, smi in fps if fp is not None]
    if len(valid) < 2:
        return {smi: None for _, smi in fps}

    from rdkit import DataStructs

    fp_list = [f for f, _ in valid]
    smi_list = [s for _, s in valid]

    if workers > 1:
        from multiprocessing import Pool

        chunk_size = max(50, len(smi_list) // (workers * 4))
        indices = list(range(len(fp_list)))
        with Pool(workers, initializer=_init_worker, initargs=(fp_list,)) as pool:
            sim_results = list(
                tqdm(
                    pool.imap_unordered(_compute_one_tanimoto, indices, chunksize=chunk_size),
                    total=len(indices),
                    desc="Tanimoto",
                    unit="mol",
                )
            )
        # Reconstruct ordered result: idx → max_sim
        result_map = {smi_list[idx]: sim for idx, sim in sim_results}
    else:
        result_map = {}
        for i, (fp_i, smi_i) in enumerate(valid):
            sims = DataStructs.BulkTanimotoSimilarity(fp_i, fp_list)
            sims[i] = -1
            result_map[smi_i] = max(sims)

    for fp, smi in fps:
        if fp is None and smi not in result_map:
            result_map[smi] = None
    return result_map


def _compute_one_tanimoto(i):
    """Worker: compute max Tanimoto for molecule i against all others."""
    from rdkit import DataStructs

    fp_list = _GLOBAL_FP_LIST
    sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list)
    sims[i] = -1
    return (i, max(sims))


def _init_worker(fp_list):
    global _GLOBAL_FP_LIST
    _GLOBAL_FP_LIST = fp_list


# -- Funnel report --------------------------------------------------------

STAGE_ORDER = [
    ("Energy positive", "energy_status", "energy_positive"),
    ("Energy prefilter (OLS)", "energy_status", "flagged_ols"),
    ("Chemical filter", "filter_status", "rejected"),
    ("Stereo filter", "stereo_status", "removed_enantiomer"),
    ("Reorder", "reorder_status", "failed"),
    ("Conformer filter", "conformer_status", "removed_conformer"),
]


def _print_funnel(rows):
    """Print sequential funnel: how many molecules pass each stage."""
    pool = set(range(len(rows)))
    n_input = len(pool)
    cum_pass = n_input

    print("\nCuration funnel")
    print(f"  {'Stage':22s} {'Cum. pass':>10s} {'New rejected':>13s}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 13}")
    print(f"  {'Input':22s} {n_input:10d} {'—':>13s}")

    for stage_name, col, fail_val in STAGE_ORDER:
        failing = {i for i in pool if str(rows[i].get(col, "")) == fail_val}
        n_fail = len(failing)
        pool -= failing
        cum_pass -= n_fail
        print(f"  {stage_name:22s} {cum_pass:10d} {n_fail:13d}")

    print(f"  {'Final curated':22s} {cum_pass:10d} {'—':>13s}")
    print()


# -- Main ----------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 8 — Descriptors and diversity statistics."
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="conformer_batches",
        help="Input Parquet batch directory (default: conformer_batches/).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="stats",
        help="Output directory for stats and plots (default: stats/).",
    )
    p.add_argument(
        "--tanimoto", action="store_true", help="Compute Tanimoto similarity (slower, O(n²))."
    )
    p.add_argument(
        "--workers", type=int, default=1, help="Parallel workers for Tanimoto (if --tanimoto)."
    )
    p.add_argument(
        "--exclude",
        type=str,
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Exclude rows where COLUMN == VALUE (repeatable).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from lib.provenance import record_run

    record_run(args.output_dir, "09_stats")

    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Parse exclude rules
    exclude_rules = {}
    for rule in args.exclude:
        if "=" in rule:
            col, val = rule.split("=", 1)
            exclude_rules[col.strip()] = val.strip()

    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")
    if exclude_rules:
        print(f"Excluding rows where: {exclude_rules}")

    # -- Accumulate all molecules -----------------------------------------
    all_rows = []

    total_raw = 0
    for fname in batch_files:
        path = os.path.join(args.input_dir, fname)
        batch = read_batch(path)
        for row in batch:
            if _should_exclude(row, exclude_rules):
                continue
            all_rows.append(row)
        total_raw += len(batch)

    total = len(all_rows)
    n_excluded = total_raw - total
    if n_excluded:
        print(f"  Excluded {n_excluded} rows by filter rules, {total} processed")
    print(f"  {total} molecules loaded")

    # -- Deduplicate by CompoundID (keep best conformer per molecule) ----
    # Pre-index: CID → list of rows
    cid_to_rows = {}
    for row in all_rows:
        cid = row.get("CompoundID", "")
        if cid:
            cid_to_rows.setdefault(cid, []).append(row)

    seen_cids = set()
    unique_rows = []
    for row in all_rows:
        cid = row.get("CompoundID", "")
        if not cid:
            unique_rows.append(row)
        elif cid not in seen_cids:
            seen_cids.add(cid)
            # Pick conformer_status=kept if available, else first
            candidates = cid_to_rows[cid]
            best = next(
                (r for r in candidates if r.get("conformer_status") == "kept"), candidates[0]
            )
            unique_rows.append(best)
    n_unique = len(unique_rows)
    if n_unique < total:
        print(f"  Deduplicated to {n_unique} unique compounds")
    all_rows = unique_rows
    total = n_unique

    # -- Compute descriptors ----------------------------------------------
    print("Computing descriptors ...")
    mol_wt_list, tpsa_list, logp_list, nrot_list, nat_list = [], [], [], [], []
    all_smiles = []
    failed = 0

    for row in all_rows:
        smi = row.get("CanonicalSMILES", "")
        all_smiles.append(smi)
        mw, tpsa, logp, nrot, nat = compute_descriptors(smi)
        if mw is None:
            failed += 1
        mol_wt_list.append(mw)
        tpsa_list.append(tpsa)
        logp_list.append(logp)
        nrot_list.append(nrot)
        nat_list.append(nat)

    if failed:
        print(f"  {failed} descriptor failures (SMILES unparseable)")

    # -- Compute Tanimoto (optional) --------------------------------------
    tanimoto_map = {}
    if args.tanimoto:
        print("Computing Tanimoto NN (Morgan r=2, 2048 bits) ...")
        fps = compute_fingerprints(all_smiles)
        tanimoto_map = nearest_neighbour_tanimoto(fps, workers=args.workers)
        tanimoto_vals = [tanimoto_map.get(smi) for smi in all_smiles]
        n_t1 = sum(1 for v in tanimoto_vals if v is not None and v == 1.0)
    else:
        tanimoto_vals = [None] * total
        n_t1 = 0

    # -- Write CSV --------------------------------------------------------
    import csv

    csv_path = os.path.join(args.output_dir, "stats_summary.csv")

    # Build dynamic headers: metadata columns (excluding mol_block) + descriptors
    exclude = {"mol_block", "_mol"}
    meta_keys = [k for k in all_rows[0].keys() if k not in exclude and not k.startswith("_")]
    desc_keys = ["MolWt", "TPSA", "logP", "nrot", "num_atoms", "max_tanimoto"]
    headers = meta_keys + desc_keys

    all_energies = [row.get("Energy_Ha", np.nan) for row in all_rows]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, row in enumerate(all_rows):
            meta_vals = [row.get(k, "") for k in meta_keys]
            desc_vals = [
                mol_wt_list[i],
                tpsa_list[i],
                logp_list[i],
                nrot_list[i],
                nat_list[i],
                tanimoto_vals[i],
            ]
            writer.writerow(meta_vals + desc_vals)
    print(f"\n  Stats → {csv_path}")

    # -- Histograms -------------------------------------------------------
    _plot_hist(nat_list, "Number of atoms (NAT)", plots_dir, "hist_nat.pdf")
    _plot_hist(
        [mw for mw in mol_wt_list if mw is not None],
        "Molecular weight (Da)",
        plots_dir,
        "hist_molwt.pdf",
    )
    _plot_hist([t for t in tpsa_list if t is not None], "TPSA (Å²)", plots_dir, "hist_tpsa.pdf")
    _plot_hist(
        [e for e in all_energies if not np.isnan(e)], "Energy (Ha)", plots_dir, "hist_energy.pdf"
    )
    if args.tanimoto:
        _plot_hist(
            [t for t in tanimoto_vals if t is not None],
            "Max Tanimoto (nearest neighbour)",
            plots_dir,
            "hist_tanimoto.pdf",
        )

    # -- Funnel report ----------------------------------------------------
    _print_funnel(all_rows)

    # -- Summary ----------------------------------------------------------
    print("\nReport")
    print(f"  Total molecules:       {total_raw}")
    if n_excluded:
        print(f"  Excluded by rules:     {n_excluded}")
    print(f"  Processed:             {total}")
    if mol_wt_list.count(None) < total:
        valid_mw = [mw for mw in mol_wt_list if mw is not None]
        print(
            f"  MolWt:  mean={np.mean(valid_mw):.1f}  median={np.median(valid_mw):.1f}  "
            f"min={np.min(valid_mw):.1f}  max={np.max(valid_mw):.1f}"
        )
    if tpsa_list.count(None) < total:
        valid_t = [t for t in tpsa_list if t is not None]
        print(
            f"  TPSA:   mean={np.mean(valid_t):.1f}  median={np.median(valid_t):.1f}  "
            f"min={np.min(valid_t):.1f}  max={np.max(valid_t):.1f}"
        )
    if args.tanimoto:
        valid_tani = [t for t in tanimoto_vals if t is not None]
        if valid_tani:
            print(
                f"  Tanimoto NN: mean={np.mean(valid_tani):.3f}  "
                f"median={np.median(valid_tani):.3f}  "
                f"max={np.max(valid_tani):.3f}"
            )
        if n_t1:
            print(f"  T=1.0 pairs:       {n_t1}")


def _plot_hist(data, xlabel, plots_dir, filename):
    """Save a histogram to plots_dir/filename. Drops None/NaN entries
    (failed descriptors, missing energies)."""
    data = [d for d in data if d is not None and not (isinstance(d, float) and np.isnan(d))]
    if not data:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(plots_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot → {path}")


if __name__ == "__main__":
    raise SystemExit(main())
