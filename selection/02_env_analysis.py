#!/usr/bin/env python3
"""
02_env_analysis.py — descriptors.parquet → coverage analysis (stage 2).

Cheap, re-runnable.  Produces the artifacts a human reads to set the floor
parameters (B_floor / K / w) before running 03_select.py:

    env_frequency.csv  — env_id, element, n_atoms, n_mols  (rarity-sorted)
    cost_curve.csv     — molecules spent vs rare envs covered (set-cover knee)
    forced_choice.csv  — rarest envs: single-carrier (forced) vs multi (choice)
    bond_types.csv     — element pair × bond order counts (Q2 report, pool-level)
    cost_curve.png     — knee plot (if matplotlib available)

Usage:
    python3 02_env_analysis.py -i descriptors.parquet -o <out_dir> [--max-carriers 10]
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from selection.lib import coverage as cov
from selection.lib import schema


def read_descriptors(path):
    return pd.read_parquet(path)


def build_env_frequency(df):
    """env-level frequency table, rarity-sorted."""
    env_elem, env_natoms, env_nmols = cov.env_frequency(
        df["atom_envs"].tolist(), df["heavy_elements"].tolist(),
    )
    rows = [
        {"env_id": eid, "element": env_elem[eid],
         "n_atoms": env_natoms[eid], "n_mols": env_nmols[eid]}
        for eid in env_elem
    ]
    hist = pd.DataFrame(rows, columns=schema.ENV_FREQUENCY_COLUMNS)
    return hist.sort_values(
        ["n_mols", "n_atoms", "env_id"], ascending=[True, True, True],
    ).reset_index(drop=True)


def build_cost_curve(df):
    """(mols_picked, envs_covered) cumulative set-cover curve."""
    env_elem, env_natoms, env_nmols = cov.env_frequency(
        df["atom_envs"].tolist(), df["heavy_elements"].tolist(),
    )
    xs, ys = cov.greedy_set_cover_curve(
        df["atom_envs"].tolist(), env_nmols, env_natoms,
    )
    return pd.DataFrame({"mols_picked": xs, "envs_covered": ys})


def build_forced_choice(df, max_carriers):
    """Rarest envs classified forced (1 carrier) vs choice (2..max_carriers)."""
    env_elem, env_natoms, env_nmols = cov.env_frequency(
        df["atom_envs"].tolist(), df["heavy_elements"].tolist(),
    )
    env2mols = cov.env_to_mols(df["atom_envs"].tolist())
    cids = df["CompoundID"].tolist()
    rows = []
    for eid in sorted(env_nmols, key=lambda e: (env_nmols[e], env_natoms.get(e, 0), e)):
        n = env_nmols[eid]
        if n > max_carriers:
            break
        carriers = env2mols[eid]
        rows.append({
            "env_id": eid,
            "element": env_elem[eid],
            "n_mols": n,
            "pick_type": "forced" if n == 1 else "choice",
            "carrier_ids": ";".join(cids[i] for i in carriers),
        })
    return pd.DataFrame(rows, columns=schema.FORCED_CHOICE_COLUMNS)


def build_bond_types(df):
    """Aggregate per-bond 'A-B:order' strings across the pool."""
    counts = Counter()
    for bonds in df["bonds"].tolist():
        for b in bonds:
            counts[b] += 1
    return pd.DataFrame(
        [{"bond": b, "count": c} for b, c in counts.most_common()],
        columns=schema.BOND_TYPES_COLUMNS,
    )


def plot_cost_curve(cost, out_dir):
    """Knee plot — optional; skipped if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping cost_curve.png)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost["mols_picked"], cost["envs_covered"], marker=".", ms=3)
    ax.set_xlabel("molecules picked")
    ax.set_ylabel("rare envs covered")
    ax.set_title("Set-cover cost curve (knee = where B_floor should sit)")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_curve.png", dpi=120)
    plt.close(fig)


def print_summary(df, hist, forced_choice, bond_types):
    print("\n=== env_analysis summary ===")
    print(f"Molecules: {len(df):,}")
    print(f"Unique atom envs: {len(hist):,}")
    print(f"Single-carrier (forced) envs: {(hist['n_mols'] == 1).sum():,}")
    print("\nEnv frequency by element (rarest elements):")
    by_elem = hist.groupby("element")["n_mols"].agg(["count", "min", "max"])
    print(by_elem.sort_values("min").to_string())
    print("\nTop bond types:")
    print(bond_types.head(10).to_string(index=False))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True,
                   help="Path to descriptors.parquet.")
    p.add_argument("-o", "--output", required=True,
                   help="Output directory (analysis/ CSVs written here).")
    p.add_argument("--max-carriers", type=int, default=10,
                   help="Forced/choice table: only envs with ≤ this many "
                        "carrier molecules (default 10).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = read_descriptors(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = build_env_frequency(df)
    cost = build_cost_curve(df)
    fc = build_forced_choice(df, args.max_carriers)
    bonds = build_bond_types(df)

    hist.to_csv(out_dir / "env_frequency.csv", index=False)
    cost.to_csv(out_dir / "cost_curve.csv", index=False)
    fc.to_csv(out_dir / "forced_choice.csv", index=False)
    bonds.to_csv(out_dir / "bond_types.csv", index=False)
    plot_cost_curve(cost, out_dir)

    print_summary(df, hist, fc, bonds)
    print(f"\nWrote env_frequency.csv, cost_curve.csv, forced_choice.csv, "
          f"bond_types.csv → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
