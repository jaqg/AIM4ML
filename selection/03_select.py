#!/usr/bin/env python3
"""
03_select.py — floor → diversity → depth passes + reports (stage 3).

Reads descriptors.parquet and produces the ~N-molecule selection (SDB):

    1. Floor pass    — coverage-constrained diversity (weighted greedy),
                       forced single-carrier picks first (§3.3).  Budget set by
                       --floor-budget (fraction of N) or --floor-k (explicit K).
    2. Diversity pass — L1 Bemis-Murcko scaffold partition (two-tier, ff/tier1-min,
                       sqrt-prop within tier 1) + L2 MaxMin on Morgan ECFP4 (§3.4).
    3. Depth pass    — bump thin envs (multiplicity < --target-mult), NN-Tanimoto
                       guard, capped by --depth-budget (§3.5).
    4. Reports       — coverage per element + frequency band, thin-coverage list,
                       bond-type coverage, property JS divergence (§3.9).

Passes are independent and each runnable alone for inspection via --passes
(e.g. `--passes floor`).

Outputs (to --output dir):
    selection.parquet         — selected molecules + pass/tier/score/order labels
    reports/coverage.csv      — env coverage by element and by frequency band
    reports/thin_coverage.csv — single-instance envs (targeting list for phase 2)
    reports/bond_types.csv    — element pair × order, pool vs selected
    reports/property_js.csv   — JS divergence (molwt, tpsa, energy)
    selection_summary.json    — run parameters + per-pass counts

Usage:
    python3 03_select.py -i descriptors.parquet -o <out_dir> --n 10000
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from selection.lib import coverage as cov
from selection.lib import diversity as div
from selection.lib import descriptors as dsc
from selection.lib import schema

ACYCLIC = "__acyclic__"
PASSES = ("floor", "diversity", "depth")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True,
                   help="Path to descriptors.parquet.")
    p.add_argument("-o", "--output", required=True,
                   help="Output directory.")
    p.add_argument("--n", type=int, default=10_000,
                   help="Total selection budget (number of molecules).")
    p.add_argument("--passes", type=str, default="floor,diversity,depth",
                   help="Comma-separated subset of floor,diversity,depth "
                        "(each runnable alone for inspection).")
    p.add_argument("--floor-budget", type=float, default=0.15,
                   help="Fraction of n for the floor pass (molecule budget).")
    p.add_argument("--floor-k", type=int, default=None,
                   help="Alternative: cover the K rarest envs (molecule count "
                        "falls out of set cover). Overrides --floor-budget.")
    p.add_argument("--w", type=float, default=0.4,
                   help="Floor-pass coverage weight in [0,1]; (1-w) on diversity.")
    p.add_argument("--depth-budget", type=float, default=0.10,
                   help="Fraction of n reserved for the depth pass.")
    p.add_argument("--target-mult", type=int, default=2,
                   help="Depth pass deepens envs to at least this multiplicity.")
    p.add_argument("--depth-nn-sim", type=float, default=0.6,
                   help="Depth pass NN guard: reject candidates whose max "
                        "Tanimoto to selected exceeds this.")
    p.add_argument("--tier1-min", type=int, default=50,
                   help="Min scaffold family size for per-scaffold tier-1 MaxMin.")
    p.add_argument("--ff", type=float, default=0.30,
                   help="Fixed fraction of the diversity budget allocated to tier 1.")
    return p.parse_args(argv)


def _scaf_key(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ACYCLIC
    return s


def run_floor(atom_envs, env_nmols, env_natoms, fp_matrix, counts, max_sim,
              args):
    """Floor pass. Returns (selected indices, scores)."""
    scores = []
    if args.floor_k is not None:
        rare_order = sorted(
            env_nmols, key=lambda e: (env_nmols[e], env_natoms.get(e, 0), e),
        )
        target = rare_order[:args.floor_k]
        sel, _ = cov.floor_pass(
            atom_envs, env_nmols, env_natoms, fp_matrix,
            len(atom_envs), args.w, target_envs=target,
            max_sim=max_sim, counts=counts, scores=scores,
        )
    else:
        b_floor = int(round(args.floor_budget * args.n))
        sel, _ = cov.floor_pass(
            atom_envs, env_nmols, env_natoms, fp_matrix,
            b_floor, args.w, max_sim=max_sim, counts=counts, scores=scores,
        )
    return sel, scores


def run_diversity(df, fp_matrix, counts, max_sim, excluded, budget_div, args):
    """Diversity pass. Returns (selected indices, scores, tier labels)."""
    n = len(df)
    tier_of = {}
    pool_idx = [i for i in range(n) if i not in excluded]
    if not pool_idx or budget_div <= 0:
        return [], [], {}

    pool_scaffold = [_scaf_key(s) for s in df["scaffold"].iloc[pool_idx].tolist()]
    groups = defaultdict(list)
    for gi, s in zip(pool_idx, pool_scaffold):
        groups[s].append(gi)

    t1_set = {s for s, v in groups.items() if len(v) > args.tier1_min}
    k_tier1 = int(np.floor(args.ff * budget_div))

    selected = []
    scores = []

    # Tier 1: per-scaffold sqrt-prop budget, MaxMin within each group.
    t1_groups = {s: v for s, v in groups.items() if s in t1_set}
    if t1_groups:
        sqrt_sum = sum(np.sqrt(len(v)) for v in t1_groups.values())
        budgets = {
            s: max(1, int(round(k_tier1 * np.sqrt(len(v)) / sqrt_sum)))
            for s, v in t1_groups.items()
        }
        for s in sorted(t1_groups):
            g_idx = t1_groups[s]
            mask = np.zeros(n, dtype=bool)
            mask[g_idx] = True
            seed = div.centroid_seed_masked(fp_matrix, mask)
            kk = min(budgets[s], len(g_idx))
            for i in div.maxmin_masked(fp_matrix, counts, mask, kk, seed,
                                       max_sim=max_sim, scores=scores):
                selected.append(i)
                tier_of[i] = "tier1"

    # Tier 2: global MaxMin on the merged pool, with the REMAINING diversity
    # budget (unused tier-1 allocation flows here).
    k_tier2 = max(0, budget_div - len(selected))
    t2_idx = [gi for gi, s in zip(pool_idx, pool_scaffold) if s not in t1_set]
    if t2_idx and k_tier2 > 0:
        mask = np.zeros(n, dtype=bool)
        mask[t2_idx] = True
        seed = div.centroid_seed_masked(fp_matrix, mask)
        kk = min(k_tier2, len(t2_idx))
        for i in div.maxmin_masked(fp_matrix, counts, mask, kk, seed,
                                   max_sim=max_sim, scores=scores):
            selected.append(i)
            tier_of[i] = "tier2"

    return selected, scores, tier_of


def run_depth(atom_envs, env2mols, fp_matrix, counts, max_sim,
              selected_set, env_nmols, args):
    """Depth pass. Returns (selected indices, scores = thin-env gain)."""
    n_depth = int(round(args.depth_budget * args.n))
    if n_depth <= 0:
        return [], []

    sel_mult = Counter()
    for i in selected_set:
        for eid in set(e for sub in atom_envs[i] for e in sub):
            sel_mult[eid] += 1
    thin = {eid for eid, c in sel_mult.items() if c < args.target_mult}
    thin |= {eid for eid in env_nmols if eid not in sel_mult}

    depth_sel = []
    depth_scores = []
    while len(depth_sel) < n_depth:
        best = None
        best_gain = 0
        best_sim = np.inf
        for eid in thin:
            for mi in env2mols.get(eid, []):
                if mi in selected_set:
                    continue
                gain = sum(1 for sub in atom_envs[mi] for e in sub if e in thin)
                if gain == 0:
                    continue
                if max_sim[mi] > args.depth_nn_sim:
                    continue
                if gain > best_gain or (gain == best_gain and max_sim[mi] < best_sim):
                    best, best_gain, best_sim = mi, gain, max_sim[mi]
        if best is None:
            break
        selected_set.add(best)
        depth_sel.append(best)
        depth_scores.append(best_gain)
        for eid in set(e for sub in atom_envs[best] for e in sub):
            sel_mult[eid] = sel_mult.get(eid, 0) + 1
            if sel_mult[eid] >= args.target_mult:
                thin.discard(eid)
        sim = div.sim_to_reference(fp_matrix, counts, best)
        np.maximum(max_sim, sim, out=max_sim)
    return depth_sel, depth_scores


def build_selection(df, order_map, pass_map, tier_map, score_map):
    rows = []
    for i in sorted(order_map, key=order_map.get):
        r = df.iloc[i]
        rows.append({
            "CompoundID": r["CompoundID"],
            "smiles": r["smiles"],
            "mol_block": r["mol_block"],
            "pass": pass_map[i],
            "tier": tier_map.get(i, ""),
            "score": score_map.get(i),
            "selection_order": order_map[i],
        })
    return pd.DataFrame(rows, columns=schema.SELECTION_COLUMNS)


def build_coverage_report(atom_envs, elements, env_elem, env_nmols,
                          selected_set):
    sel_envs = set()
    for i in selected_set:
        for sub in atom_envs[i]:
            sel_envs.update(sub)

    # per element
    elem_pool = defaultdict(set)
    for mol_envs, mol_elems in zip(atom_envs, elements):
        for sub, el in zip(mol_envs, mol_elems):
            for eid in sub:
                elem_pool[el].add(eid)
    elem_rows = [
        {"dimension": "element", "key": el,
         "n_envs_pool": len(pool), "n_envs_selected": len(pool & sel_envs),
         "coverage": len(pool & sel_envs) / len(pool) if pool else 0.0}
        for el, pool in sorted(elem_pool.items())
    ]

    # per frequency band
    bands = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"),
             (21, 100, "21-100"), (101, None, ">100")]
    band_rows = []
    for lo, hi, label in bands:
        pool = {eid for eid, c in env_nmols.items()
                if c >= lo and (hi is None or c <= hi)}
        if not pool:
            continue
        sel = pool & sel_envs
        band_rows.append({"dimension": "band", "key": label,
                          "n_envs_pool": len(pool), "n_envs_selected": len(sel),
                          "coverage": len(sel) / len(pool)})
    return pd.DataFrame(elem_rows + band_rows)


def build_thin_coverage(df, atom_envs, env_elem, env_nmols, env2mols,
                        selected_set):
    cids = df["CompoundID"].tolist()
    rows = []
    for eid, c in env_nmols.items():
        if c != 1:
            continue
        carrier = env2mols[eid][0]
        rows.append({
            "env_id": eid,
            "element": env_elem[eid],
            "carrier_id": cids[carrier],
            "covered": carrier in selected_set,
        })
    return pd.DataFrame(rows)


def build_bond_coverage(df, selected_set):
    pool_c = Counter()
    sel_c = Counter()
    for i, bonds in enumerate(df["bonds"].tolist()):
        for b in bonds:
            pool_c[b] += 1
            if i in selected_set:
                sel_c[b] += 1
    rows = [
        {"bond": b, "pool_count": pool_c[b], "selected_count": sel_c.get(b, 0),
         "coverage": sel_c.get(b, 0) / pool_c[b]}
        for b in sorted(pool_c)
    ]
    return pd.DataFrame(rows)


def build_property_js(df, selected_set):
    props = {"molwt": df["molwt"].to_numpy(),
             "tpsa": df["tpsa"].to_numpy(),
             "energy_ha": df["energy_ha"].to_numpy()}
    sel_idx = sorted(selected_set)
    rows = []
    for name, pool in props.items():
        js = cov.js_divergence(pool, pool[sel_idx])
        rows.append({"property": name, "js_divergence": js})
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    enabled = set(p for p in args.passes.split(",") if p)
    for p in enabled:
        if p not in PASSES:
            raise SystemExit(f"unknown pass '{p}' (valid: {', '.join(PASSES)})")

    out_dir = Path(args.output)
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    n = len(df)
    atom_envs = df["atom_envs"].tolist()
    elements = df["heavy_elements"].tolist()

    print(f"[1/5] Loading descriptors ({n:,} molecules) ...")
    env_elem, env_natoms, env_nmols = cov.env_frequency(atom_envs, elements)
    env2mols = cov.env_to_mols(atom_envs)
    fp_matrix = div.onbits_to_matrix(df["morgan_fp"].tolist(), dsc.NBITS, np.float32)
    counts = fp_matrix.sum(axis=1).astype(np.float32)
    max_sim = np.zeros(n, dtype=np.float32)

    floor_sel = []
    floor_scores = []
    div_sel = []
    div_scores = []
    depth_sel = []
    depth_scores = []
    tier_map = {}

    selected_set = set()

    if "floor" in enabled:
        print("[2/5] Floor pass ...")
        floor_sel, floor_scores = run_floor(
            atom_envs, env_nmols, env_natoms, fp_matrix, counts, max_sim, args)
        selected_set |= set(floor_sel)

    if "diversity" in enabled:
        print("[3/5] Diversity pass ...")
        b_depth = int(round(args.depth_budget * args.n))
        budget_div = max(0, args.n - len(selected_set) - b_depth)
        div_sel, div_scores, tier_map = run_diversity(
            df, fp_matrix, counts, max_sim, selected_set, budget_div, args)
        selected_set |= set(div_sel)

    if "depth" in enabled:
        print("[4/5] Depth pass ...")
        depth_sel, depth_scores = run_depth(
            atom_envs, env2mols, fp_matrix, counts, max_sim,
            selected_set, env_nmols, args)
        selected_set |= set(depth_sel)

    # assemble labels
    pass_map = {}
    score_map = {}
    for i, s in zip(floor_sel, floor_scores):
        pass_map[i] = "floor"
        score_map[i] = s
    for i, s in zip(div_sel, div_scores):
        pass_map[i] = "diversity"
        score_map[i] = s
    for i, s in zip(depth_sel, depth_scores):
        pass_map[i] = "depth"
        score_map[i] = s
    order_map = {i: k for k, i in enumerate(selected_set, 1)}

    print("[5/5] Writing selection + reports ...")
    sel = build_selection(df, order_map, pass_map, tier_map, score_map)
    sel.to_parquet(out_dir / "selection.parquet", engine="pyarrow", index=False)

    build_coverage_report(atom_envs, elements, env_elem, env_nmols,
                          selected_set).to_csv(reports_dir / "coverage.csv",
                                               index=False)
    build_thin_coverage(df, atom_envs, env_elem, env_nmols, env2mols,
                        selected_set).to_csv(reports_dir / "thin_coverage.csv",
                                             index=False)
    build_bond_coverage(df, selected_set).to_csv(
        reports_dir / "bond_types.csv", index=False)
    build_property_js(df, selected_set).to_csv(
        reports_dir / "property_js.csv", index=False)

    summary = {
        "n_requested": args.n,
        "n_selected": len(selected_set),
        "n_floor": len(floor_sel),
        "n_diversity": len(div_sel),
        "n_depth": len(depth_sel),
        "passes": sorted(enabled),
        "floor_budget": None if args.floor_k is not None
                        else int(round(args.floor_budget * args.n)),
        "floor_k": args.floor_k,
        "w": args.w,
        "depth_budget": int(round(args.depth_budget * args.n)),
        "target_mult": args.target_mult,
        "tier1_min": args.tier1_min,
        "ff": args.ff,
    }
    (out_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    print(f"Done. Selected {len(selected_set):,} / {args.n:,} "
          f"(floor={len(floor_sel)}, diversity={len(div_sel)}, depth={len(depth_sel)})")
    print(f"  → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
