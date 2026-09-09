#!/usr/bin/env python3
"""
verify_twins.py — Check whether T=1.0 Tanimoto pairs are true structural
duplicates or ECFP4-blind stereoisomers.

ECFP4 ignores chirality (@ / @@ markers). Two enantiomers with different
canonical SMILES (different CompoundID) can have identical ECFP4 fingerprints
→ T=1.0 despite being genuinely distinct compounds in the curation.

Usage:
    python3 tools/verify_twins.py stats/stats_summary.csv [--max-pairs 20]
"""

import argparse
import pandas as pd
from rdkit import Chem
import re


def strip_chirality(smi):
    """Remove chirality markers (@ / @@) from a SMILES string."""
    return re.sub(r'@+', '', smi)


def are_stereoisomers(smi_a, smi_b):
    """True if SMILES differ only in chirality markers."""
    return strip_chirality(smi_a) == strip_chirality(smi_b)


def are_tautomers(smi_a, smi_b):
    """Heuristic: same heavy-atom graph but different H positions.
       Check if MolToSmiles after RemoveHs + sanitize produces same string."""
    mol_a = Chem.MolFromSmiles(smi_a, sanitize=False)
    mol_b = Chem.MolFromSmiles(smi_b, sanitize=False)
    if mol_a is None or mol_b is None:
        return False
    try:
        Chem.SanitizeMol(mol_a)
        Chem.SanitizeMol(mol_b)
        a_no_h = Chem.RemoveHs(mol_a)
        b_no_h = Chem.RemoveHs(mol_b)
        return Chem.MolToSmiles(a_no_h, isomericSmiles=False) == \
               Chem.MolToSmiles(b_no_h, isomericSmiles=False)
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(
        description="Verify T=1.0 Tanimoto pairs are structural duplicates or stereoisomers."
    )
    p.add_argument("csv", type=str, help="stats_summary.csv path.")
    p.add_argument("--max-pairs", type=int, default=20,
                   help="Max pairs to print details for (default: 20).")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    # Ensure we have needed columns
    needed = {"CompoundID", "CanonicalSMILES", "max_tanimoto"}
    missing = needed - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return

    # Only look at unique CompoundIDs (Tanimoto computed on deduped set)
    df = df.drop_duplicates(subset="CompoundID")

    # T=1.0 pairs: we need pairwise comparison. The max_tanimoto column
    # gives nearest-neighbour for each row. We can't reconstruct pairs from
    # just the max score — we need the actual pairwise matrix.
    #
    # Instead: find rows with max_tanimoto == 1.0. The partner is the nearest
    # neighbour. We can't identify the exact partner without the full matrix,
    # but we can check the CanonicalSMILES of T=1.0 compounds.

    twins = df[df["max_tanimoto"] == 1.0].copy()

    if len(twins) == 0:
        print("No T=1.0 pairs found.")
        return

    print(f"T=1.0 rows (unique CompoundIDs): {len(twins)}")
    print()

    # Check: if all T=1.0 compounds have the SAME canonical SMILES → duplicate
    # Actually for ECFP4, T=1.0 can happen for enantiomer pairs.
    # We need to know which pairs form T=1.0.
    #
    # Strategy: group by heavy-atom skeleton (CanonicalSMILES without @).
    # Two compounds with identical skeleton but different chirality → likely
    # the T=1.0 pair.

    twins["skeleton"] = twins["CanonicalSMILES"].apply(strip_chirality)
    skeleton_groups = twins.groupby("skeleton")

    n_stereo = 0
    n_identical = 0
    n_other = 0
    printed = 0

    for skeleton, group in skeleton_groups:
        if len(group) < 2:
            n_other += 1
            continue

        smis = group["CanonicalSMILES"].unique()
        if len(smis) == 1:
            # Same CompoundID appearing multiple times (shouldn't happen after dedup)
            n_identical += 1
            continue

        # Multiple different SMILES with same skeleton → stereoisomer pair
        n_stereo += 1

        if printed < args.max_pairs:
            print(f"  Group size: {len(group)}, Unique SMILES: {len(smis)}")
            for smi in smis[:4]:
                print(f"    {smi}")
            print()
            printed += 1

    print("─" * 60)
    print("Summary")
    print(f"  Stereoisomer groups (same skeleton, different @/@): {n_stereo}")
    print(f"  Identical SMILES groups (true duplicates):        {n_identical}")
    print(f"  Singletons (no partner in T=1.0 set):            {n_other}")
    print(f"  Total T=1.0 compounds:                           {len(twins)}")

    if n_stereo > 0:
        n_pairs = sum(len(g) for _, g in skeleton_groups if len(g) >= 2 and len(g["CanonicalSMILES"].unique()) > 1)
        print(f"\n  → {n_pairs} of {len(twins)} T=1.0 compounds ({100*n_pairs/len(twins):.1f}%)")
        print(f"    are stereoisomers indistinguishable by ECFP4.")
        print(f"    These are GENUINELY different compounds — not curation errors.")

    # Also check: are there pairs where skeleton differs but T=1.0?
    # That would indicate a real problem (different connectivity → same FP).
    # To find these, we'd need the full pairwise matrix. For now, skip.


if __name__ == "__main__":
    main()
