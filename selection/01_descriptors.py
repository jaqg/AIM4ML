#!/usr/bin/env python3
"""
01_descriptors.py — curated Parquet → descriptors.parquet (stage 1).

Reads the curated pipeline output (stage 8 ``conformer_batches``, batched
Parquet), dedups to unique CompoundID, computes selection descriptors from
each ``mol_block`` → RDKit Mol, and writes a single ``descriptors.parquet``.

Why mol_block (not SMILES): the explicit-H canonical SMILES re-sanitizes
~0.5% of QM40 (nitro / N-oxide / quaternary-N valence); the curated mol_block
carries the resolved topology + bond orders.

Usage:
    python3 01_descriptors.py -i <curated_dir> -o <out_dir> [--metric morgan|soap]

--metric soap is reserved for the D21 experiment (blocked on DScribe/ASE).
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from rdkit import Chem

from curation.lib.parquet_io import read_batch
from selection.lib import descriptors as dsc
from selection.lib import schema


def _as_str(v):
    """pandas/parquet may return None or float NaN for missing strings."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)


def _as_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Rare elements whose loss to a skip could open a coverage hole in the
# rare-env floor (§3.1).  CHNO are common enough that a single skip is
# negligible; these are the elements the floor exists to protect.
RARE_ELEMENTS = {"S", "F", "Cl", "P", "Si", "Br", "I"}


def skipped_elements(mol_block):
    """Heavy elements of a molecule that failed full sanitization.

    Fallback: parse with sanitize=False (topology only) — enough to recover
    element symbols even when valence checks fail.  Returns a sorted list.
    """
    try:
        m = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=True)
        if m is None:
            return []
        # removeHs needs sanitization to work reliably; filter H explicitly.
        return sorted({a.GetSymbol() for a in m.GetAtoms()
                       if a.GetSymbol() != "H"})
    except Exception:
        return []


def compute_row(r):
    """Descriptor dict for one curated row; None if mol unparseable."""
    mol = dsc.parse_mol(r.get("mol_block"))
    if mol is None:
        return None

    smiles = _as_str(r.get("CanonicalSMILES")) or _as_str(r.get("SMILES"))
    molwt, tpsa = dsc.molwt_tpsa(mol)
    return {
        "CompoundID": _as_str(r.get("CompoundID")),
        "smiles": smiles,
        "mol_block": r.get("mol_block"),
        "formula": _as_str(r.get("Formula")),
        "nat_heavy": int(mol.GetNumAtoms()),
        "molwt": float(molwt),
        "tpsa": float(tpsa),
        "energy_ha": _as_float(r.get("Energy_Ha")),
        "scaffold": dsc.scaffold_smiles(mol),
        "morgan_fp": dsc.morgan_onbits(mol),
        "atom_envs": dsc.atom_envs(mol),
        "heavy_elements": dsc.heavy_elements(mol),
        "bonds": dsc.bond_strings(mol),
    }


def parse_allowed_elements(spec):
    """Parse a --allowed-elements value into a validated symbol set.

    Case-insensitive, whitespace-tolerant, deduplicated.  Unknown symbols
    raise ValueError (typo'd scope must fail loudly, not filter silently).
    Heavy elements only — H never appears in ``heavy_elements``.
    """
    symbols = {tok.strip().capitalize() for tok in spec.split(",")}
    symbols.discard("")
    if not symbols:
        raise ValueError("empty --allowed-elements spec")
    pt = Chem.GetPeriodicTable()
    for s in sorted(symbols):
        try:
            pt.GetAtomicNumber(s)
        except Exception:
            raise ValueError(f"unknown element symbol in --allowed-elements: {s!r}")
    return symbols


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True,
                   help="Directory of curated Parquet batches.")
    p.add_argument("-o", "--output", required=True,
                   help="Output directory (descriptors.parquet written here).")
    p.add_argument("--metric", choices=["morgan", "soap"], default="morgan",
                   help="Descriptor metric. 'soap' is reserved (D21, needs DScribe).")
    p.add_argument("--allowed-elements", default=None, metavar="SYMS",
                   help="Heavy-element whitelist, e.g. 'C,N,O,S'. Molecules "
                        "carrying any element outside the list are excluded "
                        "before the env histogram (D64 scope). Default: no "
                        "element filter.")
    p.add_argument("--min-atoms", type=int, default=None, metavar="N",
                   help="Minimum heavy-atom count, inclusive (all heavy atoms, "
                        "no element split). Default: no lower bound.")
    p.add_argument("--max-atoms", type=int, default=None, metavar="N",
                   help="Maximum heavy-atom count, inclusive. Default: no upper bound.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.metric == "soap":
        raise SystemExit(
            "SOAP metric not implemented (D21 trial blocked on DScribe/ASE install). "
            "Use --metric morgan."
        )

    if (args.min_atoms is not None and args.max_atoms is not None
            and args.min_atoms > args.max_atoms):
        raise SystemExit(
            f"error: --min-atoms ({args.min_atoms}) > --max-atoms "
            f"({args.max_atoms})")

    allowed = None
    if args.allowed_elements is not None:
        try:
            allowed = parse_allowed_elements(args.allowed_elements)
        except ValueError as e:
            raise SystemExit(f"error: {e}")

    files = sorted(Path(args.input).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no .parquet files in {args.input}")

    rows = []
    skipped = []
    n_total = 0
    n_dup = 0
    n_skipped = 0
    n_excl_elem = 0
    n_excl_size = 0
    excl_elem_counts = {}  # offending symbol -> n molecules
    seen = set()
    for f in files:
        for r in read_batch(str(f)):
            n_total += 1
            cid = _as_str(r.get("CompoundID"))
            key = cid if cid is not None else f"<no-cid-{n_total}>"
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            d = compute_row(r)
            if d is None:
                n_skipped += 1
                elems = skipped_elements(r.get("mol_block"))
                skipped.append({
                    "CompoundID": _as_str(r.get("CompoundID")),
                    "formula": _as_str(r.get("Formula")),
                    "elements": ",".join(elems),
                    "rare_elements": ",".join(sorted(set(elems) & RARE_ELEMENTS)),
                })
                continue
            # D64 scope filter — element reason takes precedence over size.
            if allowed is not None:
                outside = {e for e in d["heavy_elements"] if e not in allowed}
                if outside:
                    n_excl_elem += 1
                    for e in outside:
                        excl_elem_counts[e] = excl_elem_counts.get(e, 0) + 1
                    continue
            nat = d["nat_heavy"]
            if (args.min_atoms is not None and nat < args.min_atoms) or \
               (args.max_atoms is not None and nat > args.max_atoms):
                n_excl_size += 1
                continue
            rows.append(d)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "descriptors.parquet"
    df = pd.DataFrame(rows, columns=schema.DESCRIPTOR_COLUMNS)
    df.to_parquet(out_path, engine="pyarrow", index=False)

    print(f"  input rows: {n_total:,}  (dedup removed {n_dup:,})")
    print(f"  descriptors: {len(rows):,}  (skipped {n_skipped:,} unparseable)")

    # Element histogram of the INCLUDED pool (what stages 2-3 actually see).
    if rows:
        elem_n = {}
        for d in rows:
            for e in set(d["heavy_elements"]):
                elem_n[e] = elem_n.get(e, 0) + 1
        hist = ", ".join(
            f"{e}={elem_n[e]}/{len(rows)} ({100 * elem_n[e] / len(rows):.0f}%)"
            for e in sorted(elem_n))
        print(f"  elements: {hist}")

    # Scope-exclusion breakdown (D64) — element reason first, then size.
    if n_excl_elem:
        per = ", ".join(f"{e}={n}" for e, n in
                        sorted(excl_elem_counts.items()))
        print(f"  excluded by element: {n_excl_elem:,}  ({per})")
    if n_excl_size:
        bounds = []
        if args.min_atoms is not None:
            bounds.append(f"min={args.min_atoms}")
        if args.max_atoms is not None:
            bounds.append(f"max={args.max_atoms}")
        print(f"  excluded by size: {n_excl_size:,}  ({', '.join(bounds)})")

    if skipped:
        skips_df = pd.DataFrame(skipped)
        skips_df.to_csv(out_dir / "skipped.csv", index=False)
        risky = skips_df[skips_df["rare_elements"] != ""]
        print(f"  → skipped.csv written ({len(skipped)} rows)")
        if len(risky):
            print(f"  ⚠ {len(risky)} skipped molecule(s) carry rare elements "
                  f"({', '.join(sorted(RARE_ELEMENTS))}) — possible env "
                  f"coverage hole; check before trusting the rare-env floor:")
            for _, row in risky.iterrows():
                print(f"      {row['CompoundID']}  {row['formula']}  "
                      f"rare={row['rare_elements']}")
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
