#!/usr/bin/env python3
"""
02_identity.py — Realspace stage 2: geometry_key dedup + near-dup clustering.

Operates on parquet batches from 01_ingest. Two dedup classes (worker plan
Rev 1, T4), both keep-first under deterministic order (batch files sorted,
rows in file order):

1. EXACT dups — same geometry_key = bit-identical registry entry (re-
   delivery of the same geometry + electronic state). First occurrence
   kept, the rest dropped to <rejects-dir>/identity_rejected.parquet +
   .REJECTED sentinel (dedup_status="exact_dup").

   IDENTITY INVERSION vs graph track (documented, not a bug): graph 04
   KEEPS exact dups because same SMILES = distinct conformers of one
   compound worth retaining; realspace DROPS them because the same
   geometry_key IS the same registry entry (registry-literal reading) —
   a re-delivery carries no new information.

2. NEAR dups — same composition group (Hill formula + charge +
   multiplicity = registry identity minus geometry), raw Kabsch RMSD
   (conformer_rmsd, curation/realspace/identity.py) < --near-dup-threshold
   (default 0.25 A). Greedy keep-first clustering: each row is compared
   against the group's representatives so far; below-threshold to ANY
   representative → dropped to <rejects-dir>/identity_removed.parquet +
   .REMOVED sentinel (dedup_status="near_dup"), cluster_id = the NEAREST
   kept representative's id; otherwise the row becomes a new
   representative. RMSD limits (greedy upper bound, proper rotations
   only, raw mode) are documented in identity.py — a sub-threshold value
   proves near-dup; false negatives degrade gracefully (both kept).

Canonical columns: geometry_id, formula and the composition key are
RECOMPUTED from row geometry on load (row_to_frame + geometry_key +
hill_formula). The stored columns are ingest provenance, not authority;
mismatches are counted and reported (non-fatal — canonical values are
written to the output batches).

cluster_id semantics (mirror graph 08): kept rows carry their own id,
dropped rows carry the id of the nearest kept representative. Numbering
is GLOBAL and monotonic in encounter order (graph 08 numbers per
CompoundID group; realspace has no scoping column, so registry-wide
unique ids are strictly more informative — same semantics, wider scope).

Near-dup is nuclear-geometry-only: per-atom charges do NOT enter the
decision (conformer_rmsd takes symbols + coords). Total charge and
multiplicity enter via the composition key — two rows with different
total charge are never compared. Per-atom-charge-aware near-dup =
documented extension.

--force-keep-rejected retains BOTH dropped classes (and error rows) in
the output batches with dedup_status/cluster_id recorded; rejects
parquet + sentinels are written regardless (audit trail, mirror
graph 04/08).

Representative = keep-first by deterministic order; energy-aware
selection is a documented extension (xyz headers do not guarantee
energies — see identity.py). Clustering compares only within composition
groups (graph-08 complexity profile); RMSD cost is O(k^2) per group of
k members, 0 for single-member groups.

Rows whose geometry cannot be decoded (corrupt coords/symbols/charge)
are rejected with dedup_status="error" and a reason — the stage does not
crash on a bad row (mirror graph 04 mol_corrupt handling).

Usage:
    python 02_identity.py -i batches/ -o deduped_batches/ \\
        --rejects-dir rejects/02_identity [--near-dup-threshold 0.25] \\
        [--force-keep-rejected]
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from curation.realspace.identity import NEAR_DUP_ANGSTROM, conformer_rmsd, geometry_key
from curation.realspace.schema import (
    hill_formula,
    row_to_frame,
    write_realspace_batch,
)
from lib.parquet_io import read_batch

THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 5.0


def _canonical(row):
    """(geometry_key, hill_formula) recomputed from row geometry.

    Raises (TypeError, ValueError) on undecodable rows (corrupt coords,
    missing columns, non-integer charge) — caller records a reject.
    """
    frame = row_to_frame(row)
    charge = int(row["charge"])
    multiplicity = int(row["multiplicity"])
    key = geometry_key(frame.symbols, frame.coords, charge, multiplicity)
    return key, hill_formula(frame.symbols)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AIM4ML realspace 02_identity — geometry_key dedup + near-dup clustering."
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="batches",
        help="Input Parquet batch directory (default: batches/).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="deduped_batches",
        help="Output directory (default: deduped_batches/).",
    )
    p.add_argument(
        "--rejects-dir",
        type=str,
        default="rejects/02_identity",
        help="Rejects directory (default: rejects/02_identity/).",
    )
    p.add_argument(
        "--force-keep-rejected",
        action="store_true",
        help="Keep rejected rows (exact_dup / near_dup / error) in the output "
        "Parquet batches with dedup_status recorded (default: drop them).",
    )
    p.add_argument(
        "--near-dup-threshold",
        type=float,
        default=NEAR_DUP_ANGSTROM,
        help=f"Near-dup RMSD threshold in Angstrom (default: {NEAR_DUP_ANGSTROM}, "
        f"range [{THRESHOLD_MIN}, {THRESHOLD_MAX}]).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from lib.provenance import record_run

    record_run(args.output_dir, "02_identity")

    if not (THRESHOLD_MIN <= args.near_dup_threshold <= THRESHOLD_MAX):
        print(
            f"Error: --near-dup-threshold must be in range "
            f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}], got {args.near_dup_threshold}",
            file=sys.stderr,
        )
        sys.exit(1)

    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Near-dup threshold: {args.near_dup_threshold} A")
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    # -- Load + canonicalize (recompute key/formula from row geometry) ----
    entries = []
    n_key_mismatch = 0
    n_formula_mismatch = 0
    for fname in batch_files:
        for row in read_batch(os.path.join(args.input_dir, fname)):
            e = {
                "fname": fname,
                "row": row,
                "key": None,
                "formula": None,
                "group": None,
                "status": None,
                "cluster_id": None,
                "reason": None,
                "rmsd": None,
            }
            try:
                key, formula = _canonical(row)
            except (KeyError, TypeError, ValueError) as exc:
                e["status"] = "error"
                e["reason"] = f"row decode failed: {exc}"
                entries.append(e)
                continue
            if row.get("geometry_id") not in (None, key):
                n_key_mismatch += 1
            if row.get("formula") not in (None, formula):
                n_formula_mismatch += 1
            row["geometry_id"] = key
            row["formula"] = formula
            e["key"] = key
            e["formula"] = formula
            e["group"] = (formula, int(row["charge"]), int(row["multiplicity"]))
            entries.append(e)

    if n_key_mismatch or n_formula_mismatch:
        print(
            f"  WARNING: stored columns differed from recomputed canonical values "
            f"for {n_key_mismatch} geometry_id / {n_formula_mismatch} formula rows "
            f"(canonical values written)"
        )

    # -- Pass 1: exact dedup (keep-first) ---------------------------------
    seen = {}  # geometry_id -> first entry with that key
    survivors = []
    n_exact = 0
    for e in entries:
        if e["status"] == "error":
            continue
        first = seen.get(e["key"])
        if first is None:
            seen[e["key"]] = e
            survivors.append(e)
        else:
            e["status"] = "exact_dup"
            e["_rep"] = first
            e["reason"] = f"duplicate geometry_id {e['key']} (first: {first['row']['source_id']})"
            n_exact += 1

    # -- Pass 2: near-dup clustering within composition groups -------------
    # Greedy keep-first; representatives compared pairwise (graph-08 profile).
    groups = {}
    for e in survivors:
        groups.setdefault(e["group"], []).append(e)

    next_cluster_id = 1
    n_near = 0
    n_kept = 0
    for group in groups.values():  # dict insertion order = deterministic
        if len(group) == 1:
            e = group[0]
            e["status"] = "kept"
            e["cluster_id"] = next_cluster_id
            next_cluster_id += 1
            n_kept += 1
            continue
        reps = []
        for e in group:
            frame = row_to_frame(e["row"])
            best = None  # (rmsd, rep_entry) — nearest kept representative
            for rep in reps:
                d = conformer_rmsd(frame.symbols, frame.coords, rep["_syms"], rep["_coords"])
                if best is None or d < best[0]:
                    best = (d, rep)
            if best is not None and best[0] < args.near_dup_threshold:
                e["status"] = "near_dup"
                e["cluster_id"] = best[1]["cluster_id"]
                e["rmsd"] = best[0]
                e["reason"] = (
                    f"near-dup of {best[1]['row']['source_id']} "
                    f"(RMSD {best[0]:.4f} A < {args.near_dup_threshold} A)"
                )
                n_near += 1
            else:
                e["status"] = "kept"
                e["cluster_id"] = next_cluster_id
                next_cluster_id += 1
                e["_syms"] = frame.symbols
                e["_coords"] = frame.coords
                reps.append(e)
                n_kept += 1

    # -- Pass 3: exact-dup rows inherit their representative's cluster_id -
    for e in entries:
        if e["status"] == "exact_dup":
            e["cluster_id"] = e["_rep"]["cluster_id"]

    # -- Write output batches (same filenames; dropped rows filtered) -----
    written = {}
    for e in entries:
        row = e["row"]
        row["dedup_status"] = e["status"]
        row["cluster_id"] = e["cluster_id"]
        if e["status"] == "kept" or args.force_keep_rejected:
            written.setdefault(e["fname"], []).append(row)

    n_written = 0
    for fname in batch_files:
        rows = written.get(fname, [])
        if not rows:
            continue  # empty batches are not written (01_ingest precedent)
        out_path = os.path.join(args.output_dir, fname)
        write_realspace_batch(out_path, rows)
        print(f"  Wrote {out_path} ({len(rows)} frames)")
        n_written += len(rows)

    # -- Rejects: exact (+error) and near classes, own parquet + sentinel --
    exact_rows = []
    near_rows = []
    for e in entries:
        if e["status"] in ("exact_dup", "error"):
            row = e["row"]
            row["reason"] = e["reason"]
            exact_rows.append(row)
        elif e["status"] == "near_dup":
            row = e["row"]
            row["reason"] = e["reason"]
            near_rows.append(row)

    if exact_rows:
        reject_path = os.path.join(args.rejects_dir, "identity_rejected.parquet")
        write_realspace_batch(reject_path, exact_rows)
        print(f"  {len(exact_rows)} exact_dup/error rows -> {reject_path}")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(os.path.join(args.rejects_dir, ".REJECTED"), "w") as f:
            f.write(f"{len(exact_rows)} frames rejected\n")

    if near_rows:
        remove_path = os.path.join(args.rejects_dir, "identity_removed.parquet")
        write_realspace_batch(remove_path, near_rows)
        print(f"  {len(near_rows)} near_dup rows -> {remove_path}")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(os.path.join(args.rejects_dir, ".REMOVED"), "w") as f:
            f.write(f"{len(near_rows)} frames removed\n")

    # -- Report ------------------------------------------------------------
    n_error = sum(1 for e in entries if e["status"] == "error")
    print("\nReport")
    print(f"  Total rows:              {len(entries)}")
    print(f"  Kept:                    {n_kept}")
    print(f"  Exact duplicates:        {n_exact}")
    print(f"  Near duplicates:         {n_near}")
    print(f"  Errors:                  {n_error}")
    print(f"  Composition groups:      {len(groups)}")
    print(f"  Near-dup threshold:      {args.near_dup_threshold} A")
    print(f"  Written to output:       {n_written}")
    if not args.force_keep_rejected and (n_exact + n_near + n_error):
        print(
            f"  Dropped from output:     {n_exact + n_near + n_error} "
            f"(use --force-keep-rejected to keep)"
        )


if __name__ == "__main__":
    raise SystemExit(main())
