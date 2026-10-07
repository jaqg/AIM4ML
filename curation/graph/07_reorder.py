#!/usr/bin/env python3
"""
07_reorder.py — Stage 7: Canonical atom reordering.

Reorders atoms in mol_block via RDKit CanonicalRankAtoms (portable,
no external dependencies).  Updates mol_block in-place and adds a
'reorder_status' column ("ok" / "failed").

Usage:
    python3 07_reorder.py -i stereo_batches/ -o reordered_batches/
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

from lib.parquet_io import read_batch, write_batch
from lib.sdf_io import write_reject_sdf

# -- RDKit backend --------------------------------------------------------


def reorder_rdkit(mol_block):
    """Canonical rank atoms via RDKit, renumber, return new mol_block."""
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    # Compute implicit valence before ranking: unsanitized mols raise
    # "getNumImplicitHs() called without preceding call to calcImplicitValence()"
    # inside CanonicalRankAtoms for nitrile/imine N with explicit-H mol_blocks.
    mol.UpdatePropertyCache(strict=False)
    # Chem.CanonicalRankAtoms: include isotopes, break ties with coords
    ranking = list(
        Chem.CanonicalRankAtoms(
            mol,
            breakTies=True,
            includeChirality=True,
            includeIsotopes=True,
        )
    )
    mol_reordered = Chem.RenumberAtoms(mol, ranking)
    block = Chem.MolToMolBlock(mol_reordered)
    m_end = block.index("M  END") + len("M  END")
    return block[:m_end]


# -- Main ----------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AIM4ML Stage 7 — Canonical atom reordering.")
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="stereo_batches",
        help="Input Parquet batch directory (default: stereo_batches/).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="reordered_batches",
        help="Output directory (default: reordered_batches/).",
    )
    p.add_argument(
        "--rejects-dir",
        type=str,
        default="rejects/07_reorder",
        help="Rejected molecules SDF (default: rejects/07_reorder/).",
    )
    p.add_argument(
        "--workers", type=int, default=1, help="Parallel workers for reordering (default: 1)."
    )
    p.add_argument(
        "--force-keep-rejected",
        action="store_true",
        help="Keep molecules that fail reordering (default: drop them).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from lib.provenance import record_run

    record_run(args.output_dir, "07_reorder")

    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    reorder_fn = reorder_rdkit
    total_ok = 0
    total_failed = 0
    failed_rows = []

    # Worker function for parallel reordering
    def _reorder_row(row):
        new_block = reorder_fn(row["mol_block"])
        if new_block is None:
            row["reorder_status"] = "failed"
        else:
            row["mol_block"] = new_block
            row["reorder_status"] = "ok"
        return row

    for fname in batch_files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        batch = read_batch(in_path)

        if args.workers > 1:
            out_rows = []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_reorder_row, row) for row in batch]
                for future in as_completed(futures):
                    out_rows.append(future.result())
        else:
            out_rows = [_reorder_row(row) for row in batch]

        for row in out_rows:
            if row["reorder_status"] == "ok":
                total_ok += 1
            else:
                total_failed += 1
                failed_rows.append(row)

        if not args.force_keep_rejected:
            out_rows = [r for r in out_rows if r["reorder_status"] == "ok"]

        write_batch(out_path, out_rows)

    # -- Reject SDF -------------------------------------------------------
    if failed_rows:
        reject_path = os.path.join(args.rejects_dir, "reorder_failed.sdf")
        write_reject_sdf(reject_path, failed_rows, reject_reason="reorder_failed")
        print(f"  {len(failed_rows)} failed → {reject_path}")

    # -- Report ------------------------------------------------------------
    total = total_ok + total_failed
    print("\nReport")
    print(f"  Total:   {total}")
    print(f"  OK:      {total_ok}")
    print(f"  Failed:  {total_failed}")

    if total_failed:
        sentinel = os.path.join(args.rejects_dir, ".FAILED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{total_failed} molecules failed\n")
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
