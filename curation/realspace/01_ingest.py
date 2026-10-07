#!/usr/bin/env python3
"""
01_ingest.py — Realspace stage 1: xyz/extxyz dir -> Parquet batches.

Flat input directory + glob pattern; per-source knowledge lives upstream in
the converters (handoff T4). Deterministic order: files sorted by name,
frames in file order. Writes ALL T2 columns (worker plan Rev 1):

    geometry_id, symbols, n_atoms, coords, charges, formula, charge,
    multiplicity, source_id, source_file, source_index, header

Contracts:
- source_id   = "<basename>:<frame_index>" (basename = file name incl.
                extension, e.g. "heh.xyz:0"); source_file/source_index are
                kept separately for provenance queries.
- charge / multiplicity: the xyz reader invents NOTHING (missing stays
  missing); ingest fills the explicit defaults charge=0, multiplicity=1
  when the extxyz header lacks them, so the Parquet columns are always
  populated. Header values must parse as integers — otherwise the frame is
  rejected with a reason.
- geometry_id = geometry_key(symbols, coords, charge, multiplicity) (P3
  identity). Frames failing identity validation (e.g. unknown element
  symbol) are rejected with a reason.
- formula     = Hill-system formula derived at ingest (schema.hill_formula).

Rejects (mirrors graph-track convention):
- The xyz reader is STRICT: one malformed frame fails the whole file
  (read_xyz_frames raises XyzFormatError). Rejection granularity is
  therefore FILE-level for parse failures (reason carries the file:line
  context) and FRAME-level for semantic failures (bad header integers,
  unknown element). Rejects are written as a Parquet batch
  (<rejects-dir>/ingest_rejected.parquet) with source_id + reason columns,
  plus a .REJECTED sentinel line when anything was rejected.

Usage:
    python 01_ingest.py INPUT_DIR -o batches/ --pattern "*.xyz" --batch-size 5000
"""

import argparse
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from curation.realspace.identity import geometry_key
from curation.realspace.schema import frame_to_row, write_realspace_batch
from lib.xyz_io import XyzFormatError, read_xyz_frames


def _header_int(header, key, default):
    """Header value -> int; (value, None) or (None, reason)."""
    if key not in header:
        return default, None
    try:
        val = float(header[key])
    except ValueError:
        return None, f"non-numeric {key} in header: {header[key]!r}"
    if not val.is_integer():
        return None, f"non-integer {key} in header: {header[key]!r}"
    return int(val), None


def _frames_to_rows(frames, basename):
    """Frames of one file -> (rows, rejects). Frame-level semantic checks."""
    rows = []
    rejects = []
    for idx, frame in enumerate(frames):
        source_id = f"{basename}:{idx}"
        charge, reason = _header_int(frame.header, "charge", 0)
        multiplicity, m_reason = _header_int(frame.header, "multiplicity", 1)
        if reason is None:
            reason = m_reason
        if reason is not None:
            rejects.append(
                {
                    "source_id": source_id,
                    "source_file": basename,
                    "source_index": idx,
                    "reason": reason,
                }
            )
            continue
        try:
            gid = geometry_key(frame.symbols, frame.coords, charge, multiplicity)
        except ValueError as exc:
            rejects.append(
                {
                    "source_id": source_id,
                    "source_file": basename,
                    "source_index": idx,
                    "reason": str(exc),
                }
            )
            continue
        rows.append(
            frame_to_row(
                frame,
                geometry_id=gid,
                charge=charge,
                multiplicity=multiplicity,
                source_id=source_id,
                source_file=basename,
                source_index=idx,
            )
        )
    return rows, rejects


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AIM4ML realspace 01_ingest — xyz dir to Parquet batches."
    )
    p.add_argument("input_dir", type=str, help="Flat directory of xyz/extxyz files.")
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="batches",
        help="Directory for Parquet batch files (default: batches/).",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*.xyz",
        help="Glob pattern within input_dir (default: *.xyz).",
    )
    p.add_argument(
        "-b", "--batch-size", type=int, default=5000, help="Frames per batch (default: 5000)."
    )
    p.add_argument(
        "--rejects-dir",
        type=str,
        default="rejects/01_ingest",
        help="Rejects directory (default: rejects/01_ingest/).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from lib.provenance import record_run

    record_run(args.output_dir, "01_ingest")
    os.makedirs(args.output_dir, exist_ok=True)

    paths = sorted(
        p for p in glob.glob(os.path.join(args.input_dir, args.pattern)) if os.path.isfile(p)
    )

    all_rows = []
    all_rejects = []
    n_frames_seen = 0
    for path in paths:
        basename = os.path.basename(path)
        try:
            frames = read_xyz_frames(path)
        except XyzFormatError as exc:
            all_rejects.append(
                {
                    "source_id": basename,
                    "source_file": basename,
                    "source_index": None,
                    "reason": str(exc),
                }
            )
            continue
        n_frames_seen += len(frames)
        rows, rejects = _frames_to_rows(frames, basename)
        all_rows.extend(rows)
        all_rejects.extend(rejects)

    # -- Split into fixed-size batches (deterministic order) ---------------
    n_batches = 0
    written = 0
    batch_rows = []
    for row in all_rows:
        batch_rows.append(row)
        if len(batch_rows) >= args.batch_size:
            path = os.path.join(args.output_dir, f"batch_{n_batches:04d}.parquet")
            write_realspace_batch(path, batch_rows)
            print(f"  Wrote {path} ({len(batch_rows)} frames)")
            written += len(batch_rows)
            batch_rows = []
            n_batches += 1
    if batch_rows:
        path = os.path.join(args.output_dir, f"batch_{n_batches:04d}.parquet")
        write_realspace_batch(path, batch_rows)
        print(f"  Wrote {path} ({len(batch_rows)} frames)")
        written += len(batch_rows)

    # -- Rejects -----------------------------------------------------------
    if all_rejects:
        reject_path = os.path.join(args.rejects_dir, "ingest_rejected.parquet")
        write_realspace_batch(reject_path, all_rejects)
        print(f"  {len(all_rejects)} rejected -> {reject_path}")
        sentinel = os.path.join(args.rejects_dir, ".REJECTED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{len(all_rejects)} frames rejected\n")

    # -- Report ------------------------------------------------------------
    print("\nReport")
    print(f"  Input dir:        {args.input_dir}")
    print(f"  Files scanned:    {len(paths)}")
    print(f"  Frames read:      {n_frames_seen}")
    print(f"  Rows written:     {written}")
    print(f"  Rejected:         {len(all_rejects)}")
    print(f"  Batches:          {n_batches}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Output dir:       {args.output_dir}")
    if not paths:
        print(f"  WARNING: no files matched {args.pattern!r} in {args.input_dir}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
