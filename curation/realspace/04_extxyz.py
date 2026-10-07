#!/usr/bin/env python3
"""04_extxyz.py — Stage 4: realspace delivery files (extXYZ).

Reads curated realspace Parquet batches (default: ``filtered_batches``) and
writes one extended-XYZ file per batch (default: ``extxyz``), geometry
delivered through ``lib/xyz_io.write_xyz_frames`` — the same writer whose
round-trip contract ``lib/xyz_io.read_xyz_frames`` tests pin, so every
delivery file is guaranteed readable by the 01_ingest reader (full-chain
round trip).

Neutral tags (worker plan Rev 1.x, T4 table — minimal set now)::

    geometry_id  charge  multiplicity  elements  source_id  source_file
    source_index

``elements`` carries the Hill formula (composition-group key upstream).
No campaign-specific tags anywhere in the default path (no Family=...,
no energies, no IQARIS manifest fields) — zero campaign knowledge.

Per-atom charges: delivered when the source frame carried them (the
comment line then gains a ``charge:R:1`` column automatically, via
lib/xyz_io). The stored ``header`` column is intentionally NOT forwarded:
delivery emits exactly the neutral tag set above, nothing else.

Extension hook: the IQARIS manifest tag set is gated on an external spec
not in the realspace handoff. When it lands, extend the tag dict below
(or add a ``--family``-style option) — deliberately NOT via
curation/common/10_extxyz.py, whose graph behavior is frozen (planner
ruling, T4).

Rows whose geometry cannot be decoded (should be impossible after
03_filter, but Parquet is a text format) are counted as Failed and
skipped — delivery never crashes on a single bad row (02_identity
error-row precedent). Batches left with zero frames (e.g. ``--exclude``
removed everything) are NOT written (02_identity empty-batch precedent).
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from curation.realspace.schema import row_to_frame
from lib.parquet_io import read_batch
from lib.xyz_io import write_xyz_frames


def _should_exclude(row, rules):
    """Return True if row matches any exclude rule (graph 10_extxyz mirror)."""
    for col, val in rules.items():
        if str(row.get(col, "")) == val:
            return True
    return False


def _delivery_frame(row):
    """Parquet row -> XyzFrame with the neutral realspace tag set."""
    frame = row_to_frame(row)
    frame.header = {
        "geometry_id": str(row.get("geometry_id", "")),
        "charge": str(row.get("charge", 0)),
        "multiplicity": str(row.get("multiplicity", 1)),
        "elements": str(row.get("formula", "")),
        "source_id": str(row.get("source_id", "")),
        "source_file": str(row.get("source_file", "")),
        "source_index": str(row.get("source_index", 0)),
    }
    return frame


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AIM4ML realspace — build extXYZ delivery files.")
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="filtered_batches",
        help="Input Parquet batch directory (default: filtered_batches/).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="extxyz",
        help="Output directory for extXYZ files (default: extxyz/).",
    )
    p.add_argument(
        "--exclude",
        type=str,
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Exclude rows where COLUMN == VALUE (repeatable). "
        "Example: --exclude filter_status=rejected",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from lib.provenance import record_run

    record_run(args.output_dir, "04_extxyz")

    if not os.path.isdir(args.input_dir):
        print(f"Input dir not found: {args.input_dir}")
        sys.exit(1)
    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    exclude_rules = {}
    for rule in args.exclude:
        if "=" in rule:
            col, val = rule.split("=", 1)
            exclude_rules[col.strip()] = val.strip()
    if exclude_rules:
        print(f"Excluding rows where: {exclude_rules}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    total_frames = 0
    total_excluded = 0
    total_failed = 0
    total_files = 0

    for fname in batch_files:
        in_path = os.path.join(args.input_dir, fname)
        batch = read_batch(in_path)

        frames = []
        for row in batch:
            if _should_exclude(row, exclude_rules):
                total_excluded += 1
                continue
            try:
                frames.append(_delivery_frame(row))
            except (KeyError, TypeError, ValueError) as exc:
                total_failed += 1
                sid = row.get("source_id", "?")
                print(f"  Failed to decode {sid}: {type(exc).__name__}: {exc}")

        if not frames:
            continue  # empty delivery file would be schema noise; skip (02_identity precedent)

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output_dir, f"{base}.extxyz")
        tmp = out_path + ".tmp." + str(os.getpid())
        write_xyz_frames(tmp, frames)
        os.replace(tmp, out_path)
        total_files += 1
        total_frames += len(frames)

    print("\nReport")
    print(f"  Input rows:   {total_frames + total_excluded + total_failed}")
    print(f"  Frames:       {total_frames}")
    print(f"  Files:        {total_files}")
    print(f"  Excluded:     {total_excluded}")
    print(f"  Failed:       {total_failed}")
    print(f"  Output dir:   {args.output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
