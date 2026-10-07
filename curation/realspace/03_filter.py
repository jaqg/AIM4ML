#!/usr/bin/env python3
"""
03_filter.py — Realspace stage 3: generic property filter.

Filters parquet batches from 02_identity on cheap registry properties —
no chemistry perception (realspace rows carry geometry + state labels,
not perceived bonds; graph 03_filter's SMILES/mol_block machinery does
not apply here). Four checks, first failing reason wins:

  1. elements whitelist — every symbol in the frame must be in --elements
  2. atom count         — n_atoms in [--min-atoms, --max-atoms]
  3. total charge       — charge in [--charge-range MIN:MAX]
  4. multiplicity       — multiplicity in [--multiplicity-range MIN:MAX]

WHITELIST SEMANTICS DIVERGE FROM GRAPH 03_FILTER (documented): graph's
--allowed-elements lists HEAVY elements only (H implicitly allowed,
heavy-atom counts govern hydrogenation). Realspace --elements is an
EXACT whitelist of all allowed symbols, because --min-atoms/--max-atoms
count ALL atoms — implicit-H would silently bypass atom-count policy.
List H explicitly if allowed ("--elements H,Li").

PRESETS (mirror graph PRESETS dict; zero campaign knowledge):
  none               — all checks off (DEFAULT; registry ships clean)
  neutral            — charge-range 0:0
  closed_shell       — multiplicity-range 1:1
  neutral_closed_shell — both

Explicit CLI ranges override the preset's range for that property
(--elements/--min-atoms/--max-atoms have no preset component). All
defaults are None/off: the stage is a no-op pass-through unless the
operator supplies policy — no campaign knowledge anywhere (acceptance
criterion 3).

filter_status convention mirrors graph 03_filter: "ok" / "rejected".
Rejected rows are dropped from the output (unless --force-keep-rejected)
and written to <rejects-dir>/filter_rejected.parquet with filter_reason
+ full row context + .REJECTED sentinel. Rows whose geometry cannot be
decoded are "rejected" with reason row_decode_failed (realspace has no
mol-repair step; undecodable rows are garbage in the registry —
01_ingest already guards xyz inputs, this guards hand-crafted batches).

Filter decisions are deterministic scalar comparisons — rows are never
compared against each other, so ordering and batch layout do not affect
results.

Usage:
    python 03_filter.py -i deduped_batches/ -o filtered_batches/ \\
        [--preset none] [--charge-range -1:0] [--multiplicity-range 1:1] \\
        [--elements H,Li] [--min-atoms 1] [--max-atoms 8] \\
        [--force-keep-rejected]
"""

import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from curation.realspace.schema import row_to_frame, write_realspace_batch
from lib.parquet_io import read_batch

PRESETS = {
    "none": {"charge_range": None, "multiplicity_range": None},
    "neutral": {"charge_range": (0, 0), "multiplicity_range": None},
    "closed_shell": {"charge_range": None, "multiplicity_range": (1, 1)},
    "neutral_closed_shell": {"charge_range": (0, 0), "multiplicity_range": (1, 1)},
}

_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")


def _parse_range(value, flag_name):
    """'MIN:MAX' -> (min, max) ints; validates spec shape + MIN <= MAX."""
    if value is None:
        return None
    parts = str(value).split(":")
    if len(parts) != 2:
        print(f"Error: {flag_name} must be MIN:MAX, got '{value}'", file=sys.stderr)
        sys.exit(1)
    try:
        lo, hi = int(parts[0]), int(parts[1])
    except ValueError:
        print(f"Error: {flag_name} bounds must be integers, got '{value}'", file=sys.stderr)
        sys.exit(1)
    if lo > hi:
        print(f"Error: {flag_name} MIN must be <= MAX, got {lo}:{hi}", file=sys.stderr)
        sys.exit(1)
    return (lo, hi)


def _parse_elements(value):
    """'Li,H' -> sorted symbol list; validates shape + normalization."""
    if value is None:
        return None
    symbols = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        token = token[0].upper() + token[1:].lower()  # 'li' -> 'Li'
        if not _ELEMENT_RE.match(token):
            print(f"Error: --elements: bad element symbol '{token}'", file=sys.stderr)
            sys.exit(1)
        symbols.append(token)
    if not symbols:
        print("Error: --elements must list at least one element", file=sys.stderr)
        sys.exit(1)
    return sorted(set(symbols))


def check_row(
    symbols,
    n_atoms,
    charge,
    multiplicity,
    allowed,
    min_atoms,
    max_atoms,
    charge_range,
    multiplicity_range,
):
    """Run enabled checks in fixed order; return (status, reason).

    First failing check wins (graph 03_filter convention). All-None
    policy = pass-through.
    """
    if allowed is not None:
        forbidden = sorted(set(symbols) - set(allowed))
        if forbidden:
            return "rejected", f"forbidden_elements:{','.join(forbidden)}"
    if min_atoms is not None and n_atoms < min_atoms:
        return "rejected", f"n_atoms={n_atoms}<{min_atoms}"
    if max_atoms is not None and n_atoms > max_atoms:
        return "rejected", f"n_atoms={n_atoms}>{max_atoms}"
    if charge_range is not None and not charge_range[0] <= charge <= charge_range[1]:
        side = "<" if charge < charge_range[0] else ">"
        bound = charge_range[0] if charge < charge_range[0] else charge_range[1]
        return "rejected", f"charge={charge}{side}{bound}"
    if multiplicity_range is not None and not (
        multiplicity_range[0] <= multiplicity <= multiplicity_range[1]
    ):
        side = "<" if multiplicity < multiplicity_range[0] else ">"
        bound = (
            multiplicity_range[0] if multiplicity < multiplicity_range[0] else multiplicity_range[1]
        )
        return "rejected", f"multiplicity={multiplicity}{side}{bound}"
    return "ok", ""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AIM4ML realspace 03_filter — generic property filter "
        "(elements / atom count / charge / multiplicity)."
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="deduped_batches",
        help="Input Parquet batch directory (default: deduped_batches/).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="filtered_batches",
        help="Output directory (default: filtered_batches/).",
    )
    p.add_argument(
        "--rejects-dir",
        type=str,
        default="rejects/03_filter",
        help="Rejects directory (default: rejects/03_filter/).",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=list(PRESETS.keys()),
        help="Filter preset (default: none — all checks off). "
        "Explicit CLI ranges override the preset per property.",
    )
    p.add_argument(
        "--elements",
        type=str,
        default=None,
        help="Comma-separated allowed symbols — EXACT whitelist of ALL "
        "allowed elements (list H explicitly if allowed; e.g. H,Li).",
    )
    p.add_argument(
        "--min-atoms", type=int, default=None, help="Minimum number of atoms (all atoms)."
    )
    p.add_argument(
        "--max-atoms", type=int, default=None, help="Maximum number of atoms (all atoms)."
    )
    p.add_argument(
        "--charge-range",
        type=str,
        default=None,
        help="Total charge range MIN:MAX (e.g. -1:1). Overrides preset. "
        "For negative bounds use the --charge-range=-1:1 '=' form "
        "(argparse rejects bare leading-dash values).",
    )
    p.add_argument(
        "--multiplicity-range",
        type=str,
        default=None,
        help="Spin multiplicity range MIN:MAX (e.g. 1:3). Overrides preset.",
    )
    p.add_argument(
        "--force-keep-rejected",
        action="store_true",
        help="Keep rejected rows in the output Parquet batches with "
        "filter_status recorded (default: drop them).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from lib.provenance import record_run

    record_run(args.output_dir, "03_filter")

    allowed = _parse_elements(args.elements)
    charge_range = _parse_range(args.charge_range, "--charge-range")
    multiplicity_range = _parse_range(args.multiplicity_range, "--multiplicity-range")
    preset = PRESETS[args.preset]
    if charge_range is None:
        charge_range = preset["charge_range"]
    if multiplicity_range is None:
        multiplicity_range = preset["multiplicity_range"]

    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Preset: {args.preset}")
    print(f"  elements:        {allowed}")
    print(f"  atoms:           [{args.min_atoms}, {args.max_atoms}]")
    print(f"  charge:          {charge_range}")
    print(f"  multiplicity:    {multiplicity_range}")
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    total_ok = 0
    total_rejected = 0
    total_decode_failed = 0
    rejected_rows = []
    reason_counts = {}
    n_written = 0

    for fname in batch_files:
        out_rows = []
        for row in read_batch(os.path.join(args.input_dir, fname)):
            try:
                frame = row_to_frame(row)
                n_atoms = len(frame.symbols)
                charge = int(row["charge"])
                multiplicity = int(row["multiplicity"])
            except (KeyError, TypeError, ValueError) as exc:
                status, reason = "rejected", f"row_decode_failed:{exc}"
                total_decode_failed += 1
            else:
                status, reason = check_row(
                    frame.symbols,
                    n_atoms,
                    charge,
                    multiplicity,
                    allowed,
                    args.min_atoms,
                    args.max_atoms,
                    charge_range,
                    multiplicity_range,
                )

            row["filter_status"] = status
            if status == "rejected":
                row["filter_reason"] = reason
                rejected_rows.append(row)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                total_rejected += 1
                if args.force_keep_rejected:
                    out_rows.append(row)
            else:
                total_ok += 1
                out_rows.append(row)

        if out_rows:
            write_realspace_batch(os.path.join(args.output_dir, fname), out_rows)
            print(f"  Wrote {os.path.join(args.output_dir, fname)} ({len(out_rows)} frames)")
            n_written += len(out_rows)

    # -- Rejects parquet + sentinel (always, audit trail) ------------------
    if rejected_rows:
        reject_path = os.path.join(args.rejects_dir, "filter_rejected.parquet")
        write_realspace_batch(reject_path, rejected_rows)
        print(f"  {len(rejected_rows)} rejected -> {reject_path}")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(os.path.join(args.rejects_dir, ".REJECTED"), "w") as f:
            f.write(f"{len(rejected_rows)} frames rejected\n")

    # -- Report ------------------------------------------------------------
    total = total_ok + total_rejected
    print("\nReport")
    print(f"  Total:          {total}")
    print(f"  Ok:             {total_ok}")
    print(f"  Rejected:       {total_rejected}")
    print(f"  Decode failed:  {total_decode_failed}")
    print(f"  Written:        {n_written}")
    if reason_counts:
        print("\nRejection breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
