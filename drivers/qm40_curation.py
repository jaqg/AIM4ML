#!/usr/bin/env python3
"""
qm40_curation.py — QM40 graph-track curation driver.

Source: QM40 raw CSVs → converters/convert_qm40.py → qm40_input.sdf.
Track: graph (Lewis/SMILES identity; CompoundID = MD5 canonical SMILES, D62/D70).
Chain: former curation/Makefile stages 0–10, flag-for-flag — no --max-fragments
(D69: complexes/salts flow through, tagged by n_fragments), --tanimoto off by
default, WORKERS/BATCH_SIZE defaults as in the Makefile.

Usage (from repo root or curation/):
    python drivers/qm40_curation.py [--mode sample|full] [--workers N] [--dry-run]
    make -C curation qm40      # alias table
"""

from chain import driver_parser, run_driver


def main(argv=None):
    parser = driver_parser(
        description=__doc__,
        family="QM40",
        input_name="qm40_input",
    )
    args = parser.parse_args(argv)
    run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
