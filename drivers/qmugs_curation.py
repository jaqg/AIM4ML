#!/usr/bin/env python3
"""
qmugs_curation.py — QMugs graph-track curation driver.

Source: QMugs structures → converters/convert_qmugs.py → qmugs_input.sdf.
Track: graph (Lewis/SMILES identity; CompoundID = MD5 canonical SMILES, D62/D70).
Chain: same stage 0–10 chain as QM40 (former curation/Makefile, flag-for-flag) —
only family metadata and input name differ. No --max-fragments (D69).

Usage (from repo root or curation/):
    python drivers/qmugs_curation.py [--mode sample|full] [--workers N] [--dry-run]
    make -C curation qmugs     # alias table
"""

from chain import driver_parser, run_driver


def main(argv=None):
    parser = driver_parser(
        description=__doc__,
        family="QMUGS",
        input_name="qmugs_input",
    )
    args = parser.parse_args(argv)
    run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
