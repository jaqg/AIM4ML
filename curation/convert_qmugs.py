#!/usr/bin/env python3
"""
convert_qmugs.py — Convert raw QMugs data to pipeline-standard input SDF.

Reads QMugs summary.csv + per-conformer SDF files and writes a single SDF
with standardised property tags per the AIM4ML curation pipeline contract.

Input (raw QMugs):
    summary.csv                        — chembl_id, conf_id, smiles, DFT/GFN2 props
    structures/CHEMBL{id}/conf_XX.sdf  — 3D geometry + bonds (V2000, real bond orders)

Output:
    A single SDF, one entry per conformer. Properties written as > <TAG> data fields:
        Energy_Ha, FormalCharge, Multiplicity   (required)
        SMILES, SourceID                         (recommended)
        HOMO_Ha, LUMO_Ha, HL_Gap_Ha             (optional pass-through)
        PartialCharges                           (absent — QMugs has no per-atom charges)

Molecule construction:
    Conformer SDF read directly (already contains atoms, 3D coords, and real
    bond orders). No bond-order resolution needed — pass through as-is.

Assumptions:
    QMugs molecules are neutral closed-shell singlets -> FormalCharge=0,
    Multiplicity=1. QMugs stores no charge/spin data; pipeline stage 3
    (filter) verifies neutral+closed-shell downstream.

Usage:
    python3 convert_qmugs.py --data-dir /datos_pool/mldata1/QMdatasets/QMugs
    python3 convert_qmugs.py --data-dir ... --sample 1000
    python3 convert_qmugs.py --data-dir ... --output /path/to/out.sdf
"""

import argparse
import os
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter
from tqdm import tqdm

# -- Paths ---------------------------------------------------------------

DEFAULT_DATA_DIR = "/datos_pool/mldata1/QMdatasets/QMugs"

# -- Helpers -------------------------------------------------------------


def sdf_path(data_dir, chembl_id, conf_id):
    return os.path.join(data_dir, "structures", chembl_id, f"{conf_id}.sdf")


def load_mol(path):
    """Read a conformer SDF preserving original geometry and bond orders."""
    try:
        return Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
    except Exception:
        return None


# -- Main ----------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert raw QMugs data to pipeline-standard input SDF."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"QMugs data directory (default: {DEFAULT_DATA_DIR}).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Convert only the first N conformers (for testing).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output SDF path (default: <data-dir>/qmugs_input.sdf).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    summary_csv = os.path.join(data_dir, "summary.csv")
    if not os.path.isfile(summary_csv):
        print(f"ERROR: summary.csv not found at {summary_csv}")
        sys.exit(1)

    # Resolve output path
    if args.output:
        out_sdf = args.output
    else:
        out_sdf = os.path.join(data_dir, "qmugs_input.sdf")
    out_dir = os.path.dirname(out_sdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # -- Load summary ----------------------------------------------------
    print(f"Reading {summary_csv} ...")
    df = pd.read_csv(summary_csv, dtype={"chembl_id": str, "conf_id": str})
    if args.sample:
        df = df.head(args.sample)
    n_rows = len(df)
    print(f"  {n_rows} conformers to convert")

    # -- Convert ---------------------------------------------------------
    print(f"\nWriting {out_sdf} ...")
    writer = SDWriter(out_sdf)
    written = 0
    skipped = 0

    for _, row in tqdm(
        df.iterrows(), total=n_rows, desc="Converting", unit="conf", file=sys.stdout
    ):
        chembl_id = row["chembl_id"]
        conf_id = row["conf_id"]

        path = sdf_path(data_dir, chembl_id, conf_id)
        if not os.path.isfile(path):
            skipped += 1
            continue

        mol = load_mol(path)
        if mol is None:
            skipped += 1
            continue

        # Drop QMugs-native props, keep only pipeline-standard tags
        for prop in list(mol.GetPropNames()):
            mol.ClearProp(prop)

        mol.SetProp("_Name", f"{chembl_id}_{conf_id}")
        mol.SetProp("Energy_Ha", f"{row['DFT_TOTAL_ENERGY']:.8f}")
        mol.SetProp("FormalCharge", "0")
        mol.SetProp("Multiplicity", "1")
        mol.SetProp("SMILES", row["smiles"])
        mol.SetProp("SourceID", f"{chembl_id}_{conf_id}")
        mol.SetProp("HOMO_Ha", f"{row['DFT_HOMO_ENERGY']:.6f}")
        mol.SetProp("LUMO_Ha", f"{row['DFT_LUMO_ENERGY']:.6f}")
        mol.SetProp("HL_Gap_Ha", f"{row['DFT_HOMO_LUMO_GAP']:.6f}")

        writer.write(mol)
        written += 1

    writer.close()

    # -- Report ----------------------------------------------------------
    print("\nDone.")
    print(f"  Written:    {written}")
    print(f"  Skipped:    {skipped}")
    print(f"  Output:     {out_sdf}")


if __name__ == "__main__":
    main()
