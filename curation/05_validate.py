#!/usr/bin/env python3
"""
05_validate.py — Stage 5: Integrity cross-checks on curated Parquet batches.

Ensures data consistency across columns and reconstructs mol from mol_block
to verify the stored representation is valid. Dataset-agnostic — no reference
to raw source files.

Checks (per molecule):
  1. mol_block reparses correctly
  2. num_atoms / num_bonds match reconstructed mol
  3. CanonicalSMILES round-trip: canonical SMILES from mol_block matches stored
  4. Energy_Ha is finite
  5. CompoundID consistency: computed from CanonicalSMILES matches stored

Aggregate checks:
  6. CompoundID uniqueness across all batches (no collisions)
  7. Flags molecules with filter_status="rejected" still present (warn only)

Usage:
    python3 05_validate.py -i deduped_batches/
"""

import argparse
import hashlib
import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from lib.parquet_io import read_batch
from lib.sdf_io import write_reject_sdf

# -- Validate one molecule -----------------------------------------------


def validate_row(row):
    """
    Run integrity checks on a single Parquet row.

    Returns (status, errors):
      status  — "ok" | "invalid"
      errors  — list of error strings (empty if ok)
    """
    errors = []

    # 1 — mol_block reparses
    mol = Chem.MolFromMolBlock(row["mol_block"], sanitize=False, removeHs=False)
    if mol is None:
        return "invalid", ["mol_block: could not parse"]
    n_atoms_mol = mol.GetNumAtoms()
    n_bonds_mol = mol.GetNumBonds()

    # 2 — atom/bond counts match metadata
    if n_atoms_mol != row.get("num_atoms"):
        errors.append(f"num_atoms mismatch: stored={row.get('num_atoms')}, mol={n_atoms_mol}")
    if n_bonds_mol != row.get("num_bonds"):
        errors.append(f"num_bonds mismatch: stored={row.get('num_bonds')}, mol={n_bonds_mol}")

    # 3 — CanonicalSMILES round-trip (sanitize first to match Stage 4, which
    # canonicalizes a sanitized mol — aromatic, not kekule)
    stored_smi = row.get("CanonicalSMILES")
    if stored_smi:
        try:
            mol_san = Chem.Mol(mol)
            Chem.SanitizeMol(mol_san)
            recomputed = Chem.MolToSmiles(mol_san)
            if recomputed != stored_smi:
                errors.append(
                    f"CanonicalSMILES mismatch: stored={stored_smi}, recomputed={recomputed}"
                )
        except Exception:
            errors.append("CanonicalSMILES: kekulize failed during validation")

    # 4 — Energy_Ha is finite
    energy = row.get("Energy_Ha")
    if energy is None or (isinstance(energy, float) and (math.isnan(energy) or math.isinf(energy))):
        errors.append("Energy_Ha: not finite")

    # 5 — CompoundID consistency (Stage 4: MD5(canonical | atrop_key))
    cid = row.get("CompoundID")
    stored_smi = row.get("CanonicalSMILES")
    if cid and stored_smi:
        atrop = row.get("AtropisomerKey") or ""
        cid_input = stored_smi + (f"|{atrop}" if atrop else "")
        recomputed_cid = hashlib.md5(cid_input.encode()).hexdigest()
        if recomputed_cid != cid:
            errors.append(f"CompoundID mismatch: stored={cid}, recomputed={recomputed_cid}")

    if errors:
        return "invalid", errors
    return "ok", []


# -- Main ----------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="AIM4ML Stage 5 — Integrity cross-checks.")
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="deduped_batches",
        help="Input Parquet batch directory (default: deduped_batches/).",
    )
    p.add_argument(
        "--rejects-dir",
        type=str,
        default="rejects/05_validate",
        help="Invalid molecules SDF directory (default: rejects/05_validate/).",
    )
    p.add_argument(
        "--skip", action="store_true", help="Skip validation entirely (for general pipeline runs)."
    )
    return p.parse_args()


def main():
    args = parse_args()
    from lib.provenance import record_run

    record_run(args.input_dir, "05_validate")

    if args.skip:
        print("Validation skipped (--skip).")
        sys.exit(0)

    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    total_ok = 0
    total_invalid = 0
    invalid_rows = []
    all_cids = {}  # cid → (batch, idx) for collision detection
    cid_collisions = 0
    rejected_present = 0

    for fname in batch_files:
        in_path = os.path.join(args.input_dir, fname)
        batch = read_batch(in_path)

        for idx, row in enumerate(batch):
            # Warn if rejected molecules still in the stream
            if row.get("filter_status") == "rejected":
                rejected_present += 1

            status, errors = validate_row(row)
            row["validate_status"] = status
            if errors:
                row["validate_errors"] = "; ".join(errors)

            if status == "ok":
                total_ok += 1
            else:
                total_invalid += 1
                invalid_rows.append(row)

            # Check CompoundID collisions
            cid = row.get("CompoundID")
            if cid and status == "ok":
                if cid in all_cids:
                    cid_collisions += 1
                else:
                    all_cids[cid] = (fname, idx)

    total = total_ok + total_invalid

    # -- Reject SDF -------------------------------------------------------
    if invalid_rows:
        reject_path = os.path.join(args.rejects_dir, "validate_invalid.sdf")
        write_reject_sdf(reject_path, invalid_rows, reject_reason="validate_invalid")
        print(f"  {len(invalid_rows)} invalid → {reject_path}")

    # -- Report ------------------------------------------------------------
    print("\nReport")
    print(f"  Total:          {total}")
    print(f"  OK:             {total_ok}")
    print(f"  Invalid:        {total_invalid}")
    print(f"  CID collisions: {cid_collisions}")

    if rejected_present:
        print(
            f"\n  ⚠ {rejected_present} molecules with filter_status='rejected' "
            f"still present in batches (pipeline carries them through)"
        )

    if cid_collisions:
        print(f"\n  ⚠ {cid_collisions} CompoundID collisions detected")

    if total_invalid:
        sentinel = os.path.join(args.rejects_dir, ".INVALID")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{total_invalid} invalid, {cid_collisions} collisions\n")
        sys.exit(1)

    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
