#!/usr/bin/env python3
"""
04_dedup.py — Stage 4: Compute canonical SMILES, assign CompoundID,
             and remove conformer duplicates.

For each molecule in the input Parquet batches:
  1. Reconstructs mol from mol_block (bonds already set by filter stage).
  2. Computes canonical SMILES via RDKit (kekulization → SMILES).
  3. If kekulization fails and a SMILES tag is present, runs template
     fallback via AssignBondOrdersFromTemplate.
  4. Computes CompoundID = MD5(canonical SMILES).
  5. Detects conformer duplicates: molecules sharing the same CompoundID
     (keep the first occurrence, flag the rest).
  6. Updates mol_block with corrected bond representation on success.

Usage:
    python3 04_dedup.py -i curated_batches/ -o deduped_batches/
"""

import argparse
import hashlib
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

from lib.parquet_io import read_batch, write_batch
from lib.rdkit_version import check_min_rdkit
from lib.sdf_io import write_reject_sdf

# -- Core logic -----------------------------------------------------------


def canonicalize_and_assign(mol_block, smiles_tag=None):
    """
    Compute canonical SMILES from mol_block.  Falls back to SMILES template
    if kekulization fails.

    Returns (canonical_smiles, updated_mol_block, status, reason).
      canonical_smiles  — canonical SMILES string, or None on failure
      updated_mol_block — mol block with corrected bonds (or original on failure)
      status            — "ok" | "bond_assignment_failed" | "mol_corrupt"
      reason            — human-readable detail
    """
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if mol is None:
        return None, mol_block, "mol_corrupt", "could not parse mol_block"

    # -- Primary path: sanitize + canonicalize from existing bonds ---------
    # Sanitize on a COPY so invalid mols raise (and leave `mol` pristine for
    # the template fallback). Sanitize validates valence/charges and perceives
    # aromaticity, so a neutral N with 4 bonds fails loudly instead of being
    # written through as a broken mol_block.
    try:
        mol_san = Chem.Mol(mol)
        Chem.SanitizeMol(mol_san)
        can_smi = Chem.MolToSmiles(mol_san)
        new_block = _mol_to_block(mol_san)
        return can_smi, new_block, "ok", ""
    except Exception:
        pass  # sanitize/kekulize failed → try template fallback

    # -- Template fallback (requires SMILES tag) --------------------------
    if (
        smiles_tag is None
        or (isinstance(smiles_tag, float) and math.isnan(smiles_tag))
        or smiles_tag == ""
    ):
        return None, mol_block, "bond_assignment_failed", "kekulize failed, no SMILES for template"

    template = Chem.MolFromSmiles(smiles_tag)
    if template is None:
        return None, mol_block, "bond_assignment_failed", "SMILES tag unparseable"

    try:
        # Match explicit-H mol against explicit-H template (AddHs), implicit-H
        # against implicit-H template — mirrors 03_filter fix.
        has_h = any(a.GetSymbol() == "H" for a in mol.GetAtoms())
        tpl = Chem.AddHs(template) if has_h else template
        mol_assigned = AllChem.AssignBondOrdersFromTemplate(tpl, mol)
        Chem.SanitizeMol(mol_assigned)
        can_smi = Chem.MolToSmiles(mol_assigned)
        new_block = _mol_to_block(mol_assigned)
        return can_smi, new_block, "ok", ""
    except Exception:
        return None, mol_block, "bond_assignment_failed", "template fallback failed"


def _mol_to_block(mol):
    """Extract V2000 mol block from an RDKit Mol."""
    block = Chem.MolToMolBlock(mol)
    m_end = block.index("M  END") + len("M  END")
    return block[:m_end]


def _extract_atropisomer_key(mol):
    """Return a canonical string encoding atropisomer bond descriptors.

    Format: 'bond_idx:descriptor_int;bond_idx:descriptor_int' sorted by
    bond_idx.  Empty string if no atropisomer bonds (with specified stereo).

    Read from the in-memory mol object, NOT from a SMILES round-trip, so we
    are unaffected by RDKit atropisomer canonicalization bugs (#7427 etc.).
    """
    Chem.SanitizeMol(mol, catchErrors=True)
    si = Chem.FindPotentialStereo(mol)
    entries = []
    for info in si:
        if info.type == Chem.StereoType.Bond_Atropisomer:
            if info.descriptor != Chem.rdchem.StereoDescriptor.NoValue:
                entries.append(f"{info.centeredOn}:{int(info.descriptor)}")
    if not entries:
        return ""
    return ";".join(sorted(entries))


# -- Main ----------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 4 — Dedup: canonical SMILES + CompoundID + conformer removal."
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="curated_batches",
        help="Input Parquet batch directory (default: curated_batches/).",
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
        default="rejects/04_dedup",
        help="Rejected molecules SDF directory (default: rejects/04_dedup/).",
    )
    p.add_argument(
        "--force-keep-rejected",
        action="store_true",
        help="Keep rejected molecules (bond_assignment_failed, mol_corrupt) "
        "in the output Parquet batches (default: drop them).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from lib.provenance import record_run

    record_run(args.output_dir, "04_dedup")

    if not check_min_rdkit():
        print(
            f"ERROR: RDKit {rdBase.rdkitVersion} < 2024.03 — "
            f"atropisomer stereochemistry unsupported",
            file=sys.stderr,
        )
        sys.exit(1)

    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    seen_ids = set()
    total_ok = 0
    total_dup = 0
    total_fail = 0
    total_corrupt = 0
    n_written = 0
    rejected_rows = []

    for fname in batch_files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        batch = read_batch(in_path)
        out_rows = []

        for row in batch:
            smiles_raw = row.get("SMILES")
            can_smi, new_block, status, reason = canonicalize_and_assign(
                row["mol_block"],
                smiles_raw,
            )

            if status == "ok":
                row["CanonicalSMILES"] = can_smi
                row["mol_block"] = new_block

                # Reconstruct mol (includes explicit H) for formula + atrop key
                dedup_mol = Chem.MolFromMolBlock(new_block, sanitize=False, removeHs=False)

                # Atropisomer key: preserve atrop-enantiomers through dedup.
                # Canonical SMILES strips atrop stereo, so the key is the only
                # thing separating atrop-enantiomers at this stage.
                atrop_key = ""
                if dedup_mol is not None:
                    atrop_key = _extract_atropisomer_key(dedup_mol)

                cid_input = can_smi + "|" + atrop_key if atrop_key else can_smi
                cid = hashlib.md5(cid_input.encode()).hexdigest()
                row["CompoundID"] = cid
                row["AtropisomerKey"] = atrop_key

                # Compute molecular formula from mol (includes explicit H).
                # UpdatePropertyCache computes implicit valence — required by
                # CalcMolFormula on the unsanitized (sanitize=False) re-parse.
                if dedup_mol is not None:
                    dedup_mol.UpdatePropertyCache(strict=False)
                    from rdkit.Chem import rdMolDescriptors

                    row["Formula"] = rdMolDescriptors.CalcMolFormula(dedup_mol)

                if cid in seen_ids:
                    row["dedup_status"] = "conformer_duplicate"
                    total_dup += 1
                    rejected_rows.append(row)
                else:
                    row["dedup_status"] = "ok"
                    row["ICONF"] = 1
                    seen_ids.add(cid)
                    total_ok += 1
            elif status == "mol_corrupt":
                row["dedup_status"] = "mol_corrupt"
                total_corrupt += 1
                rejected_rows.append(row)
            else:  # bond_assignment_failed
                row["dedup_status"] = status
                row["dedup_reason"] = reason
                total_fail += 1
                rejected_rows.append(row)

            # Write row to output only if:
            #   - dedup_status in ("ok", "conformer_duplicate")
            #     (conformer dups must flow to Stage 8 for RMSD selection)
            #   - --force-keep-rejected is set (keep everything for debugging)
            keep_in_output = (
                row["dedup_status"] in ("ok", "conformer_duplicate") or args.force_keep_rejected
            )
            if keep_in_output:
                out_rows.append(row)
                n_written += 1

        write_batch(out_path, out_rows)

    # -- Reject SDF -------------------------------------------------------
    if rejected_rows:
        reject_path = os.path.join(args.rejects_dir, "dedup_rejected.sdf")
        write_reject_sdf(reject_path, rejected_rows, reject_reason="dedup_rejected")
        print(f"  {len(rejected_rows)} rejected → {reject_path}")

    # -- Report ------------------------------------------------------------
    total = total_ok + total_dup + total_fail + total_corrupt
    print("\nReport")
    print(f"  Total:                 {total}")
    print(f"  Unique (ok):           {total_ok}")
    print(f"  Conformer duplicates:  {total_dup}")
    print(f"  Bond assignment fail:  {total_fail}")
    print(f"  Mol corrupt:           {total_corrupt}")
    print(f"  Written to output:     {n_written}")
    if not args.force_keep_rejected and (total_fail + total_corrupt):
        print(
            f"  Dropped from output:   {total_fail + total_corrupt} "
            f"(use --force-keep-rejected to keep)"
        )

    if total_dup or total_fail:
        sentinel = os.path.join(args.rejects_dir, ".REJECTED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{total_dup + total_fail} molecules rejected\n")


if __name__ == "__main__":
    raise SystemExit(main())
