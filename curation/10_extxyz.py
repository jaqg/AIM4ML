#!/usr/bin/env python3
"""
10_extxyz.py — Stage 10: Build extended XYZ trajectory files.

Reads curated Parquet batches and writes batched extXYZ files (one per
batch) with key:value metadata in the comment line.  ExtXYZ is the final
delivery format for MLIP frameworks (MACE, NequIP, SchNetPack).

extXYZ format:
  Line 1         — number of atoms
  Line 2         — key:value,key:value,... metadata
  Lines 3–NAT+2  — element x y z (four columns, space-separated)

Metadata keys:
  SourceID, CompoundID, Formula, nat, CNSO, chrg, mult, e, smiles

Usage:
    python3 09_extxyz.py -i reordered_batches/ -o extxyz/
"""

import argparse
import os
import sys
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

from lib.parquet_io import read_batch

CNSO_ELEMENTS = {"C", "N", "S", "O"}


def _should_exclude(row, rules):
    """Return True if row matches any exclude rule."""
    for col, val in rules.items():
        if str(row.get(col, "")) == val:
            return True
    return False


# -- extXYZ writer --------------------------------------------------------


def mol_block_to_extxyz(mol_block, row, family="QM40"):
    """
    Convert a V2000 mol block + Parquet metadata row to extXYZ text.

    Returns the full extXYZ frame as a string (nat + metadata + atom lines),
    or None if the mol_block cannot be parsed.
    """
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if mol is None:
        return None

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conf = mol.GetConformer()
    nat = len(atoms)

    # Atom composition
    counts = Counter(atoms)
    formula = row.get("Formula")
    if not formula:
        from rdkit.Chem import rdMolDescriptors

        mol.UpdatePropertyCache(strict=False)
        formula = rdMolDescriptors.CalcMolFormula(mol)
    cnso = sum(counts.get(el, 0) for el in CNSO_ELEMENTS)

    # Metadata
    src_id = row.get("SourceID", "")
    cid = row.get("CompoundID", "")
    charge = row.get("FormalCharge", 0)
    mult = row.get("Multiplicity", 1)
    energy = row.get("Energy_Ha", "")
    smiles = row.get("CanonicalSMILES", "")
    iconf = row.get("ICONF", 1)

    # Descriptors from CanonicalSMILES (lightweight, no external CSV needed)
    tpsa_val, logp_val, nrot_val = "", "", ""
    if smiles:
        dmol = Chem.MolFromSmiles(smiles, sanitize=False)
        if dmol is not None:
            dmol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(dmol)
            try:
                tpsa_val = f"{Descriptors.TPSA(dmol):.2f}"
                logp_val = f"{Descriptors.MolLogP(dmol):.2f}"
                nrot_val = str(Descriptors.NumRotatableBonds(dmol))
            except Exception:
                pass

    meta = (
        f"SourceID,{src_id},CompoundID,{cid},"
        f"Formula,{formula},Nat,{nat},CNSO,{cnso},"
        f"chrg,{charge},mult,{mult},"
        f"E,{energy},fmax,,Family,{family},"
        f"smiles,{smiles},"
        f"tpsa,{tpsa_val},logp,{logp_val},nrot,{nrot_val},"
        f"nfrag,1,iconf,{iconf}"
    )

    # Atom lines
    lines = [f"{nat}\n", f"{meta}\n"]
    for i, sym in enumerate(atoms):
        pos = conf.GetAtomPosition(i)
        lines.append(f"{sym:2s}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}\n")

    return "".join(lines)


# -- Main ----------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="AIM4ML Stage 9 — Build extXYZ trajectory files.")
    p.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="conformer_batches",
        help="Input Parquet batch directory (default: conformer_batches/).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="extxyz",
        help="Output directory for extXYZ files (default: extxyz/).",
    )
    p.add_argument(
        "--family",
        type=str,
        default="QM40",
        help="Dataset family name in metadata (default: QM40).",
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
    return p.parse_args()


def main():
    args = parse_args()
    from lib.provenance import record_run

    record_run(args.output_dir, "10_extxyz")

    batch_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    # Parse exclude rules
    exclude_rules = {}
    for rule in args.exclude:
        if "=" in rule:
            col, val = rule.split("=", 1)
            exclude_rules[col.strip()] = val.strip()
    if exclude_rules:
        print(f"Excluding rows where: {exclude_rules}")

    total_ok = 0
    total_excluded = 0
    total_failed = 0

    for fname in batch_files:
        in_path = os.path.join(args.input_dir, fname)
        batch = read_batch(in_path)

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output_dir, f"{base}.extxyz")

        frames = []
        for row in batch:
            # Apply exclude rules
            if _should_exclude(row, exclude_rules):
                total_excluded += 1
                continue

            frame = mol_block_to_extxyz(row["mol_block"], row, family=args.family)
            if frame is None:
                total_failed += 1
            else:
                frames.append(frame)
                total_ok += 1

        tmp = out_path + ".tmp." + str(os.getpid())
        with open(tmp, "w") as f:
            f.writelines(frames)
        os.replace(tmp, out_path)

    # -- Report ------------------------------------------------------------
    total = total_ok + total_failed + total_excluded
    print("\nReport")
    print(f"  Total:        {total}")
    print(f"  Written:      {total_ok}")
    print(f"  Excluded:     {total_excluded}")
    print(f"  Failed:       {total_failed}")
    print(f"  Output dir:   {args.output_dir}")


if __name__ == "__main__":
    main()
