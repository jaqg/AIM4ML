#!/usr/bin/env python3
"""
03_filter.py — Stage 3: Chemical filter (neutral / non-zwitterion / closed-shell).

For every molecule in the input Parquet batches:
  1. Reconstructs the RDKit Mol from mol_block (unsanitized).
  2. Assigns bond orders + formal charges from the SMILES template
     (AssignBondOrdersFromTemplate) — SMILES is authoritative; 3D geometry
     (DetermineBonds) is the fallback when SMILES is missing or template
     matching fails. Geometry reconstruction loses formal charges for
     charge-separated species (zwitterions).
  3. Sanitizes the molecule to obtain per-atom formal charges and radical counts.
  4. Checks three criteria:
      - Neutral:         net formal charge == 0
      - Non-zwitterion:  no true zwitterion (non-adjacent +/− charge separation)
      - Closed-shell:    radical electrons == 0
  5. Writes a filter_status column ("ok" / "rejected") and updates the
     mol_block with the sanitized representation (correct bond types).

Usage:
    python3 03_filter.py -i filtered_batches/ -o curated_batches/
"""

import os
import sys
import argparse
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

def check_composition(mol_block, allowed_elements, min_heavy, max_heavy,
                      min_fragments, max_fragments):
    """
    Fast composition check before expensive DetermineBondOrders.

    Returns (passes: bool, reason: str, n_fragments: int|None).
    n_fragments is always computed (when mol parses) so the caller can
    store it regardless of which check rejects the row.
    """
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if mol is None:
        return False, "mol_corrupt", None

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    elements = set(atoms)
    heavy_atoms = sum(1 for a in atoms if a != "H")

    # Fragment count FIRST — computed before any check, so n_fragments
    # is always available for storage even when another check rejects.
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    n_fragments = len(frags)

    if allowed_elements is not None:
        # H is implicitly allowed — mol_blocks carry explicit H, but the
        # heavy-atom count (--min-heavy/--max-heavy) already governs
        # hydrogenation.  --allowed-elements lists HEAVY elements.
        allowed = set(allowed_elements)
        allowed.add("H")
        forbidden = elements - allowed
        if forbidden:
            return False, f"forbidden_elements:{','.join(sorted(forbidden))}", n_fragments

    if min_heavy is not None and heavy_atoms < min_heavy:
        return False, f"heavy_atoms={heavy_atoms}<{min_heavy}", n_fragments

    if max_heavy is not None and heavy_atoms > max_heavy:
        return False, f"heavy_atoms={heavy_atoms}>{max_heavy}", n_fragments

    if min_fragments is not None and n_fragments < min_fragments:
        return False, f"fragments={n_fragments}<{min_fragments}", n_fragments

    if max_fragments is not None and n_fragments > max_fragments:
        formulas = _fragment_formulas(mol, frags)
        return False, f"fragments={n_fragments}>{max_fragments}[{';'.join(formulas)}]", n_fragments

    return True, "", n_fragments


def _fragment_formulas(mol, frags):
    """
    Hill-ordered formula per fragment, e.g. ['C9H8O4', 'H2O'].

    Counts implicit hydrogens via a sanitized copy so formulas are complete
    for both implicit-H and explicit-H mol_blocks. Explicit H atoms are
    skipped (their count is recovered from heavy-atom GetTotalNumHs).
    """
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m, catchErrors=True)
    except Exception:
        m = mol  # fall back to explicit-atom-only counts

    formulas = []
    for frag in frags:
        elems = Counter()
        for i in frag:
            a = m.GetAtomWithIdx(i)
            if a.GetSymbol() == "H":
                continue  # counted via GetTotalNumHs on heavy atoms
            elems[a.GetSymbol()] += 1
            hs = a.GetTotalNumHs()
            if hs:
                elems["H"] += hs
        parts = []
        for e in ("C", "H"):  # Hill order: C, H first
            if e in elems:
                n = elems.pop(e)
                parts.append(e if n == 1 else f"{e}{n}")
        for e in sorted(elems):  # rest alphabetical
            n = elems[e]
            parts.append(e if n == 1 else f"{e}{n}")
        formulas.append("".join(parts))
    return formulas

from lib.parquet_io import read_batch, write_batch
from lib.sdf_io import write_reject_sdf
from lib.parallel import parallel_map


# -- Presets --------------------------------------------------------------

PRESETS = {
    "neutral_closed_shell": {
        "charge":       True,   # net formal charge == 0
        "zwitterion":   True,   # no true zwitterion (non-adjacent charge separation)
        "radicals":     True,   # radical electrons == 0
    },
    "neutral": {
        "charge":       True,
        "zwitterion":   False,
        "radicals":     False,
    },
    "none": {
        "charge":       False,
        "zwitterion":   False,
        "radicals":     False,
    },
}


# -- Filter logic ---------------------------------------------------------

def _has_true_zwitterion(mol):
    """True zwitterion = formal charge on an atom with no adjacent
    oppositely-charged atom. Adjacent +/− (nitro [N+]-[O-], N-oxide,
    sulfoxide, azide) is a polar-bond representation, not a zwitterion."""
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc == 0:
            continue
        if not any(n.GetFormalCharge() * fc < 0
                   for n in atom.GetNeighbors()):
            return True
    return False


def process_molecule(row, checks):
    """
    Process a single molecule row: reconstruct mol, assign bond orders and
    formal charges, sanitize, check enabled criteria.

    Bond orders + formal charges come from the SMILES template
    (AssignBondOrdersFromTemplate) when available — the SMILES is the
    authoritative molecular identity. 3D-geometry reconstruction
    (DetermineBonds) loses formal charges for charge-separated species
    (zwitterions), so it is only the fallback when SMILES is missing or
    template matching fails.

    Parameters
    ----------
    row : dict
        Parquet row with mol_block, FormalCharge, SMILES.
    checks : dict
        Boolean flags for 'charge', 'zwitterion', 'radicals'.

    Returns
    -------
    (mol, filter_status, reason)
    """
    mol = Chem.MolFromMolBlock(row["mol_block"], sanitize=False,
                               removeHs=False)
    if mol is None:
        return None, "mol_corrupt", "could not parse mol_block"

    original_smi = row.get("SMILES")
    template = Chem.MolFromSmiles(original_smi) if original_smi else None

    # Step 1: assign bond orders + formal charges. SMILES authoritative;
    # geometry (DetermineBonds) is the fallback.
    if template is not None:
        try:
            # Match explicit-H mol against an explicit-H template (AddHs),
            # implicit-H mol against the implicit-H template.
            has_h = any(a.GetSymbol() == "H" for a in mol.GetAtoms())
            tpl = Chem.AddHs(template) if has_h else template
            mol = AllChem.AssignBondOrdersFromTemplate(tpl, mol)
            if mol is None:
                raise ValueError("AssignBondOrdersFromTemplate returned None")
        except Exception:
            # Fallback: geometry-based bond determination.
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=int(row["FormalCharge"]))
            except Exception:
                pass
    else:
        # No SMILES: geometry-based bond determination.
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=int(row["FormalCharge"]))
        except Exception:
            pass

    # Step 2: sanitize to compute radicals, aromaticity
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        return mol, "mol_corrupt", "sanitization failed"

    # Step 3: check enabled criteria
    atoms = list(mol.GetAtoms())

    if checks.get("charge"):
        net_charge = sum(atom.GetFormalCharge() for atom in atoms)
        if net_charge != 0:
            return mol, "rejected", f"charge={net_charge:+d}"

    if checks.get("zwitterion"):
        if _has_true_zwitterion(mol):
            return mol, "rejected", "zwitterion"

    if checks.get("radicals"):
        radicals = sum(atom.GetNumRadicalElectrons() for atom in atoms)
        if radicals > 0:
            return mol, "rejected", f"radicals={radicals}"

    # Step 4: topology validation — perceived SMILES vs original SMILES
    if original_smi and template is not None:
        try:
            mol_no_h = Chem.RemoveHs(Chem.Mol(mol))
            perceived = Chem.MolToSmiles(mol_no_h, isomericSmiles=False)
            original_flat = Chem.MolToSmiles(template, isomericSmiles=False)
            if perceived != original_flat:
                return mol, "topology_warning", "canonical mismatch"
        except Exception as e:
            return mol, "topology_warning", f"topology check failed: {e}"

    return mol, "ok", ""


# -- Parallel worker -----------------------------------------------------

_CHECKS = None
_ALLOWED_ELEMENTS = None
_MIN_HEAVY = None
_MAX_HEAVY = None
_MIN_FRAGMENTS = None
_MAX_FRAGMENTS = None


def _process_row_parallel(row):
    """Worker: composition check + full molecule processing for one row.

    Returns a dict with n_fragments, status, reason, and (for ok rows)
    the sanitized mol_block + atom/bond counts.
    """
    comp_ok, comp_reason, n_fragments = check_composition(
        row["mol_block"], _ALLOWED_ELEMENTS, _MIN_HEAVY, _MAX_HEAVY,
        _MIN_FRAGMENTS, _MAX_FRAGMENTS,
    )
    result = {"n_fragments": n_fragments}
    if not comp_ok:
        result["status"] = "rejected"
        result["reason"] = comp_reason
        return result

    mol, status, reason = process_molecule(row, _CHECKS)
    result["status"] = status
    result["reason"] = reason
    if status == "ok" and mol is not None:
        try:
            block = Chem.MolToMolBlock(mol)
            m_end = block.index("M  END") + len("M  END")
            result["mol_block"] = block[:m_end]
        except Exception:
            pass  # keep original mol_block if serialization fails
        result["num_atoms"] = mol.GetNumAtoms()
        result["num_bonds"] = mol.GetNumBonds()
    return result


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 3 — Chemical filter (neutral / non-zwitterion / closed-shell)."
    )
    p.add_argument("-i", "--input-dir", type=str, default="filtered_batches",
                   help="Input Parquet batch directory (default: filtered_batches/).")
    p.add_argument("-o", "--output-dir", type=str, default="curated_batches",
                   help="Output directory (default: curated_batches/).")
    p.add_argument("--rejects-dir", type=str, default="rejects/03_filter",
                   help="Rejected molecules SDF directory (default: rejects/03_filter/).")
    p.add_argument("--preset", type=str, default="neutral_closed_shell",
                   choices=list(PRESETS.keys()),
                   help="Filter preset: neutral_closed_shell (default), neutral, none.")
    p.add_argument("--allowed-elements", type=str, default=None,
                   help="Comma-separated allowed HEAVY elements (H is always "
                        "allowed; e.g. C,N,O,S). Molecules with any other "
                        "element are rejected.")
    p.add_argument("--min-heavy", type=int, default=None,
                   help="Minimum number of heavy (non-H) atoms.")
    p.add_argument("--max-heavy", type=int, default=None,
                   help="Maximum number of heavy (non-H) atoms.")
    p.add_argument("--min-fragments", type=int, default=None,
                   help="Minimum number of connected components (fragments). "
                        "Molecules with fewer fragments are rejected.")
    p.add_argument("--max-fragments", type=int, default=None,
                   help="Maximum number of connected components (fragments). "
                        "Molecules with more fragments are rejected. "
                        "Use 1 for single-molecule-only policy.")
    p.add_argument("--force-keep-rejected", action="store_true",
                   help="Keep rejected molecules in output (default: drop them).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for per-molecule processing (default: 1).")
    return p.parse_args()


def main():
    args = parse_args()
    from lib.provenance import record_run
    record_run(args.output_dir, "03_filter")
    checks = PRESETS[args.preset]
    allowed_elements = None
    if args.allowed_elements:
        allowed_elements = [e.strip() for e in args.allowed_elements.split(",")]

    batch_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".parquet")
    )
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Preset: {args.preset}")
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    total_ok = 0
    total_rejected = 0
    total_mol_corrupt = 0
    total_topology_warning = 0
    rejected_rows = []
    reason_counts = {}
    n_dropped = 0

    # Set worker globals once (fixed per run); forked workers inherit via COW.
    global _CHECKS, _ALLOWED_ELEMENTS, _MIN_HEAVY, _MAX_HEAVY, _MIN_FRAGMENTS, _MAX_FRAGMENTS
    _CHECKS = checks
    _ALLOWED_ELEMENTS = allowed_elements
    _MIN_HEAVY = args.min_heavy
    _MAX_HEAVY = args.max_heavy
    _MIN_FRAGMENTS = args.min_fragments
    _MAX_FRAGMENTS = args.max_fragments

    for fname in batch_files:
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        batch = read_batch(in_path)
        out_rows = []

        results = parallel_map(_process_row_parallel, batch,
                               n_workers=args.workers)

        for row, result in zip(batch, results):
            row["n_fragments"] = result["n_fragments"]
            status = result["status"]
            row["filter_status"] = status
            reason = result.get("reason")
            if reason:
                row["filter_reason"] = reason

            if status == "ok":
                if result.get("mol_block"):
                    row["mol_block"] = result["mol_block"]
                row["num_atoms"] = result.get("num_atoms", row["num_atoms"])
                row["num_bonds"] = result.get("num_bonds", row["num_bonds"])
                total_ok += 1
            elif status == "rejected":
                total_rejected += 1
                rejected_rows.append(row)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            elif status == "mol_corrupt":
                total_mol_corrupt += 1
            else:  # topology_warning
                total_topology_warning += 1

            out_rows.append(row)

        if not args.force_keep_rejected:
            n_before = len(out_rows)
            out_rows = [r for r in out_rows if r.get("filter_status") != "rejected"]
            n_dropped += n_before - len(out_rows)

        write_batch(out_path, out_rows)

    # -- Reject SDF -------------------------------------------------------
    if rejected_rows:
        reject_path = os.path.join(args.rejects_dir, "filter_rejected.sdf")
        write_reject_sdf(reject_path, rejected_rows,
                         reject_reason="filter_rejected")
        print(f"  {len(rejected_rows)} rejected → {reject_path}")

    # -- Report ------------------------------------------------------------
    total = total_ok + total_rejected + total_mol_corrupt + total_topology_warning
    print(f"\nReport")
    print(f"  Total:             {total}")
    print(f"  OK:                {total_ok} ({100*total_ok/total:.2f}%)" if total else "  OK: 0")
    print(f"  Rejected:          {total_rejected} ({100*total_rejected/total:.2f}%)" if total else "  Rejected: 0")
    print(f"  Mol corrupt:       {total_mol_corrupt}")
    print(f"  Topology warning:  {total_topology_warning}")
    if n_dropped:
        print(f"  Dropped:      {n_dropped} rejected (use --force-keep-rejected to keep)")

    if reason_counts:
        print(f"\nRejection breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    if total_rejected:
        sentinel = os.path.join(args.rejects_dir, ".REJECTED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{total_rejected} molecules rejected\n")


if __name__ == "__main__":
    main()
