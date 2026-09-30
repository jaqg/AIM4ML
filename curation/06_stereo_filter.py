#!/usr/bin/env python3
"""
06_stereo_filter.py — Stage 6: Enantiomer detection (keep by default).

Groups molecules by flat SMILES (stereo markers stripped).  Within each
multi-member group, checks whether any pair are enantiomers by inverting
all chiral stereocenters (tetrahedral + atropisomer) of one molecule and
comparing to the other.  Double bonds are NOT inverted — E/Z isomers are
diastereomers, not enantiomers (mirror reflection keeps E/Z).

Molecules are reconstructed from the mol_block column (preserves
tetrahedral, double-bond, and atropisomer stereo); CanonicalSMILES is the
fallback if mol_block is missing.

Flags:
  --remove-enantiomers — opt-in: remove one enantiomer per detected pair
                          (default KEEPS all enantiomers — D09-overturn)
  --robust-stereo       — always use substructure-based stereo comparison
                          (slower, bypasses RDKit canonicalization bugs)

Auto-fallback: if a molecule contains a spiro/bridgehead stereocenter or
an atropisomer bond, the robust path is used for that group automatically.

Usage:
    python3 06_stereo_filter.py -i deduped_batches/ -o stereo_batches/
    python3 06_stereo_filter.py --remove-enantiomers  # legacy: drop one per pair
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.rdchem import BondStereo, ChiralType

RDLogger.DisableLog("rdApp.*")

from lib.parquet_io import read_batch, write_batch
from lib.rdkit_version import check_min_rdkit
from lib.sdf_io import write_reject_sdf

# -- Stereo helpers -------------------------------------------------------


def flat_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return smiles
    Chem.SanitizeMol(mol, catchErrors=True)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _mol_from_row(row, can_smi):
    """Reconstruct mol from mol_block, falling back to CanonicalSMILES.

    MolFromMolBlock with default sanitize=True preserves tetrahedral,
    E/Z (perceived from 2D/3D coordinates), and atropisomer stereo.
    """
    mb = row.get("mol_block")
    if mb:
        mol = Chem.MolFromMolBlock(mb)  # sanitize=True, removeHs=True
        if mol is not None:
            return mol
    return Chem.MolFromSmiles(can_smi)


def has_tetrahedral_stereo(mol):
    return any(
        atom.GetChiralTag()
        in (
            ChiralType.CHI_TETRAHEDRAL_CW,
            ChiralType.CHI_TETRAHEDRAL_CCW,
        )
        for atom in mol.GetAtoms()
    )


def invert_all_stereo(mol):
    """Invert all CHIRAL stereochemistry (mirror image operation).

    Tetrahedral: CW ↔ CCW
    Atropisomer (axial): CW ↔ CCW

    Double bonds are NOT inverted — E/Z isomers are diastereomers, not
    enantiomers (mirror reflection keeps E/Z unchanged).
    """
    rw = Chem.RWMol(Chem.Mol(mol))

    # Atom (tetrahedral) chirality
    for atom in rw.GetAtoms():
        chi = atom.GetChiralTag()
        if chi == ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CCW)
        elif chi == ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)

    # Bond (atropisomer / axial) chirality
    for bond in rw.GetBonds():
        stereo = bond.GetStereo()
        if stereo == BondStereo.STEREOATROPCW:
            bond.SetStereo(BondStereo.STEREOATROPCCW)
        elif stereo == BondStereo.STEREOATROPCCW:
            bond.SetStereo(BondStereo.STEREOATROPCW)

    return rw.GetMol()


def _has_spiro_chiral(mol):
    """Heuristic: atom belongs to ≥2 SSSR rings AND has chiral tag.

    Triggers robust comparison for RDKit bug #9391 (canonical SMILES
    round-trip flips chirality on spiro systems with chiral N+).
    """
    sssr = Chem.GetSSSR(mol)
    if not sssr:
        return False
    # Count ring memberships per atom
    ring_count = [0] * mol.GetNumAtoms()
    for ring in sssr:
        for idx in ring:
            ring_count[idx] += 1
    for atom in mol.GetAtoms():
        if ring_count[atom.GetIdx()] >= 2:
            chi = atom.GetChiralTag()
            if chi in (ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW):
                return True
    return False


def _has_atropisomer(mol):
    """True if mol has atropisomer (axial) bond stereo."""
    si = Chem.FindPotentialStereo(mol)
    return any(info.type == Chem.StereoType.Bond_Atropisomer for info in si)


def _stereo_descriptors(mol):
    """Return dict mapping (kind, idx) → stereo descriptor.

    kind: 'tetrahedral' | 'double_bond' | 'atropisomer'.
    Uses FindPotentialStereo so descriptors are independent of
    canonicalization order.
    """
    Chem.SanitizeMol(mol, catchErrors=True)
    si = Chem.FindPotentialStereo(mol)
    desc = {}
    for info in si:
        if info.descriptor == Chem.rdchem.StereoDescriptor.NoValue:
            continue  # potential-but-unspecified center (not real stereo)
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            desc[("tetrahedral", info.centeredOn)] = info.descriptor
        elif info.type == Chem.StereoType.Bond_Double:
            desc[("double_bond", info.centeredOn)] = info.descriptor
        elif info.type == Chem.StereoType.Bond_Atropisomer:
            bond = mol.GetBondWithIdx(info.centeredOn)
            desc[("atropisomer", info.centeredOn)] = bond.GetStereo()
    return desc


def are_enantiomers_robust(mol_a, mol_b):
    """Substructure-based enantiomer check.

    Enantiomers iff:
      1. same connectivity (atom mapping via substructure match)
      2. all CHIRAL centers (tetrahedral + atropisomer) OPPOSITE
      3. all double bonds EQUAL (E/Z is diastereomeric, mirror keeps it)
    """
    # Connectivity-only match (no stereo constraints)
    match = mol_a.GetSubstructMatch(mol_b)
    if not match:
        return False

    # Build forward mapping: mol_a idx → mol_b idx
    a_to_b = {i: match[i] for i in range(len(match))}

    # Bond mapping: mol_a bond → mol_b bond via atom endpoints
    bond_map = {}
    for bond_a in mol_a.GetBonds():
        ib = a_to_b.get(bond_a.GetBeginAtomIdx())
        jb = a_to_b.get(bond_a.GetEndAtomIdx())
        if ib is None or jb is None:
            return False
        bond_b = mol_b.GetBondBetweenAtoms(ib, jb)
        if bond_b is None:
            return False
        bond_map[bond_a.GetIdx()] = bond_b.GetIdx()

    desc_a = _stereo_descriptors(mol_a)
    desc_b = _stereo_descriptors(mol_b)

    # Double bonds must be EQUAL (not opposite)
    for (kind, idx_a), d_a in desc_a.items():
        if kind != "double_bond":
            continue
        idx_b = bond_map.get(idx_a)
        d_b = desc_b.get(("double_bond", idx_b)) if idx_b is not None else None
        if d_b is None or d_a != d_b:
            return False

    # Chiral centers must ALL be OPPOSITE
    chiral_keys = [(k, i) for (k, i) in desc_a if k in ("tetrahedral", "atropisomer")]
    if not chiral_keys:
        return False
    n_opposite = 0
    for kind, idx_a in chiral_keys:
        if kind == "tetrahedral":
            idx_b = a_to_b.get(idx_a)
        else:  # atropisomer
            idx_b = bond_map.get(idx_a)
        if idx_b is None:
            return False
        d_a = desc_a[(kind, idx_a)]
        d_b = desc_b.get((kind, idx_b))
        if d_b is None:
            return False
        if _are_opposite_descriptors(d_a, d_b):
            n_opposite += 1
    return n_opposite == len(chiral_keys)


def _are_opposite_descriptors(d_a, d_b):
    """Check if two CHIRAL stereo descriptors are opposite.

    Tetrahedral: StereoDescriptor enum (Tet_CW ↔ Tet_CCW).
    Atropisomer: BondStereo enum (STEREOATROPCW ↔ STEREOATROPCCW).
    Double bonds excluded — E/Z is diastereomeric, not enantiomeric.
    """
    SD = Chem.rdchem.StereoDescriptor
    BS = Chem.BondStereo
    opp_pairs = {
        (SD.Tet_CW, SD.Tet_CCW),
        (SD.Tet_CCW, SD.Tet_CW),
        (BS.STEREOATROPCW, BS.STEREOATROPCCW),
        (BS.STEREOATROPCCW, BS.STEREOATROPCW),
    }
    return (d_a, d_b) in opp_pairs


# -- Main ----------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="AIM4ML Stage 6 — Enantiomer filter (SMILES-based).")
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
        default="stereo_batches",
        help="Output directory (default: stereo_batches/).",
    )
    p.add_argument(
        "--rejects-dir",
        type=str,
        default="rejects/06_stereo_filter",
        help="Rejected molecules SDF directory (default: rejects/06_stereo_filter/).",
    )
    p.add_argument(
        "--remove-enantiomers",
        action="store_true",
        help="Remove one enantiomer per detected pair "
        "(default: keep all enantiomers — D09-overturn).",
    )
    p.add_argument(
        "--robust-stereo",
        action="store_true",
        help="Always use substructure-based stereo comparison (slower, correct).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from lib.provenance import record_run

    record_run(args.output_dir, "06_stereo_filter")
    args.keep_enantiomers = not args.remove_enantiomers

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

    # -- Pass 1: accumulate rows, build flat-SMILES groups ---------------
    all_rows = []
    total_mols = 0

    for fname in batch_files:
        path = os.path.join(args.input_dir, fname)
        batch = read_batch(path)
        for row in batch:
            all_rows.append(row)
            row["_batch_file"] = fname
        total_mols += len(batch)

    print(f"  Loaded {total_mols} molecules")

    # Build mol dict: index → mol (from mol_block, fallback CanonicalSMILES)
    flat_groups = {}  # flat_smiles → list of (row_idx, CompoundID)
    idx_to_mol = {}

    for idx, row in enumerate(all_rows):
        cid = row.get("CompoundID")
        can_smi = row.get("CanonicalSMILES")
        if not can_smi:
            row["stereo_status"] = "skipped_no_smiles"
            continue
        mol = _mol_from_row(row, can_smi)
        if mol is None:
            row["stereo_status"] = "skipped_no_smiles"
            continue
        # Detect multi-fragment complexes → skip enantiomer detection
        frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(frags) > 1:
            row["stereo_status"] = "complex"
            continue
        idx_to_mol[idx] = mol
        flat = flat_smiles(can_smi)
        flat_groups.setdefault(flat, []).append((idx, cid))

    # -- Pass 2: detect enantiomer pairs within each group ----------------
    n_removed = 0
    n_kept = 0
    n_complex = 0
    rejected_rows = []
    n_dropped = 0
    n_pairs_detected = 0
    n_atrop_pairs_detected = 0
    n_robust_fallback = 0

    multi_groups = {k: v for k, v in flat_groups.items() if len(v) > 1}
    print(f"  Stereo groups (>1 member): {len(multi_groups)}")

    for flat_smi, members in multi_groups.items():
        mols = [idx_to_mol.get(idx) for idx, _ in members]

        # Group-level dispatch: robust if ANY member has spiro-chiral
        # or atropisomer stereo (canonical-SMILES path unreliable there).
        needs_robust = args.robust_stereo or any(
            m is not None and (_has_spiro_chiral(m) or _has_atropisomer(m)) for m in mols
        )
        if needs_robust and not args.robust_stereo:
            n_robust_fallback += 1

        if needs_robust:
            # -- Robust path: pairwise substructure comparison -----------
            n = len(members)
            keep = [True] * n
            for i in range(n):
                if not keep[i]:
                    continue
                mol_i = mols[i]
                if mol_i is None:
                    continue
                for j in range(i + 1, n):
                    if not keep[j]:
                        continue
                    idx_j = members[j][0]
                    mol_j = mols[j]
                    if mol_j is None:
                        continue
                    if are_enantiomers_robust(mol_i, mol_j):
                        n_pairs_detected += 1
                        if _has_atropisomer(mol_i) or _has_atropisomer(mol_j):
                            n_atrop_pairs_detected += 1
                        if not args.keep_enantiomers:
                            keep[j] = False
                            all_rows[idx_j]["stereo_status"] = "removed_enantiomer"
                            n_removed += 1
                            rejected_rows.append(all_rows[idx_j])
        else:
            # -- Fast path: O(n) enantiomer-pair hashing ----------------
            # Collapse conformer duplicates by canonical SMILES first
            # (identical stereo → identical SMILES), then bucket unique
            # SMILES.  Removing a stereoisomer removes ALL its conformers
            # (order-independent full pair collapse).
            smi_to_idxs = {}
            for k, (idx, _) in enumerate(members):
                m = mols[k]
                if m is None:
                    continue
                if not has_tetrahedral_stereo(m):
                    continue  # achiral → no enantiomer
                s = Chem.MolToSmiles(m, isomericSmiles=True)
                smi_to_idxs.setdefault(s, []).append(idx)

            buckets = {}
            for s, idxs in smi_to_idxs.items():
                m = idx_to_mol[idxs[0]]
                s_inv = Chem.MolToSmiles(invert_all_stereo(m), isomericSmiles=True)
                buckets.setdefault(min(s, s_inv), []).append(s)

            for key, smis in buckets.items():
                if len(smis) >= 2:  # == 2: one unique enantiomer pair
                    n_pairs_detected += 1
                    if not args.keep_enantiomers:
                        for s in smis[1:]:
                            for idx in smi_to_idxs[s]:
                                all_rows[idx]["stereo_status"] = "removed_enantiomer"
                                n_removed += 1
                                rejected_rows.append(all_rows[idx])

        # -- Mark kept (unified — works for both paths and both modes) ---
        for k, (idx, _) in enumerate(members):
            if all_rows[idx].get("stereo_status") is None:
                all_rows[idx]["stereo_status"] = "kept"
                n_kept += 1

    # Mark kept for single-member groups
    for flat_smi, members in flat_groups.items():
        if len(members) == 1:
            idx = members[0][0]
            if all_rows[idx].get("stereo_status") is None:
                all_rows[idx]["stereo_status"] = "kept"
                n_kept += 1

    for row in all_rows:
        if row.get("stereo_status") == "complex":
            n_complex += 1

    # -- Write output batches ---------------------------------------------
    batch_groups = {}
    for row in all_rows:
        fname = row.pop("_batch_file")
        batch_groups.setdefault(fname, []).append(row)

    for fname, rows in batch_groups.items():
        out_path = os.path.join(args.output_dir, fname)
        n_before = len(rows)
        rows = [r for r in rows if r.get("stereo_status") != "removed_enantiomer"]
        n_dropped += n_before - len(rows)
        write_batch(out_path, rows)

    # -- Reject SDF -------------------------------------------------------
    if rejected_rows:
        reject_path = os.path.join(args.rejects_dir, "stereo_removed.sdf")
        write_reject_sdf(reject_path, rejected_rows, reject_reason="removed_enantiomer")
        print(f"  {len(rejected_rows)} enantiomers → {reject_path}")

    # -- Report ------------------------------------------------------------
    total = total_mols
    print("\nReport")
    print(f"  Total:                 {total}")
    print(f"  Kept:                  {n_kept}")
    print(f"  Removed enantiomers:   {n_removed}")
    print(f"  Enantiomer pairs detected: {n_pairs_detected}")
    if n_atrop_pairs_detected:
        print(f"  Atropisomer pairs detected: {n_atrop_pairs_detected}")
    if args.keep_enantiomers:
        print("  (enantiomers kept by default; use --remove-enantiomers to drop one per pair)")
    print(f"  Complexes (skipped):   {n_complex}")
    if n_robust_fallback:
        print(f"  Robust fallback groups: {n_robust_fallback}")
    if n_dropped:
        print(f"  Dropped:               {n_dropped} removed_enantiomer")

    if n_removed:
        sentinel = os.path.join(args.rejects_dir, ".REMOVED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{n_removed} enantiomers removed\n")


if __name__ == "__main__":
    main()
