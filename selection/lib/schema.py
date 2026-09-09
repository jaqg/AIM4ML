"""
schema.py — column definitions for selection-phase Parquet/CSV outputs.

Selection-specific columns only.  Curated-Parquet columns live in
``scripts/lib/schema.py`` (reused, not duplicated).
"""

# descriptors.parquet — written by 01_descriptors.py
DESCRIPTOR_COLUMNS = [
    "CompoundID",    # str   — MD5 canonical-SMILES identity (from curation)
    "smiles",        # str   — explicit-H canonical SMILES (from curation)
    "mol_block",     # str   — SDF V2000 block (geometry + bond orders), kept
                     #         for PM7-stage geometry export
    "formula",       # str   — molecular formula (from curation, if present)
    "nat_heavy",     # int   — heavy-atom count
    "molwt",         # float — molecular weight (RDKit)
    "tpsa",          # float — topological polar surface area (RDKit)
    "energy_ha",     # float — SDF Energy_Ha tag (passthrough)
    "scaffold",      # str   — Bemis-Murcko scaffold SMILES (None = acyclic)
    "morgan_fp",     # list[int]    — ECFP4 on-bit indices (chirality ON)
    "atom_envs",     # list[list[int]] — unfolded env ids per heavy atom
    "heavy_elements",  # list[str]  — element symbol per heavy atom
    "bonds",         # list[str]    — 'A-B:order' per bond
]

# selection.parquet — written by 03_select.py
SELECTION_COLUMNS = [
    "CompoundID",    # str — selected molecule
    "smiles",        # str
    "mol_block",     # str — for PM7-stage geometry export
    "pass",          # str — floor | diversity | depth
    "tier",          # str — tier1 | tier2 (diversity pass only, else "")
    "score",         # float — pass-specific selection score
    "selection_order",  # int — global greedy order (1 = first selected)
]

# analysis CSVs — written by 02_env_analysis.py
ENV_FREQUENCY_COLUMNS = ["env_id", "element", "n_atoms", "n_mols"]
COST_CURVE_COLUMNS = ["mols_picked", "envs_covered"]
FORCED_CHOICE_COLUMNS = ["env_id", "element", "n_mols", "pick_type", "carrier_ids"]
BOND_TYPES_COLUMNS = ["bond", "count"]
