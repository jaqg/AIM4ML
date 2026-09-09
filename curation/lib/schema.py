"""
schema.py — Canonical SDF property tag definitions for the AIM4ML curation pipeline.

Every stage references these constants. A single source of truth so changing
a tag name or type only requires an edit here.
"""

import hashlib

# -- Required tags (pipeline will reject molecules missing these) ----------

REQUIRED_TAGS = {
    "Energy_Ha":    float,
    "FormalCharge": int,
    "Multiplicity": int,
}

# -- Recommended tags (pipeline warns but does not reject) -----------------

RECOMMENDED_TAGS = {
    "SMILES":   str,
    "SourceID": str,
}

# -- Optional pass-through tags (validated if present, ignored if absent) ---

OPTIONAL_TAGS = {
    "HOMO_Ha":        float,
    "LUMO_Ha":        float,
    "HL_Gap_Ha":      float,
    "PartialCharges": str,          # space-separated floats
}

# -- Pipeline-computed tags (added by stages, not present in input SDF) -----

COMPUTED_TAGS = {
    "CompoundID":       str,
    "CanonicalSMILES":  str,
    "ICONF":            int,
    "sdf_status":       str,
    "stereo_status":    str,
    "MolWt":            float,
    "TPSA":             float,
    "logP":             float,
    "nrot":             int,
}

# -- Utility ---------------------------------------------------------------

def compound_id(canonical_smiles):
    """
    CompoundID = MD5(canonical SMILES).
    Stable, deterministic, SMILES-defined molecular identity.
    """
    return hashlib.md5(canonical_smiles.encode("utf-8")).hexdigest()


# -- Parquet schema ------------------------------------------------------
# Column names and dtypes for the internal Parquet batch files.
# Register every column written by any pipeline stage here so dtype
# inference never falls back to string on all-null batches.

PARQUET_COLUMNS = {
    # -- Input properties (from SDF tags) --
    "Energy_Ha":        "float64",
    "FormalCharge":     "int32",
    "Multiplicity":     "int16",
    "SMILES":           "string",
    "SourceID":         "string",
    "HOMO_Ha":          "float64",
    "LUMO_Ha":          "float64",
    "HL_Gap_Ha":        "float64",
    "PartialCharges":   "string",

    # -- Structural metadata (Stage 1) --
    "num_atoms":        "int32",
    "num_bonds":        "int32",
    "mol_block":        "string",

    # -- Stage 2: energy prefilter --
    "energy_status":    "string",

    # -- Stage 3: chemical filter --
    "filter_status":    "string",
    "filter_reason":    "string",
    "n_fragments":      "int32",

    # -- Stage 4: dedup --
    "CanonicalSMILES":  "string",
    "CompoundID":       "string",
    "AtropisomerKey":   "string",
    "ICONF":            "int32",
    "Formula":          "string",
    "dedup_status":     "string",
    "dedup_reason":     "string",

    # -- Stage 6: stereo filter --
    "stereo_status":    "string",

    # -- Stage 7: reorder --
    "reorder_status":   "string",

    # -- Stage 8: conformer filter --
    "conformer_status": "string",
    "cluster_id":       "int32",
}

# Columns that are always present (no nulls after split).
PARQUET_REQUIRED = ["mol_block", "num_atoms", "num_bonds",
                    "Energy_Ha", "FormalCharge", "Multiplicity"]

# All non-required columns — may be null if absent at that stage.
PARQUET_OPTIONAL = [c for c in PARQUET_COLUMNS if c not in PARQUET_REQUIRED]
