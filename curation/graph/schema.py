"""
schema.py — Graph-track identity semantics (Lewis/SMILES world).

CompoundID = MD5(canonical SMILES): stable, deterministic, SMILES-defined
molecular identity (D62: SMILES authoritative for bonds/charges).

Identity-neutral tag/column registries shared with the realspace track live
in lib/schema.py.
"""

import hashlib


def compound_id(canonical_smiles):
    """
    CompoundID = MD5(canonical SMILES).
    Stable, deterministic, SMILES-defined molecular identity.
    """
    return hashlib.md5(canonical_smiles.encode("utf-8")).hexdigest()
