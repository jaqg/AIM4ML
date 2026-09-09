"""
util.py — shared helpers for selection tests (curated-parquet fixtures).
"""

import hashlib

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def curated_row(smi, energy=-500.0):
    """One curated-parquet row, mirroring the real schema.

    Includes every ``scripts/lib/schema.PARQUET_COLUMNS`` key — ``read_batch``
    indexes the base columns unconditionally, so they must all be present.
    """
    mol = Chem.MolFromSmiles(smi)
    molh = Chem.AddHs(mol)
    canon = Chem.MolToSmiles(molh)
    return {
        "CompoundID": hashlib.md5(canon.encode()).hexdigest(),
        "Energy_Ha": energy,
        "FormalCharge": 0,
        "Multiplicity": 1,
        "SMILES": Chem.MolToSmiles(mol),      # implicit-H original
        "SourceID": "TEST",
        "HOMO_Ha": None,
        "LUMO_Ha": None,
        "HL_Gap_Ha": None,
        "PartialCharges": None,
        "num_atoms": molh.GetNumAtoms(),
        "num_bonds": molh.GetNumBonds(),
        "n_fragments": 1,
        "mol_block": Chem.MolToMolBlock(molh),
        "AtropisomerKey": "",
        "CanonicalSMILES": canon,
        "Formula": rdMolDescriptors.CalcMolFormula(molh),
    }


def curated_df(smiles_list):
    return pd.DataFrame([curated_row(s) for s in smiles_list])
