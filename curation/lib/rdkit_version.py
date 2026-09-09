"""rdkit_version.py — RDKit minimum-version gate for stereochemistry features."""
import re
from rdkit import rdBase

MIN_RDKIT = (2024, 3)  # atropisomer support (PR #6903)


def check_min_rdkit(min_version=MIN_RDKIT):
    """True if the running RDKit is >= min_version (year, month)."""
    v = rdBase.rdkitVersion  # e.g. '2026.03.3'
    m = re.match(r"(\d+)\.(\d+)", v)
    if not m:
        return False
    return (int(m.group(1)), int(m.group(2))) >= min_version
