"""
schema.py — Realspace-track column contract + text codecs (Lewis/SMILES-free).

Identity-neutral registries shared with the graph track live in lib/schema.py.
Graph-track identity semantics live in curation/graph/schema.py. This module
owns the realspace registry (columns written by realspace stages) and the
text codecs that move geometry through Parquet.

Storage contract (worker plan Rev 1, T2)
---------------------------------------
Nested-list columns are DEAD: ``lib/parquet_io._make_table`` stringifies list
values (lib is frozen, additive-only). All realspace columns are therefore
scalar and flow through ``lib/parquet_io.write_batch``/``read_batch``
unchanged — same precedent as the graph track storing geometry as the
``mol_block`` text column:

- ``coords``    : text, one ``x y z`` per atom line, ``%.6f`` fixed decimals
                  (registry identity needs 10^-3 A only; 10^-6 storage is a
                  strict superset — see lib/xyz_io.py writer policy).
- ``symbols``   : text, space-joined IUPAC symbols (atom order = coords order).
- ``charges``   : text, space-joined ``%.6f`` per-atom charges; NULL when the
                  source frame carries no per-atom charges.
- ``formula``   : Hill-system formula derived at ingest (composition-group key).
- ``header``    : full extxyz comment header as a JSON string (sorted keys).

Registry warning policy
-----------------------
Realspace columns are intentionally NOT registered in ``lib/schema.py``
PARQUET_COLUMNS (lib frozen — additive-only). ``lib/parquet_io`` emits
``UserWarning: Columns not registered in PARQUET_COLUMNS: ...`` on every write
carrying them. Stages MUST write through ``write_realspace_batch`` here, which
suppresses exactly that message and turns every other warning into an error
("loud on unexpected warnings").
"""

import contextlib
import json
import warnings
from collections import Counter

import numpy as np

from lib.parquet_io import write_batch
from lib.xyz_io import XyzFrame

# -- Realspace Parquet column registry -------------------------------------
# Mirrors lib/schema.py layout. Every column written by realspace stages,
# with its intended dtype (scalar — the lib inference path maps these
# exactly: str -> string, int -> int32).
REALSPACE_COLUMNS = {
    # -- Identity + geometry (ingest 01) --
    "geometry_id": "string",  # geometry_key MD5 hexdigest (computed at ingest)
    "symbols": "string",  # space-joined element symbols
    "n_atoms": "int32",  # == len(symbols) == coord line count
    "coords": "string",  # "%.6f" x y z lines
    "charges": "string",  # space-joined "%.6f"; NULL if frame had none
    "formula": "string",  # Hill formula, derived at ingest
    "charge": "int32",  # total charge; ingest fills 0 when header lacks it
    "multiplicity": "int32",  # ingest fills 1 when header lacks it
    # -- Provenance (ingest 01) --
    "source_id": "string",  # "<basename>:<frame_index>"
    "source_file": "string",  # input file name
    "source_index": "int32",  # frame index within source file
    "header": "string",  # extxyz comment header as JSON (sorted keys)
    # -- Stage-added later --
    "dedup_status": "string",  # 02_identity: "kept" / "exact_dup" / "near_dup"
    "cluster_id": "int32",  # 02_identity: near-dup cluster representative
}

# Columns that may legitimately be NULL in a realspace batch.
REALSPACE_OPTIONAL = ["charges", "dedup_status", "cluster_id"]


# -- Text codecs (frame <-> parquet row) ------------------------------------


def coords_to_text(coords: np.ndarray) -> str:
    """(N, 3) float64 -> "%.6f" x y z lines (one per atom)."""
    return "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in coords)


def coords_from_text(text: str) -> np.ndarray:
    """\"%.6f\" x y z lines -> (N, 3) float64. Value-exact for %.6f-representable input."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.array([[float(v) for v in ln.split()] for ln in lines], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords text must be 'x y z' per line, got shape {arr.shape}")
    return arr


def symbols_to_text(symbols: list[str]) -> str:
    """List of element symbols -> space-joined text."""
    return " ".join(symbols)


def symbols_from_text(text: str) -> list[str]:
    """Space-joined element symbols -> list."""
    return text.split()


def charges_to_text(charges: np.ndarray | None) -> str | None:
    """Per-atom charges -> space-joined "%.6f" text; None stays None."""
    if charges is None:
        return None
    return " ".join(f"{q:.6f}" for q in charges)


def charges_from_text(text: str | None) -> np.ndarray | None:
    """Space-joined "%.6f" charges -> (N,) float64; None/empty stays None."""
    if text is None or not text.strip():
        return None
    return np.array([float(v) for v in text.split()], dtype=np.float64)


def header_to_json(header: dict[str, str]) -> str:
    """Header dict -> deterministic JSON string (sorted keys)."""
    return json.dumps(header, sort_keys=True)


def header_from_json(text: str | None) -> dict[str, str]:
    """JSON string -> header dict; None/empty -> {}."""
    if text is None or not text.strip():
        return {}
    return json.loads(text)


def hill_formula(symbols: list[str]) -> str:
    """Hill-system formula from element symbols.

    Hill order: C first, H second (only when carbon is present), all other
    elements alphabetical. Without carbon, ALL elements are alphabetical
    (H included, e.g. HeH+ -> "HHe"). Counts of 1 are omitted (Li4 -> "Li4").
    """
    counts = Counter(symbols)
    if not counts:
        return ""
    parts: list[str] = []
    if "C" in counts:
        for el in ("C", "H"):
            if el in counts:
                n = counts.pop(el)
                parts.append(el if n == 1 else f"{el}{n}")
    for el in sorted(counts):
        n = counts[el]
        parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts)


def frame_to_row(
    frame: XyzFrame,
    *,
    geometry_id: str,
    charge: int,
    multiplicity: int,
    source_id: str,
    source_file: str,
    source_index: int,
) -> dict:
    """XyzFrame + provenance -> one realspace Parquet row (all T2 columns).

    geometry_id comes from curation.realspace.identity.geometry_key; charge /
    multiplicity are passed EXPLICITLY (ingest fills header defaults first —
    this function never invents them).
    """
    n_atoms = len(frame.symbols)
    if frame.coords.shape != (n_atoms, 3):
        raise ValueError(f"coords shape {frame.coords.shape} does not match {n_atoms} symbols")
    if frame.charges is not None and len(frame.charges) != n_atoms:
        raise ValueError(f"charges length {len(frame.charges)} does not match {n_atoms} symbols")
    return {
        "geometry_id": geometry_id,
        "symbols": symbols_to_text(frame.symbols),
        "n_atoms": n_atoms,
        "coords": coords_to_text(frame.coords),
        "charges": charges_to_text(frame.charges),
        "formula": hill_formula(frame.symbols),
        "charge": charge,
        "multiplicity": multiplicity,
        "source_id": source_id,
        "source_file": source_file,
        "source_index": source_index,
        "header": header_to_json(frame.header),
    }


def row_to_frame(row: dict) -> XyzFrame:
    """One realspace Parquet row -> XyzFrame (symbols/coords/charges/header)."""
    return XyzFrame(
        symbols=symbols_from_text(row["symbols"]),
        coords=coords_from_text(row["coords"]),
        charges=charges_from_text(row.get("charges")),
        header=header_from_json(row.get("header")),
    )


# -- Scoped Parquet write ---------------------------------------------------


@contextlib.contextmanager
def _ignore_unregistered_columns_warning():
    """Suppress ONLY the known lib warning; every other warning becomes an error.

    lib/parquet_io warns (UserWarning) about columns outside the frozen
    PARQUET_COLUMNS registry — expected for realspace columns. Anything else
    firing during a write is unexpected and must surface loudly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # unexpected warnings -> raised
        # Checked first (filters are consulted front-to-back): known message ignored.
        warnings.filterwarnings(
            "ignore",
            message=r"^Columns not registered in PARQUET_COLUMNS:",
            category=UserWarning,
        )
        yield


def write_realspace_batch(path: str, rows: list[dict]) -> None:
    """Write realspace rows via lib/parquet_io.write_batch with scoped warnings.

    Realspace columns are intentionally unregistered in frozen lib/schema.py;
    this wrapper suppresses exactly that notice and escalates everything else.
    """
    with _ignore_unregistered_columns_warning():
        write_batch(path, rows)
