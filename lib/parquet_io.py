"""
parquet_io.py — Read and write Parquet batch files for the AIM4ML pipeline.

Each Parquet batch is a table with one row per molecule.  The molecule
itself is stored as an SDF V2000 text block in the `mol_block` column;
all other columns are metadata extracted from the input SDF tags.

Public functions:
    write_batch(path, rows)   → write a list of dicts to a Parquet file
    read_batch(path)            → read a Parquet file → list of dicts
"""

import math
import os
import warnings

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import PARQUET_COLUMNS, PARQUET_OPTIONAL

# Column order for the Parquet file.
_COL_ORDER = list(PARQUET_COLUMNS.keys())


def _make_table(rows):
    """Convert a list of dicts into a PyArrow Table with a dynamic schema.

    Base columns come from PARQUET_COLUMNS.  Any additional keys present
    in the rows are included with type inferred from the first non-None value.
    """
    if not rows:
        arrays = {
            col: pa.array([], type=pa.from_numpy_dtype(dtype) if dtype != "string" else pa.string())
            for col, dtype in PARQUET_COLUMNS.items()
        }
        return pa.table(arrays)

    # Discover extra columns (not in base schema)
    known_cols = set(PARQUET_COLUMNS.keys())
    extra_cols = []
    for row in rows:
        for key in row:
            if key not in known_cols and key not in extra_cols:
                extra_cols.append(key)

    # Warn if columns are not registered (development safety net)
    if extra_cols:
        warnings.warn(
            f"Columns not registered in PARQUET_COLUMNS: {extra_cols}. "
            f"Add them to schema.py to ensure consistent dtypes across batches.",
            UserWarning,
            stacklevel=3,
        )

    # Full column order: base first, then extras
    full_order = list(PARQUET_COLUMNS.keys()) + extra_cols

    # Build column arrays
    data = {}
    for col in full_order:
        values = [row.get(col) for row in rows]
        if col in PARQUET_COLUMNS:
            dtype = PARQUET_COLUMNS[col]
        else:
            # Infer dtype: if any value is a string, treat as string.
            # Otherwise use first non-None value.
            has_string = any(isinstance(v, str) for v in values if v is not None)
            if has_string:
                dtype = "string"
            else:
                first = next((v for v in values if v is not None), None)
                if isinstance(first, bool):
                    dtype = "bool"
                elif isinstance(first, int):
                    dtype = "int32"
                elif isinstance(first, float):
                    dtype = "float64"
                else:
                    dtype = "string"

        if dtype == "string":
            data[col] = pa.array(
                [str(v) if v is not None else None for v in values], type=pa.string()
            )
        elif dtype == "bool":
            data[col] = pa.array(
                [bool(v) if v is not None else None for v in values], type=pa.bool_()
            )
        elif dtype.startswith("float"):
            data[col] = pa.array(
                [float(v) if v is not None else None for v in values], type=pa.float64()
            )
        elif dtype.startswith("int"):
            # Null ints round-trip as float NaN through pandas to_pandas();
            # treat both None and NaN as null here.
            data[col] = pa.array(
                [
                    int(v)
                    if v is not None and not (isinstance(v, float) and math.isnan(v))
                    else None
                    for v in values
                ],
                type=pa.int32(),
            )
        else:
            data[col] = pa.array(
                [str(v) if v is not None else None for v in values], type=pa.string()
            )

    return pa.table(data)


def write_batch(path, rows):
    """Write a batch of molecule rows to a Parquet file (atomically).

    Parameters
    ----------
    path : str
        Path to the output .parquet file.
    rows : list of dict
        Each dict has keys matching PARQUET_COLUMNS.

    Writes to a temp file then os.replace() so a killed process never
    leaves a truncated .parquet at `path`.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table = _make_table(rows)
    tmp = path + ".tmp." + str(os.getpid())
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def read_batch(path):
    """Read a Parquet batch file and return a list of dict rows.

    Parameters
    ----------
    path : str
        Path to the .parquet file.

    Returns
    -------
    list of dict
        Each dict has keys matching PARQUET_COLUMNS.
    """
    table = pq.read_table(path)
    df = table.to_pandas()
    # Ensure correct column order and fill missing optional columns
    for col in PARQUET_OPTIONAL:
        if col not in df.columns:
            df[col] = None
    # Base columns first, then any extra columns
    base_cols = list(PARQUET_COLUMNS.keys())
    extra_cols = [c for c in df.columns if c not in base_cols]
    rows = df[base_cols + extra_cols].to_dict(orient="records")
    # Pandas 3.0 returns NaN for missing strings instead of None.
    # Normalize NaN → None for string columns so downstream code
    # that checks `if value` works correctly.
    for row in rows:
        for key, val in row.items():
            if isinstance(val, float) and math.isnan(val):
                row[key] = None
    return rows
