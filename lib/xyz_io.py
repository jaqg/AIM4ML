"""
xyz_io.py — Read/write extxyz and plain xyz files (identity-neutral I/O).

Lives in lib/ because it is pure format I/O, shared by any track. Geometry
identity semantics (geometry_key, RMSD) live in curation/realspace/ (mirrors
compound_id in curation/graph/schema.py).

Formats
-------
extxyz: first line = atom count; second line = whitespace-separated
``key=value`` pairs (values may be double-quoted to contain spaces, e.g.
``Lattice="10 0 0 0 10 0 0 0 10"``); remaining lines = atom records.
Per-atom columns are declared by the ``Properties`` header key, e.g.
``species:S:1:pos:R:3:charge:R:1`` (name:type:count triplets). This reader
honors ``species`` (element symbols), ``pos`` (x y z) and ``charge``
(per-atom, optional). Any other per-atom column is ignored: its declaration
survives in ``header["Properties"]`` but the data is not loaded.

plain xyz: the comment line is free text; when it is not a full set of
``key=value`` pairs it is stored verbatim as ``header["title"]``. Plain xyz
atom lines must be exactly ``symbol x y z`` (extra columns are rejected —
sources delivering richer bare xyz must normalize upstream).

Policies (settled in the realspace worker plan, Rev 1)
------------------------------------------------------
- The reader invents NO defaults: a frame without ``charge`` /
  ``multiplicity`` header keys stays without them. Ingest fills defaults
  explicitly, later (nothing silently invented).
- Non-finite coordinates or per-atom charges (nan/inf) are rejected: they
  poison the canonical (Z, x, y, z) atom sort used by geometry identity.
- The writer emits coordinates and per-atom charges with ``%.6f`` fixed
  decimals (delivery convention, mirrors curation/common/10_extxyz.py).
  Written text is lossy beyond 10^-6 by design — registry identity needs
  10^-3 A only — but the float64 round trip is *value-identical* for any
  input that is %.6f-representable: parse -> write -> parse yields the
  identical float64 array. Test fixtures pin exactly that class.
- ``Properties`` in the output header is always derived from frame data
  (symbols + coords + charges), never trusted from ``header``.
- Frames whose header is exactly ``{"title": ...}`` (non-empty title, no
  other keys, no per-atom charges) are written as plain xyz: the comment
  stays free text so plain sources round-trip verbatim.
- Malformed input raises XyzFormatError (a ValueError) carrying file:line
  context.
"""

import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Sequence

import numpy as np

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class XyzFormatError(ValueError):
    """Malformed xyz/extxyz input. Message carries file:line context."""


@dataclass(slots=True)
class XyzFrame:
    """One xyz/extxyz frame: atom symbols + coordinates + optional extras.

    coords: (N, 3) float64. charges: (N,) float64, optional per-atom charge.
    header: comment-line key/values as strings (extxyz pairs, or {"title":
    ...} for plain xyz). The reader never invents defaults: missing
    charge/multiplicity stay missing from this dict.
    """

    symbols: list[str]
    coords: np.ndarray
    charges: np.ndarray | None = None
    header: dict[str, str] = field(default_factory=dict)


def _parse_comment(line: str) -> dict[str, str]:
    """Parse an xyz comment line into a header dict.

    Every whitespace-separated token must be an ``identifier=value`` pair
    (values may carry quotes/spaces, handled by shlex); otherwise the whole
    line is kept verbatim as ``{"title": ...}``.
    """
    stripped = line.strip()
    if not stripped:
        return {}
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        # Unbalanced quotes -> free text, not a key=value header.
        return {"title": stripped}
    pairs: dict[str, str] = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep or not _KEY_RE.match(key):
            return {"title": stripped}
        pairs[key] = value
    return pairs


def _parse_properties(spec: str, lineno: int) -> list[tuple[str, str, int]]:
    """Parse an extxyz Properties spec into (name, type, count) triplets."""
    parts = spec.split(":")
    if len(parts) % 3 != 0:
        raise XyzFormatError(
            f"line {lineno}: Properties must be name:type:count triplets: {spec!r}"
        )
    cols: list[tuple[str, str, int]] = []
    for i in range(0, len(parts), 3):
        name, typ, count_s = parts[i], parts[i + 1], parts[i + 2]
        if not _KEY_RE.match(name):
            raise XyzFormatError(f"line {lineno}: bad Properties column name {name!r}")
        try:
            count = int(count_s)
        except ValueError as exc:
            raise XyzFormatError(
                f"line {lineno}: bad Properties column count {count_s!r} in {spec!r}"
            ) from exc
        if count < 1:
            raise XyzFormatError(f"line {lineno}: Properties column count must be >= 1: {spec!r}")
        cols.append((name, typ, count))
    return cols


def read_xyz_frames(path: str | Path) -> list[XyzFrame]:
    """Read all xyz/extxyz frames from a file. Raises XyzFormatError on malformed input."""
    path = Path(path)
    lines = path.read_text().splitlines()
    frames: list[XyzFrame] = []
    i, n = 0, len(lines)
    while True:
        # Blank separator lines between frames are tolerated.
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            return frames
        nat_line = lines[i].strip()
        try:
            nat = int(nat_line)
        except ValueError as exc:
            raise XyzFormatError(
                f"{path}:{i + 1}: atom count is not an integer: {nat_line!r}"
            ) from exc
        if nat < 0:
            raise XyzFormatError(f"{path}:{i + 1}: negative atom count: {nat}")
        i += 1
        if i >= n:
            raise XyzFormatError(f"{path}:{i + 1}: missing comment line after atom count {nat}")
        header = _parse_comment(lines[i])
        comment_lineno = i + 1
        i += 1

        spec = header.get("Properties")
        if spec is None:
            cols: list[tuple[str, str, int]] = [("species", "S", 1), ("pos", "R", 3)]
        else:
            cols = _parse_properties(spec, comment_lineno)
            names = [c[0] for c in cols]
            if "species" not in names or "pos" not in names:
                raise XyzFormatError(
                    f"{path}:{comment_lineno}: Properties must declare species and pos: {spec!r}"
                )
            for name, _typ, count in cols:
                want = {"species": 1, "pos": 3, "charge": 1}.get(name)
                if want is not None and count != want:
                    raise XyzFormatError(
                        f"{path}:{comment_lineno}: {name} column must have count {want}: {spec!r}"
                    )
        n_values = sum(c[2] for c in cols)
        has_charge = any(c[0] == "charge" for c in cols)

        symbols: list[str] = []
        coords: list[list[float]] = []
        charges: list[float] = []
        for _ in range(nat):
            if i >= n:
                raise XyzFormatError(
                    f"{path}: EOF: expected {nat} atom lines, frame comment at line {comment_lineno}"
                )
            tokens = lines[i].split()
            if len(tokens) != n_values:
                raise XyzFormatError(
                    f"{path}:{i + 1}: atom line has {len(tokens)} columns, "
                    f"expected {n_values}: {lines[i]!r}"
                )
            sym: str | None = None
            xyz: list[float] | None = None
            q: float | None = None
            col = 0
            for name, _typ, count in cols:
                values = tokens[col : col + count]
                col += count
                if name == "species":
                    sym = values[0]
                elif name == "pos":
                    try:
                        xyz = [float(v) for v in values]
                    except ValueError as exc:
                        raise XyzFormatError(
                            f"{path}:{i + 1}: non-numeric coordinate: {lines[i]!r}"
                        ) from exc
                elif name == "charge":
                    try:
                        q = float(values[0])
                    except ValueError as exc:
                        raise XyzFormatError(
                            f"{path}:{i + 1}: non-numeric charge: {lines[i]!r}"
                        ) from exc
                # Other per-atom columns: ignored (documented in module docstring).
            if sym is None or xyz is None:
                raise XyzFormatError(f"{path}:{i + 1}: atom line missing species/pos: {lines[i]!r}")
            if not all(math.isfinite(v) for v in xyz) or (q is not None and not math.isfinite(q)):
                raise XyzFormatError(
                    f"{path}:{i + 1}: non-finite coordinate or charge: {lines[i]!r}"
                )
            symbols.append(sym)
            coords.append(xyz)
            if has_charge:
                if q is None:
                    raise XyzFormatError(f"{path}:{i + 1}: missing charge column: {lines[i]!r}")
                charges.append(q)
            i += 1

        frames.append(
            XyzFrame(
                symbols=symbols,
                coords=np.asarray(coords, dtype=np.float64).reshape(nat, 3),
                charges=np.asarray(charges, dtype=np.float64) if has_charge else None,
                header=header,
            )
        )


def _quote_value(value: str) -> str:
    """Quote a header value when it contains whitespace (extxyz convention)."""
    return f'"{value}"' if any(ch.isspace() for ch in value) else value


def _comment_line(frame: XyzFrame) -> str:
    """Build the comment line for a frame.

    Bare-title frames (header == {"title": non-empty}, no other keys, no
    per-atom charges) are written as plain xyz free text. Everything else
    gets a key=value line whose Properties entry is derived from frame data.
    """
    title = frame.header.get("title")
    extra = [k for k in frame.header if k not in ("title", "Properties")]
    if title and not extra and "Properties" not in frame.header and frame.charges is None:
        return title
    props = "Properties=species:S:1:pos:R:3"
    if frame.charges is not None:
        props += ":charge:R:1"
    parts = [props]
    for key, value in frame.header.items():
        if key == "Properties":
            continue  # always derived from data, never trusted from header
        parts.append(f"{key}={_quote_value(value)}")
    return " ".join(parts)


def write_xyz_frame(fh: IO[str], frame: XyzFrame) -> None:
    """Write one frame to an open text handle (coords/charges as %.6f)."""
    nat = len(frame.symbols)
    if frame.coords.shape != (nat, 3):
        raise XyzFormatError(f"coords shape {frame.coords.shape} does not match {nat} symbols")
    if frame.charges is not None and len(frame.charges) != nat:
        raise XyzFormatError(f"charges length {len(frame.charges)} does not match {nat} symbols")
    out = [f"{nat}\n", f"{_comment_line(frame)}\n"]
    for idx, sym in enumerate(frame.symbols):
        x, y, z = frame.coords[idx]
        line = f"{sym} {x:.6f} {y:.6f} {z:.6f}"
        if frame.charges is not None:
            line += f" {frame.charges[idx]:.6f}"
        out.append(line + "\n")
    fh.write("".join(out))


def write_xyz_frames(path: str | Path, frames: Sequence[XyzFrame]) -> None:
    """Write frames to a file (blank line between frames). Value-identical float64 round trip for %.6f-representable inputs."""
    path = Path(path)
    with path.open("w") as fh:
        for n, frame in enumerate(frames):
            if n:
                fh.write("\n")
            write_xyz_frame(fh, frame)
