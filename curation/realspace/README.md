# curation/realspace — geometry-key identity track

Curation stages for realspace (3D geometry) pipelines: identity keyed on
molecular *geometry* (structural keys derived from 3D coordinates) instead
of the graph track's canonical-SMILES identity (CompoundID = MD5 of
canonical SMILES, D62).

Motivation: the IQARIS extension targets non-organic species that cannot be
represented as RDKit-sanitized SMILES graphs — diboranes/carboranes
(3-center bonds), noble-gas species, organometallic aggregates (dative
bonds, metal valence), hydrate clusters. The graph track is dispatched
*around* at intake; these stages operate on geometry keys and converge with
the graph track at the shared output stages (extxyz delivery, stats).

Architecture: class-conditional dispatch (D70) — track declared at intake by
the converter/driver; shared machinery lives in top-level `lib/`; genuinely
shared stages in `curation/common/`. Graph behavior is frozen: this track
never modifies `lib/` files except the realspace-added `lib/xyz_io.py`, and
deliberately does not extend `curation/common/10_extxyz.py`.

## Stage chain

```
input .xyz/.extxyz dir
    │
    ▼
01_ingest.py          batches/          (Parquet, geometry + provenance)
    │
    ▼
02_identity.py        deduped_batches/  (exact-dup reject + near-dup cluster)
    │
    ▼
03_filter.py          filtered_batches/ (generic scalar checks, presets)
    │
    ▼
04_extxyz.py          extxyz/           (delivery files, neutral tags)
```

Run from the repo root (stage scripts bootstrap `sys.path` themselves):

```bash
python curation/realspace/01_ingest.py input_xyz_dir -o batches
python curation/realspace/02_identity.py -i batches -o deduped_batches
python curation/realspace/03_filter.py -i deduped_batches -o filtered_batches
python curation/realspace/04_extxyz.py -i filtered_batches -o extxyz
```

Each stage records provenance (`lib.provenance.record_run`) to the parent of
its output dir; reruns append. All stages are deterministic given fixed
inputs (identity depends only on the row set; filter is serial; delivery is
per-batch serial).

## Identity model — `identity.py`

`geometry_key(frame)`:

1. round every coordinate to 1e-3 Å (float-noise tolerance),
2. sort atoms by `(Z, x, y, z)` (canonical order),
3. serialize `%.3f` coordinates + `|charge|` + multiplicity,
4. MD5 → `geometry_id` (hex digest).

Pinned golden: HeH⁺ (charge 1, mult 1) → `98e3caa9339f0b4e832717e0921ca7d1`.

`conformer_rmsd(...)` — Kabsch superposition, proper rotations only,
greedy atom re-matching (≤ 5 passes), element-multiset check (→ `inf` on
mismatch). Used by 02_identity for near-duplicate detection; nuclear
geometry only (per-atom charges are not compared — documented extension
point).

**Inverted-superposition note:** `curation/graph/08_conformer_filter.py`
contains a Kabsch bug (`cb @ rot.T`) tracked as a separate follow-up; this
track intentionally does not touch graph code.

## Stages

### 01_ingest
- `source_id = "<basename>:<frame_index>"` (file-level rejects use bare
  `<basename>`); `charge`/`multiplicity` defaults (0/1) filled explicitly
  from the extxyz header; `formula` in Hill notation; `header` preserved
  verbatim as JSON; per-atom `charges` optional (NULL when absent).
- Plain-xyz input: every frame accepted with the default charge/multiplicity.

### 02_identity (dedup, drops by default)
- **Exact dups:** `geometry_key` recomputed from row geometry on load —
  stored `geometry_id` is provenance, not authority (mismatches counted,
  warned, never fatal; canonical values written). Keep-first →
  `identity_rejected.parquet` + `.REJECTED` sentinel.
- **Near dups:** greedy clustering vs cluster representatives (full
  argmin over kept representatives), raw `conformer_rmsd` <
  `--near-dup-threshold` (default 0.25 Å) → `identity_removed.parquet` +
  `.REMOVED` sentinel; `cluster_id` = nearest kept representative.
- **cluster_id is global** (registry-wide unique); numbering depends on
  batch encounter order — deterministic for a fixed input order.
- `--force-keep-rejected` retains all classes with `dedup_status` /
  `cluster_id` recorded; corrupt rows never crash the stage (`error` class
  in rejects). Empty output batches are not written.

### 03_filter (generic scalar filter, serial)
- First-failing-reason order: `--elements` → `--min-atoms`/`--max-atoms` →
  `--charge-range` → `--multiplicity-range`.
- Presets: `none` (DEFAULT — zero campaign knowledge), `neutral`,
  `closed_shell`, `neutral_closed_shell`; explicit CLI ranges override
  preset per property.
- `--elements` is an **exact whitelist**: unlike graph 03_filter there is
  no implicit-H pass — `--min-atoms`/`--max-atoms` count ALL atoms, so
  implicit H would silently bypass atom-count policy. List H explicitly.
- Ranges must use `=` form: `--charge-range=-1:1` (argparse parses a bare
  `-1:1` as a flag). Rejected rows → `filter_rejected.parquet` +
  `.REJECTED`; `filter_status` ∈ {ok, rejected}.

### 04_extxyz (delivery)
- One `.extxyz` per batch, written through `lib/xyz_io.write_xyz_frames`
  (same writer/reader pair pinned by round-trip tests — every delivery
  file is re-ingestable by 01_ingest).
- Neutral tag set only: `geometry_id`, `charge`, `multiplicity`,
  `elements` (Hill), `source_id`, `source_file`, `source_index`. The
  stored `header` column is not forwarded; no campaign knowledge in the
  default path. Per-atom charges delivered automatically when present.
- IQARIS manifest tag set is gated on an external spec — extend the tag
  dict in this stage when it lands (NOT `curation/common/10_extxyz.py`,
  whose graph behavior is frozen).
- `--exclude COLUMN=VALUE` (repeatable), e.g.
  `--exclude filter_status=rejected` after a `--force-keep-rejected`
  upstream run.

## Data conventions

- Parquet columns (`schema.py`): scalars are text/int columns; `coords` /
  `symbols` / `charges` are space-separated text (lib/parquet_io is frozen
  and stringifies nested lists); stage-added cols: `dedup_status`,
  `cluster_id`, `filter_status`, `filter_reason`.
- Warnings policy: the known unregistered-column UserWarning is suppressed
  exactly at `write_realspace_batch` call sites; every other warning is an
  error in tests.
- Tests never write to the CWD: any run that can produce rejects passes an
  explicit `--rejects-dir` under `tmp_path`.

## Tests

`curation/tests/test_realspace_*.py` + `test_xyz_io.py` — run with the
main repo's venv (`python -m pytest curation/tests -q`). Includes a
full-chain smoke (ingest → identity → filter → extxyz) asserting golden
geometry ids, tag delivery, and per-stage provenance.
