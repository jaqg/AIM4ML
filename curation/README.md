# AIM4ML Curation Pipeline

Reproducible curation pipeline for quantum-chemistry molecular datasets. Converts raw SDF input through 11 validation, filtering, deduplication, and formatting stages into extended XYZ trajectory files ready for machine-learned interatomic potential (MLIP) training with MACE, NequIP, SchNetPack, and similar frameworks.

The pipeline is organized as identity-model *tracks* (D70, class-conditional
dispatch): **graph** (Lewis/SMILES world — CompoundID = MD5 of canonical
SMILES, D62) and **realspace** (geometry-key world — implemented: 01_ingest →
04_extxyz geometry-key pipeline for IQARIS-style species that cannot be
canonical-SMILES-represented; see `curation/realspace/README.md`). The track
is declared at intake by the
converter/driver; shared machinery lives in top-level `lib/`, genuinely
shared stages in `curation/common/`. This README documents the graph track.

## Pipeline Architecture

```
input.sdf
  │
  ▼
[0] validate       ── contract check (required tags, types)
[1] split          ── SDF → Parquet batches
[2] energy_prefilter ── OLS atom-type outlier detection (MAD z‑score)
[3] filter         ── neutral / non‑zwitterion / closed‑shell
[4] dedup          ── canonical SMILES + CompoundID + conformer dedup
[5] validate       ── integrity cross‑checks
[6] stereo_filter  ── enantiomer removal
[7] reorder        ── canonical atom ordering (RDKit CanonicalRankAtoms)
[8] conformer_filter ── conformer RMSD pruning
[9] stats          ── descriptors + histograms + Tanimoto diversity
[10] extxyz        ── extended XYZ trajectory files (MLIP‑ready)
  │
  ▼
extxyz/*.xyz   +   stats/stats_summary.csv   +   stats/plots/
```

## Stage Table (graph track)

| Stage | Script | Track | Purpose |
|-------|--------|-------|---------|
| 0 | `curation/graph/00_validate.py` | graph | contract check (required tags, types) |
| 1 | `curation/common/01_split.py` | common | SDF → Parquet batches |
| 2 | `curation/graph/02_energy_prefilter.py` | graph | OLS atom-type outlier detection (MAD z‑score) |
| 3 | `curation/graph/03_filter.py` | graph | neutral / non‑zwitterion / closed‑shell |
| 4 | `curation/graph/04_dedup.py` | graph | canonical SMILES + CompoundID + conformer dedup |
| 5 | `curation/graph/05_validate.py` | graph | integrity cross‑checks |
| 6 | `curation/graph/06_stereo_filter.py` | graph | enantiomer removal |
| 7 | `curation/graph/07_reorder.py` | graph | canonical atom ordering (RDKit CanonicalRankAtoms) |
| 8 | `curation/graph/08_conformer_filter.py` | graph | conformer RMSD pruning |
| 9 | `curation/graph/09_stats.py` | graph | descriptors + histograms + Tanimoto diversity |
| 10 | `curation/common/10_extxyz.py` | common | extended XYZ trajectory files (MLIP‑ready) |

Realspace-track stages (`curation/realspace/`, see its README): `01_ingest`,
`02_identity` (geometry-key dedup: exact-dup reject + near-dup clustering),
`03_filter` (generic scalar filter with presets), `04_extxyz` (delivery,
neutral tags), plus `schema.py`/`identity.py` libraries. Stage output
directory names are unchanged by the track split.

## Dependencies

- Python ≥ 3.10
- [RDKit](https://www.rdkit.org/) ≥ 2024.03
- NumPy, pandas, pyarrow (Parquet I/O)
- Matplotlib (stats plots)
- OpenBabel (optional, for legacy QM40 scripts)

Install with conda:

```bash
conda create -n aim4ml python=3.11 rdkit numpy pandas pyarrow matplotlib -c conda-forge
conda activate aim4ml
```

## Modular Usage

Each stage is a standalone script — run independently, swap backends, tune thresholds, or repurpose for different datasets without touching the drivers.

**Invocation convention (pick one, be consistent):** stages are invoked by
script path from the repo root, e.g. `python curation/graph/03_filter.py`.
Each stage bootstraps the repo root onto `sys.path` itself, so this works
from any working directory. Full-pipeline runs go through the drivers
(`python drivers/qm40_curation.py`), which call the same
stages in-process.

### Input/Output conventions

All stages 2–10 use Parquet batches internally (zstd‑compressed). Stages 0–1 convert SDF ↔ Parquet. Each script accepts at minimum:

| Flag | Meaning |
|------|---------|
| `-i` / `--input-dir` | Directory of Parquet batches (default varies by stage) |
| `-o` / `--output-dir` | Directory for output Parquet batches |
| `--rejects-dir` | Directory for rejected‑molecule SDFs |
| `--force-keep-rejected` | Keep rejected rows in Parquet output (default: drop) |

### Stage-by-stage flags

#### Stage 0 — `graph/00_validate.py`
```bash
python3 curation/graph/00_validate.py input.sdf [-o output.sdf] [--rejects-dir rejects/00_validate] [--lenient]
```
| Flag | Effect |
|------|--------|
| `input.sdf` | Positional: path to raw SDF |
| `-o` | Clean output SDF (default: `<input>_valid.sdf`) |
| `--lenient` | Warn instead of rejecting on missing required tags |

#### Stage 1 — `common/01_split.py`
```bash
python3 curation/common/01_split.py input.sdf [-o batches/] [-b 5000]
```
| Flag | Effect |
|------|--------|
| `-b` / `--batch-size` | Molecules per Parquet batch (default: 5000) |

#### Stage 2 — `graph/02_energy_prefilter.py`
```bash
python3 curation/graph/02_energy_prefilter.py -i batches/ -o filtered_batches/ \
    [--threshold 3.5] [--atom-types H C N O F S Cl Br] [--skip] [--force-keep-rejected]
```
| Flag | Effect |
|------|--------|
| `--threshold` | MAD z‑score cut‑off (default: 3.5). Higher = fewer flagged |
| `--atom-types` | Atom symbols to include in the OLS model (default: H,C,N,O,F,S,Cl,Br) |
| `--skip` | Pass‑through all molecules (no energy filtering) |

#### Stage 3 — `graph/03_filter.py`
```bash
python3 curation/graph/03_filter.py -i filtered_batches/ -o curated_batches/ \
    [--preset neutral_closed_shell] [--allowed-elements C,H,N,O,F,S,Cl,Br] \
    [--min-heavy 4] [--max-heavy 200] [--min-fragments N] [--max-fragments N] \
    [--force-keep-rejected]
```
| Flag | Effect |
|------|--------|
| `--preset` | Pre‑defined filter: `neutral_closed_shell`, `neutral`, or `none` |
| `--allowed-elements` | Comma‑separated allowed atomic symbols |
| `--min-heavy` | Minimum number of heavy (non‑H) atoms |
| `--max-heavy` | Maximum number of heavy (non‑H) atoms |
| `--min-fragments` / `--max-fragments` | Optional fragment-count gate (CLI opt-in; the drivers no longer set it) |

**Multi-fragment complexes are KEPT and TAGGED.** The driver default run no
longer passes `--max-fragments 1`; salts, solvates and adducts flow through
curation. Every kept row — including monomers — carries four metadata columns,
grouped next to `n_fragments` in the Parquet schema:

| Column | Content |
|------|--------|
| `fragment_formulas` | Hill-ordered formula per fragment, `;`-joined (e.g. `C9H8O4;H2O`) |
| `fragment_smiles` | Canonical SMILES per fragment, `;`-joined |
| `fragment_charges` | Net formal charge per fragment, `;`-joined (e.g. `1;-1` for salts) |
| `fragment_heavy_atoms` | Heavy-atom count per fragment, `;`-joined |

Fragment truth follows the SMILES tag when it parses (D62 priority —
converter-topology errors cannot corrupt fragment identity); mol_block
perception is only the fallback. Columns are always populated (single-item
strings for monomers, never null); empty strings only for corrupt rows and
for composition-rejected rows (dropped anyway; the rejection reason string
carries the formulas). Fragment-level *policy* (drop/keep/largest/
solute-only) is deliberately deferred to Selection — curation tags, it does
not decide.

Caveat (pre-existing, documented): rows whose SMILES tag is missing rely on
geometry-based bond determination, which fabricates formal charges for
charge-separated species — salts are then rejected by the net-charge check.
Converter output always carries SMILES tags (D62), so this affects only
tagless legacy inputs.

#### Stage 4 — `graph/04_dedup.py`
```bash
python3 curation/graph/04_dedup.py -i curated_batches/ -o deduped_batches/ [--rejects-dir rejects/04_dedup]
```
Adds `CanonicalSMILES`, `CompoundID` (MD5), `Formula`, and `conformer_duplicate` flag. Duplicate conformers (same CompoundID, same energy) are tagged but kept — they're pruned later by Stage 8.

#### Stage 5 — `graph/05_validate.py`
```bash
python3 curation/graph/05_validate.py -i deduped_batches/ [--rejects-dir rejects/05_validate] [--skip]
```
| Flag | Effect |
|------|--------|
| `--skip` | Skip integrity checks entirely |

#### Stage 6 — `graph/06_stereo_filter.py`
```bash
python3 curation/graph/06_stereo_filter.py -i deduped_batches/ -o stereo_batches/ [--force-keep-rejected]
```
Removes one enantiomer from each racemic pair. Keeps the first canonical SMILES. Molecules with multiple fragments (complexes) are tagged `complex` and passed through.

#### Stage 7 — `graph/07_reorder.py`
```bash
python3 curation/graph/07_reorder.py -i stereo_batches/ -o reordered_batches/ \
    [--rejects-dir rejects/07_reorder] [--workers 4] [--force-keep-rejected]
```
| Flag | Effect |
|------|--------|
| `--workers` | Parallel workers for row reordering (default 1) |
| `--force-keep-rejected` | Keep molecules that fail reordering (default: drop them to rejects SDF) |

#### Stage 8 — `graph/08_conformer_filter.py`
```bash
python3 curation/graph/08_conformer_filter.py -i reordered_batches/ -o conformer_batches/ \
    [--rmsd-threshold 1.0] [--force-keep-rejected]
```
| Flag | Effect |
|------|--------|
| `--rmsd-threshold` | Heavy‑atom Kabsch RMSD cutoff in Å (default: 1.0). Lower = more conformers kept |

#### Stage 9 — `graph/09_stats.py`
```bash
python3 curation/graph/09_stats.py -i conformer_batches/ -o stats/ \
    [--tanimoto] [--workers 8] [--exclude COLUMN=VALUE]
```
| Flag | Effect |
|------|--------|
| `--tanimoto` | Compute ECFP4 Tanimoto nearest‑neighbour (O(n²), multiprocessed) |
| `--workers` | CPU workers for Tanimoto |
| `--exclude` | Drop rows matching `COLUMN=VALUE` before stats (repeatable) |

#### Stage 10 — `common/10_extxyz.py`
```bash
python3 curation/common/10_extxyz.py -i conformer_batches/ -o extxyz/ \
    [--family QM40] [--exclude COLUMN=VALUE]
```
| Flag | Effect |
|------|--------|
| `--family` | Dataset name written to extXYZ metadata (default: QM40) |
| `--exclude` | Drop rows matching `COLUMN=VALUE` before writing (repeatable) |

Frame metadata includes `nfrag` (from the row's `n_fragments` — complexes
export `nfrag,2`) and `iconf`.

### Report utility

```bash
python3 tools/report_stats.py stats/stats_summary.csv \
    --total-input 162954 --rejects-dir rejects/
```

Prints curation funnel (per‑stage drop counts), NAT/MolWt/TPSA/Energy descriptor summary, and Tanimoto diversity statistics (if `--tanimoto` was used in Stage 9).

### Running individual stages

```bash
# Validate only (stage 0)
python3 curation/graph/00_validate.py my_dataset.sdf -o my_dataset_valid.sdf

# Chemically filter with custom element set (stage 3)
python3 curation/graph/03_filter.py -i batches/ -o curated/ --preset neutral \
    --allowed-elements C,H,O,N --min-heavy 6

# Tighter conformer pruning (stage 8)
python3 curation/graph/08_conformer_filter.py -i reordered/ -o conformers/ --rmsd-threshold 0.5

# Generate extXYZ with dataset name override (stage 10)
python3 curation/common/10_extxyz.py -i conformers/ -o extxyz/ --family MyDataset
```

### Adopting for a new dataset

1. Prepare an SDF with the [required tags](#input-sdf). Write a converter script (see `converters/convert_qm40.py` as a template) if your data is in another format.
2. Run `curation/graph/00_validate.py` to check the contract.
3. Adjust `--allowed-elements`, `--min-heavy`, `--max-heavy` in Stage 3 for your chemistry.
4. Set `--family` in Stage 10 for correct extXYZ metadata.
5. Optionally skip Stage 2 (`--skip`) if your dataset has no energy column or you don't trust the OLS model for your atom types.
6. Add a thin driver under `drivers/` (copy `qm40_curation.py`; set family + input name) so the full chain is one command.

## Quick Start

```bash
# From repo root
python drivers/qm40_curation.py --mode sample     # QM40 driver, sample data
python drivers/qm40_curation.py --mode full --workers 40
python drivers/qmugs_curation.py                  # QMugs driver
python drivers/qm40_curation.py --dry-run         # print stage chain, run nothing

# Per-stage (debugging; run script directly)
python curation/graph/00_validate.py <input.sdf> -o /tmp/valid.sdf
python curation/graph/03_filter.py -i batches/ -o curated/ --workers 4

# Run test suite (from repo root)
pytest curation/tests/ -v
```

No Makefile, no stamps: drivers run the full chain each invocation; every
stage records its own provenance into `<BASE>/provenance.json`
(`lib/provenance.py`).

Intermediates can be removed whenever (regenerable from input SDF):

```bash
rm -rf ../AIM4ML-workspace/samples/{batches,filtered_batches,curated_batches,\
deduped_batches,stereo_batches,reordered_batches,conformer_batches,\
extxyz,stats,rejects,qm40_input_valid.sdf}
```

## Data Contract

### Input SDF

Each molecule must carry these SDF property tags:

| Tag | Type | Status |
|-----|------|--------|
| `Energy_Ha` | float | required |
| `FormalCharge` | int | required |
| `Multiplicity` | int | required |
| `SMILES` | str | recommended |
| `SourceID` | str | recommended |
| `HOMO_Ha`, `LUMO_Ha`, `HL_Gap_Ha` | float | optional |
| `PartialCharges` | str | optional |

Molecules are stored as V2000 mol blocks with 3D coordinates and single bonds (bond orders resolved by dedup stage via `DetermineBondOrders` + SMILES template fallback).

### Output: Extended XYZ

```
natoms
SourceID=...,CompoundID=...,Formula=...,nat=...,CNSO=...,chrg=...,mult=...,e=...,smiles=...
C   0.123456  0.234567  0.345678
H   1.234567  0.345678  0.456789
...
```

Metadata keys: `SourceID`, `CompoundID`, `Formula`, `nat`, `CNSO`, `chrg`, `mult`, `e` (Energy_Ha), `smiles` (canonical).

## Source Converters & Legacy Scripts

- `converters/convert_qm40.py` — converts raw QM40 CSVs (main, xyz, bond) to pipeline-standard SDF
- `converters/convert_qmugs.py` — converts QMugs structures to pipeline-standard SDF

Untested QM40-era one-offs (`filter_qm40.py`, `energy_prefilter_qm40.py`)
live outside the repo in `AIM4ML-workspace/legacy/` — untested by design,
kept for reference only.

## Data Preservation

The pipeline is fully reproducible from `input.sdf` alone. All intermediate Parquet batches can be regenerated. See [`DATA_PRESERVATION.md`](DATA_PRESERVATION.md) for details on what to archive and what can be safely deleted.

## Repository Structure

```
AIM4ML/
├── lib/                      # identity-neutral shared machinery (D70)
│   ├── schema.py             #   tag + Parquet column registries
│   ├── sdf_io.py             #   SDF read/write helpers
│   ├── parquet_io.py         #   Parquet batch I/O
│   ├── provenance.py         #   per-stage run audit log
│   ├── parallel.py           #   multiprocessing map
│   └── rdkit_version.py      #   RDKit version gate
├── converters/               # raw dataset → pipeline-standard SDF
│   ├── convert_qm40.py
│   └── convert_qmugs.py
├── drivers/                  # per-source recipe scripts (track + flags)
│   ├── chain.py              #   shared stage-chain runner
│   ├── qm40_curation.py
│   └── qmugs_curation.py
├── curation/
│   ├── graph/                # graph-track stages (Lewis/SMILES identity)
│   │   ├── 00_validate.py … 09_stats.py
│   │   └── schema.py         #   graph identity (CompoundID = MD5 SMILES)
│   ├── common/               # shared stages (identity-neutral)
│   │   ├── 01_split.py
│   │   └── 10_extxyz.py
│   ├── realspace/            # geometry-key track (IQARIS: 01_ingest–04_extxyz)
│   ├── tests/                # pytest suite
│   ├── tools/                # utility scripts (reporting, inspection)
│   ├── DATA_PRESERVATION.md  # archival policy
│   └── README.md             # this file
└── selection/                # selection pipeline (stages + selection/lib)
```

Legacy QM40 one-offs: `AIM4ML-workspace/legacy/` (outside the repo).

## Citation

If you use this pipeline in your research, please cite the accompanying manuscript:

> Quiñonero, J. A. et al. *AIM4ML: Automated Curation Pipeline for Quantum Chemistry Datasets*. (in preparation)

## License

MIT — see [LICENSE](LICENSE)
