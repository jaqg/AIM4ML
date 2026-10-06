# AIM4ML selection pipeline

Pick ~10,000 molecules (SDB) from the curated QM40 pool (~147k, CDB) for IQA
descriptor computation, maximising **atomic-environment coverage** (the IQA
targets are local quantities).

Three stages, human in the loop between 2 and 3:

```
01_descriptors.py   curated Parquet → descriptors.parquet
02_env_analysis.py  descriptors.parquet → analysis/ (frequency, cost curve,
                    forced/choice, bond types)
[human: read analysis/cost_curve.csv, pick B_floor / K / w]
03_select.py        floor pass → diversity pass → depth pass → selection.parquet
                    + reports/
```

## Quickstart

```bash
make descriptors CURATED=/path/to/conformer_batches OUT=out
make analysis   OUT=out
# read out/analysis/cost_curve.csv → choose --floor-budget (or --floor-k) and --w
make select     OUT=out N=10000
```

## Layout

```
selection/
├── 01_descriptors.py     curated Parquet → dedup → descriptors.parquet
├── 02_env_analysis.py    descriptors.parquet → analysis/ CSVs (+ cost-curve plot)
├── 03_select.py          floor + diversity + depth passes → selection.parquet + reports/
├── lib/
│   ├── schema.py         selection-phase column definitions
│   ├── descriptors.py    ECFP atom-env + Morgan fingerprint (RDKit, deterministic)
│   ├── coverage.py       env frequency, set-cover cost curve, floor pass, JS
│   └── diversity.py      scaffold grouping, MaxMin, Tanimoto
├── Makefile
└── tests/
```

Reuses top-level `lib/` (`parquet_io.py`) for curated-Parquet I/O —
no duplicated I/O code.

## Scope filter (01_descriptors.py, D64)

Stage 1 enforces the dataset scope **before the env histogram** — otherwise
F/Cl carriers (~20% of QM40) corrupt the rare-env floor calibration:

| Flag | Default | Meaning |
|---|---|---|
| `--allowed-elements` | none | heavy-element whitelist, e.g. `C,N,O,S`; molecules carrying anything outside are excluded (per-element counts in the report). Unknown symbol → hard error. |
| `--min-atoms` | none | min heavy-atom count, inclusive (all heavy atoms, no element split) |
| `--max-atoms` | none | max heavy-atom count, inclusive |

Default (no flags) = no filter; the report always prints the element
histogram of the included pool so an unfiltered real run is visible.
Scope values belong in the Makefile invocation, not the code.

## Key design points

- **Descriptors from `mol_block` → RDKit Mol**, never re-parsed from SMILES
  (the explicit-H canonical SMILES re-sanitizes ~0.5% of QM40; the curated
  mol_block carries resolved topology + bond orders).
- **Atom envs = unfolded 32-bit Morgan ids** (not folded 2048-bit) — exact
  identity is required for the "exactly ONE carrier molecule" forced-pick logic.
- **Dedup to CompoundID** inside stage 1 (env frequency must count each
  molecule once; conformer multiplicity would corrupt the rare-env logic).
- **Passes are independent** — `03_select.py --passes floor` runs the floor
  alone for inspection.

## Parameters (03_select.py)

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 10000 | total selection budget |
| `--passes` | floor,diversity,depth | which passes to run |
| `--floor-budget` | 0.15 | fraction of n for the floor pass |
| `--floor-k` | — | alternative: cover the K rarest envs |
| `--w` | 0.4 | floor-pass coverage weight (1−w on diversity) |
| `--depth-budget` | 0.10 | fraction of n for the depth pass |
| `--target-mult` | 2 | depth pass deepens envs to this multiplicity |
| `--depth-nn-sim` | 0.6 | depth pass NN-Tanimoto guard |
| `--tier1-min` | 50 | min scaffold family size for tier-1 MaxMin |
| `--ff` | 0.30 | diversity budget fraction to tier 1 |
