# AIM4ML

Curation and selection pipelines for building the AIM4ML database of QM-quality
molecular structures (PhD Chapter 1, QTCOVI group, Universidad de Oviedo).

## Layout

| Directory | Purpose |
|-----------|---------|
| `lib/` | Identity-neutral shared machinery (D70): Parquet batch I/O, SDF I/O, provenance, parallel map, RDKit version gate, tag/column registries. Used by both identity-model tracks + selection. |
| `curation/` | Curation pipeline, split into identity-model tracks (D70): `curation/graph/` (stages 00–10 + graph identity schema — Lewis/SMILES world), `curation/common/` (shared stages `01_split`, `10_extxyz`), `curation/realspace/` (geometry-key track placeholder — IQARIS, not yet implemented), `curation/tests/`. See `curation/README.md`. |
| `converters/` | Dumb per-source converters: raw dataset → pipeline-standard SDF (`convert_qm40.py`, `convert_qmugs.py`). |
| `drivers/` | Thin per-source recipe scripts composing the stage chain; source knowledge (track + flags) legally lives here (`qm40_curation.py`, `qmugs_curation.py`). |
| `selection/` | Selection pipeline: descriptor computation, environment analysis, greedy diverse subset selection. See `selection/README.md`. |
| `selection/tests/` | Selection pipeline test suite (pytest). |

## Usage

Curation is driven by per-source thin recipes under `drivers/` (each encodes
one source → one stage chain):

```bash
python drivers/qm40_curation.py --mode sample    # QM40 on sample data
python drivers/qm40_curation.py --mode full --workers 40
python drivers/qmugs_curation.py                 # QMugs
python drivers/qm40_curation.py --dry-run        # print stage chain, run nothing
```

Workspace data paths assume the default location
`../AIM4ML-workspace/samples/` (sample mode); override with `--base`.
No Makefile — direct drivers are the interface.

## Testing

```bash
# From repo root (249 tests: 191 curation + 58 selection)
pytest curation/tests/ selection/tests/ -v

# Or per package
pytest curation/tests/ -v
pytest selection/tests/ -v
make -C selection test          # selection still ships its own Makefile
```

Coverage: **88%** (pytest-cov; subprocess measurement via
`COVERAGE_PROCESS_START=$PWD/.coveragerc` — the `.pth` hook lives in the
venv, opt-in per run). Core stages 91–99%; known skip: `07_reorder.py`
~26% — only its pure helper functions are locally tested; the CLI/main
path (batch iteration, rejects SDF, provenance) is exercised only by real
cluster runs. Legacy QM40 one-off scripts (`filter_qm40.py`,
`energy_prefilter_qm40.py`) are untested by design and live outside the
repo in `AIM4ML-workspace/legacy/`.

## Development

Lint/format: `ruff check .` + `ruff format --check .` (config in `ruff.toml`, line-length 100).

Type check (repo root on MYPYPATH; `selection/lib` checked separately — same module name `lib`):

```bash
MYPYPATH=. .venv/bin/mypy -p lib -p curation.graph -p curation.common
MYPYPATH=selection .venv/bin/mypy -p lib   # selection/lib
```

Optional git hook (ruff on staged files before every commit; full tests not run — run pytest before pushing):

```bash
git config core.hooksPath .git-hooks
```

## License

MIT — see `LICENSE`.
