# AIM4ML

Curation and selection pipelines for building the AIM4ML database of QM-quality
molecular structures (PhD Chapter 1, QTCOVI group, Universidad de Oviedo).

## Layout

| Directory | Purpose |
|-----------|---------|
| `curation/` | 11-stage curation pipeline (00–10): validation, splitting, energy prefilter, property filtering, deduplication, stereo filtering, conformer filtering, RDKit/AMBER reordering, statistics, extXYZ export. Includes QMugs/QM40 converters. See `curation/README.md`. |
| `curation/tests/` | Curation pipeline test suite (pytest). |
| `selection/` | Selection pipeline: descriptor computation, environment analysis, greedy diverse subset selection. See `selection/README.md`. |
| `selection/tests/` | Selection pipeline test suite (pytest). |

## Usage

Each pipeline is driven by its `Makefile`:

```bash
cd curation && make      # run curation stages
cd selection && make     # run selection stages
```

Sample data paths in Makefiles assume workspace location:
`../AIM4ML-workspace/samples/`

## Testing

```bash
# From repo root (237 tests: 179 curation + 58 selection)
pytest curation/tests/ selection/tests/ -v

# Or per package
pytest curation/tests/ -v
pytest selection/tests/ -v
make -C curation test
make -C selection test
```

Coverage: **88%** (pytest-cov; subprocess measurement via
`COVERAGE_PROCESS_START=$PWD/.coveragerc` — the `.pth` hook lives in the
venv, opt-in per run). Core stages 91–99%; known skip: `07_reorder.py`
~22% — its antechamber call paths need the AMBER binary, cluster-only
(tested there via real runs; local tests cover the pure-Python parts).
Legacy QM40 one-off converters are untested by design.

## Development

Lint/format: `ruff check .` + `ruff format --check .` (config in `ruff.toml`, line-length 100).

Type check (libs only; the two `lib` packages must be checked separately — same module name):

```bash
MYPYPATH=curation  .venv/bin/mypy -p lib
MYPYPATH=selection .venv/bin/mypy -p lib
```

Optional git hook (ruff on staged files before every commit; full tests not run — run pytest before pushing):

```bash
git config core.hooksPath .git-hooks
```

## License

MIT — see `LICENSE`.
