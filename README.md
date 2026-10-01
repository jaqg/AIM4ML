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
# From repo root (all 181 tests: 123 curation + 58 selection)
pytest curation/tests/ selection/tests/ -v

# Or per package
pytest curation/tests/ -v
pytest selection/tests/ -v
make -C curation test
make -C selection test
```

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
