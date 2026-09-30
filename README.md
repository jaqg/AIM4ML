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
# From repo root
cd .. && pytest curation/tests/ -v
cd .. && pytest selection/tests/ -v

# Or from within each directory
cd curation && make test
cd selection && make test
```

## License

MIT — see `LICENSE`.
