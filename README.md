# AIM4ML

Curation and selection pipelines for building the AIM4ML database of QM-quality
molecular structures (PhD Chapter 1, QTCOVI group, Universidad de Oviedo).

## Layout

| Directory | Purpose |
|-----------|---------|
| `curation/` | 11-stage curation pipeline (00–10): validation, splitting, energy prefilter, property filtering, deduplication, stereo filtering, conformer filtering, RDKit/AMBER reordering, statistics, extXYZ export. Includes QMugs/QM40 converters. See `curation/README.md`. |
| `selection/` | Selection pipeline: descriptor computation, environment analysis, greedy diverse subset selection. See `selection/README.md`. |

## Usage

Each pipeline is driven by its `Makefile`:

```bash
cd curation && make      # run curation stages
cd selection && make     # run selection stages
```

## License

MIT — see `LICENSE`.
