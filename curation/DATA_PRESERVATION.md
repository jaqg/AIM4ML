# AIM4ML curation pipeline — Data preservation

## What to keep

| File / directory | Description |
|------------------|-------------|
| `input.sdf` | Original curated input (single SDF, all standard tags) |
| `extxyz/` | Final extXYZ trajectory files (MLIP-ready, one per batch) |
| `stats/stats_summary.csv` | Full per-molecule descriptor table with metadata |
| `stats/plots/` | Histograms (NAT, MolWt, TPSA, Energy) |
| `rejects/` | Rejected molecules per stage (debug, audit) |

## What can be deleted

All intermediate Parquet batch directories (reproducible from `input.sdf`):

| Directory | Stage |
|-----------|-------|
| `batches/` | After split (1) |
| `filtered_batches/` | After energy prefilter (2) |
| `curated_batches/` | After filter (3) |
| `deduped_batches/` | After dedup (4) |
| `stereo_batches/` | After stereo filter (6) |
| `reordered_batches/` | After reorder (7) |
| `conformer_batches/` | After conformer filter (8) |

There are no stamp files: the per-source drivers (`drivers/qm40_curation.py`,
`drivers/qmugs_curation.py`) run the full stage chain on every invocation.
Per-stage provenance is recorded by each stage into `<BASE>/provenance.json`
(`lib/provenance.py`), so audit trails do not depend on stamps.

## Reproducibility

The full pipeline is reproducible from `input.sdf` alone:

```bash
# Alias table (curation/Makefile)
make qm40 ARGS="--mode full --workers 40"

# Or the driver directly (from repo root)
python drivers/qm40_curation.py --mode full --workers 40
```

All intermediate data is regenerated identically on re-run (intermediate
directory names are unchanged: `batches/`, `filtered_batches/`, …).
