# AIM4ML curation pipeline — Makefile alias table (D70 track layout).
#
# Run from curation/ directory:
#   make qm40       → drivers/qm40_curation.py  (graph track, QM40 source)
#   make qmugs      → drivers/qmugs_curation.py (graph track, QMugs source)
#   make sample     → qm40 driver on sample data (local paths)
#   make test       → run test suite
#
# Driver options pass through, e.g.:
#   make qm40 ARGS="--mode full --workers 40"
#   make sample ARGS="--dry-run"
#
# Per-stage pass-throughs (debugging; no stamps — every run executes):
#   make stage3 ARGS="-i <in> -o <out> [--rejects-dir ...] [--workers N]"
#
# Intermediate data in ../../AIM4ML-workspace/samples (sample mode, default)
# or ../../AIM4ML-workspace (full mode). Use cluster paths via --base.
# No stamp files: provenance is recorded per stage into <BASE>/provenance.json.

SHELL  := /bin/bash
PYTHON := python3
REPO   := ..
DRIVERS := $(REPO)/drivers
SCRIPTS := graph
COMMON  := common
ARGS    :=

.PHONY: all qm40 qmugs sample test clean help

all: sample

qm40:
	$(PYTHON) $(DRIVERS)/qm40_curation.py $(ARGS)

qmugs:
	$(PYTHON) $(DRIVERS)/qmugs_curation.py $(ARGS)

sample:
	$(PYTHON) $(DRIVERS)/qm40_curation.py --mode sample $(ARGS)

test:
	cd $(REPO) && $(PYTHON) -m pytest curation/tests/ -v

clean:
	rm -rf ../../AIM4ML-workspace/samples/qm40_input_valid.sdf \
	       ../../AIM4ML-workspace/samples/batches \
	       ../../AIM4ML-workspace/samples/filtered_batches \
	       ../../AIM4ML-workspace/samples/curated_batches \
	       ../../AIM4ML-workspace/samples/deduped_batches \
	       ../../AIM4ML-workspace/samples/stereo_batches \
	       ../../AIM4ML-workspace/samples/reordered_batches \
	       ../../AIM4ML-workspace/samples/conformer_batches \
	       ../../AIM4ML-workspace/samples/extxyz \
	       ../../AIM4ML-workspace/samples/stats \
	       ../../AIM4ML-workspace/samples/rejects \
	       ../../AIM4ML-workspace/samples/.stamps
	@echo "Cleaned."

help:
	@echo "AIM4ML curation (alias table — drivers own the stage chain):"
	@echo "  make qm40 [ARGS=\"--mode full --workers 40\"]"
	@echo "  make qmugs [ARGS=\"...\"]"
	@echo "  make sample [ARGS=\"...\"]     (qm40 on sample data)"
	@echo "  make test"
	@echo "  make clean"
	@echo ""
	@echo "Per-stage pass-throughs (debugging):"
	@for i in 0 1 2 3 4 5 6 7 8 9 10; do echo "  make stage$$i ARGS=\"...\""; done

# -- Per-stage pass-throughs (debugging) ----------------------------------
# Paths are relative to curation/; pass inputs/outputs via ARGS.

stage0:
	$(PYTHON) $(SCRIPTS)/00_validate.py $(ARGS)

stage1:
	$(PYTHON) $(COMMON)/01_split.py $(ARGS)

stage2:
	$(PYTHON) $(SCRIPTS)/02_energy_prefilter.py $(ARGS)

stage3:
	$(PYTHON) $(SCRIPTS)/03_filter.py $(ARGS)

stage4:
	$(PYTHON) $(SCRIPTS)/04_dedup.py $(ARGS)

stage5:
	$(PYTHON) $(SCRIPTS)/05_validate.py $(ARGS)

stage6:
	$(PYTHON) $(SCRIPTS)/06_stereo_filter.py $(ARGS)

stage7:
	$(PYTHON) $(SCRIPTS)/07_reorder.py $(ARGS)

stage8:
	$(PYTHON) $(SCRIPTS)/08_conformer_filter.py $(ARGS)

stage9:
	$(PYTHON) $(SCRIPTS)/09_stats.py $(ARGS)

stage10:
	$(PYTHON) $(COMMON)/10_extxyz.py $(ARGS)
