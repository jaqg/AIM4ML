# AIM4ML curation pipeline — Makefile
#
# Run from curation/ directory:
#   make                    → full pipeline on sample data (local paths)
#   make clean              → remove all generated files
#   make test               → run test suite
#
# Pipeline stages (sequential):
#   0=validate  1=split  2=energy_prefilter  3=filter
#   4=dedup     5=validate  6=stereo_filter  7=reorder
#   8=conformer_filter  9=stats  10=extxyz
#
# Intermediate data in ../AIM4ML-workspace/samples/ (sample mode, default).
# Use cluster paths or symlink for full data.
# Stamp files in $(SAMPLES)/.stamps/ track completed stages.

SHELL  := /bin/bash
PYTHON := python3

# Mode: sample (default) or full
MODE   := sample
WORKERS := 1
BATCH_SIZE := 5000
TANIMOTO := 0
FAMILY := QM40
INPUT_NAME := qm40_input

# Paths
SCRIPTS  := .
ifeq ($(MODE),full)
BASE      := ../AIM4ML-workspace
INPUT_SDF := $(BASE)/$(INPUT_NAME).sdf
else
BASE      := ../AIM4ML-workspace/samples
INPUT_SDF := $(BASE)/$(INPUT_NAME).sdf
endif

STAMPS       := $(BASE)/.stamps
VALID_SDF    := $(BASE)/$(INPUT_NAME)_valid.sdf
BATCHES      := $(BASE)/batches
FILTERED     := $(BASE)/filtered_batches
CURATED      := $(BASE)/curated_batches
DEDUPED      := $(BASE)/deduped_batches
STEREO       := $(BASE)/stereo_batches
REORDERED    := $(BASE)/reordered_batches
CONFORMER    := $(BASE)/conformer_batches
EXTXYZ       := $(BASE)/extxyz
STATS        := $(BASE)/stats
REJECTS      := $(BASE)/rejects

# -- Targets --------------------------------------------------------------

.PHONY: all test clean help version git-meta

all: version extxyz stats    # full pipeline

version:
	@$(PYTHON) -c "from rdkit import rdBase; print('RDKit version:', rdBase.rdkitVersion)"

git-meta:
	@git rev-parse HEAD > .git-commit 2>/dev/null || echo "unknown" > .git-commit
	@if git status --porcelain 2>/dev/null | grep -q .; then echo "true" > .git-dirty; else echo "false" > .git-dirty; fi
	@echo "  .git-commit and .git-dirty updated"

help:
	@echo "Targets:"
	@echo "  make          full pipeline (sample data)"
	@echo "  make test     run pytest"
	@echo "  make clean    remove generated data"
	@echo ""
	@echo "Pipeline stages:"
	@for i in 0 1 2 3 4 5 6 7 8 9 10; do echo "  make stage$$i"; done

test:
	cd .. && $(PYTHON) -m pytest curation/tests/ -v

clean:
	rm -rf $(VALID_SDF) $(BATCHES) $(FILTERED) $(CURATED) $(DEDUPED) \
	       $(STEREO) $(REORDERED) $(CONFORMER) $(EXTXYZ) $(STATS) \
	       $(REJECTS) $(STAMPS)
	@echo "Cleaned."

# -- Stage 0: validate ----------------------------------------------------

$(STAMPS)/00_validate.done: $(INPUT_SDF)
	@mkdir -p $(STAMPS)
	$(PYTHON) $(SCRIPTS)/00_validate.py $(INPUT_SDF) -o $(VALID_SDF) \
	          --rejects-dir $(REJECTS)/00_validate
	touch $@

stage0: $(STAMPS)/00_validate.done

# -- Stage 1: split -------------------------------------------------------

$(STAMPS)/01_split.done: $(STAMPS)/00_validate.done
	@mkdir -p $(BATCHES)
	$(PYTHON) $(SCRIPTS)/01_split.py $(VALID_SDF) -o $(BATCHES) --batch-size $(BATCH_SIZE)
	touch $@

stage1: $(STAMPS)/01_split.done

# -- Stage 2: energy prefilter --------------------------------------------

$(STAMPS)/02_energy_prefilter.done: $(STAMPS)/01_split.done
	@mkdir -p $(FILTERED)
	$(PYTHON) $(SCRIPTS)/02_energy_prefilter.py -i $(BATCHES) -o $(FILTERED) \
	          --rejects-dir $(REJECTS)/02_energy_prefilter
	touch $@

stage2: $(STAMPS)/02_energy_prefilter.done

# -- Stage 3: filter ------------------------------------------------------

$(STAMPS)/03_filter.done: $(STAMPS)/02_energy_prefilter.done
	@mkdir -p $(CURATED)
	$(PYTHON) $(SCRIPTS)/03_filter.py -i $(FILTERED) -o $(CURATED) \
	          --rejects-dir $(REJECTS)/03_filter \
	          --max-fragments 1 --workers $(WORKERS)
	touch $@

stage3: $(STAMPS)/03_filter.done

# -- Stage 4: dedup -------------------------------------------------------

$(STAMPS)/04_dedup.done: $(STAMPS)/03_filter.done
	@mkdir -p $(DEDUPED)
	$(PYTHON) $(SCRIPTS)/04_dedup.py -i $(CURATED) -o $(DEDUPED) \
	          --rejects-dir $(REJECTS)/04_dedup
	touch $@

stage4: $(STAMPS)/04_dedup.done

# -- Stage 5: validate ----------------------------------------------------

$(STAMPS)/05_validate.done: $(STAMPS)/04_dedup.done
	$(PYTHON) $(SCRIPTS)/05_validate.py -i $(DEDUPED) \
	          --rejects-dir $(REJECTS)/05_validate
	touch $@

stage5: $(STAMPS)/05_validate.done

# -- Stage 6: stereo filter -----------------------------------------------

$(STAMPS)/06_stereo_filter.done: $(STAMPS)/05_validate.done
	@mkdir -p $(STEREO)
	$(PYTHON) $(SCRIPTS)/06_stereo_filter.py -i $(DEDUPED) -o $(STEREO) \
	          --rejects-dir $(REJECTS)/06_stereo_filter
	touch $@

stage6: $(STAMPS)/06_stereo_filter.done

# -- Stage 7: reorder -----------------------------------------------------

$(STAMPS)/07_reorder.done: $(STAMPS)/06_stereo_filter.done
	@mkdir -p $(REORDERED)
	$(PYTHON) $(SCRIPTS)/07_reorder.py -i $(STEREO) -o $(REORDERED) \
	          --rejects-dir $(REJECTS)/07_reorder
	touch $@

stage7: $(STAMPS)/07_reorder.done

# -- Stage 8: conformer filter --------------------------------------------

$(STAMPS)/08_conformer_filter.done: $(STAMPS)/07_reorder.done
	@mkdir -p $(CONFORMER)
	$(PYTHON) $(SCRIPTS)/08_conformer_filter.py -i $(REORDERED) -o $(CONFORMER) \
	          --rejects-dir $(REJECTS)/08_conformer_filter --workers $(WORKERS)
	touch $@

stage8: $(STAMPS)/08_conformer_filter.done

# -- Stage 9: stats -------------------------------------------------------

$(STAMPS)/09_stats.done: $(STAMPS)/08_conformer_filter.done
ifeq ($(TANIMOTO),1)
	$(PYTHON) $(SCRIPTS)/09_stats.py -i $(CONFORMER) -o $(STATS) --tanimoto --workers $(WORKERS)
else
	$(PYTHON) $(SCRIPTS)/09_stats.py -i $(CONFORMER) -o $(STATS)
endif
	touch $@

stage9: $(STAMPS)/09_stats.done

stats: $(STAMPS)/09_stats.done

# -- Stage 10: extxyz -----------------------------------------------------

extxyz: $(STAMPS)/10_extxyz.done

$(STAMPS)/10_extxyz.done: $(STAMPS)/08_conformer_filter.done
	@mkdir -p $(EXTXYZ)
	$(PYTHON) $(SCRIPTS)/10_extxyz.py -i $(CONFORMER) -o $(EXTXYZ) --family $(FAMILY)
	touch $@
