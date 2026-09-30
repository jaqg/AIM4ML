# AIM4ML selection pipeline — orchestration (no full-chain; human in the loop).
#
#   make descriptors CURATED=<dir> OUT=<dir>
#   make analysis   OUT=<dir>
#   [ human: read <OUT>/analysis/cost_curve.csv, pick B_floor / K / w ]
#   make select     OUT=<dir> N=10000 [SELECT_ARGS="--floor-budget 0.15 --w 0.4"]
#   make test

PYTHON := python3
SELDIR := .

# Override at the command line, e.g.:
#   make descriptors CURATED=/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/08-conformer OUT=out
CURATED := ../AIM4ML-workspace/samples/conformer_batches
OUT     := out
N       := 10000
# Extra flags passed to 03_select.py (floor/depth/diversity params).
SELECT_ARGS := --floor-budget 0.15 --w 0.4 --depth-budget 0.10 --tier1-min 50 --ff 0.30
# D64 scope filter (stage 1). Set per run, e.g.:
#   make descriptors CURATED=... OUT=out SCOPE_ARGS="--allowed-elements C,N,O,S --min-atoms 10 --max-atoms 20"
# Empty default = no filter (debug); real runs must set it explicitly.
SCOPE_ARGS :=

.PHONY: all descriptors analysis select test clean help

help:
	@echo "AIM4ML selection pipeline (3 stages, human in the loop):"
	@echo "  make descriptors CURATED=<dir> OUT=<dir>"
	@echo "  make analysis   OUT=<dir>"
	@echo "  [human: read \$$(OUT)/analysis/cost_curve.csv, pick B_floor/K/w]"
	@echo "  make select     OUT=<dir> N=10000 [SELECT_ARGS=...]"
	@echo "  make test"

descriptors:
	$(PYTHON) $(SELDIR)/01_descriptors.py -i $(CURATED) -o $(OUT) $(SCOPE_ARGS)

analysis:
	$(PYTHON) $(SELDIR)/02_env_analysis.py -i $(OUT)/descriptors.parquet -o $(OUT)/analysis

select:
	$(PYTHON) $(SELDIR)/03_select.py -i $(OUT)/descriptors.parquet -o $(OUT) \
	          --n $(N) $(SELECT_ARGS)

test:
	cd .. && $(PYTHON) -m pytest selection/tests/ -v

clean:
	rm -rf $(OUT)
