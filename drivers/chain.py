"""
chain.py — Shared graph-track curation stage chain for the per-source drivers.

Source knowledge (track, family, input name) lives in drivers/<source>_curation.py;
this module only encodes the stage chain exactly as the former curation/Makefile
stages 0–10 ran it (flag-for-flag, incl. the absence of --max-fragments per D69).

No stamp files: each driver invocation runs the full chain; per-stage provenance
is recorded by every stage itself into <BASE>/provenance.json (lib/provenance).
"""

import argparse
import importlib
import os
import shlex
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Stage module locations post-track-split (D70 layout).
_COMMON_STAGES = {"01_split", "10_extxyz"}


def workspace_base(mode, override=None):
    """Pipeline base dir: AIM4ML-workspace[/samples], or an explicit override."""
    if override:
        return override
    root = os.path.join(os.path.dirname(_REPO_ROOT), "AIM4ML-workspace")
    return os.path.join(root, "samples") if mode == "sample" else root


def build_stages(base, input_name, family, workers, batch_size=5000, tanimoto=False):
    """Return [(stage_name, argv), ...] replicating the former Makefile chain.

    Order matches the Makefile's `all: version extxyz stats` — stage 10 (extxyz)
    runs before stage 9 (stats), both consuming stage 8 output.
    """
    valid = os.path.join(base, f"{input_name}_valid.sdf")
    rejects = os.path.join(base, "rejects")
    batches = os.path.join(base, "batches")
    filtered = os.path.join(base, "filtered_batches")
    curated = os.path.join(base, "curated_batches")
    deduped = os.path.join(base, "deduped_batches")
    stereo = os.path.join(base, "stereo_batches")
    reordered = os.path.join(base, "reordered_batches")
    conformer = os.path.join(base, "conformer_batches")
    extxyz = os.path.join(base, "extxyz")
    stats = os.path.join(base, "stats")

    stages = [
        (
            "00_validate",
            [
                os.path.join(base, f"{input_name}.sdf"),
                "-o",
                valid,
                "--rejects-dir",
                os.path.join(rejects, "00_validate"),
            ],
        ),
        ("01_split", [valid, "-o", batches, "--batch-size", str(batch_size)]),
        (
            "02_energy_prefilter",
            [
                "-i",
                batches,
                "-o",
                filtered,
                "--rejects-dir",
                os.path.join(rejects, "02_energy_prefilter"),
            ],
        ),
        (
            "03_filter",
            [
                "-i",
                filtered,
                "-o",
                curated,
                "--rejects-dir",
                os.path.join(rejects, "03_filter"),
                "--workers",
                str(workers),
            ],
        ),
        (
            "04_dedup",
            ["-i", curated, "-o", deduped, "--rejects-dir", os.path.join(rejects, "04_dedup")],
        ),
        (
            "05_validate",
            ["-i", deduped, "--rejects-dir", os.path.join(rejects, "05_validate")],
        ),
        (
            "06_stereo_filter",
            [
                "-i",
                deduped,
                "-o",
                stereo,
                "--rejects-dir",
                os.path.join(rejects, "06_stereo_filter"),
            ],
        ),
        (
            "07_reorder",
            ["-i", stereo, "-o", reordered, "--rejects-dir", os.path.join(rejects, "07_reorder")],
        ),
        (
            "08_conformer_filter",
            [
                "-i",
                reordered,
                "-o",
                conformer,
                "--rejects-dir",
                os.path.join(rejects, "08_conformer_filter"),
                "--workers",
                str(workers),
            ],
        ),
        # Makefile `all:` order: extxyz before stats.
        ("10_extxyz", ["-i", conformer, "-o", extxyz, "--family", family]),
        (
            "09_stats",
            ["-i", conformer, "-o", stats]
            + (["--tanimoto", "--workers", str(workers)] if tanimoto else []),
        ),
    ]
    return stages


def run_stages(stages, dry_run=False):
    """Run the stage chain in-process via each stage's main(argv).

    Stage mains call sys.exit() on both success (0) and failure — SystemExit(0)
    is treated as success here; any other code propagates and stops the chain.
    sys.argv is spoofed per stage so provenance record_run captures the exact
    stage invocation, as it did under the Makefile subprocess model.
    """
    for name, argv in stages:
        pkg = "curation.common" if name in _COMMON_STAGES else "curation.graph"
        if dry_run:
            print(f"DRY-RUN: python {pkg}/{name}.py {shlex.join(argv)}")
            continue
        print(f"\n=== stage {name} ===")
        mod = importlib.import_module(f"{pkg}.{name}")
        saved_argv = sys.argv
        sys.argv = [f"{name}.py", *argv]
        try:
            mod.main(argv)
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        finally:
            sys.argv = saved_argv
    print("\nPipeline complete.")


def driver_parser(description, family, input_name):
    """Common argparse surface for the per-source drivers."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="sample: AIM4ML-workspace/samples; full: AIM4ML-workspace (default: sample).",
    )
    p.add_argument("--workers", type=int, default=1, help="Worker processes (default: 1).")
    p.add_argument("--batch-size", type=int, default=5000, help="Split batch size (default: 5000).")
    p.add_argument(
        "--tanimoto",
        action="store_true",
        help="Add Tanimoto computation to stage 9 (off by default, as in the Makefile).",
    )
    p.add_argument(
        "--base",
        default=None,
        help="Override the pipeline base dir (default: derived from --mode).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stage chain without executing.",
    )
    p.set_defaults(_family=family, _input_name=input_name)
    return p


def run_driver(args):
    base = workspace_base(args.mode, args.base)
    stages = build_stages(
        base,
        input_name=args._input_name,
        family=args._family,
        workers=args.workers,
        batch_size=args.batch_size,
        tanimoto=args.tanimoto,
    )
    run_stages(stages, dry_run=args.dry_run)
