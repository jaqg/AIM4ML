#!/usr/bin/env python3
"""
summarize_run.py — One-page run summary extracted from curation.log.

Parses the per-stage Report blocks, rejection breakdowns, fitted atomic
energies, descriptor stats, and flags anomalies (non-reconciling counters,
kept-but-uncounted rows, CID collisions).

Usage:
    python3 summarize_run.py <run_dir>          # print to terminal (default)
    python3 summarize_run.py <run_dir> --save   # also write summary.md in run dir
    python3 summarize_run.py <run_dir> --md PATH  # write markdown to PATH
    python3 summarize_run.py curation.log       # pass the log file directly
"""

import argparse
import os
import re
import sys

# script filename → (label, total_key, keep_key, reject_key)
_STAGE_META = {
    "00_validate.py":         ("0  validate",         "Total",          "Valid",         "Rejected"),
    "01_split.py":            ("1  split",            "Total molecules","Written",       None),
    "02_energy_prefilter.py": ("2  energy prefilter", "Total",          "OK",            "Flagged (OLS)"),
    "03_filter.py":           ("3  filter",           "Total",          "OK",            "Rejected"),
    "04_dedup.py":            ("4  dedup",            "Total",          "Unique (ok)",   "Conformer duplicates"),
    "05_validate.py":         ("5  validate",         "Total",          "OK",            "Invalid"),
    "06_stereo_filter.py":    ("6  stereo filter",    "Total",          "Kept",          "Removed enantiomers"),
    "07_reorder.py":          ("7  reorder",          "Total",          "OK",            "Failed"),
    "08_conformer_filter.py": ("8  conformer filter", "Total molecules","Kept",          "Removed conformers"),
    "09_stats.py":            ("9  stats",            "Total molecules","Processed",     None),
    "10_extxyz.py":           ("10 extxyz",           "Total",          "Written",       "Failed"),
}


def _as_int(value):
    if value is None:
        return None
    m = re.search(r"-?\d+", str(value))
    return int(m.group()) if m else None


def _split_stages(text):
    """Split log into stage blocks keyed by script filename."""
    stages = []
    current = None
    for line in text.splitlines():
        m = re.match(r"^\s*python3\s+\./(\d\d_\w+\.py)", line)
        if m:
            if current is not None:
                stages.append(current)
            current = {"script": m.group(1), "lines": []}
        elif current is not None:
            current["lines"].append(line)
    if current is not None:
        stages.append(current)
    return stages


def _extract_report(lines):
    """Return dict of '  key: value' lines under a 'Report' header."""
    d = {}
    for i, line in enumerate(lines):
        if line.strip() == "Report":
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and ":" in lines[j]:
                key, _, val = lines[j].strip().partition(":")
                d[key.strip()] = val.strip()
                j += 1
            break
    return d


def _extract_breakdown(lines):
    """Return dict of '    reason: count' lines under 'Rejection breakdown:'."""
    d = {}
    for i, line in enumerate(lines):
        if line.strip() == "Rejection breakdown:":
            j = i + 1
            while j < len(lines) and lines[j].startswith("    "):
                key, _, val = lines[j].strip().partition(":")
                if key.strip() and val.strip():
                    d[key.strip()] = val.strip()
                j += 1
            break
    return d


def _extract_energies(text):
    """Fitted atomic energies: {'C': -38.11, 'H': -0.60, ...}."""
    d = {}
    for m in re.finditer(r"^\s*e_(\w+)\s*=\s*(-?\d+\.?\d*)", text, re.M):
        d[m.group(1)] = float(m.group(2))
    return d


def _extract_residual(text):
    m = re.search(r"Residual median:\s*(-?\d+\.?\d*)\s*Ha\s+MAD:\s*(-?\d+\.?\d*)\s*Ha", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_log(text):
    """Parse curation.log into a structured summary dict."""
    stages = _split_stages(text)
    funnel = []
    breakdowns = []
    energies = {}
    residual = None
    descriptors = {}

    for st in stages:
        meta = _STAGE_META.get(st["script"])
        if meta is None:
            continue
        label, total_key, keep_key, reject_key = meta
        report = _extract_report(st["lines"])

        row = {
            "num": int(st["script"][:2]),
            "label": label,
            "total": _as_int(report.get(total_key)),
            "kept": _as_int(report.get(keep_key)) if keep_key else None,
            "rejected": _as_int(report.get(reject_key)) if reject_key else None,
            "raw": report,
        }
        funnel.append(row)

        bd = _extract_breakdown(st["lines"])
        if bd:
            breakdowns.append((label, bd))

        if st["script"] == "02_energy_prefilter.py":
            energies = _extract_energies("\n".join(st["lines"]))
            residual = _extract_residual("\n".join(st["lines"]))
        if st["script"] == "09_stats.py":
            for key in ("MolWt", "TPSA"):
                if key in report:
                    descriptors[key] = report[key]

    anomalies = _compute_anomalies(funnel)
    funnel.sort(key=lambda r: r["num"])
    return {
        "funnel": funnel,
        "breakdowns": breakdowns,
        "energies": energies,
        "residual": residual,
        "descriptors": descriptors,
        "anomalies": anomalies,
    }


def _compute_anomalies(funnel):
    """Flag non-reconciling counters and kept-but-uncounted rows."""
    anomalies = []
    by_label = {r["label"]: r for r in funnel}

    # Stage 6: skipped_no_smiles = total - kept - removed - complexes
    s6 = by_label.get("6  stereo filter")
    if s6:
        total = s6["total"]
        kept = s6["kept"] or 0
        removed = s6["rejected"] or 0
        complexes = _as_int(s6["raw"].get("Complexes (skipped)")) or 0
        skipped = (total or 0) - kept - removed - complexes
        if skipped > 0:
            anomalies.append(
                f"stage 6: {skipped} skipped_no_smiles (kept, not in Kept counter)"
            )

    # Stage 3: mol_corrupt / topology_warning rows are kept (only 'rejected' dropped)
    s3 = by_label.get("3  filter")
    if s3:
        corrupt = _as_int(s3["raw"].get("Mol corrupt")) or 0
        topo = _as_int(s3["raw"].get("Topology warning")) or 0
        if corrupt or topo:
            anomalies.append(
                f"stage 3: {corrupt + topo} non-ok rows kept "
                f"({corrupt} mol_corrupt, {topo} topology_warning)"
            )

    # Stage 5: CID collisions (conformer duplicates from dedup)
    s5 = by_label.get("5  validate")
    if s5:
        collisions = _as_int(s5["raw"].get("CID collisions"))
        if collisions:
            anomalies.append(
                f"stage 5: {collisions} CID collisions (conformer duplicates)"
            )

    # Stage 2: energy-positive should be 0
    s2 = by_label.get("2  energy prefilter")
    if s2:
        pos = _as_int(s2["raw"].get("Energy positive"))
        if pos:
            anomalies.append(f"stage 2: {pos} energy-positive molecules (unexpected)")

    return anomalies


# -- Renderers ------------------------------------------------------------

def _fmt_count(v):
    return "-" if v is None else str(v)


def _funnel_md(summary):
    lines = ["| Stage | Total | Kept/OK | Rejected/Removed |", "|---|---|---|---|"]
    for r in summary["funnel"]:
        lines.append(
            f"| {r['label']} | {_fmt_count(r['total'])} | "
            f"{_fmt_count(r['kept'])} | {_fmt_count(r['rejected'])} |"
        )
    return "\n".join(lines)


def _funnel_txt(summary):
    lines = ["Stage              Total       Kept/OK     Rejected/Removed"]
    for r in summary["funnel"]:
        lines.append(
            f"{r['label']:<18} {_fmt_count(r['total']):>10}  "
            f"{_fmt_count(r['kept']):>10}  {_fmt_count(r['rejected']):>10}"
        )
    return "\n".join(lines)


def render(summary, as_markdown):
    parts = []

    parts.append("# AIM4ML curation run summary" if as_markdown else "AIM4ML curation run summary")
    parts.append("")

    # Funnel
    parts.append("## Funnel" if as_markdown else "FUNNEL")
    parts.append(_funnel_md(summary) if as_markdown else _funnel_txt(summary))

    # Rejection breakdowns
    if summary["breakdowns"]:
        parts.append("")
        parts.append("## Rejection breakdowns" if as_markdown else "REJECTION BREAKDOWNS")
        for label, bd in summary["breakdowns"]:
            parts.append(f"### {label.strip()}" if as_markdown else f"\n{label.strip()}:")
            for reason, count in bd.items():
                if as_markdown:
                    parts.append(f"- `{reason}`: {count}")
                else:
                    parts.append(f"  {reason:<24} {count}")

    # Energies
    if summary["energies"]:
        parts.append("")
        parts.append("## Energy prefilter" if as_markdown else "ENERGY PREFILTER")
        e = summary["energies"]
        en_str = "  ".join(f"{sym}={v:.4f}" for sym, v in e.items())
        if as_markdown:
            parts.append("```")
            parts.append(en_str)
            parts.append("```")
        else:
            parts.append("  " + en_str)
        if summary["residual"]:
            med, mad = summary["residual"]
            parts.append(f"  Residual median: {med:.4f} Ha   MAD: {mad:.4f} Ha")

    # Descriptors
    if summary["descriptors"]:
        parts.append("")
        parts.append("## Descriptors" if as_markdown else "DESCRIPTORS")
        for key, val in summary["descriptors"].items():
            parts.append(f"  {key}: {val}")

    # Anomalies
    parts.append("")
    if summary["anomalies"]:
        parts.append("## Anomalies" if as_markdown else "ANOMALIES")
        for a in summary["anomalies"]:
            parts.append(f"- ⚠ {a}" if as_markdown else f"  ⚠ {a}")
    else:
        parts.append("## Anomalies" if as_markdown else "ANOMALIES")
        parts.append("  none" if not as_markdown else "- none")

    return "\n".join(parts) + "\n"


# -- Main -----------------------------------------------------------------

def _resolve_log(path):
    if os.path.isfile(path):
        return path
    log = os.path.join(path, "curation.log")
    if os.path.isfile(log):
        return log
    sys.exit(f"ERROR: no curation.log found at {path!r} (dir or .log file)")


def main():
    p = argparse.ArgumentParser(
        description="One-page AIM4ML run summary from curation.log."
    )
    p.add_argument("run", help="Run directory (or curation.log path).")
    p.add_argument("--save", action="store_true",
                   help="Also write summary.md into the run directory.")
    p.add_argument("--md", type=str, default=None,
                   help="Write markdown to this path (implies no terminal print).")
    args = p.parse_args()

    log = _resolve_log(args.run)
    with open(log, "r") as fh:
        text = fh.read()

    summary = parse_log(text)

    if args.md:
        with open(args.md, "w") as fh:
            fh.write(render(summary, as_markdown=True))
        print(f"Wrote {args.md}")
        return

    print(render(summary, as_markdown=False))

    if args.save:
        run_dir = os.path.dirname(log) if os.path.isfile(args.run) else args.run
        out = os.path.join(run_dir, "summary.md")
        with open(out, "w") as fh:
            fh.write(render(summary, as_markdown=True))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
