#!/usr/bin/env python3
"""
plot_histograms.py — Publication-quality MolWt and Tanimoto histograms.

Reads stats_summary.csv and produces two histogram panels in a single figure.
Tanimoto is computed on unique CompoundIDs (same as 09_stats.py).

Usage:
    python3 tools/plot_histograms.py stats/stats_summary.csv
    python3 tools/plot_histograms.py stats/stats_summary.csv -o figures/
    python3 tools/plot_histograms.py stats/stats_summary.csv --latex  # LaTeX fonts (PDF + PGF)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# -- Plotting -------------------------------------------------------------


def plot_molwt(ax, df, color):
    """Molecular weight histogram."""
    mw = df["MolWt"].dropna()
    bins = np.arange(100, 600, 20)
    ax.hist(mw, bins=bins, color=color, edgecolor="white", linewidth=0.3, alpha=0.85)

    mean = mw.mean()
    median = mw.median()

    ax.axvline(mean, color="#D55E00", linestyle="--", linewidth=1.0, label=f"Mean = {mean:.0f} Da")
    ax.axvline(
        median, color="#009E73", linestyle=":", linewidth=1.0, label=f"Median = {median:.0f} Da"
    )

    ax.set_xlabel("Molecular weight / Da")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(100, 600)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(50))


def plot_tanimoto(ax, df, color, no_t1=False):
    """Max Tanimoto histogram (unique CompoundIDs)."""
    # Deduplicate by CompoundID (same as 09_stats.py before Tanimoto)
    if "CompoundID" in df.columns:
        df = df.drop_duplicates(subset="CompoundID")

    tani = df["max_tanimoto"].dropna()
    n_t1 = int((tani == 1.0).sum())

    if no_t1 and n_t1 > 0:
        tani = tani[tani < 1.0]
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(tani, bins=bins, color=color, edgecolor="white", linewidth=0.3, alpha=0.85)

    mean = tani.mean()
    median = tani.median()

    ax.axvline(mean, color="#D55E00", linestyle="--", linewidth=1.0, label=f"Mean = {mean:.3f}")
    ax.axvline(
        median, color="#009E73", linestyle=":", linewidth=1.0, label=f"Median = {median:.3f}"
    )

    # Annotate T=1.0 spike (only if shown)
    if not no_t1 and n_t1 > 0:
        ax.annotate(
            f"T=1.0: {n_t1} ({100 * n_t1 / len(tani):.1f}%)",
            xy=(1.0, n_t1),
            xytext=(0.78, n_t1 * 1.3),
            fontsize=7,
            color="#CC79A7",
            arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=0.8),
        )

    ax.set_xlabel("Max Tanimoto (ECFP4)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0, 0.95 if no_t1 else 1.0)


# -- Main ----------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Publication-quality MolWt and Tanimoto histograms.")
    p.add_argument("csv", type=str, help="Path to stats_summary.csv.")
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory (default: figures/).",
    )
    p.add_argument(
        "--latex",
        action="store_true",
        help="Enable LaTeX rendering (requires LaTeX + cm-super). Outputs PDF + PGF.",
    )
    p.add_argument(
        "--style",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "histograms.mplstyle"),
        help="Matplotlib style file (default: tools/histograms.mplstyle).",
    )
    p.add_argument(
        "--color-molwt",
        type=str,
        default="#56B4E9",
        help="MolWt histogram color (default: #56B4E9).",
    )
    p.add_argument(
        "--no-t1", action="store_true", help="Exclude T=1.0 pairs from Tanimoto histogram."
    )
    p.add_argument(
        "--color-tanimoto",
        type=str,
        default="#56B4E9",
        help="Tanimoto histogram color (default: #56B4E9).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # -- Load style -------------------------------------------------------
    if os.path.isfile(args.style):
        plt.style.use(args.style)
    else:
        print(f"Warning: style file not found: {args.style}")

    # Override for histograms
    plt.rcParams.update(
        {
            "figure.figsize": (10, 4),
            "figure.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
        }
    )

    if not args.latex:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                "mathtext.fontset": "dejavusans",
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "pgf.rcfonts": False,
            }
        )

    # -- Load data --------------------------------------------------------
    if not os.path.isfile(args.csv):
        print(f"Error: CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # -- Create figure ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.35)

    plot_molwt(ax1, df, args.color_molwt)
    ax1.set_title("(a) Molecular weight distribution")

    plot_tanimoto(ax2, df, args.color_tanimoto, args.no_t1)
    ax2.set_title("(b) Max Tanimoto nearest-neighbour")

    # -- Save -------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    if args.latex:
        pdf_path = os.path.join(args.output_dir, "histograms.pdf")
        pgf_path = os.path.join(args.output_dir, "histograms.pgf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(pgf_path, format="pgf", bbox_inches="tight")
        print(f"Saved PDF: {pdf_path}")
        print(f"Saved PGF: {pgf_path}")
    else:
        pdf_path = os.path.join(args.output_dir, "histograms.pdf")
        png_path = os.path.join(args.output_dir, "histograms.png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", bbox_inches="tight")
        print(f"Saved PDF: {pdf_path}")
        print(f"Saved PNG: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
