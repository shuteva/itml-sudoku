"""
statistics_analysis.py
Mia Shuteva – 89231346

Generates runtime and accuracy plots for the three Sudoku algorithms.
Run after sudoku_solver.py has produced results_summary.csv,
or call analyse(results_list) directly from the pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


COLORS = {
    "CSP (Backtracking + MRV)":  "#2E86AB",
    "ANN (Convolutional)":        "#E84855",
    "Hybrid (ANN → CSP)":         "#3BB273",
}

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid":       True,
        "grid.alpha":      0.3,
        "figure.dpi":      150,
    })


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_accuracy_bar(df: pd.DataFrame):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))

    algos   = df["algorithm"].tolist()
    accvals = df["accuracy"].tolist()
    colors  = [COLORS.get(a, "#888") for a in algos]

    bars = ax.bar(algos, accvals, color=colors, width=0.5, zorder=3)

    for bar, val in zip(bars, accvals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Solver Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.set_xticklabels(algos, wrap=True)
    plt.tight_layout()

    path = OUTPUT_DIR / "accuracy_bar.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved {path}")
    return path


def plot_runtime_bar(df: pd.DataFrame):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))

    algos   = df["algorithm"].tolist()
    means   = df["mean_runtime_s"].tolist()
    stds    = df["std_runtime_s"].tolist()
    colors  = [COLORS.get(a, "#888") for a in algos]

    bars = ax.bar(algos, means, yerr=stds, color=colors, width=0.5,
                  capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#444"},
                  zorder=3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + max(means) * 0.01,
            f"{mean*1000:.1f} ms",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("Mean Runtime per Puzzle  (±1 std dev)", fontsize=13, fontweight="bold")
    ax.set_xticklabels(algos, wrap=True)
    plt.tight_layout()

    path = OUTPUT_DIR / "runtime_bar.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved {path}")
    return path


def plot_tradeoff_scatter(df: pd.DataFrame):
    """Accuracy vs mean runtime scatter — the key trade-off plot."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 5))

    for _, row in df.iterrows():
        color = COLORS.get(row["algorithm"], "#888")
        ax.scatter(
            row["mean_runtime_s"] * 1000,
            row["accuracy"],
            s=120, color=color, zorder=5, label=row["algorithm"]
        )
        ax.annotate(
            row["algorithm"].split(" ")[0],
            (row["mean_runtime_s"] * 1000, row["accuracy"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=9, color=color
        )

    ax.set_xlabel("Mean Runtime (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Runtime–Accuracy Trade-off", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()

    path = OUTPUT_DIR / "tradeoff_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved {path}")
    return path


def plot_percentile_runtime(df: pd.DataFrame):
    """Bar chart comparing p50 and p95 runtimes."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))

    n     = len(df)
    x     = np.arange(n)
    w     = 0.35
    algos = df["algorithm"].tolist()

    p50 = df["p50_runtime_s"].values * 1000
    p95 = df["p95_runtime_s"].values * 1000

    b1 = ax.bar(x - w/2, p50, w, label="P50 (median)", zorder=3,
                color=[COLORS.get(a, "#888") for a in algos], alpha=0.9)
    b2 = ax.bar(x + w/2, p95, w, label="P95", zorder=3,
                color=[COLORS.get(a, "#888") for a in algos], alpha=0.5,
                hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(algos, wrap=True)
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("P50 vs P95 Runtime per Algorithm", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.4)
    plt.tight_layout()

    path = OUTPUT_DIR / "percentile_runtime.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved {path}")
    return path


# ─────────────────────────────────────────────
# COMBINED REPORT
# ─────────────────────────────────────────────

def analyse(results: list[dict] | pd.DataFrame):
    """
    Generate all plots from a list of result dicts (or DataFrame).
    Returns paths to generated figures.
    """
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results

    print("\n[Stats] Generating figures…")
    paths = [
        plot_accuracy_bar(df),
        plot_runtime_bar(df),
        plot_tradeoff_scatter(df),
        plot_percentile_runtime(df),
    ]

    # Print variance analysis
    print("\n─── Variance Analysis ─────────────────────────────────")
    for _, row in df.iterrows():
        cv = row["std_runtime_s"] / row["mean_runtime_s"] if row["mean_runtime_s"] else 0
        print(f"  {row['algorithm']:30s}  CV={cv:.2f}  "
              f"range=[{row['min_runtime_s']*1000:.2f}, {row['max_runtime_s']*1000:.2f}] ms")

    return paths


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    summary_path = Path("results_summary.csv")
    if not summary_path.exists():
        print(f"[Error] '{summary_path}' not found. Run sudoku_solver.py first.")
        raise SystemExit(1)

    df = pd.read_csv(summary_path)
    analyse(df)
    print("\n[Done] All figures saved to ./figures/")
