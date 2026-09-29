"""Plotting helpers for reports/figures.

Colours follow a fixed categorical order (blue first) so the same model or series keeps
the same colour across every figure. Text stays in neutral ink rather than series colour.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import StrMethodFormatter

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def set_style() -> None:
    """Apply a light, recessive house style to matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.prop_cycle": plt.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def plot_one_way(
    table: pl.DataFrame,
    factor: str,
    value: str = "frequency",
    value_label: str = "Claims per policy-year",
    overall: float | None = None,
    rotate_labels: bool = False,
    ordered: bool = True,
) -> plt.Figure:
    """Observed value by factor level (top) with the exposure behind it (bottom).

    Two stacked panels on a shared x-axis rather than one chart with two y-axes, so
    each measure is read against its own scale. Levels with little exposure are the
    ones whose observed value is noisy, which the lower panel makes visible. Unordered
    factors (e.g. region codes) are drawn as points only, since a joining line would
    suggest an order that does not exist.
    """
    levels = [str(v) for v in table[factor].to_list()]
    x = np.arange(len(levels))
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 5), gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08}
    )
    top.plot(
        x, table[value].to_numpy(), color=SERIES[0], marker="o",
        linestyle="-" if ordered else "none", markersize=5 if ordered else 7,
    )
    if overall is not None:
        top.axhline(
            overall, color=TEXT_SECONDARY, linewidth=1, linestyle="--", label="Portfolio average"
        )
        top.legend(loc="best", fontsize=8, labelcolor=TEXT_SECONDARY)
    top.set_ylabel(value_label)
    top.set_ylim(bottom=0)
    top.set_title(f"{factor}: observed {value_label.lower()}, exposure-weighted")

    bottom.bar(x, table["exposure"].to_numpy(), color=SERIES[0], alpha=0.35, width=0.8)
    bottom.set_ylabel("Exposure\n(policy-years)")
    bottom.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    bottom.set_xticks(x, levels, rotation=90 if rotate_labels else 0)
    if len(levels) > 30:
        for i, label in enumerate(bottom.get_xticklabels()):
            label.set_visible(i % 5 == 0)
    bottom.set_xlabel(factor)
    return fig


def plot_severity_distribution(amounts: np.ndarray, markers: dict[str, float]) -> plt.Figure:
    """Histogram of individual claim amounts on a log scale, with reference lines."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(0, np.log10(amounts.max()) + 0.1, 80)
    ax.hist(amounts, bins=bins, color=SERIES[0], alpha=0.8, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for label, value in markers.items():
        ax.axvline(value, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.annotate(
            label, (value, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
            textcoords="offset points", color=TEXT_SECONDARY, fontsize=8,
        )
    ax.set_xlabel("Claim amount (EUR, log scale)")
    ax.set_ylabel("Number of claims (log scale)")
    ax.set_title("Individual claim amounts")
    return fig
