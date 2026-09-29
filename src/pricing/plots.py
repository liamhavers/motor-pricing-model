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
            "axes.titleweight": "bold",
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


# Fixed colour per model, used in every comparison figure.
MODEL_COLOURS = {"GLM": SERIES[0], "GBM": SERIES[1]}


def plot_relativities(
    glm_rel: pl.DataFrame, one_way: pl.DataFrame, factor: str, levels: list[str]
) -> plt.Figure:
    """GLM relativities (with 95% intervals) against one-way relativities for one factor.

    The one-way relativity is each level's observed frequency divided by the base level's,
    ignoring every other factor. The gap between the two is what the other factors explain.
    """
    rel = glm_rel.filter(pl.col("factor") == factor)
    rel = rel.join(pl.DataFrame({"level": levels, "pos": range(len(levels))}), on="level").sort("pos")
    base = rel.filter(pl.col("is_base"))["level"][0]
    ow = one_way.with_columns(pl.col(factor).cast(pl.String))
    base_freq = ow.filter(pl.col(factor) == base)["frequency"][0]
    ow = ow.with_columns((pl.col("frequency") / base_freq).alias("rel"))
    ow = ow.join(rel.select("level", "pos"), left_on=factor, right_on="level").sort("pos")

    x = rel["pos"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(1, color=TEXT_SECONDARY, linewidth=1)
    ax.plot(
        ow["pos"].to_numpy(), ow["rel"].to_numpy(), color=SERIES[2], marker="s",
        linestyle="--", linewidth=1.5, label="One-way (observed, no adjustment)",
    )
    y = rel["relativity"].to_numpy()
    err = np.vstack([y - rel["lower_95"].to_numpy(), rel["upper_95"].to_numpy() - y])
    ax.errorbar(
        x, y, yerr=err, color=MODEL_COLOURS["GLM"], marker="o", capsize=3,
        label="GLM relativity with 95% interval",
    )
    ax.set_xticks(x, rel["level"].to_list())
    ax.set_xlabel(f"{factor} (base level: {base})")
    ax.set_ylabel("Relativity to base level")
    ax.set_title(f"{factor}: GLM relativities against one-way")
    ax.legend(loc="best", fontsize=8, labelcolor=TEXT_SECONDARY)
    return fig


def plot_lorenz(curves: dict[str, tuple[np.ndarray, np.ndarray]], ginis: dict[str, float]) -> plt.Figure:
    """Ordered Lorenz curves for several models on one set of policies."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle="--", label="Random (Gini 0)")
    for name, (x, y) in curves.items():
        step = max(1, len(x) // 2000)
        ax.plot(x[::step], y[::step], color=MODEL_COLOURS.get(name), label=f"{name} (Gini {ginis[name]:.3f})")
    ax.set_xlabel("Cumulative share of exposure\n(policies sorted from lowest to highest predicted rate)")
    ax.set_ylabel("Cumulative share of observed claims")
    ax.set_title("Lorenz curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(axis="both")
    ax.legend(loc="upper left", fontsize=8, labelcolor=TEXT_SECONDARY)
    return fig


def plot_calibration(
    tables: dict[str, pl.DataFrame],
    ylabel: str = "Claims per policy-year",
    xlabel: str = "Band of predicted rate (equal exposure)",
    title: str = "Calibration, exposure-weighted",
) -> plt.Figure:
    """Observed against predicted by band of prediction, one panel per model.

    Takes tables from `evaluation.calibration_table`. Each model's bands come from sorting
    on its own predictions, so the panels share a y scale but not their policies.
    """
    fig, axes = plt.subplots(1, len(tables), figsize=(4.5 * len(tables), 4), sharey=True)
    for ax, (name, t) in zip(np.atleast_1d(axes), tables.items()):
        ax.plot(t["band"], t["predicted_mean"], color=MODEL_COLOURS.get(name), marker="o", label="Predicted")
        ax.plot(
            t["band"], t["observed_mean"], color=TEXT_PRIMARY, marker="D", linestyle="none",
            markersize=6, label="Observed",
        )
        ax.set_xticks(t["band"].to_list())
        ax.set_xlabel(xlabel)
        ax.set_title(name)
        ax.legend(loc="upper left", fontsize=8, labelcolor=TEXT_SECONDARY)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", color=TEXT_PRIMARY)
    return fig
