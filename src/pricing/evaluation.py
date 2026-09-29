"""Deviance, Lorenz/Gini, calibration and double-lift."""

import numpy as np
import polars as pl


def one_way_table(df: pl.DataFrame, factor: str | pl.Expr, name: str | None = None) -> pl.DataFrame:
    """Exposure-weighted observed frequency, severity and pure premium by level of one factor.

    `factor` is a column name or an expression (e.g. a clipped or binned column). Frequency
    is claims per policy-year (total claims / total exposure), not the mean of per-policy
    rates, so short-exposure policies do not get the same weight as full-year ones.
    """
    expr = pl.col(factor) if isinstance(factor, str) else factor
    name = name or (factor if isinstance(factor, str) else expr.meta.output_name())
    return (
        df.group_by(expr.alias(name))
        .agg(
            pl.len().alias("policies"),
            pl.col("Exposure").sum().alias("exposure"),
            pl.col("ClaimNb").sum().alias("claims"),
            pl.col("ClaimAmount").sum().alias("claim_amount"),
        )
        .with_columns(
            (pl.col("claims") / pl.col("exposure")).alias("frequency"),
            (pl.col("claim_amount") / pl.col("claims")).alias("severity"),
            (pl.col("claim_amount") / pl.col("exposure")).alias("pure_premium"),
        )
        .sort(name)
    )


def poisson_deviance(claims, pred_rate, exposure) -> float:
    """Mean Poisson deviance per policy-year, exposure-weighted.

    Compares observed claims with predicted claims (rate x exposure), summed over policies
    and divided by total exposure. Lower is better; zero means predictions equal observed
    counts exactly. Policies with no claims contribute 2 x predicted claims.
    """
    y = np.asarray(claims, dtype=float)
    mu = np.asarray(pred_rate, dtype=float) * np.asarray(exposure, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(term - (y - mu)) / np.sum(exposure))


def deviance_explained(claims, pred_rate, exposure, null_rate: float) -> float:
    """Share of the null model's deviance removed by the model (1 - D_model / D_null).

    The null model predicts `null_rate` (the training claim frequency) for every policy.
    """
    null = poisson_deviance(claims, np.full(len(claims), null_rate), exposure)
    return 1 - poisson_deviance(claims, pred_rate, exposure) / null


def lorenz_curve(observed, pred_rate, exposure) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz curve: policies sorted from lowest to highest predicted rate.

    Returns (cumulative share of exposure, cumulative share of observed claims or cost).
    For severity, pass claim cost, predicted cost per claim and claim count instead.
    A model that ranks risk well puts few claims in the low-predicted policies, so the
    curve sags far below the diagonal.
    """
    order = np.argsort(pred_rate, kind="stable")
    w = np.asarray(exposure, dtype=float)[order]
    y = np.asarray(observed, dtype=float)[order]
    x_cum = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    y_cum = np.concatenate([[0.0], np.cumsum(y) / y.sum()])
    return x_cum, y_cum


def gini(observed, pred_rate, exposure) -> float:
    """Gini coefficient from the ordered Lorenz curve: 1 - 2 x area under the curve.

    Zero means the model ranks no better than random; higher means better risk ranking.
    """
    x, y = lorenz_curve(observed, pred_rate, exposure)
    return float(1 - 2 * np.trapezoid(y, x))


def calibration_table(observed, pred, weight, n_bins: int = 10) -> pl.DataFrame:
    """Observed against predicted by band of prediction, each band carrying equal weight.

    `observed` is each row's total (claims, or claim cost), `pred` the predicted amount per
    unit of weight (claims per policy-year, or cost per claim) and `weight` the units
    (exposure, or number of claims). Rows are sorted on `pred` and cut into `n_bins` bands
    of equal total weight, so for frequency each band holds a tenth of the exposure, not a
    tenth of the policies. `observed_mean` and `predicted_mean` are per unit of weight.
    """
    order = np.argsort(pred, kind="stable")
    w = np.asarray(weight, dtype=float)[order]
    cum = np.cumsum(w) / w.sum()
    band = np.minimum((cum * n_bins - 1e-12).astype(int), n_bins - 1)
    df = pl.DataFrame(
        {
            "band": band + 1,
            "weight": w,
            "observed": np.asarray(observed, dtype=float)[order],
            "predicted": np.asarray(pred, dtype=float)[order] * w,
        }
    )
    return (
        df.group_by("band")
        .agg(pl.col("weight", "observed", "predicted").sum())
        .with_columns(
            (pl.col("observed") / pl.col("weight")).alias("observed_mean"),
            (pl.col("predicted") / pl.col("weight")).alias("predicted_mean"),
        )
        .sort("band")
    )


def balance(observed, pred, weight) -> float:
    """Total predicted divided by total observed (1.0 is perfect balance).

    Frequency: (claims, rate, exposure). Severity: (claim cost, cost per claim, claim count).
    """
    return float(np.sum(np.asarray(pred) * np.asarray(weight)) / np.sum(observed))


def gamma_deviance(avg_claim, pred, weight) -> float:
    """Mean Gamma deviance per claim, weighted by number of claims.

    Each term depends on the ratio of observed to predicted, not the difference, so a
    EUR 500 miss on a EUR 1,000 claim counts as much as a EUR 5,000 miss on a EUR 10,000
    claim. That suits costs whose spread grows with their size. Lower is better.
    """
    y = np.asarray(avg_claim, dtype=float)
    mu = np.asarray(pred, dtype=float)
    w = np.asarray(weight, dtype=float)
    return float(2 * np.sum(w * (-np.log(y / mu) + (y - mu) / mu)) / w.sum())
