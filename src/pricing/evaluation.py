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


def calibration_table(claims, pred_rate, exposure, n_bins: int = 10) -> pl.DataFrame:
    """Observed vs predicted frequency by band of predicted rate, with equal exposure per band.

    Bands are formed by sorting policies on predicted rate and cutting cumulative exposure
    into `n_bins` equal parts, so each band carries a tenth of the exposure, not a tenth of
    the policies.
    """
    order = np.argsort(pred_rate, kind="stable")
    w = np.asarray(exposure, dtype=float)[order]
    cum = np.cumsum(w) / w.sum()
    band = np.minimum((cum * n_bins - 1e-12).astype(int), n_bins - 1)
    df = pl.DataFrame(
        {
            "band": band + 1,
            "exposure": w,
            "claims": np.asarray(claims, dtype=float)[order],
            "pred_claims": np.asarray(pred_rate, dtype=float)[order] * w,
        }
    )
    return (
        df.group_by("band")
        .agg(pl.col("exposure", "claims", "pred_claims").sum())
        .with_columns(
            (pl.col("claims") / pl.col("exposure")).alias("observed_freq"),
            (pl.col("pred_claims") / pl.col("exposure")).alias("predicted_freq"),
        )
        .sort("band")
    )


def balance(claims, pred_rate, exposure) -> float:
    """Total predicted claims divided by total observed claims (1.0 is perfect balance)."""
    return float(np.sum(np.asarray(pred_rate) * np.asarray(exposure)) / np.sum(claims))
