"""Deviance, Lorenz/Gini, calibration and double-lift."""

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
