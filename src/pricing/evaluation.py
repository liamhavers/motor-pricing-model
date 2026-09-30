"""Deviance, Lorenz/Gini, calibration and double-lift."""

import numpy as np
import polars as pl
from sklearn.metrics import mean_tweedie_deviance


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


def tweedie_deviance(cost, pred_pp, exposure, power: float) -> float:
    """Mean Tweedie deviance of pure premium per policy-year, exposure-weighted.

    Compares observed cost per policy-year (cost / exposure) with the predicted pure
    premium. Power 1 is Poisson deviance and power 2 is Gamma deviance; values in between
    suit claim cost, which is zero for most policies and positive and skewed for the rest.
    Deviances at different powers are on different scales, so models are only compared at
    the same power.
    """
    e = np.asarray(exposure, dtype=float)
    return float(
        mean_tweedie_deviance(np.asarray(cost, dtype=float) / e, pred_pp, sample_weight=e, power=power)
    )


def double_lift_table(observed, pred_a, pred_b, weight, n_bins: int = 10) -> pl.DataFrame:
    """Observed against two models' predictions, by band of the ratio pred_b / pred_a.

    Rows are sorted on how much model B's prediction differs from model A's and cut into
    `n_bins` bands of equal weight. Band 1 is where B is lowest relative to A, the last band
    where B is highest. Each series is divided by its own overall average, so a value of 1.2
    means 20% above that series' portfolio average; this removes any difference in overall
    level and leaves only the differences in shape.

    Where the models disagree, the observed line shows which one was right: if observed
    follows B's line across the bands, B's extra differentiation is real.
    """
    w = np.asarray(weight, dtype=float)
    obs = np.asarray(observed, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    order = np.argsort(b / a, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    band = np.empty(len(w), dtype=int)
    band[order] = np.minimum((cum * n_bins - 1e-12).astype(int), n_bins - 1) + 1
    df = pl.DataFrame({"band": band, "weight": w, "observed": obs, "a": a * w, "b": b * w})
    overall = {c: df[c].sum() / w.sum() for c in ("observed", "a", "b")}
    return (
        df.group_by("band")
        .agg(pl.col("weight", "observed", "a", "b").sum())
        .with_columns(
            (pl.col("b") / pl.col("a")).alias("ratio_b_to_a"),
            *[(pl.col(c) / pl.col("weight") / overall[c]).alias(f"{c}_index") for c in ("observed", "a", "b")],
            (pl.col("observed") / pl.col("a")).alias("observed_over_a"),
            (pl.col("observed") / pl.col("b")).alias("observed_over_b"),
        )
        .sort("band")
    )


def bootstrap_difference(
    metric, observed, pred_a, pred_b, weight, n_boot: int = 200, seed: int = 42
) -> tuple[float, float, float]:
    """Difference metric(B) - metric(A) with a 95% bootstrap interval.

    Policies are resampled with replacement and both models are scored on the same
    resample each time, so the interval reflects how much the difference could move with a
    different sample of policies. Returns (difference on the full data, lower, upper).
    """
    observed, pred_a, pred_b, weight = (np.asarray(x, dtype=float) for x in (observed, pred_a, pred_b, weight))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(observed), len(observed))
        diffs.append(metric(observed[i], pred_b[i], weight[i]) - metric(observed[i], pred_a[i], weight[i]))
    full = metric(observed, pred_b, weight) - metric(observed, pred_a, weight)
    return float(full), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def pearson_dispersion(claims, pred_rate, exposure, n_params: int) -> float:
    """Pearson chi-squared divided by residual degrees of freedom for a Poisson model.

    A Poisson model assumes variance equals the mean, which makes this about 1. Values
    above 1 mean the data varies more than Poisson allows (overdispersion): predictions are
    unaffected, but standard errors and confidence intervals are too narrow by a factor of
    the square root of this value.
    """
    y = np.asarray(claims, dtype=float)
    mu = np.asarray(pred_rate, dtype=float) * np.asarray(exposure, dtype=float)
    return float(np.sum((y - mu) ** 2 / mu) / (len(y) - n_params))


def top_share(observed, pred, weight, share: float = 0.1) -> float:
    """Share of observed claims (or cost) in the `share` of exposure predicted riskiest."""
    order = np.argsort(pred, kind="stable")[::-1]
    w = np.asarray(weight, dtype=float)[order]
    y = np.asarray(observed, dtype=float)[order]
    top = np.cumsum(w) <= share * w.sum()
    return float(y[top].sum() / y.sum())
