"""GLM fitting (Poisson, Gamma, Tweedie) and relativities."""

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper


def fit_poisson_glm(
    X: pd.DataFrame, claims: np.ndarray, exposure: np.ndarray
) -> GLMResultsWrapper:
    """Poisson GLM with a log link and log(exposure) as an offset.

    The model is  E[claims] = exposure * exp(X @ beta), i.e.
    log E[claims] = log(exposure) + X @ beta. The offset enters with a fixed coefficient
    of 1, so the coefficients describe the claim rate per policy-year.
    """
    model = sm.GLM(
        np.asarray(claims, dtype=float),
        X,
        family=sm.families.Poisson(),
        offset=np.log(np.asarray(exposure, dtype=float)),
    )
    return model.fit()


def fit_gamma_glm(X: pd.DataFrame, avg_claim: np.ndarray, claim_count: np.ndarray) -> GLMResultsWrapper:
    """Gamma GLM with a log link on average claim cost, weighted by number of claims.

    The weights are variance weights: the average of n claims is less variable than a
    single claim (its variance is divided by n), so it is given n times the weight.
    """
    model = sm.GLM(
        np.asarray(avg_claim, dtype=float),
        X,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=np.asarray(claim_count, dtype=float),
    )
    return model.fit()


def fit_tweedie_glm(
    X: pd.DataFrame, cost: np.ndarray, exposure: np.ndarray, power: float
) -> GLMResultsWrapper:
    """Tweedie GLM with a log link on claim cost per policy-year, weighted by exposure.

    The target is (capped) claim cost divided by exposure, the pure premium per
    policy-year, with exposure as the variance weight: a full-year policy's cost per year is
    less noisy than a one-month policy's.

    Fitted with L-BFGS rather than statsmodels' default IRLS. On this data IRLS's memory use
    grows with each iteration (over 10 GB for power 1.9 on the full training set), while
    L-BFGS stays under 3 GB. With the tight tolerances below the two agree to within
    0.001% on every relativity.
    """
    model = sm.GLM(
        np.asarray(cost, dtype=float) / np.asarray(exposure, dtype=float),
        X,
        family=sm.families.Tweedie(var_power=power, link=sm.families.links.Log()),
        var_weights=np.asarray(exposure, dtype=float),
    )
    return model.fit(method="lbfgs", maxiter=5000, factr=1e2, pgtol=1e-10)


def predict(results: GLMResultsWrapper, X: pd.DataFrame) -> np.ndarray:
    """exp(X @ beta): claims per policy-year for frequency, cost per claim for severity,
    cost per policy-year for Tweedie.

    No offset is applied, so a frequency prediction is for one year of exposure.
    """
    return np.exp(X.to_numpy() @ results.params.to_numpy())


def drop_factor_tests(
    fit, X: pd.DataFrame, factors: list[str], full: GLMResultsWrapper
) -> pl.DataFrame:
    """F-test for each factor: refit without its columns and compare deviance.

    F = (increase in deviance / columns removed) / dispersion of the full model. A small
    p-value means the factor explains more than chance would. `fit` takes a design matrix
    and returns fitted results on the same rows and target as `full`.
    """
    rows = []
    for factor in factors:
        cols = [c for c in X.columns if c.startswith(f"{factor}[")]
        reduced = fit(X.drop(columns=cols))
        f_stat = (reduced.deviance - full.deviance) / len(cols) / full.scale
        rows.append(
            {
                "factor": factor,
                "columns": len(cols),
                "deviance_increase": reduced.deviance - full.deviance,
                "F": f_stat,
                "p_value": float(stats.f.sf(f_stat, len(cols), full.df_resid)),
            }
        )
    return pl.DataFrame(rows).sort("p_value")


def relativities(
    results: GLMResultsWrapper, levels: dict[str, list[str]], base: dict[str, str]
) -> pl.DataFrame:
    """Exponentiated coefficients with 95% confidence intervals, one row per factor level.

    A relativity of 1.3 means 30% more claims per policy-year (frequency) or a 30% higher
    cost per claim (severity) than the base level of that factor, with every other factor
    held fixed. Base levels are included with relativity 1.
    """
    ci = results.conf_int()
    rows = []
    for factor, lvls in levels.items():
        for lvl in lvls:
            name = f"{factor}[{lvl}]"
            if lvl == base[factor]:
                coef, lo, hi = 0.0, 0.0, 0.0
            else:
                coef, lo, hi = results.params[name], ci.loc[name, 0], ci.loc[name, 1]
            rows.append(
                {
                    "factor": factor,
                    "level": lvl,
                    "is_base": lvl == base[factor],
                    "relativity": float(np.exp(coef)),
                    "lower_95": float(np.exp(lo)),
                    "upper_95": float(np.exp(hi)),
                }
            )
    return pl.DataFrame(rows)
