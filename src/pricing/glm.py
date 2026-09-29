"""GLM fitting (Poisson, Gamma, Tweedie) and relativities."""

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
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


def predict_rate(results: GLMResultsWrapper, X: pd.DataFrame) -> np.ndarray:
    """Predicted claims per policy-year (offset of zero, i.e. one year of exposure)."""
    return np.exp(X.to_numpy() @ results.params.to_numpy())


def relativities(
    results: GLMResultsWrapper, levels: dict[str, list[str]], base: dict[str, str]
) -> pl.DataFrame:
    """Exponentiated coefficients with 95% confidence intervals, one row per factor level.

    A relativity of 1.3 means 30% more claims per policy-year than the base level of that
    factor, with every other factor held fixed. Base levels are included with relativity 1.
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
