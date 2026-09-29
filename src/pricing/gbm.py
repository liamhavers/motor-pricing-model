"""LightGBM fitting: Poisson for frequency (with exposure), Gamma for severity.

Frequency
---------

Exposure enters through `init_score`, LightGBM's per-row starting value for the raw
(log-scale) prediction. With a Poisson objective the model is

    log E[claims] = init_score + sum of trees(x)

so setting init_score = log(exposure) is exactly the GLM offset: the trees learn the log of
the claim rate per policy-year, and exposure scales it up or down per policy.

The alternative, used by some references, is to model the rate (claims / exposure) with
exposure as the sample weight. For the Poisson loss the two give the same gradients, so
they fit the same model. init_score is used here because it mirrors the GLM directly.

LightGBM skips its usual "start from the average" step when init_score is given, so the
log of the training claim frequency is added to the offset. Otherwise the first few hundred
trees are spent learning the overall level.

Severity
--------
The target is average claim cost per policy (capped), with a Gamma objective and the
number of claims as the sample weight, mirroring the Gamma GLM. There is no offset, so
LightGBM's usual start from the (weighted) average applies.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from pricing.data import RANDOM_SEED

BASE_PARAMS = {
    "learning_rate": 0.05,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 10,
    "seed": RANDOM_SEED,
    "deterministic": True,
    "verbose": -1,
}


def _offset(exposure: np.ndarray, base_rate: float) -> np.ndarray:
    return np.log(np.asarray(exposure, dtype=float)) + np.log(base_rate)


def _cv(params: dict, data: lgb.Dataset, nfold: int, max_rounds: int) -> tuple[int, float]:
    result = lgb.cv(
        {**BASE_PARAMS, **params},
        data,
        num_boost_round=max_rounds,
        nfold=nfold,
        stratified=False,
        seed=RANDOM_SEED,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    scores = result[f"valid {params['metric']}-mean"]
    return len(scores), float(scores[-1])


def cv_poisson_gbm(
    X: pd.DataFrame,
    claims: np.ndarray,
    exposure: np.ndarray,
    params: dict,
    nfold: int = 5,
    max_rounds: int = 3000,
) -> tuple[int, float]:
    """K-fold CV with early stopping, for choosing tree settings and the number of trees.

    Returns (best number of trees, mean CV score). The score is LightGBM's Poisson metric,
    the negative log-likelihood per policy. It differs from Poisson deviance only by a term
    that depends on the data, not the model, so ranking settings by either gives the same
    order. Model-to-model comparisons use `evaluation.poisson_deviance` instead.
    """
    base_rate = claims.sum() / exposure.sum()
    data = lgb.Dataset(X, label=claims, init_score=_offset(exposure, base_rate))
    return _cv({"objective": "poisson", "metric": "poisson", **params}, data, nfold, max_rounds)


def cv_gamma_gbm(
    X: pd.DataFrame,
    avg_claim: np.ndarray,
    claim_count: np.ndarray,
    params: dict,
    nfold: int = 5,
    max_rounds: int = 3000,
) -> tuple[int, float]:
    """As `cv_poisson_gbm`, for severity. The score is LightGBM's Gamma deviance metric, which
    ranks settings in the same order as `evaluation.gamma_deviance` but is on its own scale.
    """
    data = lgb.Dataset(X, label=avg_claim, weight=claim_count)
    return _cv(
        {"objective": "gamma", "metric": "gamma_deviance", **params}, data, nfold, max_rounds
    )


@dataclass
class FrequencyGBM:
    """A LightGBM Poisson model for claim frequency, with exposure as an offset."""

    params: dict = field(default_factory=dict)
    num_boost_round: int = 500
    booster: lgb.Booster | None = None
    base_rate: float = 0.0

    def fit(self, X: pd.DataFrame, claims: np.ndarray, exposure: np.ndarray) -> "FrequencyGBM":
        self.base_rate = float(claims.sum() / exposure.sum())
        data = lgb.Dataset(X, label=claims, init_score=_offset(exposure, self.base_rate))
        self.booster = lgb.train(
            {**BASE_PARAMS, "objective": "poisson", **self.params},
            data,
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted claims per policy-year. predict() ignores init_score, so the base rate
        is added back and no exposure term is included (one year of exposure)."""
        raw = self.booster.predict(X, raw_score=True)
        return np.exp(raw + np.log(self.base_rate))


@dataclass
class SeverityGBM:
    """A LightGBM Gamma model for average claim cost, weighted by number of claims."""

    params: dict = field(default_factory=dict)
    num_boost_round: int = 500
    booster: lgb.Booster | None = None

    def fit(self, X: pd.DataFrame, avg_claim: np.ndarray, claim_count: np.ndarray) -> "SeverityGBM":
        data = lgb.Dataset(X, label=avg_claim, weight=claim_count)
        self.booster = lgb.train(
            {**BASE_PARAMS, "objective": "gamma", **self.params},
            data,
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted cost per claim."""
        return self.booster.predict(X)
