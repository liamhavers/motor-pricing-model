import numpy as np
import pandas as pd
import pytest

from pricing.gbm import FrequencyGBM
from pricing.glm import fit_poisson_glm, predict_rate


@pytest.fixture
def simulated():
    """Two groups with true claim rates 0.1 and 0.2 per year and varied exposure."""
    rng = np.random.default_rng(0)
    n = 40_000
    group = rng.integers(0, 2, n)
    exposure = rng.uniform(0.05, 1.0, n)
    rate = np.where(group == 1, 0.2, 0.1)
    claims = rng.poisson(rate * exposure)
    return group, exposure, claims


def observed_rate(claims, exposure, mask):
    """Claims per policy-year in one group: what a correctly offset model should reproduce."""
    return claims[mask].sum() / exposure[mask].sum()


def test_glm_offset_recovers_rate_per_year(simulated):
    group, exposure, claims = simulated
    X = pd.DataFrame({"const": 1.0, "g": group.astype(float)})
    res = fit_poisson_glm(X, claims, exposure)
    rate = predict_rate(res, X)
    for g in (0, 1):
        assert rate[group == g][0] == pytest.approx(observed_rate(claims, exposure, group == g))
    assert np.exp(res.params["g"]) == pytest.approx(2.0, rel=0.1)
    assert np.sum(rate * exposure) == pytest.approx(claims.sum(), rel=1e-6)


def test_gbm_offset_recovers_rate_per_year(simulated):
    group, exposure, claims = simulated
    X = pd.DataFrame({"g": group})
    model = FrequencyGBM(params={"num_leaves": 2, "min_data_in_leaf": 20}, num_boost_round=300)
    rate = model.fit(X, claims, exposure).predict_rate(X)
    for g in (0, 1):
        expected = observed_rate(claims, exposure, group == g)
        assert rate[group == g].mean() == pytest.approx(expected, rel=0.01)
