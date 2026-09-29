import numpy as np
import pandas as pd
import pytest

from pricing.gbm import FrequencyGBM
from pricing.glm import fit_poisson_glm, predict


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
    rate = predict(res, X)
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


def test_gamma_glm_weighted_mean_matches_claim_weighted_average():
    from pricing.glm import fit_gamma_glm

    avg = np.array([1000.0, 2000.0, 4000.0])
    count = np.array([3.0, 1.0, 1.0])
    res = fit_gamma_glm(pd.DataFrame({"const": np.ones(3)}), avg, count)
    # Intercept-only Gamma GLM recovers the average cost per claim, not per policy.
    assert np.exp(res.params["const"]) == pytest.approx((3 * 1000 + 2000 + 4000) / 5)


def test_severity_gbm_recovers_group_means():
    from pricing.gbm import SeverityGBM

    rng = np.random.default_rng(4)
    group = rng.integers(0, 2, 20_000)
    mean = np.where(group == 1, 3000.0, 1000.0)
    avg = rng.gamma(shape=2.0, scale=mean / 2.0)
    count = np.ones_like(avg)
    model = SeverityGBM({"num_leaves": 2, "min_data_in_leaf": 20}, 300).fit(pd.DataFrame({"g": group}), avg, count)
    pred = model.predict(pd.DataFrame({"g": [0, 1]}))
    assert pred[0] == pytest.approx(avg[group == 0].mean(), rel=0.02)
    assert pred[1] == pytest.approx(avg[group == 1].mean(), rel=0.02)


@pytest.fixture
def simulated_cost():
    """Compound Poisson-Gamma claim cost: two groups with pure premiums 100 and 300 a year."""
    rng = np.random.default_rng(5)
    n = 60_000
    group = rng.integers(0, 2, n)
    exposure = rng.uniform(0.1, 1.0, n)
    freq = np.where(group == 1, 0.15, 0.1)
    sev = np.where(group == 1, 2000.0, 1000.0)
    counts = rng.poisson(freq * exposure)
    cost = np.array([rng.gamma(0.5, s / 0.5, k).sum() for k, s in zip(counts, sev)])
    return group, exposure, cost


def test_tweedie_glm_recovers_group_pure_premiums(simulated_cost):
    from pricing.glm import fit_tweedie_glm

    group, exposure, cost = simulated_cost
    X = pd.DataFrame({"const": 1.0, "g": group.astype(float)})
    rate = predict(fit_tweedie_glm(X, cost, exposure, power=1.8), X)
    for g in (0, 1):
        expected = cost[group == g].sum() / exposure[group == g].sum()
        assert rate[group == g][0] == pytest.approx(expected, rel=1e-3)


def test_tweedie_gbm_is_rebased_to_training_total(simulated_cost):
    from pricing.gbm import TweedieGBM

    group, exposure, cost = simulated_cost
    X = pd.DataFrame({"g": group})
    model = TweedieGBM(1.8, {"num_leaves": 2, "min_data_in_leaf": 20}, 300).fit(X, cost, exposure)
    assert np.sum(model.predict(X) * exposure) == pytest.approx(cost.sum())
    pred = model.predict(pd.DataFrame({"g": [0, 1]}))
    assert pred[1] / pred[0] == pytest.approx(3.0, rel=0.15)
