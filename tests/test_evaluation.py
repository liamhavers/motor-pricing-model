import numpy as np
import polars as pl
import pytest

from pricing.evaluation import one_way_table


def test_one_way_table_is_exposure_weighted():
    df = pl.DataFrame(
        {
            "Area": ["A", "A", "B"],
            "Exposure": [1.0, 0.25, 0.5],
            "ClaimNb": [1, 0, 1],
            "ClaimAmount": [100.0, 0.0, 300.0],
        }
    )
    out = one_way_table(df, "Area")
    a = out.filter(pl.col("Area") == "A").row(0, named=True)
    # 1 claim over 1.25 policy-years, not the mean of per-policy rates (0.5).
    assert a["frequency"] == pytest.approx(0.8)
    assert a["severity"] == pytest.approx(100.0)
    assert a["pure_premium"] == pytest.approx(80.0)


def test_one_way_table_accepts_expression():
    df = pl.DataFrame({"x": [1, 7, 12], "Exposure": [1.0] * 3, "ClaimNb": [0] * 3,
                       "ClaimAmount": [0.0] * 3})
    out = one_way_table(df, pl.col("x") // 10 * 10, "x_band")
    assert out["x_band"].to_list() == [0, 10]
    assert out["policies"].to_list() == [2, 1]


def test_poisson_deviance_matches_sklearn_rate_form():
    from sklearn.metrics import mean_poisson_deviance

    from pricing.evaluation import poisson_deviance

    rng = np.random.default_rng(1)
    e = rng.uniform(0.1, 1, 1000)
    rate = rng.uniform(0.05, 0.3, 1000)
    y = rng.poisson(rate * e)
    expected = mean_poisson_deviance(y / e, rate, sample_weight=e)
    assert poisson_deviance(y, rate, e) == pytest.approx(expected)
    assert poisson_deviance(y, y / e + 1e-300, e) == pytest.approx(0, abs=1e-9)


def test_gini_orders_models():
    from pricing.evaluation import gini

    rng = np.random.default_rng(2)
    rate = rng.uniform(0.01, 0.5, 50_000)
    e = np.ones_like(rate)
    y = rng.poisson(rate)
    assert gini(y, rate, e) > 0.2
    assert abs(gini(y, rng.uniform(size=rate.size), e)) < 0.02


def test_calibration_bands_have_equal_exposure():
    from pricing.evaluation import calibration_table

    rng = np.random.default_rng(3)
    e = rng.uniform(0.1, 1, 10_000)
    rate = rng.uniform(0.05, 0.3, 10_000)
    y = rng.poisson(rate * e)
    t = calibration_table(y, rate, e, n_bins=10)
    assert t.height == 10
    assert t["weight"].to_numpy() == pytest.approx(e.sum() / 10, rel=0.01)
    assert t["observed"].sum() == y.sum()


def test_gamma_deviance_is_zero_when_exact_and_scale_free():
    from pricing.evaluation import gamma_deviance

    y = np.array([100.0, 1000.0, 5000.0])
    w = np.array([1.0, 2.0, 1.0])
    assert gamma_deviance(y, y, w) == pytest.approx(0)
    # Same relative error at any scale gives the same deviance.
    assert gamma_deviance(y, 1.5 * y, w) == pytest.approx(gamma_deviance(10 * y, 15 * y, w))
