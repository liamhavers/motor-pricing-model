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


def test_double_lift_bands_follow_ratio_and_index_to_one():
    from pricing.evaluation import double_lift_table

    rng = np.random.default_rng(6)
    n = 20_000
    a = rng.uniform(0.05, 0.2, n)
    b = a * rng.uniform(0.5, 2.0, n)
    w = rng.uniform(0.1, 1, n)
    y = rng.poisson(b * w)  # B is the true model
    t = double_lift_table(y, a, b, w, n_bins=5)
    assert t["ratio_b_to_a"].is_sorted()
    assert t["weight"].to_numpy() == pytest.approx(w.sum() / 5, rel=0.01)
    for c in ("observed_index", "a_index", "b_index"):
        assert np.average(t[c].to_numpy(), weights=t["weight"].to_numpy()) == pytest.approx(1.0)
    # Observed follows the true model: it rises across bands, A's index falls.
    assert t["observed_index"][-1] > t["observed_index"][0]
    assert t["a_index"][-1] < t["a_index"][0]


def test_pearson_dispersion_near_one_for_poisson_data():
    from pricing.evaluation import pearson_dispersion

    rng = np.random.default_rng(7)
    rate = rng.uniform(0.05, 0.3, 100_000)
    e = rng.uniform(0.1, 1, rate.size)
    assert pearson_dispersion(rng.poisson(rate * e), rate, e, 1) == pytest.approx(1.0, abs=0.03)


def test_bootstrap_difference_contains_full_difference():
    from pricing.evaluation import bootstrap_difference, gini

    rng = np.random.default_rng(8)
    rate = rng.uniform(0.01, 0.5, 20_000)
    e = np.ones_like(rate)
    y = rng.poisson(rate)
    noisy = rate * rng.uniform(0.5, 1.5, rate.size)
    diff, lo, hi = bootstrap_difference(gini, y, noisy, rate, e, n_boot=100)
    assert lo < diff < hi
    assert lo > 0  # the true rate ranks better than a noisy version


def test_top_share():
    from pricing.evaluation import top_share

    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0])
    pred = np.arange(10.0)
    assert top_share(y, pred, np.ones(10), 0.1) == pytest.approx(1.0)


def test_leaf_rules_describe_integer_splits():
    from sklearn.tree import DecisionTreeRegressor

    from pricing.evaluation import leaf_rules

    X = np.array([[a, b] for a in range(18, 80) for b in (50, 100)], dtype=float)
    y = (X[:, 0] < 30) * 1.0 + (X[:, 1] > 50) * 2.0
    tree = DecisionTreeRegressor(max_depth=2).fit(X, y)
    rules = set(leaf_rules(tree, ["DrivAge", "BonusMalus"]).values())
    # Only 50 and 100 occur, so the tree splits at the midpoint 75: BonusMalus <= 75.
    assert "BonusMalus < 76, DrivAge < 30" in rules
    assert any("DrivAge >= 30" in r for r in rules)


def test_mispricing_table_shift_sign():
    from pricing.evaluation import mispricing_table

    df = pl.DataFrame({"seg": ["a", "b"], "exposure": [1.0, 1.0], "observed": [50.0, 150.0],
                       "glm": [100.0, 100.0], "gbm": [60.0, 140.0]})
    t = mispricing_table(df, "seg")
    assert t["shift"].to_list() == [-40.0, 40.0]
    assert t["observed_over_glm"].to_list() == [0.5, 1.5]
