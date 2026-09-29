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
