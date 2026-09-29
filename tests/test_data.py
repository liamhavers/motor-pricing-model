import polars as pl
import pytest

from pricing.data import (
    MAX_CLAIM_NB,
    aggregate_severity,
    build_policy_table,
    clean_frequency,
    data_quality_summary,
    split_train_test,
)


@pytest.fixture
def freq():
    return pl.DataFrame(
        {
            "IDpol": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ClaimNb": [0, 1, 2, 9, 1],
            "Exposure": [0.5, 1.0, 1.5, 0.1, 0.3],
            "Area": ["A", "B", "C", "D", "E"],
            "VehBrand": ["B1", "B2", "B1", "B2", "B1"],
            "VehGas": ["'Diesel'", "'Regular'", "'Diesel'", "'Regular'", "'Diesel'"],
            "Region": ["R11"] * 5,
        }
    )


@pytest.fixture
def sev():
    # Policy 5 has a claim but no severity rows; policy 99 has severity but no policy.
    return pl.DataFrame(
        {
            "IDpol": [2, 3, 3, 4, 4, 4, 99],
            "ClaimAmount": [100.0, 200.0, 5000.0, 10.0, 20.0, 30.0, 1.0],
        }
    )


def test_clean_frequency_caps_and_strips(freq):
    out = clean_frequency(freq)
    assert out["ClaimNb"].max() == MAX_CLAIM_NB
    assert out["Exposure"].max() == 1.0
    assert set(out["VehGas"]) == {"Diesel", "Regular"}
    assert out["IDpol"].dtype == pl.Int64


def test_aggregate_severity_caps_per_claim(sev):
    out = aggregate_severity(sev, large_loss_cap=1000.0).sort("IDpol")
    row = out.filter(pl.col("IDpol") == 3).row(0, named=True)
    assert row["SevNb"] == 2
    assert row["ClaimAmount"] == 1200.0
    assert row["ClaimAmountExcess"] == 4000.0


def test_aggregate_severity_without_cap_has_no_excess(sev):
    out = aggregate_severity(sev)
    assert out["ClaimAmountExcess"].sum() == 0
    assert out["ClaimAmount"].sum() == sev["ClaimAmount"].sum()


def test_build_policy_table(freq, sev):
    out = build_policy_table(freq, sev).sort("IDpol")
    assert out.height == freq.height
    assert out["ClaimNb"].to_list() == [0, 1, 2, 3, 0]
    assert out["SevNb"].to_list() == [0, 1, 2, 3, 0]
    assert out["ClaimAmount"].to_list() == [0.0, 100.0, 5200.0, 60.0, 0.0]
    # Every policy with a claim has a positive cost, which the Gamma model requires.
    assert (out.filter(pl.col("ClaimNb") > 0)["ClaimAmount"] > 0).all()


def test_capping_preserves_total(freq, sev):
    out = build_policy_table(freq, sev, large_loss_cap=1000.0)
    uncapped = build_policy_table(freq, sev)
    total = out["ClaimAmount"].sum() + out["ClaimAmountExcess"].sum()
    assert total == pytest.approx(uncapped["ClaimAmount"].sum())


def test_data_quality_summary(freq, sev):
    s = data_quality_summary(freq, sev)
    assert s["claim_nb_above_cap"] == 1
    assert s["exposure_above_cap"] == 1
    assert s["policies_with_claims_but_no_severity"] == 1
    assert s["severity_policies_not_in_freq"] == 1


def test_split_is_disjoint_and_reproducible():
    df = pl.DataFrame({"IDpol": range(100)})
    train, test = split_train_test(df)
    assert train.height == 80 and test.height == 20
    assert set(train["IDpol"]).isdisjoint(test["IDpol"])
    assert split_train_test(df)[1]["IDpol"].to_list() == test["IDpol"].to_list()
