"""Loading, cleaning, joining and capping of the freMTPL2 frequency and severity data.

The frequency table has one row per policy. The severity table has one row per claim,
so it is aggregated to policy level before joining on `IDpol`.

Cleaning rules (all counts are from the full OpenML data):

- `ClaimNb` is capped at `MAX_CLAIM_NB`. A handful of policies report 5 to 16 claims,
  several of them within a few weeks of exposure. These look like data errors or fleet
  policies rather than private motor risks.
- `Exposure` is capped at 1 year. About 1,200 policies exceed 1 (max 2.01), which is not
  possible for a policy-year record.
- Policies with claims recorded in the frequency table but no rows in the severity table
  (about 9,100 policies) have their claim count set to the number of severity rows,
  i.e. zero. Otherwise the frequency model would predict claims that the severity and
  Tweedie models never see a cost for, and frequency x severity would not reconcile
  with total claim cost.
- Large losses are capped per claim, not per policy, because the cap is meant to limit
  the effect of individual large claims. The threshold is chosen in the severity stage.
"""

from pathlib import Path

import polars as pl
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
MAX_CLAIM_NB = 4
MAX_EXPOSURE = 1.0

FREQ_OPENML_ID = 41214
SEV_OPENML_ID = 41215

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RATING_FACTORS = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
]


def _fetch_cached(data_id: int, name: str, raw_dir: Path) -> pl.DataFrame:
    path = raw_dir / f"{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = fetch_openml(data_id=data_id, as_frame=True, data_home=raw_dir / "openml").frame
    df = pl.from_pandas(frame)
    df.write_parquet(path)
    return df


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the raw (freq, sev) tables, downloading from OpenML on first call."""
    freq = _fetch_cached(FREQ_OPENML_ID, "freMTPL2freq", raw_dir)
    sev = _fetch_cached(SEV_OPENML_ID, "freMTPL2sev", raw_dir)
    return freq, sev


def clean_frequency(freq: pl.DataFrame) -> pl.DataFrame:
    """Fix types, strip stray quotes from VehGas, and cap ClaimNb and Exposure."""
    return freq.with_columns(
        pl.col("IDpol").cast(pl.Int64),
        pl.col("ClaimNb").cast(pl.Int64).clip(upper_bound=MAX_CLAIM_NB),
        pl.col("Exposure").clip(upper_bound=MAX_EXPOSURE),
        pl.col("Area", "VehBrand", "Region").cast(pl.String),
        pl.col("VehGas").cast(pl.String).str.strip_chars("'"),
    )


def aggregate_severity(sev: pl.DataFrame, large_loss_cap: float | None = None) -> pl.DataFrame:
    """Aggregate claim-level amounts to one row per policy.

    Returns `SevNb` (number of claim rows), `ClaimAmount` (capped total if a cap is given)
    and `ClaimAmountExcess` (the amount removed by the cap, zero without a cap).
    """
    amount = pl.col("ClaimAmount")
    capped = amount.clip(upper_bound=large_loss_cap) if large_loss_cap is not None else amount
    return (
        sev.with_columns(pl.col("IDpol").cast(pl.Int64))
        .group_by("IDpol")
        .agg(
            pl.len().cast(pl.Int64).alias("SevNb"),
            capped.sum().alias("ClaimAmount"),
            (amount - capped).sum().alias("ClaimAmountExcess"),
        )
    )


def build_policy_table(
    freq: pl.DataFrame, sev: pl.DataFrame, large_loss_cap: float | None = None
) -> pl.DataFrame:
    """Clean and join the raw tables into one row per policy.

    `ClaimNb` is the (capped) claim count for the frequency model. `SevNb` is the number
    of costed claims, used to average and weight severity. Severity rows with no matching
    policy (a handful) are dropped by the left join.
    """
    policies = clean_frequency(freq).join(
        aggregate_severity(sev, large_loss_cap), on="IDpol", how="left"
    )
    return policies.with_columns(
        pl.col("SevNb", "ClaimAmount", "ClaimAmountExcess").fill_null(0),
    ).with_columns(
        pl.min_horizontal("ClaimNb", "SevNb").alias("ClaimNb"),
    )


def data_quality_summary(freq: pl.DataFrame, sev: pl.DataFrame) -> dict[str, int]:
    """Count the issues the cleaning rules deal with, on the raw tables."""
    raw = freq.with_columns(pl.col("IDpol").cast(pl.Int64))
    sev_policy = aggregate_severity(sev)
    joined = raw.join(sev_policy, on="IDpol", how="left")
    no_sev = joined.filter((pl.col("ClaimNb") > 0) & pl.col("SevNb").is_null())
    return {
        "policies": raw.height,
        "claim_rows": sev.height,
        "claim_nb_above_cap": raw.filter(pl.col("ClaimNb") > MAX_CLAIM_NB).height,
        "exposure_above_cap": raw.filter(pl.col("Exposure") > MAX_EXPOSURE).height,
        "policies_with_claims_but_no_severity": no_sev.height,
        "claims_with_no_severity": int(no_sev["ClaimNb"].sum()),
        "severity_count_mismatch": joined.filter(
            pl.col("SevNb").is_not_null() & (pl.col("SevNb") != pl.col("ClaimNb"))
        ).height,
        "severity_policies_not_in_freq": sev_policy.join(raw, on="IDpol", how="anti").height,
    }


def split_train_test(
    df: pl.DataFrame, test_size: float = 0.2, seed: int = RANDOM_SEED
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Random 80/20 split by policy with a fixed seed. The test set is used once, at the end."""
    train_idx, test_idx = train_test_split(
        range(df.height), test_size=test_size, random_state=seed
    )
    return df[sorted(train_idx)], df[sorted(test_idx)]
