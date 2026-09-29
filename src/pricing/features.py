"""Banding and encoding of rating factors (banded for the GLM, raw for the GBM).

GLM banding choices, made on the training data one-ways:

- DrivAge: two-year bands from 18 to 25, where frequency falls steeply (28% at 18 to 11% at
  25), then 26-29 and ten-year bands. The young bands are narrow because that is where the
  money is; the ten-year bands are wide because frequency is nearly flat.
- BonusMalus: 50 on its own (57% of exposure), then bands of 10 up to 110, and 110+.
  Individual values are not monotone: values on the 5%-a-year discount path from 100
  (95, 90, ... 57, 54, 51) have much lower frequency than values in between, which are only
  reachable after a claim. Ordered bands average over this; the GBM can split on it.
- VehAge: 0, 1-5 and 6-12 (roughly flat), 13-17, 18+ (frequency falls for old vehicles).
- VehPower: each value from 4 to 11, then 12+ where exposure is thin.
- Density: bands on a log10 scale (half a decade each), since frequency rises roughly
  with log density. Area is not used: it is a coarser banding of the same quantity.
- VehBrand, VehGas, Region: kept as they are. The smallest region has about 450
  policy-years in training, so its relativity will have a wide confidence interval.

The base (reference) level of each factor is the level with the most training exposure,
so relativities read as "relative to the most common risk".
"""

from itertools import pairwise

import pandas as pd
import polars as pl

# Lower bounds of each band; a band runs up to the next lower bound minus one.
BANDS: dict[str, list[int]] = {
    "DrivAge": [18, 20, 22, 24, 26, 30, 40, 50, 60, 70, 80],
    "BonusMalus": [50, 51, 60, 70, 80, 90, 100, 110],
    "VehAge": [0, 1, 6, 13, 18],
    "VehPower": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Density": [1, 10, 32, 100, 316, 1000, 3162, 10000],
}

GLM_FACTORS = [*BANDS, "VehBrand", "VehGas", "Region"]
GBM_NUMERIC = ["DrivAge", "BonusMalus", "VehAge", "VehPower", "Density"]
GBM_CATEGORICAL = ["VehBrand", "VehGas", "Region"]
GBM_FEATURES = GBM_NUMERIC + GBM_CATEGORICAL


def band_labels(bounds: list[int]) -> list[str]:
    """Readable labels for a list of lower bounds, e.g. [18, 20, 22] -> 18-19, 20-21, 22+."""
    labels = [
        f"{lo}" if hi - lo == 1 else f"{lo}-{hi - 1}" for lo, hi in pairwise(bounds)
    ]
    return [*labels, f"{bounds[-1]}+"]


def band_column(col: str, bounds: list[int]) -> pl.Expr:
    """Map a numeric column onto its band label. Values below the first bound go in band one."""
    labels = band_labels(bounds)
    expr = pl.lit(labels[0])
    for lo, label in zip(bounds[1:], labels[1:]):
        expr = pl.when(pl.col(col) >= lo).then(pl.lit(label)).otherwise(expr)
    return expr.alias(col)


def add_glm_bands(df: pl.DataFrame) -> pl.DataFrame:
    """Replace the banded numeric factors with their band labels (as strings)."""
    return df.with_columns(band_column(col, bounds) for col, bounds in BANDS.items())


def base_levels(banded_train: pl.DataFrame, factors: list[str] = GLM_FACTORS) -> dict[str, str]:
    """The level with the most exposure for each factor, used as the GLM reference level."""
    return {
        f: banded_train.group_by(f)
        .agg(pl.col("Exposure").sum())
        .sort(["Exposure", f], descending=[True, False])[f][0]
        for f in factors
    }


def glm_levels(banded_train: pl.DataFrame, factors: list[str] = GLM_FACTORS) -> dict[str, list[str]]:
    """All levels seen in training for each factor, fixing the design matrix columns.

    Banded factors keep their band order (18-19 before 20-21); others are sorted by name.
    """
    seen = {f: set(banded_train[f].unique().to_list()) for f in factors}
    return {
        f: [lvl for lvl in band_labels(BANDS[f]) if lvl in seen[f]] if f in BANDS else sorted(seen[f])
        for f in factors
    }


def glm_design_matrix(
    banded: pl.DataFrame, levels: dict[str, list[str]], base: dict[str, str]
) -> pd.DataFrame:
    """One-hot design matrix with an intercept and the base level of each factor dropped.

    Columns are named `factor[level]`. Levels are fixed from training, so the test matrix has
    the same columns; a level not seen in training would get all zeros (the base level).
    """
    columns = {"const": pl.lit(1.0)}
    for f, lvls in levels.items():
        for lvl in lvls:
            if lvl != base[f]:
                columns[f"{f}[{lvl}]"] = (pl.col(f) == lvl).cast(pl.Float64)
    return banded.select(**columns).to_pandas()


def gbm_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """Raw features for LightGBM, with categoricals as pandas categories (native handling)."""
    out = df.select(GBM_FEATURES).to_pandas()
    for c in GBM_CATEGORICAL:
        out[c] = out[c].astype("category")
    return out
