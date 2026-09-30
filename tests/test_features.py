import polars as pl

from pricing.features import add_glm_bands, band_column, band_labels, base_levels, glm_design_matrix


def test_band_labels():
    assert band_labels([18, 20, 21, 30]) == ["18-19", "20", "21-29", "30+"]


def test_band_column_boundaries():
    df = pl.DataFrame({"age": [17, 18, 19, 20, 29, 30, 99]})
    out = df.select(band_column("age", [18, 20, 30]))["age"].to_list()
    assert out == ["18-19", "18-19", "18-19", "20-29", "20-29", "30+", "30+"]


def test_base_level_is_most_exposed():
    df = pl.DataFrame({"f": ["a", "b", "b"], "Exposure": [2.0, 0.5, 0.5]})
    assert base_levels(df, ["f"]) == {"f": "a"}


def test_design_matrix_drops_base_and_handles_unseen():
    levels = {"f": ["a", "b", "c"]}
    base = {"f": "b"}
    X = glm_design_matrix(pl.DataFrame({"f": ["a", "b", "c", "z"]}), levels, base)
    assert list(X.columns) == ["const", "f[a]", "f[c]"]
    assert X.loc[3, ["f[a]", "f[c]"]].sum() == 0  # unseen level falls back to base


def test_add_glm_bands_covers_all_values():
    df = pl.DataFrame(
        {"DrivAge": [18, 45, 100], "BonusMalus": [50, 55, 230], "VehAge": [0, 7, 100],
         "VehPower": [4, 8, 15], "Density": [1, 500, 27000]}
    )
    out = add_glm_bands(df)
    assert out["BonusMalus"].to_list() == ["50", "51-59", "110+"]
    assert out["Density"].to_list() == ["1-9", "316-999", "10000+"]


def test_intercept_only_design_matrix_has_every_row():
    X = glm_design_matrix(pl.DataFrame({"f": ["a", "b", "c"]}), {}, {})
    assert X.shape == (3, 1)


def test_off_discount_path():
    from pricing.features import off_discount_path

    df = pl.DataFrame({"BonusMalus": [50, 51, 52, 62, 76, 77, 100, 125]})
    assert df.select(off_discount_path())["OffDiscountPath"].to_list() == [0, 0, 1, 1, 0, 1, 0, 0]
