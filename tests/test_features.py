import numpy as np
import pandas as pd

from src.features.build_features import build_features, feature_columns
from src.schema import DATE, PAGE_VIEWS, TARGET, VALUE_COLUMNS


def test_target_is_mean_of_next_horizon_days(clean_df, config):
    feats = build_features(clean_df, config["features"]).set_index(DATE)
    pv = clean_df.assign(**{DATE: pd.to_datetime(clean_df[DATE])}).set_index(DATE)[PAGE_VIEWS]
    horizon = config["features"]["horizon_days"]

    t = feats.dropna(subset=[TARGET]).index[10]
    expected = pv.loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=horizon)].mean()
    assert np.isclose(feats.loc[t, TARGET], expected)


def test_features_do_not_use_future_data(clean_df, config):
    # Ubah semua data setelah tanggal `cutoff`; fitur pada `cutoff` tidak boleh berubah.
    base = build_features(clean_df, config["features"]).set_index(DATE)
    cutoff = base.index[50]

    future_mask = pd.to_datetime(clean_df[DATE]) > cutoff
    perturbed = clean_df.copy()
    perturbed.loc[future_mask, VALUE_COLUMNS] = perturbed.loc[future_mask, VALUE_COLUMNS] * 3 + 7
    changed = build_features(perturbed, config["features"]).set_index(DATE)

    cols = feature_columns(base)
    pd.testing.assert_series_equal(base.loc[cutoff, cols], changed.loc[cutoff, cols])


def test_missing_day_is_not_treated_as_consecutive(clean_df, config):
    # Hari yang hilang tidak boleh membuat pv_lag_1 mengambil nilai dua hari lalu.
    gap_date = clean_df[DATE].iloc[100]
    df = clean_df[clean_df[DATE] != gap_date]
    feats = build_features(df, config["features"]).set_index(DATE)
    next_day = pd.Timestamp(gap_date) + pd.Timedelta(days=1)
    assert next_day not in feats.index
