from datetime import date, timedelta

import numpy as np

from src.data.validate_dataset import run_checks
from src.schema import DATE, HUMIDITY, PAGE_VIEWS, TEMPERATURE


def _today_for(df):
    return max(df[DATE]) + timedelta(days=2)


def test_clean_data_passes(clean_df, config):
    assert run_checks(clean_df, config["validation"], today=_today_for(clean_df)) == []


def test_missing_column_fails(clean_df, config):
    errors = run_checks(clean_df.drop(columns=[PAGE_VIEWS]), config["validation"])
    assert any("Kolom wajib hilang" in e for e in errors)


def test_out_of_range_values_fail(clean_df, config):
    df = clean_df.copy()
    df.loc[0, HUMIDITY] = 150
    df.loc[1, PAGE_VIEWS] = -5
    errors = run_checks(df, config["validation"], today=_today_for(df))
    assert any(HUMIDITY in e for e in errors)
    assert any(PAGE_VIEWS in e for e in errors)


def test_duplicate_dates_and_missing_values_fail(clean_df, config):
    df = clean_df.copy()
    df.loc[1, DATE] = df.loc[0, DATE]
    df.loc[: len(df) // 5, TEMPERATURE] = np.nan
    errors = run_checks(df, config["validation"], today=_today_for(df))
    assert any("duplikat" in e for e in errors)
    assert any("missing value" in e for e in errors)


def test_stale_data_fails(clean_df, config):
    errors = run_checks(clean_df, config["validation"], today=date(2030, 1, 1))
    assert any("basi" in e for e in errors)
