import numpy as np
import pandas as pd
import pytest

from src.schema import DATE, HUMIDITY, PAGE_VIEWS, PRECIPITATION, TEMPERATURE
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def clean_df():
    """Data harian sintetis dengan skema yang sama seperti data/interim/dataset_clean.csv."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
    return pd.DataFrame({
        DATE: dates.date,
        PAGE_VIEWS: np.round(20 + weekly + rng.normal(0, 2, n)).clip(0),
        PRECIPITATION: rng.gamma(1.0, 5.0, n).round(1),
        TEMPERATURE: rng.normal(24, 1, n).round(1),
        HUMIDITY: rng.uniform(60, 90, n).round(),
    })
