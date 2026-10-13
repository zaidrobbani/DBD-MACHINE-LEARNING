import numpy as np
import pandas as pd

from src.monitoring.check_drift import data_drift, decide, performance_drift
from src.schema import DATE, TARGET


def _page_views(values):
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="D"), dtype=float)


def test_data_drift_flags_spike_but_not_normal_noise():
    rng = np.random.default_rng(0)
    normal = rng.normal(20, 2, 97)

    stable = data_drift(_page_views(normal), window=7, reference_days=90)
    spike = data_drift(_page_views(np.concatenate([normal[:90], normal[90:] + 30])), window=7, reference_days=90)

    assert abs(stable["zscore"]) < 3
    assert spike["zscore"] > 3


def test_data_drift_needs_enough_history():
    assert data_drift(_page_views([20] * 8), window=7, reference_days=90)["zscore"] is None


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_performance_drift_only_uses_rows_after_training():
    features = pd.DataFrame({
        DATE: pd.date_range("2026-01-01", periods=20, freq="D"),
        "x": 0.0,
        TARGET: [10.0] * 10 + [40.0] * 5 + [np.nan] * 5,
    })
    bundle = {"model": _ConstantModel(10.0), "feature_columns": ["x"], "trained_until": "2026-01-10"}

    perf = performance_drift(bundle, features, reference_mae=5.0, min_rows=5)

    assert perf["n_rows"] == 5  # 11-15 Jan: setelah training dan sudah berlabel
    assert perf["mae"] == 30.0
    assert perf["mae_ratio"] == 6.0
    assert performance_drift(bundle, features, reference_mae=5.0, min_rows=6)["mae_ratio"] is None


def test_decide(config):
    mon = config["monitoring"]
    drift = {"zscore": 5.0}
    no_drift = {"zscore": 0.5}

    assert decide(no_drift, None, None, mon)["retrain"]  # model belum ada
    assert not decide(no_drift, None, 2.0, mon)["retrain"]
    assert decide(no_drift, None, mon["max_model_age_days"] + 1, mon)["retrain"]
    assert decide(drift, None, mon["drift_cooldown_days"] + 1, mon)["retrain"]

    in_cooldown = decide(drift, None, mon["drift_cooldown_days"] - 1, mon)
    assert not in_cooldown["retrain"]
    assert in_cooldown["suppressed"]
