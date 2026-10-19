import mlflow
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.train_model import log_training
from src.monitoring.check_drift import log_drift
from src.tracking import mlflow_run

METRICS = {
    "model_type": "random_forest",
    "residual_on": "pv_mean_7",
    "params": {"n_estimators": 10, "max_depth": 3},
    "horizon_days": 7,
    "n_train": 100,
    "n_test": 30,
    "test": {"mae": 3.0, "rmse": 4.0, "r2": 0.1},
    "baseline_test": {"mae": 3.5, "rmse": 4.5, "r2": -0.2},
    "skill_vs_baseline_mae": 0.14,
    "cv": {"model_mae_mean": 3.1, "baseline_mae_mean": 3.4},
}
DRIFT_REPORT = {
    "retrain": True,
    "reasons": ["data drift: z-score page_views +4.00 (ambang ±3.0)"],
    "suppressed": [],
    "signals": {
        "data_drift": {"recent_mean": 30.0, "reference_mean": 10.0, "reference_std": 5.0, "zscore": 4.0},
        "performance_drift": {"n_rows": 0, "mae": None, "reference_mae": 3.0, "mae_ratio": None},
        "model_age_days": 5.0,
    },
}


def test_tracking_is_noop_without_tracking_uri(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)

    with mlflow_run("exp", "run") as tracker:
        tracker.log_metrics({"mae": 1.0})

    assert not any(tmp_path.iterdir())  # tidak ada folder mlruns/ yang dibuat


@pytest.fixture
def local_mlflow(monkeypatch, tmp_path):
    # Store folder lokal cukup untuk test (sqlite butuh sqlalchemy, tidak ada di mlflow-skinny).
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("RETRAIN_REASON", "jadwal mingguan")
    mlflow.set_tracking_uri(uri)
    yield mlflow.MlflowClient(uri)
    mlflow.set_tracking_uri(None)


def _only_run(client, experiment):
    runs = client.search_runs([client.get_experiment_by_name(experiment).experiment_id])
    assert len(runs) == 1
    return runs[0]


def test_log_training(local_mlflow, tmp_path):
    plot = tmp_path / "forecast_test.png"
    plot.write_bytes(b"png")

    X = pd.DataFrame({"pv_mean_7": [1.0, 2.0, 3.0]})
    model = Ridge().fit(X, [1.0, 2.0, 3.0])

    log_training("training", METRICS, "random_forest max_depth=3 s/d 2026-10-04",
                 {"test_days": 30, "cv_folds": 3}, "2026-10-04", [str(plot)], model, X)

    run = _only_run(local_mlflow, "training")
    assert run.info.run_name == "random_forest max_depth=3 s/d 2026-10-04"
    assert run.data.metrics["test_mae"] == 3.0
    assert run.data.metrics["baseline_mae"] == 3.5
    assert run.data.params["model_max_depth"] == "3"
    assert run.data.tags["trigger"] == "jadwal mingguan"
    assert [a.path for a in local_mlflow.list_artifacts(run.info.run_id)] == ["forecast_test.png"]


def test_log_drift_skips_missing_signals(local_mlflow):
    log_drift("drift", DRIFT_REPORT, "2026-10-11")

    run = _only_run(local_mlflow, "drift")
    assert run.data.metrics["zscore"] == 4.0
    assert run.data.metrics["retrain"] == 1.0
    assert "mae_ratio" not in run.data.metrics  # belum ada data berlabel baru
    assert run.data.tags["reasons"].startswith("data drift")
