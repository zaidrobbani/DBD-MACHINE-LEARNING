import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import feature_columns
from src.models.residual import ResidualRegressor
from src.schema import DATE, TARGET
from src.tracking import mlflow_run
from src.utils import config_arg_parser, load_config
from src.visualization.visualize import plot_forecast

ESTIMATORS = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": HistGradientBoostingRegressor,
    "ridge": lambda **params: make_pipeline(StandardScaler(), Ridge(**params)),
}


def build_model(model_cfg: dict, seed: int):
    model_type = model_cfg["type"]
    if model_type not in ESTIMATORS:
        raise ValueError(f"model.type tidak dikenal: {model_type}")
    model = ESTIMATORS[model_type](**model_cfg["params"][model_type], random_state=seed)
    return ResidualRegressor(model, model_cfg["baseline_feature"]) if model_cfg["residual"] else model


def temporal_train_test_split(df: pd.DataFrame, test_days: int, horizon: int):
    """Split kronologis: `test_days` terakhir jadi test set.

    Baris train yang target-nya (t+1..t+horizon) masuk ke periode test dibuang
    (purge), supaya tidak ada informasi dari periode test yang bocor ke train.
    """
    df = df.sort_values(DATE).reset_index(drop=True)
    test_start = df[DATE].max() - pd.Timedelta(days=test_days - 1)
    train = df[df[DATE] < test_start - pd.Timedelta(days=horizon)]
    test = df[df[DATE] >= test_start]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_and_evaluate(model_cfg, seed, train, test, features) -> tuple[dict, dict, np.ndarray]:
    """Latih di `train`, evaluasi di `test`, dan bandingkan dengan baseline naif."""
    model = build_model(model_cfg, seed).fit(train[features], train[TARGET])
    pred = model.predict(test[features])
    return (
        regression_metrics(test[TARGET], pred),
        regression_metrics(test[TARGET], test[model_cfg["baseline_feature"]]),
        pred,
    )


def cross_validate_temporal(model_cfg, seed, train, features, n_splits, gap) -> dict:
    # TimeSeriesSplit: setiap fold dilatih di masa lalu dan diuji di periode
    # sesudahnya (expanding window), dengan jeda `gap` hari untuk purge target.
    folds = []
    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(train), start=1):
        model_m, baseline_m, _ = fit_and_evaluate(
            model_cfg, seed, train.iloc[train_idx], train.iloc[val_idx], features
        )
        folds.append({"fold": fold, "n_train": len(train_idx), "n_val": len(val_idx),
                      "model_mae": model_m["mae"], "baseline_mae": baseline_m["mae"]})
    return {
        "folds": folds,
        "model_mae_mean": float(np.mean([f["model_mae"] for f in folds])),
        "baseline_mae_mean": float(np.mean([f["baseline_mae"] for f in folds])),
    }


def log_training(experiment: str, metrics: dict, split_cfg: dict, trained_until: str, artifacts: list[str]) -> None:
    with mlflow_run(experiment, f"{metrics['model_type']} s/d {trained_until}") as tracker:
        tracker.log_params({
            "model_type": metrics["model_type"],
            "residual_on": metrics["residual_on"],
            "horizon_days": metrics["horizon_days"],
            "test_days": split_cfg["test_days"],
            "cv_folds": split_cfg["cv_folds"],
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
            "trained_until": trained_until,
            **{f"model_{k}": v for k, v in metrics["params"].items()},
        })
        tracker.log_metrics({
            **{f"test_{k}": v for k, v in metrics["test"].items()},
            **{f"baseline_{k}": v for k, v in metrics["baseline_test"].items()},
            "skill_vs_baseline_mae": metrics["skill_vs_baseline_mae"],
            "cv_model_mae": metrics["cv"]["model_mae_mean"],
            "cv_baseline_mae": metrics["cv"]["baseline_mae_mean"],
        })
        for path in artifacts:
            tracker.log_artifact(path)


def period(df: pd.DataFrame) -> list[str]:
    return [str(df[DATE].min().date()), str(df[DATE].max().date())]


def main(config_path: str) -> None:
    config = load_config(config_path)
    paths, split_cfg, model_cfg = config["paths"], config["split"], config["model"]
    horizon = config["features"]["horizon_days"]
    seed = config["project"]["seed"]

    df = pd.read_csv(paths["processed_path"], parse_dates=[DATE])
    labeled = df.dropna(subset=[TARGET]).sort_values(DATE).reset_index(drop=True)
    features = feature_columns(labeled)

    min_rows = split_cfg["min_train_rows"] + split_cfg["test_days"]
    if len(labeled) < min_rows:
        raise SystemExit(
            f"[train_model] Data berlabel hanya {len(labeled)} baris, butuh minimal {min_rows}. "
            f"Jalankan `python -m src.ingestion.ingestion_data --backfill` dulu."
        )

    train, test = temporal_train_test_split(labeled, split_cfg["test_days"], horizon)
    print(f"[train_model] Train: {' s/d '.join(period(train))} ({len(train)} baris)")
    print(f"[train_model] Test:  {' s/d '.join(period(test))} ({len(test)} baris)")

    cv = cross_validate_temporal(model_cfg, seed, train, features, split_cfg["cv_folds"], horizon)
    print(f"[train_model] CV temporal ({split_cfg['cv_folds']} fold) MAE model: {cv['model_mae_mean']:.3f} "
          f"| baseline: {cv['baseline_mae_mean']:.3f}")

    test_metrics, baseline_metrics, test_pred = fit_and_evaluate(model_cfg, seed, train, test, features)
    skill = 1 - test_metrics["mae"] / baseline_metrics["mae"]
    print(f"[train_model] Holdout MAE model: {test_metrics['mae']:.3f} | baseline: {baseline_metrics['mae']:.3f} "
          f"| skill vs baseline: {skill:+.1%}")

    # Evaluasi di atas sudah final; model yang di-deploy dilatih ulang
    # dengan seluruh data berlabel supaya memakai informasi terbaru.
    final_data = labeled if model_cfg["refit_on_full_data"] else train
    model = build_model(model_cfg, seed).fit(final_data[features], final_data[TARGET])

    model_path = Path(paths["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": features,
        "horizon_days": horizon,
        "trained_until": period(final_data)[1],
    }, model_path)

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_cfg["type"],
        "residual_on": model_cfg["baseline_feature"] if model_cfg["residual"] else None,
        "params": model_cfg["params"][model_cfg["type"]],
        "horizon_days": horizon,
        "target": f"rata-rata page_views t+1..t+{horizon}",
        "n_train": len(train),
        "n_test": len(test),
        "train_period": period(train),
        "test_period": period(test),
        "cv": cv,
        "test": test_metrics,
        "baseline_test": baseline_metrics,
        "skill_vs_baseline_mae": float(skill),
        "beats_baseline": bool(skill > 0),
    }
    Path(paths["metrics_path"]).write_text(json.dumps(metrics, indent=2))
    plot_forecast(test[DATE], test[TARGET], test_pred, test[model_cfg["baseline_feature"]],
                  paths["forecast_plot_path"], horizon)

    log_training(config["tracking"]["training_experiment"], metrics, split_cfg, period(final_data)[1],
                 [paths["metrics_path"], paths["forecast_plot_path"]])

    print(f"[train_model] Model: {model_path} | metrik: {paths['metrics_path']} | plot: {paths['forecast_plot_path']}")
    if not metrics["beats_baseline"]:
        print("[train_model] PERINGATAN: model belum mengalahkan baseline naif.")


if __name__ == "__main__":
    args = config_arg_parser("Latih model forecasting minat DBD.").parse_args()
    main(args.config)
