"""Putuskan apakah model perlu dilatih ulang di luar jadwal mingguan.

Tiga sinyal (ambang di `monitoring` pada config):
1. Data drift: rata-rata page_views beberapa hari terakhir jauh dari rata-rata
   mingguan pada periode referensi sebelumnya (z-score).
2. Performance drift: error model pada data yang baru punya label jauh lebih
   besar dari MAE test saat training.
3. Umur model: pengaman kalau retrain terjadwal gagal.

Sinyal drift diabaikan selama cooldown setelah training, supaya lonjakan yang
bertahan berhari-hari tidak memicu training setiap hari.

Di GitHub Actions, keputusan ditulis ke $GITHUB_OUTPUT (`retrain=true|false`)
dan alasannya ke $GITHUB_STEP_SUMMARY.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.schema import DATE, PAGE_VIEWS, TARGET
from src.tracking import mlflow_run
from src.utils import config_arg_parser, load_config, reindex_daily


def data_drift(page_views: pd.Series, window: int, reference_days: int) -> dict:
    """Bandingkan rata-rata `window` hari terakhir dengan sebaran rata-rata
    `window` harian selama `reference_days` hari sebelumnya."""
    page_views = page_views.dropna()
    recent = page_views.iloc[-window:]
    reference = page_views.iloc[-(window + reference_days):-window]
    ref_means = reference.rolling(window, min_periods=window).mean().dropna()
    if len(recent) < window or len(ref_means) < 2:
        return {"recent_mean": None, "reference_mean": None, "reference_std": None, "zscore": None}

    recent_mean, ref_mean, ref_std = recent.mean(), ref_means.mean(), ref_means.std()
    if ref_std > 0:
        zscore = (recent_mean - ref_mean) / ref_std
    else:
        zscore = 0.0 if recent_mean == ref_mean else np.inf
    return {
        "recent_mean": float(recent_mean),
        "reference_mean": float(ref_mean),
        "reference_std": float(ref_std),
        "zscore": float(zscore),
    }


def performance_drift(bundle: dict, features: pd.DataFrame, reference_mae: float, min_rows: int) -> dict:
    """MAE model pada baris sesudah `trained_until` yang target-nya sudah diketahui.

    Target baris awal masih sedikit tumpang-tindih dengan target terakhir saat
    training, jadi MAE ini cenderung sedikit optimis (lebih jarang alarm palsu).
    """
    trained_until = pd.Timestamp(bundle["trained_until"])
    unseen = features[(features[DATE] > trained_until) & features[TARGET].notna()]
    if len(unseen) < min_rows:
        return {"n_rows": len(unseen), "mae": None, "reference_mae": reference_mae, "mae_ratio": None}

    pred = bundle["model"].predict(unseen[bundle["feature_columns"]])
    mae = float(np.mean(np.abs(unseen[TARGET].to_numpy() - pred)))
    return {"n_rows": len(unseen), "mae": mae, "reference_mae": reference_mae, "mae_ratio": mae / reference_mae}


def decide(data: dict, perf: dict | None, model_age_days: float | None, mon_cfg: dict) -> dict:
    if model_age_days is None:
        return {"retrain": True, "reasons": ["model belum ada"], "suppressed": []}

    reasons = []
    if model_age_days > mon_cfg["max_model_age_days"]:
        reasons.append(f"umur model {model_age_days:.1f} hari > {mon_cfg['max_model_age_days']}")

    drift = []
    if data["zscore"] is not None and abs(data["zscore"]) > mon_cfg["zscore_threshold"]:
        drift.append(f"data drift: z-score page_views {data['zscore']:+.2f} (ambang ±{mon_cfg['zscore_threshold']})")
    if perf and perf["mae_ratio"] is not None and perf["mae_ratio"] > mon_cfg["mae_ratio_threshold"]:
        drift.append(f"performance drift: MAE {perf['mae']:.2f} = {perf['mae_ratio']:.2f}x MAE test "
                     f"(ambang {mon_cfg['mae_ratio_threshold']}x)")

    in_cooldown = model_age_days < mon_cfg["drift_cooldown_days"]
    suppressed = drift if in_cooldown else []
    if not in_cooldown:
        reasons.extend(drift)
    return {"retrain": bool(reasons), "reasons": reasons, "suppressed": suppressed}


def model_age_days(metrics: dict, now: datetime) -> float:
    trained_at = datetime.fromisoformat(metrics["timestamp"])
    return (now - trained_at).total_seconds() / 86400


def summary_markdown(report: dict) -> str:
    decision = "**retrain**" if report["retrain"] else "tidak perlu retrain"
    lines = ["### Cek drift", f"Keputusan: {decision}", ""]
    lines += [f"- {r}" for r in report["reasons"]]
    lines += [f"- (diabaikan, masih cooldown) {r}" for r in report["suppressed"]]
    lines += ["", "```json", json.dumps(report["signals"], indent=2), "```"]
    return "\n".join(lines) + "\n"


def log_drift(experiment: str, report: dict, run_date: str) -> None:
    signals = report["signals"]
    data, perf = signals["data_drift"], signals["performance_drift"] or {}
    values = {
        "zscore": data["zscore"],
        "recent_mean": data["recent_mean"],
        "reference_mean": data["reference_mean"],
        "recent_mae": perf.get("mae"),
        "mae_ratio": perf.get("mae_ratio"),
        "model_age_days": signals["model_age_days"],
        "retrain": float(report["retrain"]),
    }
    with mlflow_run(experiment, f"cek drift {run_date}") as tracker:
        tracker.log_metrics({k: v for k, v in values.items() if v is not None and math.isfinite(v)})
        tracker.set_tags({
            "reasons": "; ".join(report["reasons"]) or "-",
            "suppressed": "; ".join(report["suppressed"]) or "-",
        })


def main(config_path: str) -> None:
    config = load_config(config_path)
    paths, mon_cfg = config["paths"], config["monitoring"]

    raw = reindex_daily(pd.read_csv(paths["raw_dataset"]))
    data = data_drift(raw[PAGE_VIEWS], mon_cfg["drift_window_days"], mon_cfg["reference_days"])

    perf, age = None, None
    model_path, metrics_path = Path(paths["model_path"]), Path(paths["metrics_path"])
    if model_path.exists() and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        age = model_age_days(metrics, datetime.now(timezone.utc))
        features = pd.read_csv(paths["processed_path"], parse_dates=[DATE])
        perf = performance_drift(joblib.load(model_path), features, metrics["test"]["mae"],
                                 mon_cfg["min_eval_rows"])

    report = decide(data, perf, age, mon_cfg)
    report["signals"] = {"data_drift": data, "performance_drift": perf, "model_age_days": age}
    log_drift(config["tracking"]["drift_experiment"], report, str(raw.index.max().date()))

    print(f"[check_drift] retrain={str(report['retrain']).lower()}")
    for reason in report["reasons"]:
        print(f"  - {reason}")
    for reason in report["suppressed"]:
        print(f"  - (diabaikan, masih cooldown) {reason}")

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"retrain={str(report['retrain']).lower()}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary_markdown(report))


if __name__ == "__main__":
    args = config_arg_parser("Cek drift data/performa untuk memutuskan retrain.").parse_args()
    main(args.config)
