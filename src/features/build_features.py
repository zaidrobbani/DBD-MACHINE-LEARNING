from pathlib import Path

import pandas as pd

from src.schema import DATE, HUMIDITY, PAGE_VIEWS, PRECIPITATION, TARGET, TEMPERATURE
from src.utils import config_arg_parser, load_config, reindex_daily


def build_features(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Bangun fitur time-series per hari t dan target t+1..t+horizon.

    Aturan anti-leakage: setiap fitur pada baris t hanya memakai data
    sampai hari t. Satu-satunya kolom yang melihat masa depan adalah target.
    """
    df = reindex_daily(df)
    pv = df[PAGE_VIEWS]
    out = pd.DataFrame(index=df.index)

    out["day_of_week"] = df.index.dayofweek
    out["month"] = df.index.month
    out["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    out["pv_today"] = pv
    for lag in feat_cfg["page_view_lags"]:
        out[f"pv_lag_{lag}"] = pv.shift(lag)
    windows = feat_cfg["page_view_windows"]
    for w in windows:
        out[f"pv_mean_{w}"] = pv.rolling(w, min_periods=w).mean()
    out[f"pv_std_{min(windows)}"] = pv.rolling(min(windows), min_periods=min(windows)).std()
    out["pv_trend"] = out[f"pv_mean_{min(windows)}"] - out[f"pv_mean_{max(windows)}"]

    # Efek cuaca ke populasi nyamuk/kasus DBD muncul dengan jeda 1-3 minggu,
    # jadi dipakai agregat rolling dan versi yang digeser ke belakang.
    weather_aggs = {"precip_sum": df[PRECIPITATION], "temp_mean": df[TEMPERATURE], "humidity_mean": df[HUMIDITY]}
    for w in feat_cfg["weather_windows"]:
        for name, series in weather_aggs.items():
            rolling = series.rolling(w, min_periods=w)
            out[f"{name}_{w}"] = rolling.sum() if name == "precip_sum" else rolling.mean()
    lag_window = feat_cfg["weather_lag_window"]
    for lag in feat_cfg["weather_lags"]:
        for name in ("precip_sum", "humidity_mean"):
            out[f"{name}_{lag_window}_lag_{lag}"] = out[f"{name}_{lag_window}"].shift(lag)

    # Target: rata-rata page_views pada t+1 .. t+horizon.
    horizon = feat_cfg["horizon_days"]
    out[TARGET] = pv.shift(-horizon).rolling(horizon, min_periods=horizon).mean()

    # Buang baris yang fiturnya belum lengkap (awal series / hari tanpa data).
    # Baris akhir dengan target NaN tetap disimpan: itu baris untuk prediksi.
    out = out.dropna(subset=feature_columns(out))
    return out.reset_index()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (DATE, TARGET)]


def main(config_path: str) -> None:
    config = load_config(config_path)
    paths = config["paths"]

    df = pd.read_csv(paths["interim_dataset"])
    features = build_features(df, config["features"])

    processed_path = Path(paths["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_path, index=False)

    n_labeled = int(features[TARGET].notna().sum())
    print(f"[build_features] {len(features)} baris ({n_labeled} berlabel) disimpan ke {processed_path}")


if __name__ == "__main__":
    args = config_arg_parser("Bangun fitur dari data interim.").parse_args()
    main(args.config)
