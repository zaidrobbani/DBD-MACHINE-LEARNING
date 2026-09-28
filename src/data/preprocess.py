import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_snapshots(input_dir):
    pattern = os.path.join(input_dir, "raw_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        latest = os.path.join(input_dir, "latest.csv")
        if os.path.exists(latest):
            files = [latest]
    if not files:
        return pd.DataFrame(columns=["date", "page_views", "precipitation_sum", "temperature_mean", "relative_humidity_mean", "fetched_at"])
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def standardize_types(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], format="ISO8601")
    for col in ["page_views", "precipitation_sum", "temperature_mean", "relative_humidity_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def deduplicate_by_date(df):
    df = df.sort_values("fetched_at")
    return df.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)


def interpolate_climate_columns(df):
    df = df.copy()
    climate_cols = ["precipitation_sum", "temperature_mean", "relative_humidity_mean"]
    df[climate_cols] = df[climate_cols].interpolate(method="linear", limit_direction="both")
    return df


def drop_missing_page_views(df):
    before = len(df)
    df = df.dropna(subset=["page_views"]).reset_index(drop=True)
    dropped = before - len(df)
    return df, dropped


def validate(df, cfg):
    required_columns = ["date", "page_views", "precipitation_sum", "temperature_mean", "relative_humidity_mean"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        return False, f"Kolom hilang: {missing_columns}"

    min_rows = cfg["preprocessing"]["min_rows"]
    if len(df) < min_rows:
        return False, f"Jumlah baris ({len(df)}) di bawah minimum ({min_rows})"

    if df["date"].duplicated().any():
        return False, "Masih ada tanggal duplikat setelah deduplikasi"

    missing_ratio = df[required_columns].isna().mean().max()
    threshold = cfg["preprocessing"]["missing_value_threshold"]
    if missing_ratio > threshold:
        return False, f"Rasio missing value ({missing_ratio:.2%}) melebihi ambang batas ({threshold:.2%})"

    return True, "Validasi lolos"


def save_clean_snapshot(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(output_dir, f"clean_{timestamp}.csv")
    df.to_csv(snapshot_path, index=False)
    latest_path = os.path.join(output_dir, "latest.csv")
    df.to_csv(latest_path, index=False)
    return snapshot_path, latest_path


def main():
    parser = argparse.ArgumentParser(description="Bersihkan data mentah DBD hasil ingest_data.py")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = cfg["preprocessing"]["input_dir"]
    output_dir = cfg["preprocessing"]["output_dir"]

    raw_df = load_raw_snapshots(input_dir)
    if raw_df.empty:
        print(f"Tidak ada data mentah ditemukan di {input_dir}. Jalankan ingest_data.py dahulu.", file=sys.stderr)
        sys.exit(1)

    print(f"Total baris mentah dari seluruh snapshot: {len(raw_df)}")

    df = standardize_types(raw_df)
    df = deduplicate_by_date(df)
    print(f"Setelah deduplikasi berdasarkan tanggal: {len(df)} baris")

    df = interpolate_climate_columns(df)
    df, dropped = drop_missing_page_views(df)
    print(f"Baris dibuang karena page_views kosong: {dropped}")
    print(f"Baris tersisa setelah pembersihan: {len(df)}")

    is_valid, message = validate(df, cfg)
    print(f"Validasi: {message}")
    if not is_valid:
        sys.exit(1)

    snapshot_path, latest_path = save_clean_snapshot(df, output_dir)
    print(f"Data bersih disimpan di {snapshot_path}")
    print(f"Pointer terbaru diperbarui di {latest_path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()