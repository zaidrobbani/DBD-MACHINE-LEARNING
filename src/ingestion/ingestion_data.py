import argparse
import os
import sys
from datetime import date, timedelta, datetime

import pandas as pd
import requests
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_date_range(days_back, end_offset_days):
    end_date = date.today() - timedelta(days=end_offset_days)
    start_date = end_date - timedelta(days=days_back - 1)
    return start_date, end_date


def fetch_wikipedia_pageviews(cfg, start_date, end_date):
    wiki = cfg["wikipedia"]
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{wiki['project']}/{wiki['access']}/{wiki['agent']}/{wiki['article']}/"
        f"{wiki['granularity']}/{start_str}/{end_str}"
    )
    headers = {"User-Agent": wiki["user_agent"]}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    items = response.json().get("items", [])
    records = []
    for item in items:
        ts = item["timestamp"]
        record_date = datetime.strptime(ts[:8], "%Y%m%d").date()
        records.append({"date": record_date, "page_views": item["views"]})
    return pd.DataFrame(records)


def fetch_open_meteo(cfg, start_date, end_date):
    loc = cfg["location"]
    meteo = cfg["open_meteo"]
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ",".join(meteo["daily_variables"]),
        "timezone": cfg["project"]["timezone"],
    }
    response = requests.get(meteo["base_url"], params=params, timeout=30)
    response.raise_for_status()
    daily = response.json().get("daily", {})
    if not daily:
        return pd.DataFrame(columns=["date"] + meteo["daily_variables"])
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.drop(columns=["time"])
    rename_map = {
        "temperature_2m_mean": "temperature_mean",
        "precipitation_sum": "precipitation_sum",
        "relative_humidity_2m_mean": "relative_humidity_mean",
    }
    df = df.rename(columns=rename_map)
    return df


def merge_sources(wiki_df, meteo_df):
    merged = pd.merge(wiki_df, meteo_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    columns = ["date", "page_views", "precipitation_sum", "temperature_mean", "relative_humidity_mean", "fetched_at"]
    for col in columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged[columns]


def save_snapshot(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(output_dir, f"raw_{timestamp}.csv")
    df.to_csv(snapshot_path, index=False)
    latest_path = os.path.join(output_dir, "latest.csv")
    df.to_csv(latest_path, index=False)
    return snapshot_path, latest_path


def main():
    parser = argparse.ArgumentParser(description="Ingest data dinamis DBD dari Wikipedia Pageviews dan Open-Meteo")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--days-back", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    days_back = args.days_back or cfg["ingestion"]["days_back"]
    end_offset_days = cfg["ingestion"]["end_date_offset_days"]
    output_dir = cfg["ingestion"]["output_dir"]

    start_date, end_date = build_date_range(days_back, end_offset_days)
    print(f"Mengambil data untuk rentang {start_date} sampai {end_date}")

    try:
        wiki_df = fetch_wikipedia_pageviews(cfg, start_date, end_date)
        print(f"Wikipedia Pageviews API: {len(wiki_df)} baris berhasil diambil")
    except requests.RequestException as e:
        print(f"Gagal mengambil data Wikipedia Pageviews API: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        meteo_df = fetch_open_meteo(cfg, start_date, end_date)
        print(f"Open-Meteo API: {len(meteo_df)} baris berhasil diambil")
    except requests.RequestException as e:
        print(f"Gagal mengambil data Open-Meteo API: {e}", file=sys.stderr)
        sys.exit(1)

    merged = merge_sources(wiki_df, meteo_df)
    snapshot_path, latest_path = save_snapshot(merged, output_dir)

    print(f"Snapshot baru disimpan di {snapshot_path}")
    print(f"Pointer terbaru diperbarui di {latest_path}")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()