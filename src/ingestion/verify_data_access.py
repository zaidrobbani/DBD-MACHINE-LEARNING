import requests
import pandas as pd
from datetime import date, timedelta
import os

WIKI_PROJECT = "id.wikipedia.org"
WIKI_ARTICLE = "Demam_berdarah_dengue"
WIKI_ACCESS = "all-access"
WIKI_AGENT = "all-agents"
WIKI_GRANULARITY = "daily"

OPEN_METEO_LATITUDE = -7.9666
OPEN_METEO_LONGITUDE = 112.6326
OPEN_METEO_TIMEZONE = "Asia/Jakarta"

DAYS_BACK = 14
END_DATE_OFFSET_DAYS = 2
OUTPUT_DIR = "data/raw"


def build_date_range(days_back, end_offset_days):
    end_date = date.today() - timedelta(days=end_offset_days)
    start_date = end_date - timedelta(days=days_back - 1)
    return start_date, end_date


def fetch_wikipedia_pageviews(start_date, end_date):
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{WIKI_PROJECT}/{WIKI_ACCESS}/{WIKI_AGENT}/{WIKI_ARTICLE}/"
        f"{WIKI_GRANULARITY}/{start_str}/{end_str}"
    )
    headers = {"User-Agent": "dbd-mlops-pipeline/1.0 (contact: 245150207111012@student.ub.ac.id)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    items = response.json().get("items", [])
    records = []
    for item in items:
        timestamp = item["timestamp"]
        record_date = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        records.append({"date": record_date, "page_views": item["views"]})
    return pd.DataFrame(records)


def fetch_open_meteo_climate(days_back, end_offset_days):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": OPEN_METEO_LATITUDE,
        "longitude": OPEN_METEO_LONGITUDE,
        "past_days": days_back + end_offset_days,
        "forecast_days": 1,
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
        "timezone": OPEN_METEO_TIMEZONE,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()["daily"]
    return pd.DataFrame({
        "date": payload["time"],
        "temperature_mean": payload["temperature_2m_mean"],
        "precipitation_sum": payload["precipitation_sum"],
        "relative_humidity_mean": payload["relative_humidity_2m_mean"],
    })


def merge_sources(wiki_df, climate_df):
    merged = pd.merge(wiki_df, climate_df, on="date", how="inner")
    merged["fetched_at"] = pd.Timestamp.now().isoformat()
    return merged


def save_snapshot(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp_label = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(output_dir, f"raw_{timestamp_label}.csv")
    latest_path = os.path.join(output_dir, "latest.csv")
    df.to_csv(snapshot_path, index=False)
    df.to_csv(latest_path, index=False)
    return snapshot_path, latest_path


def main():
    start_date, end_date = build_date_range(DAYS_BACK, END_DATE_OFFSET_DAYS)
    print(f"Mengambil data dari {start_date} sampai {end_date}")

    wiki_df = fetch_wikipedia_pageviews(start_date, end_date)
    print(f"Wikipedia Pageviews API berhasil diakses, {len(wiki_df)} baris diterima")

    climate_df = fetch_open_meteo_climate(DAYS_BACK, END_DATE_OFFSET_DAYS)
    climate_df = climate_df[climate_df["date"] <= end_date.isoformat()]
    print(f"Open-Meteo API berhasil diakses, {len(climate_df)} baris diterima")

    merged_df = merge_sources(wiki_df, climate_df)
    print(f"Hasil gabungan kedua sumber data: {len(merged_df)} baris")

    snapshot_path, latest_path = save_snapshot(merged_df, OUTPUT_DIR)
    print(f"Snapshot disimpan di {snapshot_path}")
    print(f"File latest diperbarui di {latest_path}")

    print(merged_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
