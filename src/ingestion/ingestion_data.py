import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.data.versioning import diff_datasets, merge_into_dataset
from src.schema import DATE, FETCHED_AT, OPEN_METEO_VARIABLES, PAGE_VIEWS, RAW_COLUMNS, WEATHER_COLUMNS
from src.utils import config_arg_parser, load_config


def build_date_range(days_back, end_offset_days, end_date=None):
    end_date = end_date or date.today() - timedelta(days=end_offset_days)
    start_date = end_date - timedelta(days=days_back - 1)
    return start_date, end_date


def fetch_wikipedia_pageviews(cfg, start_date, end_date):
    wiki = cfg["wikipedia"]
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{wiki['project']}/{wiki['access']}/{wiki['agent']}/{wiki['article']}/"
        f"{wiki['granularity']}/{start_date:%Y%m%d}/{end_date:%Y%m%d}"
    )
    response = requests.get(url, headers={"User-Agent": wiki["user_agent"]}, timeout=30)
    response.raise_for_status()
    records = [
        {DATE: datetime.strptime(item["timestamp"][:8], "%Y%m%d").date(), PAGE_VIEWS: item["views"]}
        for item in response.json().get("items", [])
    ]
    return pd.DataFrame(records, columns=[DATE, PAGE_VIEWS])


def request_open_meteo(url, cfg, start_date, end_date):
    params = {
        "latitude": cfg["location"]["latitude"],
        "longitude": cfg["location"]["longitude"],
        "start_date": f"{start_date:%Y-%m-%d}",
        "end_date": f"{end_date:%Y-%m-%d}",
        "daily": ",".join(OPEN_METEO_VARIABLES),
        "timezone": cfg["project"]["timezone"],
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    daily = response.json().get("daily", {})
    if not daily:
        return pd.DataFrame(columns=[DATE, *WEATHER_COLUMNS]).set_index(DATE)
    df = pd.DataFrame(daily).rename(columns=OPEN_METEO_VARIABLES)
    df[DATE] = pd.to_datetime(df.pop("time")).dt.date
    return df.set_index(DATE)


def fetch_open_meteo(cfg, start_date, end_date):
    # Archive API (ERA5) untuk histori panjang; beberapa hari terakhir biasanya
    # belum tersedia, jadi diisi dari Forecast API yang hanya menyimpan ~3 bulan.
    meteo = cfg["open_meteo"]
    df = request_open_meteo(meteo["archive_url"], cfg, start_date, end_date)
    df = df.reindex(pd.date_range(start_date, end_date, freq="D").date)
    if df.isna().any().any():
        recent_start = max(start_date, date.today() - timedelta(days=meteo["forecast_max_past_days"]))
        if recent_start <= end_date:
            df = df.combine_first(request_open_meteo(meteo["base_url"], cfg, recent_start, end_date))
    df.index.name = DATE
    return df.reset_index()


def merge_sources(wiki_df, meteo_df):
    merged = pd.merge(wiki_df, meteo_df, on=DATE, how="outer").sort_values(DATE).reset_index(drop=True)
    merged[FETCHED_AT] = datetime.now().isoformat(timespec="seconds")
    return merged.reindex(columns=RAW_COLUMNS)


def fetch_all(cfg, start_date, end_date):
    """Ambil dan gabungkan kedua sumber. Keluar dengan error jika salah satu API gagal."""
    sources = [("Wikipedia Pageviews API", fetch_wikipedia_pageviews), ("Open-Meteo API", fetch_open_meteo)]
    frames = []
    for name, fetch in sources:
        try:
            df = fetch(cfg, start_date, end_date)
        except requests.RequestException as e:
            print(f"Gagal mengambil data {name}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"{name}: {len(df)} baris berhasil diambil")
        frames.append(df)
    return merge_sources(*frames)


def main():
    parser = config_arg_parser("Ingest data dinamis DBD dari Wikipedia Pageviews dan Open-Meteo")
    parser.add_argument("--days-back", type=int, default=None)
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Ambil histori panjang (ingestion.history_days) untuk kebutuhan training",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="Tanggal akhir (YYYY-MM-DD). Default: hari ini - ingestion.end_date_offset_days",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ing_cfg = cfg["ingestion"]
    days_back = args.days_back or (ing_cfg["history_days"] if args.backfill else ing_cfg["days_back"])

    start_date, end_date = build_date_range(days_back, ing_cfg["end_date_offset_days"], args.end_date)
    print(f"Mengambil data untuk rentang {start_date} sampai {end_date}")
    fetched = fetch_all(cfg, start_date, end_date)

    # Dataset mentah bersifat kumulatif: data baru ditambahkan ke data lama,
    # bukan menimpanya. Riwayat setiap versi disimpan oleh DVC.
    dataset_path = Path(cfg["paths"]["raw_dataset"])
    existing = pd.read_csv(dataset_path) if dataset_path.exists() else None
    dataset = merge_into_dataset(existing, fetched)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)

    if existing is None:
        print(f"Dataset baru dibuat di {dataset_path}: {len(dataset)} baris")
    else:
        diff = diff_datasets(existing, dataset)
        print(f"Dataset diperbarui di {dataset_path}: {diff['rows_old']} -> {diff['rows_new']} baris "
              f"(+{diff['rows_added']} baru, {diff['rows_changed']} direvisi)")
    print(f"Rentang data: {dataset[DATE].min()} s/d {dataset[DATE].max()}")


if __name__ == "__main__":
    main()
