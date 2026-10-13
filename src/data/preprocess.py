import sys
from pathlib import Path

import pandas as pd

from src.data.validate_dataset import run_checks
from src.schema import DATE, PAGE_VIEWS, VALUE_COLUMNS, WEATHER_COLUMNS
from src.utils import config_arg_parser, load_config, reindex_daily


def interpolate_climate_columns(df, max_gap_days):
    # Hanya menambal celah pendek di tengah data (tanpa ekstrapolasi ke tepi);
    # celah panjang dibiarkan kosong agar ditolak validasi.
    df = df.copy()
    df[WEATHER_COLUMNS] = df[WEATHER_COLUMNS].interpolate(method="linear", limit=max_gap_days, limit_area="inside")
    return df


def main():
    args = config_arg_parser("Bersihkan dataset mentah DBD hasil ingestion_data.py").parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]

    raw_path = Path(paths["raw_dataset"])
    if not raw_path.exists():
        print(f"{raw_path} tidak ditemukan. Jalankan ingestion_data atau `dvc pull` dahulu.", file=sys.stderr)
        sys.exit(1)

    # Satu baris per hari kalender; page_views yang hilang tetap NaN (tidak
    # dibuang) supaya urutan hari tidak rusak.
    df = reindex_daily(pd.read_csv(raw_path))
    df[VALUE_COLUMNS] = df[VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = interpolate_climate_columns(df, cfg["preprocessing"]["max_interpolate_gap_days"]).reset_index()
    df[DATE] = df[DATE].dt.date
    print(f"Dataset mentah: {len(df)} hari, {int(df[PAGE_VIEWS].isna().sum())} hari tanpa page_views")

    errors = run_checks(df, cfg["validation"])
    if errors:
        print("Validasi data bersih GAGAL:", *errors, sep="\n  - ", file=sys.stderr)
        sys.exit(1)
    print("Validasi data bersih lolos")

    interim_path = Path(paths["interim_dataset"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim_path, index=False)
    print(f"Data bersih disimpan di {interim_path}")


if __name__ == "__main__":
    main()
