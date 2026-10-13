"""Cek cepat apakah kedua API bisa diakses, tanpa menyimpan file apa pun."""

from src.ingestion.ingestion_data import build_date_range, fetch_all
from src.utils import config_arg_parser, load_config


def main():
    args = config_arg_parser("Verifikasi akses Wikipedia Pageviews API dan Open-Meteo API").parse_args()
    cfg = load_config(args.config)

    start_date, end_date = build_date_range(cfg["ingestion"]["days_back"], cfg["ingestion"]["end_date_offset_days"])
    print(f"Mengambil data dari {start_date} sampai {end_date}")

    merged = fetch_all(cfg, start_date, end_date)
    print(f"Hasil gabungan kedua sumber data: {len(merged)} baris (tidak disimpan)")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
