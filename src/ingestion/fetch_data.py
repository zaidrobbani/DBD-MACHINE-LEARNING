"""Menarik data terbaru dari API eksternal dan menyimpannya ke data/raw/.

Setiap eksekusi membuat snapshot baru bertimestamp (untuk jejak audit / data
versioning sederhana) sekaligus memperbarui `data/raw/latest.csv` yang
menjadi acuan tahap berikutnya (validasi -> training).

Jalankan dari root proyek:
    python -m src.ingestion.fetch_data

Kredensial API TIDAK disimpan di config.yaml -- ambil dari environment
variable (diset sebagai GitHub Actions secret saat berjalan di CI).
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_from_api(api_url: str, params: dict, api_key: str | None) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(api_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def main(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    ing_cfg = config["ingestion"]

    api_key = os.environ.get(ing_cfg["api_key_env"], "")
    if not api_key:
        print(f"[fetch_data] Peringatan: env var {ing_cfg['api_key_env']} "
              f"kosong/tidak diset. Lanjut tanpa auth header.")

    df = fetch_from_api(ing_cfg["api_url"], ing_cfg.get("params", {}), api_key or None)

    raw_dir = Path(ing_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = raw_dir / f"{timestamp}_dataset.csv"
    latest_path = raw_dir / "latest.csv"

    df.to_csv(snapshot_path, index=False)
    df.to_csv(latest_path, index=False)

    print(f"[fetch_data] {len(df)} baris diambil dari API.")
    print(f"[fetch_data] Snapshot disimpan: {snapshot_path}")
    print(f"[fetch_data] Referensi terbaru diperbarui: {latest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ambil data terbaru dari API eksternal.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
