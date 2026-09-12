"""Mengubah data/interim/ menjadi fitur siap-latih di data/processed/.

Jalankan dari root proyek:
    python -m src.features.build_features
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan/transformasikan fitur di sini (encoding, scaling, dsb)."""
    return df


def main(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)

    interim_path = Path(config["data"]["interim_path"])
    processed_path = Path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(interim_path)
    df = build_features(df)
    df.to_csv(processed_path, index=False)

    print(f"[build_features] {len(df)} baris disimpan ke {processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangun fitur dari data interim.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
