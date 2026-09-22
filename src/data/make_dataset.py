import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df


def main(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)

    raw_path = Path(config["data"]["raw_path"])
    interim_path = Path(config["data"]["interim_path"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = clean_data(df)
    df.to_csv(interim_path, index=False)

    print(f"[make_dataset] {len(df)} baris disimpan ke {interim_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bersihkan raw dataset DBD.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
