import argparse

import pandas as pd
import yaml

from src.schema import DATE

DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser


def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Index per hari kalender: tanggal yang hilang muncul sebagai baris NaN.

    Dengan begitu shift(n)/rolling(n)/interpolate(limit=n) berarti n hari, bukan n baris.
    """
    df = df.copy()
    df[DATE] = pd.to_datetime(df[DATE])
    df = df.set_index(DATE).sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df.index.name = DATE
    return df
