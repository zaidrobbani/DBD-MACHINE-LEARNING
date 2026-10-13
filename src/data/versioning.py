"""Logika dataset kumulatif: menggabungkan data baru dan membandingkan dua versi."""

import pandas as pd

from src.schema import DATE, FETCHED_AT, RAW_COLUMNS, VALUE_COLUMNS


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE] = pd.to_datetime(df[DATE]).dt.date
    df[FETCHED_AT] = pd.to_datetime(df[FETCHED_AT], format="ISO8601")
    return df


def merge_into_dataset(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan `new` ke dataset kumulatif, satu baris per tanggal.

    Tanggal yang sudah ada diganti dengan hasil fetch terbaru, karena sumber
    bisa merevisi data (mis. cuaca dari Forecast API diganti data Archive).
    """
    frames = [_normalize(new)] if existing is None else [_normalize(existing), _normalize(new)]
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.sort_values(FETCHED_AT, kind="stable")
        .drop_duplicates(subset=DATE, keep="last")
        .sort_values(DATE)
        .reset_index(drop=True)[RAW_COLUMNS]
    )


def diff_datasets(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Ringkasan perubahan isi antara dua versi dataset (per tanggal)."""
    old_vals = _normalize(old).set_index(DATE)[VALUE_COLUMNS]
    new_vals = _normalize(new).set_index(DATE)[VALUE_COLUMNS]

    added = new_vals.index.difference(old_vals.index)
    removed = old_vals.index.difference(new_vals.index)
    common = old_vals.index.intersection(new_vals.index)

    a, b = old_vals.loc[common], new_vals.loc[common]
    changed_cells = ~((a == b) | (a.isna() & b.isna()))

    def date_range(vals):
        return [str(vals.index.min()), str(vals.index.max())] if len(vals) else [None, None]

    return {
        "rows_old": len(old_vals),
        "rows_new": len(new_vals),
        "date_range_old": date_range(old_vals),
        "date_range_new": date_range(new_vals),
        "rows_added": len(added),
        "rows_removed": len(removed),
        "rows_changed": int(changed_cells.any(axis=1).sum()),
        "changed_per_column": {c: int(n) for c, n in changed_cells.sum().items() if n},
        "added_dates": [str(d) for d in added],
    }
