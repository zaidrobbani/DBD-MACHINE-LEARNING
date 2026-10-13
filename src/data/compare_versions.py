"""Bandingkan isi dua versi dataset yang dilacak DVC.

`dvc diff` hanya menunjukkan hash file yang berubah; script ini membuka kedua
versi (dari cache DVC sesuai commit git) dan menunjukkan baris apa yang berubah.

Contoh:
    python -m src.data.compare_versions                      # HEAD vs workspace
    python -m src.data.compare_versions --old HEAD~1 --new HEAD
"""

import hashlib
import io
import json

import dvc.api
import pandas as pd

from src.data.versioning import diff_datasets
from src.utils import config_arg_parser, load_config

WORKSPACE = "workspace"


def read_version(path: str, rev: str) -> tuple[pd.DataFrame, str]:
    """Baca dataset pada revisi git tertentu (atau workspace) beserta md5-nya."""
    if rev == WORKSPACE:
        with open(path, "rb") as f:
            content = f.read()
    else:
        content = dvc.api.read(path, rev=rev, mode="rb")
    return pd.read_csv(io.BytesIO(content)), hashlib.md5(content).hexdigest()


def main():
    parser = config_arg_parser("Bandingkan dua versi dataset yang dilacak DVC.")
    parser.add_argument("--old", default="HEAD", help="Revisi git versi lama (default: HEAD)")
    parser.add_argument("--new", default=WORKSPACE, help="Revisi git versi baru (default: workspace)")
    args = parser.parse_args()

    path = load_config(args.config)["paths"]["raw_dataset"]
    old_df, old_md5 = read_version(path, args.old)
    new_df, new_md5 = read_version(path, args.new)

    diff = diff_datasets(old_df, new_df)
    added_dates = diff.pop("added_dates")

    print(f"Dataset: {path}")
    print(f"  {args.old:>10}: md5={old_md5}  {diff['rows_old']} baris  {' s/d '.join(diff['date_range_old'])}")
    print(f"  {args.new:>10}: md5={new_md5}  {diff['rows_new']} baris  {' s/d '.join(diff['date_range_new'])}")
    print(json.dumps({k: v for k, v in diff.items() if not k.startswith(("rows_old", "rows_new", "date_range"))},
                     indent=2))
    if added_dates:
        print(f"Tanggal baru: {added_dates[0]} s/d {added_dates[-1]}")


if __name__ == "__main__":
    main()
