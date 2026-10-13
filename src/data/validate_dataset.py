import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.schema import DATE, REQUIRED_COLUMNS
from src.utils import config_arg_parser, load_config


def run_checks(df: pd.DataFrame, rules: dict, today: date | None = None) -> list[str]:
    """Aturan validasi tunggal, dipakai untuk data mentah maupun data bersih."""
    errors: list[str] = []
    today = today or date.today()

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        # Pengecekan lain bergantung pada kolom wajib, jadi berhenti di sini.
        return [f"Kolom wajib hilang: {sorted(missing_cols)}"]

    if len(df) < rules["min_rows"]:
        errors.append(f"Jumlah baris ({len(df)}) di bawah minimum ({rules['min_rows']})")
    if len(df) == 0:
        return errors

    dates = pd.to_datetime(df[DATE], errors="coerce")
    if dates.isna().any():
        errors.append(f"Ada {int(dates.isna().sum())} nilai tanggal yang tidak valid")

    n_duplicate_dates = int(dates.duplicated().sum())
    if n_duplicate_dates > rules["max_duplicate_dates"]:
        errors.append(f"Ada {n_duplicate_dates} tanggal duplikat")

    missing_ratio = df[REQUIRED_COLUMNS].isna().mean()
    worst_col = missing_ratio.idxmax()
    if missing_ratio[worst_col] > rules["max_missing_ratio"]:
        errors.append(
            f"Rasio missing value kolom '{worst_col}' ({missing_ratio[worst_col]:.2%}) "
            f"melebihi batas ({rules['max_missing_ratio']:.2%})"
        )

    for col, (low, high) in rules["ranges"].items():
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        out_of_range = pd.Series(False, index=values.index)
        if low is not None:
            out_of_range |= values < low
        if high is not None:
            out_of_range |= values > high
        if out_of_range.any():
            errors.append(
                f"Kolom '{col}' punya {int(out_of_range.sum())} nilai di luar rentang [{low}, {high}]"
            )

    if dates.notna().any():
        staleness = (pd.Timestamp(today) - dates.max()).days
        if staleness > rules["max_staleness_days"]:
            errors.append(
                f"Data basi: tanggal terakhir {dates.max().date()} ({staleness} hari lalu), "
                f"batas {rules['max_staleness_days']} hari"
            )

    return errors


def main(config_path: str) -> int:
    config = load_config(config_path)
    paths = config["paths"]

    source_path = Path(paths["raw_dataset"])
    df = pd.read_csv(source_path)

    errors = run_checks(df, config["validation"])
    passed = len(errors) == 0

    has_dates = DATE in df and len(df) > 0
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path),
        "n_rows": len(df),
        "date_min": str(df[DATE].min()) if has_dates else None,
        "date_max": str(df[DATE].max()) if has_dates else None,
        "passed": passed,
        "errors": errors,
    }

    report_dir = Path(paths["validation_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    status = "LOLOS" if passed else "GAGAL"
    print(f"[validate_dataset] Validasi {status} ({len(df)} baris). Laporan: {report_path}")
    for err in errors:
        print(f"  - {err}")

    return 0 if passed else 1


if __name__ == "__main__":
    args = config_arg_parser("Validasi data hasil ingestion.").parse_args()
    raise SystemExit(main(args.config))
