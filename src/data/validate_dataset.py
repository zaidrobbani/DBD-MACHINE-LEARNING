import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_checks(df: pd.DataFrame, rules: dict) -> list[str]:
    errors: list[str] = []

    missing_cols = set(rules["required_columns"]) - set(df.columns)
    if missing_cols:
        errors.append(f"Kolom wajib hilang: {sorted(missing_cols)}")

    if len(df) < rules["min_rows"]:
        errors.append(f"Jumlah baris ({len(df)}) di bawah minimum ({rules['min_rows']})")

    missing_ratio = df.isna().mean().max() if len(df) else 1.0
    if missing_ratio > rules["max_missing_ratio"]:
        errors.append(
            f"Rasio missing value tertinggi ({missing_ratio:.2%}) melebihi "
            f"batas ({rules['max_missing_ratio']:.2%})"
        )

    duplicate_ratio = df.duplicated().mean() if len(df) else 0.0
    if duplicate_ratio > rules.get("max_duplicate_ratio", 1.0):
        errors.append(f"Rasio baris duplikat ({duplicate_ratio:.2%}) melebihi batas")

    return errors


def main(config_path: str = "config/config.yaml") -> int:
    config = load_config(config_path)
    ing_cfg = config["ingestion"]
    val_cfg = config["validation"]

    latest_path = Path(ing_cfg["raw_dir"]) / "latest.csv"
    df = pd.read_csv(latest_path)

    errors = run_checks(df, val_cfg)
    passed = len(errors) == 0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": str(latest_path),
        "n_rows": len(df),
        "passed": passed,
        "errors": errors,
    }

    report_dir = Path("data/validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"{ts}_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    status = "LOLOS" if passed else "GAGAL"
    print(f"[validate_dataset] Validasi {status}. Laporan: {report_path}")
    if not passed:
        for err in errors:
            print(f"  - {err}")

    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validasi data hasil ingestion.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
