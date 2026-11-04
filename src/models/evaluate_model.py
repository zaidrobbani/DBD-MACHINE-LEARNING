"""Gate evaluasi: model baru hanya lolos kalau semua metrik memenuhi ambang di config.

Keluar dengan kode 1 kalau ada metrik yang gagal, sehingga job berikutnya
(pendaftaran ke Model Registry) otomatis tidak dijalankan.
"""

import json
import os
from pathlib import Path

from src.utils import config_arg_parser, load_config


def evaluate(metrics: dict, thresholds: dict) -> list[dict]:
    checks = [
        ("skill_vs_baseline_mae", metrics["skill_vs_baseline_mae"], ">", thresholds["min_skill_vs_baseline"]),
        ("test_mae", metrics["test"]["mae"], "<=", thresholds["max_test_mae"]),
        ("test_rmse", metrics["test"]["rmse"], "<=", thresholds["max_test_rmse"]),
    ]
    return [
        {"metric": name, "value": value, "rule": f"{op} {limit}",
         "passed": value > limit if op == ">" else value <= limit}
        for name, value, op, limit in checks
    ]


def report_lines(results: list[dict]) -> list[str]:
    lines = ["| Metrik | Nilai | Syarat | Hasil |", "| --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['metric']} | {r['value']:.3f} | {r['rule']} | {'LOLOS' if r['passed'] else 'GAGAL'} |")
    return lines


def main(config_path: str) -> None:
    config = load_config(config_path)
    metrics = json.loads(Path(config["paths"]["metrics_path"]).read_text())
    results = evaluate(metrics, config["evaluation"])
    passed = all(r["passed"] for r in results)

    lines = report_lines(results)
    verdict = "LOLOS: model boleh didaftarkan ke Staging." if passed else "GAGAL: model tidak didaftarkan."
    print("\n".join(f"[evaluate] {line}" for line in lines))
    print(f"[evaluate] {verdict}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("### Evaluasi model\n\n" + "\n".join(lines) + f"\n\n**{verdict}**\n")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(config_arg_parser("Bandingkan metrik model dengan ambang batas.").parse_args().config)
