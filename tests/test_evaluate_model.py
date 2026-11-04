import json

import pytest
import yaml

from src.models.evaluate_model import evaluate, main

THRESHOLDS = {"min_skill_vs_baseline": 0.0, "max_test_mae": 5.99, "max_test_rmse": 7.25}


def metrics(mae=4.0, rmse=5.0, skill=0.08):
    return {"test": {"mae": mae, "rmse": rmse}, "skill_vs_baseline_mae": skill}


def write_inputs(tmp_path, values):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(values))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"metrics_path": str(metrics_path)}, "evaluation": THRESHOLDS}))
    return str(config_path)


def test_evaluate_passes_when_all_thresholds_met():
    assert all(r["passed"] for r in evaluate(metrics(), THRESHOLDS))


@pytest.mark.parametrize("values,failed", [
    (metrics(skill=0.0), "skill_vs_baseline_mae"),
    (metrics(skill=-0.1), "skill_vs_baseline_mae"),
    (metrics(mae=6.0), "test_mae"),
    (metrics(rmse=7.3), "test_rmse"),
])
def test_evaluate_flags_the_failing_metric(values, failed):
    results = evaluate(values, THRESHOLDS)
    assert [r["metric"] for r in results if not r["passed"]] == [failed]


def test_main_exits_nonzero_when_model_fails(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(write_inputs(tmp_path, metrics(skill=-0.1)))
    assert exc.value.code == 1
    assert "GAGAL" in capsys.readouterr().out


def test_main_passes_and_writes_step_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    main(write_inputs(tmp_path, metrics()))
    assert "LOLOS" in summary.read_text()
