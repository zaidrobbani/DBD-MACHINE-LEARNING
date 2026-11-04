import argparse
import subprocess
from pathlib import Path

import mlflow
import yaml
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.tracking import require_remote_tracking

MODEL_NAME = "dbd-forecaster"
EXPERIMENT = "dbd-training"
LINEAGE_PATH = Path("models/registry.yaml")
DATA_DVC = Path("data/raw/dataset.csv.dvc")


def best_run(client: MlflowClient, experiment: str) -> Run:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"[registry] Experiment '{experiment}' tidak ditemukan.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="metrics.skill_vs_baseline_mae > 0",
        order_by=["metrics.cv_model_mae ASC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"[registry] Tidak ada run yang memenuhi kriteria pada experiment '{experiment}'.")
    return runs[0]


def run_from_ci(client: MlflowClient, experiment: str, ci_run_id: str) -> Run:
    """Run MLflow yang dibuat oleh satu run GitHub Actions (tag ci_run_id)."""
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"[registry] Experiment '{experiment}' tidak ditemukan.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.ci_run_id = '{ci_run_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"[registry] Tidak ada run dengan ci_run_id {ci_run_id}.")
    return runs[0]


def logged_model_id(run: Run, artifact_name: str = "model") -> str:
    """ID logged model (m-...) milik sebuah run.

    `runs:/<run_id>/model` tidak bisa dipakai di DagsHub: MLflow 3 mencarinya
    lewat `run.outputs`, field yang tidak dikirim server DagsHub. Jadi cari
    langsung di daftar logged model experiment, lalu filter di sisi klien.
    """
    models = mlflow.search_logged_models(experiment_ids=[run.info.experiment_id], output_format="list")
    matches = [m for m in models if m.source_run_id == run.info.run_id and m.name == artifact_name]
    if not matches:
        raise SystemExit(f"[registry] Run {run.info.run_id[:8]} tidak punya logged model '{artifact_name}'.")
    return matches[0].model_id


def register(client: MlflowClient, run: Run, name: str = MODEL_NAME):
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.run_id == run.info.run_id:
            print(f"[registry] Run {run.info.run_id[:8]} sudah terdaftar sebagai Version {mv.version}")
            return mv
    mv = mlflow.register_model(f"models:/{logged_model_id(run)}", name)
    print(f"[registry] Run {run.info.run_id[:8]} didaftarkan sebagai {name} Version {mv.version}")
    return mv


def to_staging(client: MlflowClient, name: str, version: str) -> None:
    # Hanya versi Staging lama yang diarsipkan. Versi Production tidak tersentuh.
    client.transition_model_version_stage(name, version, "Staging", archive_existing_versions=True)
    print(f"[registry] {name} Version {version}: -> Staging")
    print_versions(client, name)


def print_versions(client: MlflowClient, name: str) -> None:
    print(f"[registry] Status semua versi {name}:")
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
        print(f"  Version {mv.version}: {mv.current_stage:<10} run {mv.run_id[:8]}")


def promote(client: MlflowClient, name: str, version: str) -> None:
    for stage in ("Staging", "Production"):
        client.transition_model_version_stage(
            name, version, stage, archive_existing_versions=(stage == "Production")
        )
        print(f"[registry] {name} Version {version}: -> {stage}")
    print_versions(client, name)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def data_version(run: Run) -> tuple[str, str]:
    """(commit git, md5 dataset) yang dipakai run ini.

    Run dari CI punya tag git_commit, jadi hash data dibaca dari commit itu,
    bukan dari working tree laptop yang bisa saja berisi data lain. Run dari
    laptop tidak punya tag, jadi pakai HEAD (commit dulu sebelum promote).
    """
    commit = run.data.tags.get("git_commit") or git("rev-parse", "HEAD")
    try:
        dvc_file = git("show", f"{commit}:{DATA_DVC.as_posix()}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"[registry] Commit {commit[:7]} tidak ada di repo lokal. Jalankan `git fetch` dulu.")
    return commit, yaml.safe_load(dvc_file)["outs"][0]["md5"]


def write_lineage(run: Run, mv) -> None:
    metrics = run.data.metrics
    commit, md5 = data_version(run)
    record = {
        "model_name": mv.name,
        "version": int(mv.version),
        "stage": "Production",
        "model_uri": f"models:/{mv.name}/Production",
        "mlflow_run_id": run.info.run_id,
        "trained_until": run.data.params.get("trained_until"),
        "data": {
            "path": "data/raw/dataset.csv",
            "md5": md5,
        },
        "git_commit": commit,
        "metrics": {k: round(metrics[k], 3) for k in
                    ("cv_model_mae", "cv_baseline_mae", "test_mae", "skill_vs_baseline_mae") if k in metrics},
    }
    LINEAGE_PATH.write_text(yaml.safe_dump(record, sort_keys=False))
    print(f"[registry] Lineage ditulis ke {LINEAGE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model registry dbd-forecaster")
    parser.add_argument("step", choices=["register", "staging", "promote"])
    parser.add_argument("--run-id", help="Pakai run ini, bukan run dengan CV MAE terkecil")
    parser.add_argument("--ci-run-id", help="Pakai run MLflow yang dibuat oleh run GitHub Actions ini")
    args = parser.parse_args()

    require_remote_tracking()
    client = MlflowClient()
    if args.ci_run_id:
        run, label = run_from_ci(client, EXPERIMENT, args.ci_run_id), "Run dari CI"
    elif args.run_id:
        run, label = client.get_run(args.run_id), "Run dipilih"
    else:
        run, label = best_run(client, EXPERIMENT), "Run terbaik"
    print(f"[registry] {label}: {run.info.run_id} (cv_model_mae={run.data.metrics['cv_model_mae']:.3f})")
    mv = register(client, run)
    if args.step == "staging":
        to_staging(client, mv.name, mv.version)
    if args.step == "promote":
        promote(client, mv.name, mv.version)
        write_lineage(run, mv)


if __name__ == "__main__":
    main()