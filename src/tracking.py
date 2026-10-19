"""Logging eksperimen ke MLflow (server MLflow milik DagsHub).

Hanya aktif kalau env MLFLOW_TRACKING_URI diset; kredensial dibaca MLflow dari
MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD. Tanpa env itu (unit test,
run lokal biasa) semua pemanggilan logging tidak melakukan apa-apa.
"""

import os
from contextlib import contextmanager


class _NoOpTracker:
    def __getattr__(self, name):
        return self
    def __call__(self, *args, **kwargs):
        return None


def run_tags() -> dict:
    """Tag konteks run: commit git dan alasan retrain (diisi oleh workflow)."""
    tags = {"git_commit": os.environ.get("GITHUB_SHA"), "trigger": os.environ.get("RETRAIN_REASON")}
    return {k: v for k, v in tags.items() if v}


@contextmanager
def mlflow_run(experiment: str, run_name: str):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        yield _NoOpTracker()
        return

    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=run_tags()):
        yield mlflow
