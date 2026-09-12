"""Melatih model dari data/processed/ dan menyimpan artefaknya ke models/.

Jalankan dari root proyek:
    python -m src.models.train_model
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]

    df = pd.read_csv(data_cfg["processed_path"])
    target = data_cfg["target_column"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=config["project"]["seed"],
    )

    model = RandomForestClassifier(**model_cfg["params"])
    model.fit(X_train, y_train)

    output_path = Path(model_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print(f"[train_model] Model disimpan ke {output_path}")
    print(f"[train_model] Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"[train_model] Test accuracy:  {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih model klasifikasi DBD.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
