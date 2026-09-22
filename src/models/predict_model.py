import argparse

import joblib
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(input_path: str, config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)

    model = joblib.load(config["model"]["output_path"])
    df = pd.read_csv(input_path)

    predictions = model.predict(df)
    df["prediction"] = predictions

    output_path = input_path.replace(".csv", "_predictions.csv")
    df.to_csv(output_path, index=False)

    print(f"[predict_model] Prediksi disimpan ke {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi dengan model DBD terlatih.")
    parser.add_argument("--input", required=True, help="Path ke CSV data baru.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.input, args.config)
