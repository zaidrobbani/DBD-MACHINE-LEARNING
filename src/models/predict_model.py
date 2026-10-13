import joblib
import pandas as pd

from src.schema import DATE
from src.utils import config_arg_parser, load_config


def main(config_path: str, input_path: str | None = None) -> None:
    config = load_config(config_path)
    paths = config["paths"]

    bundle = joblib.load(paths["model_path"])
    df = pd.read_csv(input_path or paths["processed_path"], parse_dates=[DATE])

    # Prediksi dibuat dari baris paling akhir: fitur hari ini -> rata-rata
    # page_views untuk `horizon_days` hari ke depan.
    latest = df.sort_values(DATE).iloc[[-1]]
    prediction = float(bundle["model"].predict(latest[bundle["feature_columns"]])[0])

    horizon = bundle["horizon_days"]
    base_date = latest[DATE].iloc[0]
    start = (base_date + pd.Timedelta(days=1)).date()
    end = (base_date + pd.Timedelta(days=horizon)).date()
    print(f"[predict_model] Data terakhir: {base_date.date()} (model dilatih s/d {bundle['trained_until']})")
    print(f"[predict_model] Prediksi rata-rata page_views {start} s/d {end}: {prediction:.1f} per hari")


if __name__ == "__main__":
    parser = config_arg_parser("Prediksi minat DBD beberapa hari ke depan.")
    parser.add_argument("--input", default=None, help="CSV fitur (default: paths.processed_path)")
    args = parser.parse_args()
    main(args.config, args.input)
