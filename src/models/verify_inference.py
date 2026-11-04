import mlflow
import numpy as np
import pandas as pd

from src.schema import DATE
from src.tracking import require_remote_tracking

URI = "models:/dbd-forecaster/Production"

require_remote_tracking()
model = mlflow.pyfunc.load_model(URI)
print(f"[verify] Model dimuat dari {URI} (run {model.metadata.run_id})")

cols = model.metadata.get_input_schema().input_names()
df = pd.read_csv("data/processed/features.csv", parse_dates=[DATE]).sort_values(DATE)
pred = float(np.ravel(model.predict(df.iloc[[-1]][cols]))[0])
print(f"[verify] Data terakhir {df[DATE].iloc[-1].date()} -> "
      f"prediksi rata-rata page_views 7 hari ke depan: {pred:.1f} per hari")