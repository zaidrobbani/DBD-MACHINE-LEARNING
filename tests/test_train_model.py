import pandas as pd

from src.features.build_features import build_features, feature_columns
from src.models.train_model import build_model, temporal_train_test_split
from src.schema import DATE, TARGET


def test_temporal_split_is_chronological_with_purge_gap(clean_df, config):
    feats = build_features(clean_df, config["features"]).dropna(subset=[TARGET])
    horizon = config["features"]["horizon_days"]
    test_days = 60

    train, test = temporal_train_test_split(feats, test_days, horizon)

    assert len(test) == test_days
    assert train[DATE].max() < test[DATE].min()
    # Target baris train terakhir (t+1..t+horizon) harus berakhir sebelum periode test.
    assert train[DATE].max() + pd.Timedelta(days=horizon) < test[DATE].min()


def test_residual_model_follows_level_shift(clean_df, config):
    # Model residual harus tetap mengikuti level baru walau tidak pernah melihatnya saat training.
    feats = build_features(clean_df, config["features"]).dropna(subset=[TARGET])
    features = feature_columns(feats)
    model_cfg = {**config["model"], "type": "random_forest", "residual": True}

    model = build_model(model_cfg, seed=0).fit(feats[features], feats[TARGET])

    shifted = feats[features].tail(20).copy()
    level_cols = [c for c in features if c.startswith(("pv_today", "pv_lag_", "pv_mean_"))]
    shifted[level_cols] = shifted[level_cols] + 100
    assert model.predict(shifted).mean() > 100
