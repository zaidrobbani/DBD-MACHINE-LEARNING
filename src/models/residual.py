from sklearn.base import BaseEstimator, RegressorMixin


class ResidualRegressor(BaseEstimator, RegressorMixin):
    """Latih model pada selisih target terhadap `base_column`.

    Level page_views bergeser antar tahun, dan model berbasis pohon tidak bisa
    memprediksi di luar rentang nilai yang pernah dilihat. Dengan memprediksi
    deviasi dari rata-rata terbaru, model tetap mengikuti level saat ini.
    """

    def __init__(self, estimator, base_column: str):
        self.estimator = estimator
        self.base_column = base_column

    def fit(self, X, y):
        self.estimator.fit(X, y - X[self.base_column])
        return self

    def predict(self, X):
        return self.estimator.predict(X) + X[self.base_column].to_numpy()
