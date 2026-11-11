from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BaselineDistanceModel:
    """
    One-class baseline distance model.

    Input is RobustScaler output fitted on good-road data, so every feature is measured in IQR units
    from the good-road median. Only upward deviations (rougher than the baseline) count, and each
    feature weighs the same:

        d2 = sum over features of max(scaled_value, 0) ** 2

    A road that is rougher on any feature always scores worse. (The previous version used a
    Mahalanobis distance; siv_kiiht has zero spread among good roads, so its covariance
    regularization 1e-6 became a weight of 1,000,000 and lateral acceleration dominated the score.)
    """

    threshold_: float
    feature_names_: list[str]
    threshold_quantile_: float

    def _prepare(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.loc[:, self.feature_names_].to_numpy(dtype=float, copy=False)
        else:
            X_arr = np.asarray(X, dtype=float)
        # Only upward deviations from baseline center contribute to badness.
        return np.clip(X_arr, a_min=0.0, a_max=None)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Sum of squared upward deviations per row (larger = rougher road)
        return np.square(self._prepare(X)).sum(axis=1)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Larger = more normal, smaller = more anomalous (sklearn-like semantics)
        d2 = self.score_samples(X)
        return self.threshold_ - d2

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        d2 = self.score_samples(X)
        return np.where(d2 > self.threshold_, -1, 1).astype(int)
