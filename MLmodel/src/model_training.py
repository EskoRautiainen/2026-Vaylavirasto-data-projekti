from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from shared.baseline_model import BaselineDistanceModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "MLmodel" / "MLfiles"

# -------------------------
# MODEL TRAINING
# -------------------------
def step_06_model_training(
    good_road_scaled: pd.DataFrame,
) -> BaselineDistanceModel:
    """
    Trains one-class baseline distance model on good-road baseline data.

    Args:
        good_road_scaled: Scaled good road baseline data

    Returns:
        trained_model: Fitted BaselineDistanceModel

    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If input dataframe is empty
    """
    if not isinstance(good_road_scaled, pd.DataFrame):
        raise TypeError("good_road_scaled must be a pandas DataFrame")
    if good_road_scaled.empty:
        raise ValueError("good_road_scaled cannot be empty")

    threshold_quantile = 0.99

    features = good_road_scaled.columns.tolist()

    print("------------------------------------------------------------")
    print("Model training started. This may take a while, please wait...")
    print()

    # Anomaly threshold: 99th percentile of the good-road distances (sum of squared upward deviations).
    model = BaselineDistanceModel(
        threshold_=0.0,
        feature_names_=features,
        threshold_quantile_=threshold_quantile,
    )
    d2 = model.score_samples(good_road_scaled)
    threshold = float(np.quantile(d2, threshold_quantile))
    model.threshold_ = threshold

    # Training-only score summary on good-road baseline.
    good_road_scores = model.decision_function(good_road_scaled[features])
    good_road_anomaly_scores = -good_road_scores

    print("Model training completed using baseline distance model")
    print()
    print("Model parameters:")
    print(f"  Threshold quantile (baseline): {threshold_quantile}")
    print(f"  Distance threshold (d2): {threshold:.6f}")
    print("  Features:")
    for feature in features:
        print(f"    {feature}")
    print()

    print("Training summary (good-road baseline only):")
    print("  Dataset used: good_road_scaled")
    print(f"  Training rows (good_road_scaled): {len(good_road_scaled)}")
    print(
        "  Good-road normality score min/median/max: "
        f"{float(np.min(good_road_scores)):.6f} / "
        f"{float(np.median(good_road_scores)):.6f} / "
        f"{float(np.max(good_road_scores)):.6f}"
    )
    print(
        "  Good-road anomaly score min/median/max: "
        f"{float(np.min(good_road_anomaly_scores)):.6f} / "
        f"{float(np.median(good_road_anomaly_scores)):.6f} / "
        f"{float(np.max(good_road_anomaly_scores)):.6f}"
    )
    print()

    model_path = OUTPUT_DIR / "anomaly_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return model
