from __future__ import annotations

import pandas as pd


# -------------------------
# FEATURE ENGINEERING
# -------------------------
def step_03_engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Check if input is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame, got {type(dataframe).__name__}"
        )

    selected_features = [
        "pys_kiiht",
        "siv_kiiht",
        "nyo_kiiht",
    ]

    # Check if required columns exist in the dataframe
    missing_features = [
        col for col in selected_features if col not in dataframe.columns
    ]
    if missing_features:
        raise ValueError(
            f"Required features are missing: {missing_features}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    engineered_dataframe = dataframe.loc[:, selected_features].copy()

    # Print preview of engineered features
    print()
    print("------------------------------------------------------------")
    print("Feature engineering selected features.")
    print()
    print(engineered_dataframe.head().to_string())
    print()

    # Production must not write feature_metadata.json: it belongs to the trained model (MLmodel).
    return engineered_dataframe
