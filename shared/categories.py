from __future__ import annotations

import numpy as np
import pandas as pd

# Documented prioritisation (doc/Technical_documentation.docx), worst first:
# Critical 4 %, Poor 4 %, Fair 32 %, Good 40 %, Excellent 20 % of the roads in a run.
# Categories are relative to the data in the run, not absolute quality classes.
CATEGORY_LABELS = ["Critical", "Poor", "Fair", "Good", "Excellent"]
CATEGORY_UPPER_BOUNDS = [0.04, 0.08, 0.40, 0.80, 1.0]  # cumulative share of roads, worst first
PRIORITY_MAP = {label: priority for priority, label in enumerate(CATEGORY_LABELS, start=1)}


def categorize(anomaly_scores) -> pd.Categorical:
    """Category per road from anomaly_score (larger = worse), in the same order as the input."""
    # Rank-based, so ties and quantized scores never produce empty or overlapping bins.
    worst_first = pd.Series(np.asarray(anomaly_scores, dtype=float)).rank(
        method="average", ascending=False, pct=True
    )
    # .array: plain positional values, so callers can't mis-align them by index
    return pd.cut(worst_first, bins=[0.0] + CATEGORY_UPPER_BOUNDS, labels=CATEGORY_LABELS).array
