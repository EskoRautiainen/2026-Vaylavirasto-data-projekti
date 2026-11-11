from shared.categories import categorize, PRIORITY_MAP


# ----------------------------------------------------------------------------------------------------
#   BUILD RESULTS
# ----------------------------------------------------------------------------------------------------
def step_06_build_results(metadata, features, predictions, scores):
    result = metadata.copy()

    result = result.join(features)

    anomaly_scores = -scores  # Same as training: larger = more anomalous / worse road

    result['anomaly_prediction'] = predictions
    result['anomaly_score'] = anomaly_scores.round(1)
    result['anomaly_type'] = [
        'Normal' if p == 1 else 'Anomaly' for p in predictions
    ]

    # Shared with training: documented split Critical 4 % / Poor 4 % / Fair 32 % / Good 40 % / Excellent 20 %.
    # Ranked on unrounded scores; categorize() returns positional values (no index alignment).
    result['anomaly_category'] = categorize(anomaly_scores)

    result['priority_score'] = result['anomaly_category'].map(PRIORITY_MAP)  # Assign priority scores

    return result
