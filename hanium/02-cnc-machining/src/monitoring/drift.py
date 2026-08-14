INPUT_DRIFT_Z_THRESHOLD = 2.0
OUTPUT_DRIFT_RATIO_THRESHOLD = 0.8


def compute_drift_status(
    recent_requests: list[dict], threshold: float, window_size: int = 10
) -> dict:
    n = len(recent_requests)
    if n < window_size:
        return {
            "n_requests_logged": n,
            "window_size": window_size,
            "sufficient_data": False,
        }

    window = recent_requests[:window_size]

    feature_names = window[0]["feature_means"].keys()
    avg_means = {
        feature: sum(r["feature_means"][feature] for r in window) / window_size
        for feature in feature_names
    }
    flagged_features = [
        {"feature": feature, "avg_scaled_mean": avg}
        for feature, avg in avg_means.items()
        if abs(avg) > INPUT_DRIFT_Z_THRESHOLD
    ]
    flagged_features.sort(key=lambda f: abs(f["avg_scaled_mean"]), reverse=True)

    avg_score = sum(r["score"] for r in window) / window_size
    ratio_to_threshold = avg_score / threshold

    return {
        "n_requests_logged": n,
        "window_size": window_size,
        "sufficient_data": True,
        "input_drift": {
            "flagged_features": flagged_features,
            "all_feature_avg_scaled_means": avg_means,
        },
        "output_drift": {
            "avg_score_recent": avg_score,
            "threshold": threshold,
            "ratio_to_threshold": ratio_to_threshold,
            "flagged": ratio_to_threshold > OUTPUT_DRIFT_RATIO_THRESHOLD,
        },
    }
