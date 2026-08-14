from monitoring.drift import compute_drift_status


def _request(feature_means: dict, score: float, label: str = "good") -> dict:
    return {"feature_means": feature_means, "score": score, "predicted_label_text": label}


def test_insufficient_data_returns_early():
    recent = [_request({"f0": 0.0}, 0.5) for _ in range(3)]

    status = compute_drift_status(recent, threshold=1.0, window_size=10)

    assert status["sufficient_data"] is False
    assert status["n_requests_logged"] == 3


def test_no_drift_when_values_near_baseline():
    recent = [_request({"f0": 0.1, "f1": -0.1}, 0.3) for _ in range(10)]

    status = compute_drift_status(recent, threshold=1.0, window_size=10)

    assert status["sufficient_data"] is True
    assert status["input_drift"]["flagged_features"] == []
    assert status["output_drift"]["flagged"] is False


def test_input_drift_flags_feature_far_from_baseline():
    recent = [_request({"f0": 3.5, "f1": 0.0}, 0.3) for _ in range(10)]

    status = compute_drift_status(recent, threshold=1.0, window_size=10)

    flagged = status["input_drift"]["flagged_features"]
    assert len(flagged) == 1
    assert flagged[0]["feature"] == "f0"


def test_output_drift_flags_when_score_near_threshold():
    recent = [_request({"f0": 0.0}, 0.9) for _ in range(10)]

    status = compute_drift_status(recent, threshold=1.0, window_size=10)

    assert status["output_drift"]["flagged"] is True
    assert status["output_drift"]["ratio_to_threshold"] == 0.9
