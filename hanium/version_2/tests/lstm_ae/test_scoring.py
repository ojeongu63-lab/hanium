import numpy as np
import pandas as pd
import pytest

from lstm_ae.scoring import (
    aggregate_window_errors_by_experiment,
    compute_thresholds,
    evaluate_experiment_predictions,
)


def test_aggregate_window_errors_by_experiment_computes_mean_max_p95():
    # experiment 1: window errors [1, 2, 3, 4, 5] (mean=3, max=5, p95=4.8)
    # experiment 2: window errors [10, 20] (mean=15, max=20, p95=19.5)
    window_errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])
    experiment_ids = np.array([1, 1, 1, 1, 1, 2, 2])

    result = aggregate_window_errors_by_experiment(window_errors, experiment_ids)

    result = result.set_index("experiment_id")
    assert result.loc[1, "mean_score"] == pytest.approx(3.0)
    assert result.loc[1, "max_score"] == pytest.approx(5.0)
    assert result.loc[1, "p95_score"] == pytest.approx(np.percentile([1, 2, 3, 4, 5], 95))
    assert result.loc[2, "mean_score"] == pytest.approx(15.0)
    assert result.loc[2, "max_score"] == pytest.approx(20.0)


def test_compute_thresholds_is_percentile_of_train_experiment_scores():
    train_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3, 4],
        "mean_score": [10.0, 20.0, 30.0, 40.0],
        "max_score": [100.0, 200.0, 300.0, 400.0],
        "p95_score": [1.0, 2.0, 3.0, 4.0],
    })

    thresholds = compute_thresholds(train_scores, percentile=90)

    assert thresholds["mean"] == pytest.approx(np.percentile([10, 20, 30, 40], 90))
    assert thresholds["max"] == pytest.approx(np.percentile([100, 200, 300, 400], 90))
    assert thresholds["p95"] == pytest.approx(np.percentile([1, 2, 3, 4], 90))


def test_evaluate_experiment_predictions_computes_precision_recall_per_method():
    eval_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3, 4],
        "mean_score": [1.0, 5.0, 1.0, 5.0],
        "max_score": [1.0, 5.0, 1.0, 5.0],
        "p95_score": [1.0, 5.0, 1.0, 5.0],
    })
    labels = pd.Series({1: 0, 2: 1, 3: 1, 4: 0})  # exp2 correctly flagged, exp3 missed, exp4 false alarm
    thresholds = {"mean": 2.0, "max": 2.0, "p95": 2.0}

    result = evaluate_experiment_predictions(eval_scores, labels, thresholds)

    for method in ["mean", "max", "p95"]:
        assert result[method]["tp"] == 1  # exp2
        assert result[method]["fp"] == 1  # exp4
        assert result[method]["fn"] == 1  # exp3
        assert result[method]["tn"] == 1  # exp1
        assert result[method]["precision"] == pytest.approx(0.5)
        assert result[method]["recall"] == pytest.approx(0.5)
