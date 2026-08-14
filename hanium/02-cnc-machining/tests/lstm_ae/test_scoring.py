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


def test_evaluate_experiment_predictions_distinguishes_methods():
    # Design: Each method has DIFFERENT scores and thresholds to detect method mix-ups.
    # If implementation accidentally uses mean_score for max, or reuses thresholds["mean"],
    # the precision/recall will differ from expected and test will catch it.
    #
    # mean_score with threshold 25:
    #   exp1(10) < 25 + label=0 → TN, exp2(40) > 25 + label=1 → TP,
    #   exp3(15) < 25 + label=1 → FN, exp4(50) > 25 + label=0 → FP
    #   → tp=1, fp=1, fn=1, tn=1, precision=0.5, recall=0.5
    #
    # max_score with threshold 50:
    #   exp1(10) < 50 + label=0 → TN, exp2(40) < 50 + label=1 → FN,
    #   exp3(60) > 50 + label=1 → TP, exp4(30) < 50 + label=0 → TN
    #   → tp=1, fp=0, fn=1, tn=2, precision=1.0, recall=0.5
    #
    # p95_score with threshold 45:
    #   exp1(30) < 45 + label=0 → TN, exp2(60) > 45 + label=1 → TP,
    #   exp3(40) < 45 + label=1 → FN, exp4(50) > 45 + label=0 → FP
    #   → tp=1, fp=1, fn=1, tn=1, precision=0.5, recall=0.5
    eval_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3, 4],
        "mean_score": [10.0, 40.0, 15.0, 50.0],
        "max_score": [10.0, 40.0, 60.0, 30.0],
        "p95_score": [30.0, 60.0, 40.0, 50.0],
    })
    labels = pd.Series({1: 0, 2: 1, 3: 1, 4: 0})
    thresholds = {"mean": 25.0, "max": 50.0, "p95": 45.0}

    result = evaluate_experiment_predictions(eval_scores, labels, thresholds)

    # mean method: tp=1, fp=1, fn=1, tn=1
    assert result["mean"]["tp"] == 1
    assert result["mean"]["fp"] == 1
    assert result["mean"]["fn"] == 1
    assert result["mean"]["tn"] == 1
    assert result["mean"]["precision"] == pytest.approx(0.5)
    assert result["mean"]["recall"] == pytest.approx(0.5)

    # max method: tp=1, fp=0, fn=1, tn=2 (different from mean!)
    assert result["max"]["tp"] == 1
    assert result["max"]["fp"] == 0
    assert result["max"]["fn"] == 1
    assert result["max"]["tn"] == 2
    assert result["max"]["precision"] == pytest.approx(1.0)
    assert result["max"]["recall"] == pytest.approx(0.5)

    # p95 method: tp=1, fp=1, fn=1, tn=1 (same as mean, but via different scores)
    assert result["p95"]["tp"] == 1
    assert result["p95"]["fp"] == 1
    assert result["p95"]["fn"] == 1
    assert result["p95"]["tn"] == 1
    assert result["p95"]["precision"] == pytest.approx(0.5)
    assert result["p95"]["recall"] == pytest.approx(0.5)


def test_evaluate_experiment_predictions_matches_labels_by_experiment_id_not_row_order():
    # Proves that label alignment is by experiment_id value, not by row position.
    # If implementation used positional alignment instead of .loc[scores.index] reindexing,
    # this test would fail.
    eval_scores = pd.DataFrame({
        "experiment_id": [4, 1, 3, 2],  # shuffled order
        "mean_score": [5.0, 1.0, 1.0, 5.0],
        "max_score": [5.0, 1.0, 1.0, 5.0],
        "p95_score": [5.0, 1.0, 1.0, 5.0],
    })
    # Labels indexed in a different order than eval_scores
    labels = pd.Series({2: 1, 4: 0, 1: 0, 3: 1})  # order: [2, 4, 1, 3]
    thresholds = {"mean": 2.0, "max": 2.0, "p95": 2.0}

    result = evaluate_experiment_predictions(eval_scores, labels, thresholds)

    # Expected confusion matrix (using experiment_id values, not row positions):
    # Exp 1: score=1 < 2 → N, label=0 → TN
    # Exp 2: score=5 > 2 → A, label=1 → TP
    # Exp 3: score=1 < 2 → N, label=1 → FN
    # Exp 4: score=5 > 2 → A, label=0 → FP
    for method in ["mean", "max", "p95"]:
        assert result[method]["tp"] == 1  # exp2
        assert result[method]["fp"] == 1  # exp4
        assert result[method]["fn"] == 1  # exp3
        assert result[method]["tn"] == 1  # exp1
        assert result[method]["precision"] == pytest.approx(0.5)
        assert result[method]["recall"] == pytest.approx(0.5)


def test_evaluate_experiment_predictions_raises_keyerror_for_missing_labels():
    # Documents expected behavior: if eval_scores contains an experiment_id
    # not present in labels, a KeyError is raised.
    # (In practice, Task 5's pipeline.py ensures this cannot occur with correct caller inputs.)
    eval_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3],
        "mean_score": [1.0, 5.0, 1.0],
        "max_score": [1.0, 5.0, 1.0],
        "p95_score": [1.0, 5.0, 1.0],
    })
    labels = pd.Series({1: 0, 2: 1})  # missing experiment_id=3
    thresholds = {"mean": 2.0, "max": 2.0, "p95": 2.0}

    with pytest.raises(KeyError):
        evaluate_experiment_predictions(eval_scores, labels, thresholds)
