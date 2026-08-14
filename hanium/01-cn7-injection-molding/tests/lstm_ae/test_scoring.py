import numpy as np
import pytest

from lstm_ae.scoring import (
    aggregate_eval_shot_errors,
    compute_threshold,
    evaluate_predictions,
    flatten_train_shot_errors,
)


def test_flatten_train_shot_errors_preserves_row_major_order():
    squared_errors = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )  # (2 windows, 3 window_size, 1 feature)

    result = flatten_train_shot_errors(squared_errors)

    assert result.shape == (6, 1)
    np.testing.assert_array_equal(result.flatten(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_aggregate_eval_shot_errors_averages_overlapping_windows():
    # window_size=2, 3 windows -> 4 shots. 1 feature for simplicity.
    # window0 covers shots [0,1] = [1,2]; window1 covers shots [1,2] = [3,4];
    # window2 covers shots [2,3] = [5,6]
    squared_errors = np.array(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
        ]
    )

    result = aggregate_eval_shot_errors(squared_errors)

    assert result.shape == (4, 1)
    # shot0: only window0 pos0 -> 1.0
    # shot1: window0 pos1 (2.0) + window1 pos0 (3.0) -> mean 2.5
    # shot2: window1 pos1 (4.0) + window2 pos0 (5.0) -> mean 4.5
    # shot3: only window2 pos1 -> 6.0
    np.testing.assert_allclose(result.flatten(), [1.0, 2.5, 4.5, 6.0])


def test_compute_threshold_is_a_percentile_of_train_shot_errors():
    # per-shot scalar error (mean over features) will be exactly [10,20,...,100]
    # (10 values, 2 identical features per row so the mean equals that value).
    # numpy's default linear interpolation for the 90th percentile of 10 sorted
    # values: index = 0.90*(10-1) = 8.1 -> between arr[8]=90 and arr[9]=100,
    # fraction 0.1 -> 90 + 0.1*(100-90) = 91.0 (independently hand-computed).
    train_shot_errors = np.array([[v, v] for v in range(10, 101, 10)])

    threshold = compute_threshold(train_shot_errors, percentile=90)

    assert threshold == pytest.approx(91.0)


def test_compute_threshold_defaults_to_95th_percentile():
    train_shot_errors = np.array([[v, v] for v in range(10, 101, 10)])

    threshold = compute_threshold(train_shot_errors)

    assert threshold == pytest.approx(np.percentile(range(10, 101, 10), 95))


def test_evaluate_predictions_computes_precision_recall_and_confusion_counts():
    # scalar error (mean over the single feature) = [0.5, 5.0, 0.5, 5.0]
    eval_shot_errors = np.array([[0.5], [5.0], [0.5], [5.0]])
    labels = np.array([0, 1, 1, 0])  # shot1 correctly flagged, shot2 missed, shot3 false alarm
    threshold = 2.0

    result = evaluate_predictions(eval_shot_errors, threshold, labels)

    assert result["tp"] == 1  # shot1: predicted 1, label 1
    assert result["fp"] == 1  # shot3: predicted 1, label 0
    assert result["fn"] == 1  # shot2: predicted 0, label 1
    assert result["tn"] == 1  # shot0: predicted 0, label 0
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
