import numpy as np
import pandas as pd
import pytest

from retraining.runner import collect_normal_batches, rescale_eval


def _write_batch(timeline_dir, batch_id, value):
    timeline_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"X_OutputPower": [value, value + 1.0]}).to_csv(
        timeline_dir / f"{batch_id}.csv", index=False
    )


def _label(batch_id, produced_day, label="good"):
    return {
        "batch_id": batch_id,
        "produced_day": produced_day,
        "arrived_day": produced_day + 7,
        "label": label,
    }


def test_collects_only_good_labels(tmp_path):
    _write_batch(tmp_path, "day05_0", 1.0)
    _write_batch(tmp_path, "day06_0", 2.0)
    labels = [_label("day05_0", 5), _label("day06_0", 6, label="bad")]

    result = collect_normal_batches(labels, tmp_path, current_day=20, lookback_days=30)

    assert len(result) == 2  # 배치 1개 × 2행
    assert result["experiment_id"].nunique() == 1


def test_respects_lookback_window(tmp_path):
    _write_batch(tmp_path, "day02_0", 1.0)
    _write_batch(tmp_path, "day25_0", 2.0)
    labels = [_label("day02_0", 2), _label("day25_0", 25)]

    result = collect_normal_batches(labels, tmp_path, current_day=30, lookback_days=10)

    assert result["experiment_id"].nunique() == 1
    assert result["X_OutputPower"].iloc[0] == 2.0


def test_each_batch_gets_distinct_experiment_id(tmp_path):
    _write_batch(tmp_path, "day05_0", 1.0)
    _write_batch(tmp_path, "day05_1", 2.0)
    labels = [_label("day05_0", 5), _label("day05_1", 5)]

    result = collect_normal_batches(labels, tmp_path, current_day=20, lookback_days=30)

    assert result["experiment_id"].nunique() == 2


def test_raises_when_no_usable_batches(tmp_path):
    labels = [_label("day05_0", 5, label="bad")]

    with pytest.raises(ValueError, match="정상 라벨 배치가 없습니다"):
        collect_normal_batches(labels, tmp_path, current_day=20, lookback_days=30)


def test_rescale_eval_is_identity_when_scalers_match():
    columns = ["X_OutputPower"]
    scaler = {"X_OutputPower": {"mean": 2.0, "std": 4.0}}
    eval_df = pd.DataFrame({"X_OutputPower": [0.5, -0.5], "label": [0, 1]})

    result = rescale_eval(eval_df, scaler, scaler, columns)

    np.testing.assert_allclose(result["X_OutputPower"], [0.5, -0.5])


def test_rescale_eval_moves_to_new_coordinates_and_keeps_labels():
    columns = ["X_OutputPower"]
    old = {"X_OutputPower": {"mean": 0.0, "std": 1.0}}
    new = {"X_OutputPower": {"mean": 10.0, "std": 2.0}}
    eval_df = pd.DataFrame({"X_OutputPower": [10.0, 12.0], "label": [0, 1]})

    result = rescale_eval(eval_df, old, new, columns)

    # raw = 10, 12  →  (raw - 10) / 2 = 0, 1
    np.testing.assert_allclose(result["X_OutputPower"], [0.0, 1.0])
    assert result["label"].tolist() == [0, 1]
