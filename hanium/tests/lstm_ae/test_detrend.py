import numpy as np
import pandas as pd
import pytest

from lstm_ae.detrend import apply_rolling_zscore


def test_rolling_zscore_matches_hand_computed_values_once_min_periods_reached():
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 50.0]})

    result = apply_rolling_zscore(df, ["a"], window=3, min_periods=2)

    # row1: rolling(10,20) -> mean=15, std=7.071067811865476 -> z=(20-15)/7.071...
    assert result["a"].iloc[1] == pytest.approx((20.0 - 15.0) / 7.0710678118654755)
    # row2: rolling(10,20,30) -> mean=20, std=10 -> z=(30-20)/10=1.0
    assert result["a"].iloc[2] == pytest.approx(1.0)
    # row3: rolling window=3 -> last 3 values (20,30,40) -> mean=30, std=10 -> z=1.0
    assert result["a"].iloc[3] == pytest.approx(1.0)
    # row4: rolling window=3 -> last 3 values (30,40,50) -> mean=40, std=10 -> z=1.0
    assert result["a"].iloc[4] == pytest.approx(1.0)


def test_rows_before_min_periods_fall_back_to_global_mean_and_std():
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 50.0]})
    global_mean = df["a"].mean()
    global_std = df["a"].std()

    result = apply_rolling_zscore(df, ["a"], window=3, min_periods=2)

    assert result["a"].iloc[0] == pytest.approx((10.0 - global_mean) / global_std)


def test_constant_window_does_not_produce_nan_or_inf():
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0, 5.0, 5.0, 10.0]})

    result = apply_rolling_zscore(df, ["a"], window=3, min_periods=2)

    assert result["a"].notna().all()
    assert np.isfinite(result["a"]).all()


def test_only_feature_columns_are_transformed():
    df = pd.DataFrame({
        "a": [10.0, 20.0, 30.0, 40.0],
        "label": [0, 0, 1, 0],
    })

    result = apply_rolling_zscore(df, ["a"], window=3, min_periods=2)

    assert result["label"].tolist() == [0, 0, 1, 0]
