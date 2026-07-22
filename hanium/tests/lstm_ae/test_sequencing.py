import numpy as np
import pandas as pd

from lstm_ae.sequencing import make_eval_windows, make_train_windows


def _make_df(num_rows: int, num_features: int) -> pd.DataFrame:
    data = {
        f"f{i}": np.arange(num_rows, dtype=np.float32) * 10 + i
        for i in range(num_features)
    }
    return pd.DataFrame(data)


def test_make_train_windows_is_non_overlapping_and_drops_remainder():
    df = _make_df(num_rows=8, num_features=2)
    feature_columns = ["f0", "f1"]

    windows = make_train_windows(df, feature_columns, window_size=3)

    assert windows.shape == (2, 3, 2)
    # window 0 = rows 0..2, window 1 = rows 3..5 (rows 6,7 dropped as remainder)
    expected_window0 = df[feature_columns].to_numpy(dtype=np.float32)[0:3]
    expected_window1 = df[feature_columns].to_numpy(dtype=np.float32)[3:6]
    np.testing.assert_array_equal(windows[0], expected_window0)
    np.testing.assert_array_equal(windows[1], expected_window1)


def test_make_eval_windows_is_overlapping_stride_one():
    df = _make_df(num_rows=8, num_features=2)
    feature_columns = ["f0", "f1"]

    windows = make_eval_windows(df, feature_columns, window_size=3)

    assert windows.shape == (6, 3, 2)
    values = df[feature_columns].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(windows[0], values[0:3])
    np.testing.assert_array_equal(windows[1], values[1:4])
    np.testing.assert_array_equal(windows[5], values[5:8])
