import numpy as np
import pandas as pd

from lstm_ae.sequencing import make_eval_windows, make_train_windows


def _make_df(experiment_rows: dict[int, int], num_features: int) -> pd.DataFrame:
    frames = []
    for experiment_id, num_rows in experiment_rows.items():
        data = {
            f"f{i}": np.arange(num_rows, dtype=np.float32) * 10 + i
            for i in range(num_features)
        }
        frame = pd.DataFrame(data)
        frame["experiment_id"] = experiment_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_make_train_windows_respects_experiment_boundaries_and_drops_remainder():
    # experiment 1: 7 rows, window_size=3 -> 2 windows (rows 0-2, 3-5), row 6 dropped
    # experiment 2: 5 rows, window_size=3 -> 1 window (rows 0-2 of exp2), rows 3-4 dropped
    df = _make_df({1: 7, 2: 5}, num_features=2)
    feature_columns = ["f0", "f1"]

    windows, experiment_ids = make_train_windows(df, feature_columns, window_size=3)

    assert windows.shape == (3, 3, 2)
    assert experiment_ids.tolist() == [1, 1, 2]

    exp1_values = df.loc[df["experiment_id"] == 1, feature_columns].to_numpy(dtype=np.float32)
    exp2_values = df.loc[df["experiment_id"] == 2, feature_columns].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(windows[0], exp1_values[0:3])
    np.testing.assert_array_equal(windows[1], exp1_values[3:6])
    np.testing.assert_array_equal(windows[2], exp2_values[0:3])


def test_make_train_windows_never_mixes_two_experiments_in_one_window():
    # experiment 1 has 4 rows (values 0,10,20,30), experiment 2 has 4 rows (values
    # 1000,1010,1020,1030) so a boundary-crossing window would be immediately obvious.
    df = pd.concat(
        [
            pd.DataFrame({"f0": [0.0, 10.0, 20.0, 30.0], "experiment_id": 1}),
            pd.DataFrame({"f0": [1000.0, 1010.0, 1020.0, 1030.0], "experiment_id": 2}),
        ],
        ignore_index=True,
    )

    windows, experiment_ids = make_train_windows(df, ["f0"], window_size=4)

    assert windows.shape == (2, 4, 1)
    assert experiment_ids.tolist() == [1, 2]
    assert windows[0].max() < 1000.0  # window 0 is entirely experiment 1
    assert windows[1].min() >= 1000.0  # window 1 is entirely experiment 2


def test_make_eval_windows_is_overlapping_within_each_experiment():
    # experiment 1: 5 rows, window_size=3 -> 3 windows; experiment 2: 4 rows -> 2 windows
    df = _make_df({1: 5, 2: 4}, num_features=1)

    windows, experiment_ids = make_eval_windows(df, ["f0"], window_size=3)

    assert windows.shape == (5, 3, 1)
    assert experiment_ids.tolist() == [1, 1, 1, 2, 2]

    exp1_values = df.loc[df["experiment_id"] == 1, ["f0"]].to_numpy(dtype=np.float32)
    exp2_values = df.loc[df["experiment_id"] == 2, ["f0"]].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(windows[0], exp1_values[0:3])
    np.testing.assert_array_equal(windows[1], exp1_values[1:4])
    np.testing.assert_array_equal(windows[2], exp1_values[2:5])
    np.testing.assert_array_equal(windows[3], exp2_values[0:3])
    np.testing.assert_array_equal(windows[4], exp2_values[1:4])
