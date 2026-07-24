import numpy as np
import pandas as pd


def make_train_windows(
    df: pd.DataFrame, feature_columns: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    all_windows = []
    all_experiment_ids = []
    for experiment_id, group in df.groupby("experiment_id", sort=True):
        values = group[feature_columns].to_numpy(dtype=np.float32)
        num_windows = len(values) // window_size
        trimmed = values[: num_windows * window_size]
        windows = trimmed.reshape(num_windows, window_size, len(feature_columns))
        all_windows.append(windows)
        all_experiment_ids.extend([experiment_id] * num_windows)
    return np.concatenate(all_windows, axis=0), np.array(all_experiment_ids)


def make_eval_windows(
    df: pd.DataFrame, feature_columns: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    all_windows = []
    all_experiment_ids = []
    for experiment_id, group in df.groupby("experiment_id", sort=True):
        values = group[feature_columns].to_numpy(dtype=np.float32)
        num_windows = len(values) - window_size + 1
        if num_windows <= 0:
            continue
        windows = np.stack([values[i : i + window_size] for i in range(num_windows)])
        all_windows.append(windows)
        all_experiment_ids.extend([experiment_id] * num_windows)
    return np.concatenate(all_windows, axis=0), np.array(all_experiment_ids)
