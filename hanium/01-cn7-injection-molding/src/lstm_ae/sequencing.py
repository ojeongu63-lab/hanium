import numpy as np
import pandas as pd


def make_train_windows(
    df: pd.DataFrame, feature_columns: list[str], window_size: int
) -> np.ndarray:
    values = df[feature_columns].to_numpy(dtype=np.float32)
    num_windows = len(values) // window_size
    trimmed = values[: num_windows * window_size]
    return trimmed.reshape(num_windows, window_size, len(feature_columns))


def make_eval_windows(
    df: pd.DataFrame, feature_columns: list[str], window_size: int
) -> np.ndarray:
    values = df[feature_columns].to_numpy(dtype=np.float32)
    num_windows = len(values) - window_size + 1
    return np.stack([values[i : i + window_size] for i in range(num_windows)])
