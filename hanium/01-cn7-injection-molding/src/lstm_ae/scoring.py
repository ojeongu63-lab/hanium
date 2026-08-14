import numpy as np


def flatten_train_shot_errors(squared_errors: np.ndarray) -> np.ndarray:
    num_windows, window_size, num_features = squared_errors.shape
    return squared_errors.reshape(num_windows * window_size, num_features)


def aggregate_eval_shot_errors(squared_errors: np.ndarray) -> np.ndarray:
    num_windows, window_size, num_features = squared_errors.shape
    num_shots = num_windows + window_size - 1
    sums = np.zeros((num_shots, num_features))
    counts = np.zeros(num_shots)
    for w in range(num_windows):
        for pos in range(window_size):
            shot_idx = w + pos
            sums[shot_idx] += squared_errors[w, pos]
            counts[shot_idx] += 1
    return sums / counts[:, None]


def compute_threshold(train_shot_errors: np.ndarray, percentile: float = 95.0) -> float:
    per_shot_scalar = train_shot_errors.mean(axis=1)
    return float(np.percentile(per_shot_scalar, percentile))


def evaluate_predictions(
    eval_shot_errors: np.ndarray, threshold: float, labels: np.ndarray
) -> dict:
    scalar_scores = eval_shot_errors.mean(axis=1)
    predictions = (scalar_scores > threshold).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
