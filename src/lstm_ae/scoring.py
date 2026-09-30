import numpy as np
import pandas as pd


def aggregate_window_errors_by_experiment(
    window_errors: np.ndarray, experiment_ids: np.ndarray
) -> pd.DataFrame:
    df = pd.DataFrame({"experiment_id": experiment_ids, "window_error": window_errors})
    grouped = df.groupby("experiment_id")["window_error"]
    result = pd.DataFrame({
        "mean_score": grouped.mean(),
        "max_score": grouped.max(),
        "p95_score": grouped.quantile(0.95),
    })
    return result.reset_index()


def aggregate_feature_errors_by_experiment(
    feature_errors: np.ndarray, experiment_ids: np.ndarray, feature_columns: list[str]
) -> pd.DataFrame:
    df = pd.DataFrame(feature_errors, columns=feature_columns)
    df["experiment_id"] = experiment_ids
    return df.groupby("experiment_id")[feature_columns].mean().reset_index()


def compute_thresholds(
    train_experiment_scores: pd.DataFrame, percentile: float = 95.0
) -> dict:
    return {
        "mean": float(np.percentile(train_experiment_scores["mean_score"], percentile)),
        "max": float(np.percentile(train_experiment_scores["max_score"], percentile)),
        "p95": float(np.percentile(train_experiment_scores["p95_score"], percentile)),
    }


def evaluate_experiment_predictions(
    eval_experiment_scores: pd.DataFrame, labels: pd.Series, thresholds: dict
) -> dict:
    results = {}
    for method in ["mean", "max", "p95"]:
        scores = eval_experiment_scores.set_index("experiment_id")[f"{method}_score"]
        predictions = (scores > thresholds[method]).astype(int)
        aligned_labels = labels.loc[scores.index]
        tp = int(((predictions == 1) & (aligned_labels == 1)).sum())
        fp = int(((predictions == 1) & (aligned_labels == 0)).sum())
        fn = int(((predictions == 0) & (aligned_labels == 1)).sum())
        tn = int(((predictions == 0) & (aligned_labels == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[method] = {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return results
