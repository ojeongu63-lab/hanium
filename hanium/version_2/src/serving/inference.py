import pandas as pd
import torch

from lstm_ae.pipeline import compute_window_errors
from lstm_ae.scoring import aggregate_window_errors_by_experiment
from lstm_ae.sequencing import make_eval_windows

DEMO_EXPERIMENT_ID = 0


def validate_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [col for col in feature_columns if col not in df.columns]


def scale_features(
    df: pd.DataFrame, feature_columns: list[str], scaler_dict: dict
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_columns:
        mean = scaler_dict[col]["mean"]
        std = scaler_dict[col]["std"]
        df[col] = (df[col] - mean) / std
    return df


def score_to_label(score: float, threshold: float) -> tuple[int, str]:
    if score > threshold:
        return 1, "bad"
    return 0, "good"


def predict_experiment(
    df: pd.DataFrame,
    model: torch.nn.Module,
    feature_columns: list[str],
    scaler_dict: dict,
    window_size: int,
    threshold: float,
    method: str,
) -> dict:
    missing = validate_columns(df, feature_columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if len(df) < window_size:
        raise ValueError(
            f"experiment has {len(df)} rows, needs at least {window_size} rows "
            f"for window_size={window_size}"
        )

    scaled = scale_features(df, feature_columns, scaler_dict)
    scaled["experiment_id"] = DEMO_EXPERIMENT_ID

    windows, experiment_ids = make_eval_windows(scaled, feature_columns, window_size)
    window_errors = compute_window_errors(model, windows)
    experiment_scores = aggregate_window_errors_by_experiment(window_errors, experiment_ids)

    score = float(experiment_scores.loc[0, f"{method}_score"])
    predicted_label, predicted_label_text = score_to_label(score, threshold)

    return {
        "predicted_label": predicted_label,
        "predicted_label_text": predicted_label_text,
        "score": score,
        "threshold": threshold,
        "method": method,
    }
