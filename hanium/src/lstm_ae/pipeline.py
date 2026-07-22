import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .detrend import apply_rolling_zscore
from .model import LSTMAutoencoder
from .scoring import (
    aggregate_eval_shot_errors,
    compute_threshold,
    evaluate_predictions,
    flatten_train_shot_errors,
)
from .sequencing import make_eval_windows, make_train_windows
from .training import train_autoencoder


def _compute_squared_errors(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32)
        reconstructed = model(x).numpy()
    return (reconstructed - windows) ** 2


def run_lstm_pipeline(
    train_csv_path: str,
    eval_csv_path: str,
    feature_columns: list[str],
    output_dir: str,
    window_size: int = 12,
    hidden_size: int = 64,
    latent_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    detrend_window: int = 200,
    detrend_min_periods: int = 30,
) -> dict:
    torch.manual_seed(random_seed)

    train_df = pd.read_csv(train_csv_path)
    eval_df = pd.read_csv(eval_csv_path)

    train_detrended = apply_rolling_zscore(
        train_df, feature_columns, window=detrend_window, min_periods=detrend_min_periods
    )
    eval_detrended = apply_rolling_zscore(
        eval_df, feature_columns, window=detrend_window, min_periods=detrend_min_periods
    )

    train_windows = make_train_windows(train_detrended, feature_columns, window_size)
    eval_windows = make_eval_windows(eval_detrended, feature_columns, window_size)

    model = LSTMAutoencoder(
        num_features=len(feature_columns), hidden_size=hidden_size, latent_dim=latent_dim
    )
    loss_history = train_autoencoder(
        model, train_windows, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )

    train_squared_errors = _compute_squared_errors(model, train_windows)
    eval_squared_errors = _compute_squared_errors(model, eval_windows)

    train_shot_errors = flatten_train_shot_errors(train_squared_errors)
    eval_shot_errors = aggregate_eval_shot_errors(eval_squared_errors)

    threshold = compute_threshold(train_shot_errors)
    labels = eval_df["label"].to_numpy()
    report = evaluate_predictions(eval_shot_errors, threshold, labels)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")

    training_config = {
        "window_size": window_size,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "random_seed": random_seed,
        "detrend_window": detrend_window,
        "detrend_min_periods": detrend_min_periods,
    }
    (out_dir / "training_config.json").write_text(json.dumps(training_config, indent=2))

    error_columns = ["shot_error"] + [f"error_{c}" for c in feature_columns]

    train_out = pd.DataFrame(
        np.column_stack([train_shot_errors.mean(axis=1), train_shot_errors]),
        columns=error_columns,
    )
    train_out.to_csv(out_dir / "train_reconstruction_errors.csv", index=False)

    eval_out = pd.DataFrame(
        np.column_stack([eval_shot_errors.mean(axis=1), eval_shot_errors]),
        columns=error_columns,
    )
    for col in ["PassOrFail", "Reason", "TimeStamp", "label"]:
        eval_out[col] = eval_df[col].to_numpy()
    eval_out.to_csv(out_dir / "eval_reconstruction_errors.csv", index=False)

    (out_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2))

    return {
        "train_shots": len(train_shot_errors),
        "eval_shots": len(eval_shot_errors),
        "final_train_loss": loss_history[-1],
        "threshold": threshold,
        **report,
    }
