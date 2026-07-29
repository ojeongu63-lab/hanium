import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import LSTMAutoencoder
from .scoring import (
    aggregate_window_errors_by_experiment,
    compute_thresholds,
    evaluate_experiment_predictions,
)
from .sequencing import make_eval_windows, make_train_windows
from .training import train_autoencoder


def _reconstruct(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32)
        return model(x).numpy()


def compute_window_errors(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    reconstructed = _reconstruct(model, windows)
    squared_errors = (reconstructed - windows) ** 2
    return squared_errors.reshape(len(windows), -1).mean(axis=1)


def compute_feature_errors(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    """윈도우별 재구성오차를 시간축만 평균 내 피처축은 남긴다. shape: (n_windows, n_features).
    compute_window_errors가 피처+시간을 전부 뭉개 스칼라로 만드는 것과 같은 재구성값을 쓰되,
    어떤 피처가 오차에 가장 크게 기여했는지 보기 위해 피처별로 분리해둔다."""
    reconstructed = _reconstruct(model, windows)
    squared_errors = (reconstructed - windows) ** 2
    return squared_errors.mean(axis=1)


def run_lstm_pipeline(
    train_csv_path: str,
    eval_csv_path: str,
    feature_columns: list[str],
    output_dir: str,
    window_size: int = 20,
    hidden_size: int = 64,
    latent_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    threshold_percentile: float = 95.0,
) -> dict:
    torch.manual_seed(random_seed)

    train_df = pd.read_csv(train_csv_path)
    eval_df = pd.read_csv(eval_csv_path)

    train_windows, train_experiment_ids = make_train_windows(
        train_df, feature_columns, window_size
    )
    eval_windows, eval_experiment_ids = make_eval_windows(
        eval_df, feature_columns, window_size
    )

    model = LSTMAutoencoder(
        num_features=len(feature_columns), hidden_size=hidden_size, latent_dim=latent_dim
    )
    loss_history = train_autoencoder(
        model, train_windows, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )

    train_window_errors = compute_window_errors(model, train_windows)
    eval_window_errors = compute_window_errors(model, eval_windows)

    train_experiment_scores = aggregate_window_errors_by_experiment(
        train_window_errors, train_experiment_ids
    )
    eval_experiment_scores = aggregate_window_errors_by_experiment(
        eval_window_errors, eval_experiment_ids
    )

    thresholds = compute_thresholds(train_experiment_scores, percentile=threshold_percentile)

    labels = eval_df.groupby("experiment_id")["label"].first()
    report = evaluate_experiment_predictions(eval_experiment_scores, labels, thresholds)

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
        "threshold_percentile": threshold_percentile,
    }
    (out_dir / "training_config.json").write_text(json.dumps(training_config, indent=2))

    pd.DataFrame(
        {"experiment_id": train_experiment_ids, "window_error": train_window_errors}
    ).to_csv(out_dir / "train_window_errors.csv", index=False)
    pd.DataFrame(
        {"experiment_id": eval_experiment_ids, "window_error": eval_window_errors}
    ).to_csv(out_dir / "eval_window_errors.csv", index=False)

    experiment_scores = eval_experiment_scores.merge(
        labels.rename("label"), on="experiment_id"
    )
    for method in ["mean", "max", "p95"]:
        experiment_scores[f"{method}_exceeds_threshold"] = (
            experiment_scores[f"{method}_score"] > thresholds[method]
        )
    experiment_scores.to_csv(out_dir / "experiment_scores.csv", index=False)

    (out_dir / "evaluation_report.json").write_text(
        json.dumps({"thresholds": thresholds, "results": report}, indent=2)
    )

    return {
        "model": model,
        "train_windows": len(train_windows),
        "eval_windows": len(eval_windows),
        "final_train_loss": loss_history[-1],
        "thresholds": thresholds,
        "results": report,
    }
