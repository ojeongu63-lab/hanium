import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import LSTMAutoencoder
from .scoring import (
    aggregate_feature_errors_by_experiment,
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


def compute_timestep_errors(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    """윈도우별 재구성오차를 피처축만 평균 내 시간축(윈도우 내 오프셋)은 남긴다.
    shape: (n_windows, window_size). 윈도우 내 어느 시점에서 재구성이 어긋났는지 보기 위함."""
    reconstructed = _reconstruct(model, windows)
    squared_errors = (reconstructed - windows) ** 2
    return squared_errors.mean(axis=2)


def fold_windows_to_timeline(per_window_values: np.ndarray) -> np.ndarray:
    """한 실험(experiment) 안에서 stride=1로 겹치는 윈도우들의 (n_windows, window_size)
    시점별 값(오차든 재구성값이든)을 실제 실험 시간축 위의 1차원 곡선으로 접는다(대각 평균).
    윈도우 w, 오프셋 o는 원본 시점 w+o를 커버하므로, 같은 원본 시점을 커버하는 모든 (w, o)의
    평균을 취한다."""
    n_windows, window_size = per_window_values.shape
    timeline_len = n_windows + window_size - 1
    sums = np.zeros(timeline_len)
    counts = np.zeros(timeline_len)
    for w in range(n_windows):
        sums[w : w + window_size] += per_window_values[w]
        counts[w : w + window_size] += 1
    return sums / counts


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
    exclude_from_ranking: list[str] | None = None,
) -> dict:
    torch.manual_seed(random_seed)
    exclude_from_ranking = exclude_from_ranking or []

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

    eval_feature_errors = compute_feature_errors(model, eval_windows)
    eval_feature_error_scores = aggregate_feature_errors_by_experiment(
        eval_feature_errors, eval_experiment_ids, feature_columns
    )
    eval_feature_error_scores.to_csv(out_dir / "eval_feature_errors.csv", index=False)

    train_feature_errors = compute_feature_errors(model, train_windows)
    train_feature_error_scores = aggregate_feature_errors_by_experiment(
        train_feature_errors, train_experiment_ids, feature_columns
    )
    feature_baseline = {
        "mean": train_feature_error_scores[feature_columns].mean().to_dict(),
        "std": train_feature_error_scores[feature_columns].std().to_dict(),
    }
    (out_dir / "feature_baseline.json").write_text(json.dumps(feature_baseline, indent=2))

    eval_timestep_errors = compute_timestep_errors(model, eval_windows)
    timelines = []
    offset = 0
    for experiment_id, group_ids in itertools.groupby(eval_experiment_ids):
        n = len(list(group_ids))
        timeline = fold_windows_to_timeline(eval_timestep_errors[offset : offset + n])
        timelines.append(pd.DataFrame({
            "experiment_id": experiment_id,
            "timestep": np.arange(len(timeline)),
            "error": timeline,
        }))
        offset += n
    eval_timeline_errors = pd.concat(timelines, ignore_index=True)
    eval_timeline_errors.to_csv(out_dir / "eval_timeline_errors.csv", index=False)

    rankable_columns = [c for c in feature_columns if c not in exclude_from_ranking]
    baseline_mean = pd.Series(feature_baseline["mean"])[rankable_columns]
    baseline_std = pd.Series(feature_baseline["std"])[rankable_columns]
    eval_feature_z_scores = (
        eval_feature_error_scores.set_index("experiment_id")[rankable_columns]
        - baseline_mean
    ) / baseline_std
    top_feature_by_experiment = eval_feature_z_scores.idxmax(axis=1)
    eval_reconstructed = _reconstruct(model, eval_windows)
    overlays = []
    offset = 0
    for experiment_id, group_ids in itertools.groupby(eval_experiment_ids):
        n = len(list(group_ids))
        feature = top_feature_by_experiment[experiment_id]
        feature_idx = feature_columns.index(feature)

        actual = eval_df.loc[eval_df["experiment_id"] == experiment_id, feature].to_numpy()
        reconstructed = fold_windows_to_timeline(
            eval_reconstructed[offset : offset + n, :, feature_idx]
        )
        overlays.append(pd.DataFrame({
            "experiment_id": experiment_id,
            "feature": feature,
            "timestep": np.arange(len(actual)),
            "actual": actual,
            "reconstructed": reconstructed,
        }))
        offset += n
    eval_overlay = pd.concat(overlays, ignore_index=True)
    eval_overlay.to_csv(out_dir / "eval_reconstruction_overlay.csv", index=False)

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
