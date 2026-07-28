import json

import numpy as np
import pandas as pd
import torch

from lstm_ae.pipeline import run_lstm_pipeline
from lstm_ae.model import LSTMAutoencoder

FEATURE_COLUMNS = ["f0", "f1", "f2"]


def _make_train_csv(path):
    # 2 experiments, 30 rows each -> with window_size=6, 5 windows per experiment
    frames = []
    for experiment_id in [101, 102]:
        data = {col: np.random.randn(30).astype(np.float32) for col in FEATURE_COLUMNS}
        frame = pd.DataFrame(data)
        frame["experiment_id"] = experiment_id
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _make_eval_csv(path):
    # 2 experiments: 201 (good, label=0), 202 (bad, label=1), 20 rows each
    frames = []
    for experiment_id, label in [(201, 0), (202, 1)]:
        data = {col: np.random.randn(20).astype(np.float32) for col in FEATURE_COLUMNS}
        frame = pd.DataFrame(data)
        frame["experiment_id"] = experiment_id
        frame["label"] = label
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def test_run_lstm_pipeline_creates_expected_output_files(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    output_dir = tmp_path / "model"

    _make_train_csv(train_path)
    _make_eval_csv(eval_path)

    summary = run_lstm_pipeline(
        train_csv_path=str(train_path),
        eval_csv_path=str(eval_path),
        feature_columns=FEATURE_COLUMNS,
        output_dir=str(output_dir),
        window_size=6,
        hidden_size=4,
        latent_dim=2,
        epochs=2,
        batch_size=4,
    )

    for name in [
        "model.pt",
        "training_config.json",
        "train_window_errors.csv",
        "eval_window_errors.csv",
        "experiment_scores.csv",
        "evaluation_report.json",
    ]:
        assert (output_dir / name).exists()

    train_errors = pd.read_csv(output_dir / "train_window_errors.csv")
    assert set(train_errors.columns) == {"experiment_id", "window_error"}
    assert len(train_errors) == 10  # 2 experiments x 5 non-overlapping windows each

    eval_errors = pd.read_csv(output_dir / "eval_window_errors.csv")
    assert len(eval_errors) == 30  # 2 experiments x (20-6+1)=15 overlapping windows each

    experiment_scores = pd.read_csv(output_dir / "experiment_scores.csv")
    assert len(experiment_scores) == 2  # one row per eval experiment
    assert set(experiment_scores["experiment_id"]) == {201, 202}
    assert set(experiment_scores.loc[experiment_scores["experiment_id"] == 201, "label"]) == {0}
    assert set(experiment_scores.loc[experiment_scores["experiment_id"] == 202, "label"]) == {1}

    report = json.loads((output_dir / "evaluation_report.json").read_text())
    assert set(report.keys()) == {"thresholds", "results"}
    assert set(report["thresholds"].keys()) == {"mean", "max", "p95"}
    for method in ["mean", "max", "p95"]:
        assert set(report["results"][method].keys()) >= {"precision", "recall", "tp", "fp", "fn", "tn"}
        assert report["results"][method]["tp"] + report["results"][method]["fn"] == 1  # 1 bad experiment
        assert report["results"][method]["tn"] + report["results"][method]["fp"] == 1  # 1 good experiment

    assert "train_windows" in summary
    assert summary["train_windows"] == 10
    assert summary["eval_windows"] == 30
    assert isinstance(summary["model"], LSTMAutoencoder)
