import json

import numpy as np
import pandas as pd
import torch

from lstm_ae.pipeline import (
    compute_feature_errors,
    compute_timestep_errors,
    compute_window_errors,
    fold_windows_to_timeline,
    run_lstm_pipeline,
)
from lstm_ae.model import LSTMAutoencoder

FEATURE_COLUMNS = ["f0", "f1", "f2"]


def test_compute_feature_errors_matches_window_errors_when_averaged():
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)
    windows = np.random.randn(5, 6, 3).astype(np.float32)

    window_errors = compute_window_errors(model, windows)
    feature_errors = compute_feature_errors(model, windows)

    assert feature_errors.shape == (5, 3)
    np.testing.assert_allclose(feature_errors.mean(axis=1), window_errors, rtol=1e-5)


def test_compute_timestep_errors_matches_window_errors_when_averaged():
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)
    windows = np.random.randn(5, 6, 3).astype(np.float32)

    window_errors = compute_window_errors(model, windows)
    timestep_errors = compute_timestep_errors(model, windows)

    assert timestep_errors.shape == (5, 6)
    np.testing.assert_allclose(timestep_errors.mean(axis=1), window_errors, rtol=1e-5)


def test_fold_windows_to_timeline_averages_overlapping_windows():
    # 2 overlapping windows of size 3 over a 4-step timeline:
    # window 0 covers t=0,1,2 ; window 1 covers t=1,2,3
    timestep_errors = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    timeline = fold_windows_to_timeline(timestep_errors)

    # t=0 -> only window 0 offset 0 -> 1.0
    # t=1 -> window 0 offset 1 (2.0), window 1 offset 0 (4.0) -> mean 3.0
    # t=2 -> window 0 offset 2 (3.0), window 1 offset 1 (5.0) -> mean 4.0
    # t=3 -> only window 1 offset 2 -> 6.0
    np.testing.assert_allclose(timeline, [1.0, 3.0, 4.0, 6.0])


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
        "eval_feature_errors.csv",
        "eval_timeline_errors.csv",
        "eval_reconstruction_overlay.csv",
        "feature_baseline.json",
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

    feature_errors = pd.read_csv(output_dir / "eval_feature_errors.csv")
    assert set(feature_errors.columns) == {"experiment_id", *FEATURE_COLUMNS}
    assert len(feature_errors) == 2  # one row per eval experiment
    assert set(feature_errors["experiment_id"]) == {201, 202}

    timeline_errors = pd.read_csv(output_dir / "eval_timeline_errors.csv")
    assert set(timeline_errors.columns) == {"experiment_id", "timestep", "error"}
    # 20 rows per experiment (folded timeline length == raw row count, 15 windows + 6 - 1 = 20)
    assert len(timeline_errors[timeline_errors["experiment_id"] == 201]) == 20
    assert len(timeline_errors[timeline_errors["experiment_id"] == 202]) == 20

    overlay = pd.read_csv(output_dir / "eval_reconstruction_overlay.csv")
    assert set(overlay.columns) == {"experiment_id", "feature", "timestep", "actual", "reconstructed"}
    assert len(overlay[overlay["experiment_id"] == 201]) == 20
    assert len(overlay[overlay["experiment_id"] == 202]) == 20
    assert set(overlay["feature"].unique()) <= set(FEATURE_COLUMNS)
    # one single feature chosen per experiment (the top contributor)
    assert overlay.groupby("experiment_id")["feature"].nunique().eq(1).all()

    feature_baseline = json.loads((output_dir / "feature_baseline.json").read_text())
    assert set(feature_baseline.keys()) == {"mean", "std"}
    assert set(feature_baseline["mean"].keys()) == set(FEATURE_COLUMNS)
    assert set(feature_baseline["std"].keys()) == set(FEATURE_COLUMNS)

    assert "train_windows" in summary
    assert summary["train_windows"] == 10
    assert summary["eval_windows"] == 30
    assert isinstance(summary["model"], LSTMAutoencoder)


def test_run_lstm_pipeline_excludes_columns_from_ranking(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    output_dir = tmp_path / "model"

    _make_train_csv(train_path)
    _make_eval_csv(eval_path)

    run_lstm_pipeline(
        train_csv_path=str(train_path),
        eval_csv_path=str(eval_path),
        feature_columns=FEATURE_COLUMNS,
        output_dir=str(output_dir),
        window_size=6,
        hidden_size=4,
        latent_dim=2,
        epochs=2,
        batch_size=4,
        exclude_from_ranking=["f0"],
    )

    overlay = pd.read_csv(output_dir / "eval_reconstruction_overlay.csv")
    assert "f0" not in set(overlay["feature"].unique())
    assert set(overlay["feature"].unique()) <= {"f1", "f2"}
