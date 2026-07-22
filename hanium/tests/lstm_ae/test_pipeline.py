import json

import numpy as np
import pandas as pd
import torch

from lstm_ae.pipeline import run_lstm_pipeline

FEATURE_COLUMNS = ["f0", "f1", "f2"]


def _make_train_csv(path, n_rows):
    data = {col: np.random.randn(n_rows).astype(np.float32) for col in FEATURE_COLUMNS}
    pd.DataFrame(data).to_csv(path, index=False)


def _make_eval_csv(path, n_rows):
    data = {col: np.random.randn(n_rows).astype(np.float32) for col in FEATURE_COLUMNS}
    data["PassOrFail"] = ["Y"] * (n_rows - 2) + ["N", "N"]
    data["Reason"] = [None] * (n_rows - 2) + ["가스", "미성형"]
    data["TimeStamp"] = [f"2020-01-01 00:00:{i:02d}" for i in range(n_rows)]
    data["label"] = [0] * (n_rows - 2) + [1, 1]
    pd.DataFrame(data).to_csv(path, index=False)


def test_run_lstm_pipeline_creates_expected_output_files(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    output_dir = tmp_path / "model"

    _make_train_csv(train_path, n_rows=60)
    _make_eval_csv(eval_path, n_rows=30)

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
        "train_reconstruction_errors.csv",
        "eval_reconstruction_errors.csv",
        "threshold.json",
        "evaluation_report.json",
    ]:
        assert (output_dir / name).exists()

    train_errors = pd.read_csv(output_dir / "train_reconstruction_errors.csv")
    assert "shot_error" in train_errors.columns
    assert all(f"error_{c}" in train_errors.columns for c in FEATURE_COLUMNS)
    assert len(train_errors) == 60 // 6 * 6  # non-overlapping, remainder dropped

    eval_errors = pd.read_csv(output_dir / "eval_reconstruction_errors.csv")
    assert len(eval_errors) == 30  # overlapping covers every eval shot
    assert "label" in eval_errors.columns
    assert "TimeStamp" in eval_errors.columns

    threshold = json.loads((output_dir / "threshold.json").read_text())
    assert "threshold" in threshold

    report = json.loads((output_dir / "evaluation_report.json").read_text())
    assert set(report.keys()) >= {"precision", "recall", "tp", "fp", "fn", "tn"}

    assert "final_train_loss" in summary
    assert summary["train_shots"] == 60 // 6 * 6
    assert summary["eval_shots"] == 30
