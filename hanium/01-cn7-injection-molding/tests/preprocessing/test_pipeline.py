import json

import pandas as pd

from preprocessing.columns import FEATURE_COLUMNS
from preprocessing.pipeline import run_pipeline


def _make_unlabeled_csv(path, n_rows):
    data = {col: [float(i % 5) for i in range(n_rows)] for col in FEATURE_COLUMNS}
    data["_id"] = [f"u{i}" for i in range(n_rows)]
    data["TimeStamp"] = [f"2020-01-01 00:{i // 60:02d}:{i % 60:02d}" for i in range(n_rows)]
    pd.DataFrame(data).to_csv(path, index=False)


def _make_labeled_csv(path):
    n_normal = 20
    n_rows = n_normal + 2
    data = {col: [float(i % 5) for i in range(n_rows)] for col in FEATURE_COLUMNS}
    data["_id"] = [f"l{i}" for i in range(n_rows)]
    data["TimeStamp"] = [f"2020-02-01 00:{i // 60:02d}:{i % 60:02d}" for i in range(n_rows)]
    data["PassOrFail"] = ["Y"] * n_normal + ["N", "N"]
    data["Reason"] = [None] * n_normal + ["가스", "미성형"]
    pd.DataFrame(data).to_csv(path, index=False)


def test_run_pipeline_creates_expected_output_files(tmp_path):
    labeled_path = tmp_path / "labeled.csv"
    unlabeled_path = tmp_path / "unlabeled.csv"
    output_dir = tmp_path / "processed"

    _make_labeled_csv(labeled_path)
    _make_unlabeled_csv(unlabeled_path, n_rows=200)

    manifest = run_pipeline(str(labeled_path), str(unlabeled_path), str(output_dir))

    for name in ["train.csv", "eval.csv", "scaler.json", "removed_outliers.csv", "manifest.json"]:
        assert (output_dir / name).exists()

    train_df = pd.read_csv(output_dir / "train.csv")
    assert list(train_df.columns) == FEATURE_COLUMNS

    eval_df = pd.read_csv(output_dir / "eval.csv")
    assert "label" in eval_df.columns
    assert "TimeStamp" in eval_df.columns
    assert eval_df["label"].sum() == 2

    scaler_dict = json.loads((output_dir / "scaler.json").read_text())
    assert set(scaler_dict.keys()) == set(FEATURE_COLUMNS)

    assert manifest["eval_label_counts"]["gas"] == 1
    assert manifest["eval_label_counts"]["misform"] == 1
