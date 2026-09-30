import json

import pandas as pd
import pytest

from preprocessing.columns import FEATURE_COLUMNS
from preprocessing.pipeline import run_pipeline


def _make_experiment_csv(path, n_rows, machining_process="Prep"):
    data = {col: [float(i % 5) + 1 for i in range(n_rows)] for col in FEATURE_COLUMNS}
    data["Machining_Process"] = [machining_process] * n_rows
    data["M_sequence_number"] = list(range(n_rows))
    data["M_CURRENT_PROGRAM_NUMBER"] = [0] * n_rows
    pd.DataFrame(data).to_csv(path, index=False)


def _make_index_csv(path):
    pd.DataFrame({
        "No": [1, 2, 3, 4],
        "material": ["aluminum"] * 4,
        "feedrate": [3, 6, 3, 6],
        "clamp_pressure": [4, 4, 3, 3],
        "tool_condition": ["unworn", "unworn", "unworn", "worn"],
        "machining_finalized": ["yes", "yes", "yes", "yes"],
        "passed_visual_inspection": ["yes", "yes", "yes", "no"],
    }).to_csv(path, index=False)


def test_run_pipeline_creates_expected_output_files(tmp_path):
    experiment_dir = tmp_path / "experiments"
    experiment_dir.mkdir()
    index_path = tmp_path / "train.csv"
    output_dir = tmp_path / "processed"

    _make_index_csv(index_path)
    _make_experiment_csv(experiment_dir / "experiment_01.csv", n_rows=10)
    _make_experiment_csv(experiment_dir / "experiment_02.csv", n_rows=10)
    _make_experiment_csv(experiment_dir / "experiment_03.csv", n_rows=8)
    _make_experiment_csv(experiment_dir / "experiment_04.csv", n_rows=6, machining_process="end")

    manifest = run_pipeline(
        experiment_index_path=str(index_path),
        experiment_dir=str(experiment_dir),
        output_dir=str(output_dir),
        train_experiment_ids=[1, 2],
        eval_good_experiment_ids=[3],
        eval_bad_experiment_ids=[4],
    )

    for name in ["train.csv", "eval.csv", "scaler.json", "manifest.json"]:
        assert (output_dir / name).exists()

    train_df = pd.read_csv(output_dir / "train.csv")
    assert list(train_df.columns) == FEATURE_COLUMNS + ["experiment_id"]
    assert len(train_df) == 20
    assert set(train_df["experiment_id"]) == {1, 2}

    eval_df = pd.read_csv(output_dir / "eval.csv")
    assert len(eval_df) == 14
    assert set(eval_df["experiment_id"]) == {3, 4}
    assert eval_df.loc[eval_df["experiment_id"] == 3, "label"].unique().tolist() == [0]
    assert eval_df.loc[eval_df["experiment_id"] == 4, "label"].unique().tolist() == [1]
    assert set(eval_df.loc[eval_df["experiment_id"] == 4, "Machining_Process"]) == {"End"}

    scaler_dict = json.loads((output_dir / "scaler.json").read_text())
    assert set(scaler_dict.keys()) == set(FEATURE_COLUMNS)

    assert manifest["experiment_split"]["train"]["rows"] == 20
    assert manifest["eval_rows"] == 14
    assert manifest["experiment_split"]["eval_bad"]["rows"] == 6


def test_run_pipeline_train_never_touches_eval_experiments(tmp_path):
    experiment_dir = tmp_path / "experiments"
    experiment_dir.mkdir()
    index_path = tmp_path / "train.csv"
    output_dir = tmp_path / "processed"

    _make_index_csv(index_path)
    for i, n in zip([1, 2, 3, 4], [10, 10, 8, 6]):
        _make_experiment_csv(experiment_dir / f"experiment_{i:02d}.csv", n_rows=n)

    run_pipeline(
        experiment_index_path=str(index_path),
        experiment_dir=str(experiment_dir),
        output_dir=str(output_dir),
        train_experiment_ids=[1, 2],
        eval_good_experiment_ids=[3],
        eval_bad_experiment_ids=[4],
    )

    train_df = pd.read_csv(output_dir / "train.csv")
    eval_df = pd.read_csv(output_dir / "eval.csv")

    assert set(train_df["experiment_id"]).isdisjoint(set(eval_df["experiment_id"]))


def test_run_pipeline_rejects_bad_labeled_experiment_in_train(tmp_path):
    experiment_dir = tmp_path / "experiments"
    experiment_dir.mkdir()
    index_path = tmp_path / "train.csv"
    output_dir = tmp_path / "processed"

    _make_index_csv(index_path)
    for i, n in zip([1, 2, 3, 4], [10, 10, 8, 6]):
        _make_experiment_csv(experiment_dir / f"experiment_{i:02d}.csv", n_rows=n)

    with pytest.raises(ValueError):
        run_pipeline(
            experiment_index_path=str(index_path),
            experiment_dir=str(experiment_dir),
            output_dir=str(output_dir),
            train_experiment_ids=[1, 2, 4],  # experiment 4 is labeled bad
            eval_good_experiment_ids=[3],
            eval_bad_experiment_ids=[],
        )
