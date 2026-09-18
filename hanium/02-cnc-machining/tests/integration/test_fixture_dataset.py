"""合成データセットがKAMP原本と同じ形状だから前処理・学習・feeder がコード変更なしに走るかどうか。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib 모드라 형제 모듈이 경로에 없다

from fixture_dataset import EXPERIMENT_SUBDIR, ROWS, write_dataset  # noqa: E402
from preprocessing.columns import DEAD_SENSOR_COLUMNS, FEATURE_COLUMNS  # noqa: E402
from preprocessing.split import (  # noqa: E402
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)


def _read(dataset_dir: Path, experiment_id: int) -> pd.DataFrame:
    return pd.read_csv(dataset_dir / EXPERIMENT_SUBDIR / f"experiment_{experiment_id:02d}.csv")


def test_index_lists_the_split_ids_with_consistent_labels(tmp_path):
    write_dataset(tmp_path)

    index = pd.read_csv(tmp_path / "train.csv").set_index("No")

    assert sorted(index.index) == sorted(
        TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS + EVAL_BAD_EXPERIMENT_IDS
    )
    for experiment_id in TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS:
        assert index.loc[experiment_id, "passed_visual_inspection"] == "yes"
    for experiment_id in EVAL_BAD_EXPERIMENT_IDS:
        assert index.loc[experiment_id, "passed_visual_inspection"] == "no"
    assert (index["machining_finalized"] == "yes").all()


def test_experiment_files_have_kamp_shape(tmp_path):
    write_dataset(tmp_path)

    df = _read(tmp_path, 1)

    assert len(df) == ROWS
    assert len(df.columns) == 48
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    assert set(DEAD_SENSOR_COLUMNS) <= set(df.columns)
    assert {"Machining_Process", "M_sequence_number", "M_CURRENT_PROGRAM_NUMBER"} <= set(df.columns)
    assert not df.isna().any().any()


def test_generation_is_deterministic(tmp_path):
    write_dataset(tmp_path / "a", seed=0)
    write_dataset(tmp_path / "b", seed=0)

    for name in ["train.csv", f"{EXPERIMENT_SUBDIR}/experiment_17.csv"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_noisy_train_and_bad_experiments_differ_from_baseline(tmp_path):
    write_dataset(tmp_path)

    base, noisy, bad = _read(tmp_path, 1), _read(tmp_path, 17), _read(tmp_path, 4)

    # 17 은 임계값을 끌어올리는 잡음 큰 train 실험(스펙 §2), 4 는 eval_bad
    assert noisy["X_ActualPosition"].diff().std() > 2 * base["X_ActualPosition"].diff().std()
    assert bad["S_OutputCurrent"].mean() > 2 * base["S_OutputCurrent"].mean()
