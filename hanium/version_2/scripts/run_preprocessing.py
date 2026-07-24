import json
from pathlib import Path

from preprocessing.pipeline import run_pipeline
from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209"


def main() -> None:
    manifest = run_pipeline(
        experiment_index_path=str(DATASET_DIR / "train.csv"),
        experiment_dir=str(DATASET_DIR / "CNC Virtual Data set _v2"),
        output_dir=str(ROOT / "data" / "processed"),
        train_experiment_ids=TRAIN_EXPERIMENT_IDS,
        eval_good_experiment_ids=EVAL_GOOD_EXPERIMENT_IDS,
        eval_bad_experiment_ids=EVAL_BAD_EXPERIMENT_IDS,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
