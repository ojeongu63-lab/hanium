import json

from config import DATA_ROOT, DATASET_DIR, EXPERIMENT_DIR
from preprocessing.pipeline import run_pipeline
from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)


def main() -> None:
    manifest = run_pipeline(
        experiment_index_path=str(DATASET_DIR / "train.csv"),
        experiment_dir=str(EXPERIMENT_DIR),
        output_dir=str(DATA_ROOT / "processed"),
        train_experiment_ids=TRAIN_EXPERIMENT_IDS,
        eval_good_experiment_ids=EVAL_GOOD_EXPERIMENT_IDS,
        eval_bad_experiment_ids=EVAL_BAD_EXPERIMENT_IDS,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
