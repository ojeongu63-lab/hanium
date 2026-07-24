from datetime import datetime, timezone

SOURCE = "CNC 비식별화 원본데이터_1209 (train.csv + experiment_01~25.csv)"


def build_manifest(
    *,
    total_rows: int,
    train_rows: int,
    eval_rows: int,
    eval_good_rows: int,
    eval_bad_rows: int,
    train_experiment_ids: list[int],
    eval_good_experiment_ids: list[int],
    eval_bad_experiment_ids: list[int],
    feature_columns: list[str],
    dead_sensor_columns: list[str],
    metadata_excluded_columns: list[str],
) -> dict:
    return {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "total_rows": total_rows,
        "experiment_split": {
            "train": {"experiment_ids": train_experiment_ids, "rows": train_rows},
            "eval_good": {"experiment_ids": eval_good_experiment_ids, "rows": eval_good_rows},
            "eval_bad": {"experiment_ids": eval_bad_experiment_ids, "rows": eval_bad_rows},
        },
        "eval_rows": eval_rows,
        "feature_columns": feature_columns,
        "dropped_columns": {"dead_sensors": dead_sensor_columns},
        "metadata_excluded_columns": metadata_excluded_columns,
    }
