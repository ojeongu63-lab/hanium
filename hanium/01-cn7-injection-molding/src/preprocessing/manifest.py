from datetime import datetime, timezone

FILTER_CONDITION = "PART_NAME LIKE 'CN7%' AND EQUIP_NAME == '650톤-우진2호기'"


def build_manifest(
    *,
    raw_labeled_rows: int,
    labeled_rows_after_dedup: int,
    raw_unlabeled_rows: int,
    unlabeled_rows_after_plasticizing_filter: int,
    train_rows_after_cleaning: int,
    removed_outlier_rows: int,
    contamination: float,
    feature_columns: list[str],
    dead_sensor_columns: list[str],
    disabled_sensor_columns: list[str],
    eval_label_counts: dict[str, int],
) -> dict:
    return {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "filter_condition": FILTER_CONDITION,
        "labeled": {
            "raw_rows": raw_labeled_rows,
            "rows_after_dedup": labeled_rows_after_dedup,
            "duplicates_removed": raw_labeled_rows - labeled_rows_after_dedup,
        },
        "unlabeled": {
            "raw_rows": raw_unlabeled_rows,
            "non_plasticizing_removed": raw_unlabeled_rows - unlabeled_rows_after_plasticizing_filter,
            "rows_after_plasticizing_filter": unlabeled_rows_after_plasticizing_filter,
            "rows_after_self_cleaning": train_rows_after_cleaning,
            "outliers_removed": removed_outlier_rows,
            "contamination": contamination,
        },
        "feature_columns": feature_columns,
        "dropped_columns": {
            "dead_sensors": dead_sensor_columns,
            "disabled_after_eval_window": disabled_sensor_columns,
        },
        "eval_label_counts": eval_label_counts,
    }
