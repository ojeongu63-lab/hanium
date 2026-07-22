from preprocessing.manifest import build_manifest


def test_build_manifest_records_row_counts_and_columns():
    manifest = build_manifest(
        raw_labeled_rows=6736,
        labeled_rows_after_dedup=3974,
        raw_unlabeled_rows=35239,
        train_rows_after_cleaning=34886,
        removed_outlier_rows=353,
        contamination=0.01,
        feature_columns=["a", "b"],
        dead_sensor_columns=["dead1"],
        disabled_sensor_columns=["disabled1"],
        eval_label_counts={"normal": 3956, "gas": 13, "misform": 5},
    )

    assert manifest["labeled"]["duplicates_removed"] == 2762
    assert manifest["unlabeled"]["outliers_removed"] == 353
    assert manifest["feature_columns"] == ["a", "b"]
    assert manifest["dropped_columns"] == {
        "dead_sensors": ["dead1"],
        "disabled_after_eval_window": ["disabled1"],
    }
    assert manifest["eval_label_counts"] == {"normal": 3956, "gas": 13, "misform": 5}
    assert "processed_at" in manifest
    assert "filter_condition" in manifest
