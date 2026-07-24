from preprocessing.manifest import build_manifest


def test_build_manifest_records_split_and_columns():
    manifest = build_manifest(
        total_rows=32048,
        train_rows=14654,
        eval_rows=17394,
        eval_good_rows=7991,
        eval_bad_rows=9403,
        train_experiment_ids=[1, 2, 3, 11, 13, 14, 15, 17],
        eval_good_experiment_ids=[12, 18, 22, 24, 25],
        eval_bad_experiment_ids=[4, 5, 6, 7, 8, 9, 10, 16, 19, 20, 21, 23],
        feature_columns=["a", "b"],
        dead_sensor_columns=["dead1"],
        metadata_excluded_columns=["meta1"],
    )

    assert manifest["total_rows"] == 32048
    assert manifest["experiment_split"]["train"] == {
        "experiment_ids": [1, 2, 3, 11, 13, 14, 15, 17],
        "rows": 14654,
    }
    assert manifest["experiment_split"]["eval_good"] == {
        "experiment_ids": [12, 18, 22, 24, 25],
        "rows": 7991,
    }
    assert manifest["experiment_split"]["eval_bad"] == {
        "experiment_ids": [4, 5, 6, 7, 8, 9, 10, 16, 19, 20, 21, 23],
        "rows": 9403,
    }
    assert manifest["eval_rows"] == 17394
    assert manifest["feature_columns"] == ["a", "b"]
    assert manifest["dropped_columns"] == {"dead_sensors": ["dead1"]}
    assert manifest["metadata_excluded_columns"] == ["meta1"]
    assert "processed_at" in manifest
    assert "source" in manifest
