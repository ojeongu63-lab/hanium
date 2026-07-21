import pandas as pd

from preprocessing.columns import (
    DEAD_SENSOR_COLUMNS,
    DISABLED_SENSOR_COLUMNS,
    FEATURE_COLUMNS,
    select_features,
)


def test_feature_columns_has_24_entries():
    assert len(FEATURE_COLUMNS) == 24


def test_dropped_sensor_columns_are_not_in_feature_columns():
    for col in DEAD_SENSOR_COLUMNS + DISABLED_SENSOR_COLUMNS:
        assert col not in FEATURE_COLUMNS


def test_select_features_returns_only_whitelisted_columns_in_order():
    data = {col: [0.0] for col in FEATURE_COLUMNS}
    data["_id"] = ["x"]
    data["PassOrFail"] = ["Y"]
    data["Mold_Temperature_1"] = [0.0]
    df = pd.DataFrame(data)

    result = select_features(df)

    assert list(result.columns) == FEATURE_COLUMNS
