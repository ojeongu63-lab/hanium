from preprocessing.columns import (
    DEAD_SENSOR_COLUMNS,
    FEATURE_COLUMNS,
    METADATA_EXCLUDED_COLUMNS,
    select_features,
)
import pandas as pd


def test_feature_columns_has_41_entries_and_no_overlap_with_dropped():
    assert len(FEATURE_COLUMNS) == 41
    assert len(set(FEATURE_COLUMNS)) == 41
    assert set(FEATURE_COLUMNS).isdisjoint(DEAD_SENSOR_COLUMNS)
    assert set(FEATURE_COLUMNS).isdisjoint(METADATA_EXCLUDED_COLUMNS)


def test_dead_and_metadata_excluded_columns_match_spec():
    assert DEAD_SENSOR_COLUMNS == [
        "Z_CurrentFeedback",
        "Z_DCBusVoltage",
        "Z_OutputCurrent",
        "Z_OutputVoltage",
    ]
    assert METADATA_EXCLUDED_COLUMNS == [
        "M_CURRENT_PROGRAM_NUMBER",
        "M_sequence_number",
        "Machining_Process",
    ]


def test_select_features_keeps_only_feature_columns_in_order():
    df = pd.DataFrame({col: [1.0] for col in FEATURE_COLUMNS + ["extra_col"]})

    result = select_features(df)

    assert list(result.columns) == FEATURE_COLUMNS
