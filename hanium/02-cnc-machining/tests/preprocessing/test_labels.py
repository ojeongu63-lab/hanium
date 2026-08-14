import pandas as pd

from preprocessing.labels import add_labels


def test_finalized_and_passed_is_good():
    df = pd.DataFrame({
        "machining_finalized": ["yes"],
        "passed_visual_inspection": ["yes"],
    })

    result = add_labels(df)

    assert result["label"].tolist() == [0]


def test_not_finalized_is_bad():
    df = pd.DataFrame({
        "machining_finalized": ["no"],
        "passed_visual_inspection": [None],
    })

    result = add_labels(df)

    assert result["label"].tolist() == [1]


def test_finalized_but_failed_visual_inspection_is_bad():
    df = pd.DataFrame({
        "machining_finalized": ["yes"],
        "passed_visual_inspection": ["no"],
    })

    result = add_labels(df)

    assert result["label"].tolist() == [1]


def test_mixed_rows_labeled_independently():
    df = pd.DataFrame({
        "machining_finalized": ["yes", "no", "yes"],
        "passed_visual_inspection": ["yes", None, "no"],
    })

    result = add_labels(df)

    assert result["label"].tolist() == [0, 1, 1]
