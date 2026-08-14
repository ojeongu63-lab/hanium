import pandas as pd

from preprocessing.dedup import remove_exact_duplicates


def test_removes_fully_identical_rows():
    df = pd.DataFrame({
        "_id": ["a", "b", "a"],
        "value": [1, 2, 1],
    })

    result = remove_exact_duplicates(df)

    assert len(result) == 2
    assert result["_id"].tolist() == ["a", "b"]


def test_keeps_rows_with_same_id_but_different_values():
    df = pd.DataFrame({
        "_id": ["a", "a"],
        "value": [1, 2],
    })

    result = remove_exact_duplicates(df)

    assert len(result) == 2
