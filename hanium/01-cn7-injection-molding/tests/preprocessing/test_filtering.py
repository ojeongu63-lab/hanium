import pandas as pd

from preprocessing.filtering import remove_non_plasticizing_shots


def test_removes_rows_with_zero_screw_rpm():
    df = pd.DataFrame({
        "Max_Screw_RPM": [30.7, 0.0, 15.2, 0.0],
        "value": [1, 2, 3, 4],
    })

    result = remove_non_plasticizing_shots(df)

    assert len(result) == 2
    assert result["value"].tolist() == [1, 3]


def test_keeps_all_rows_when_none_are_zero():
    df = pd.DataFrame({
        "Max_Screw_RPM": [30.7, 15.2],
        "value": [1, 2],
    })

    result = remove_non_plasticizing_shots(df)

    assert len(result) == 2
