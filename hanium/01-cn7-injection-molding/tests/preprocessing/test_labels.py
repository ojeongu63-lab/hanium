import pandas as pd

from preprocessing.labels import encode_labels


def test_encode_labels_maps_pass_fail_to_binary():
    df = pd.DataFrame({
        "PassOrFail": ["Y", "N", "N"],
        "Reason": [None, "가스", "미성형"],
    })

    result = encode_labels(df)

    assert result["label"].tolist() == [0, 1, 1]


def test_encode_labels_reclassifies_initial_startup_defect_as_normal():
    df = pd.DataFrame({
        "PassOrFail": ["N", "N"],
        "Reason": ["초기허용불량", "가스"],
    })

    result = encode_labels(df)

    assert result["label"].tolist() == [0, 1]
