import pandas as pd

from preprocessing.cleaning import normalize_machining_process


def test_lowercase_end_normalized_to_capitalized():
    df = pd.DataFrame({"Machining_Process": ["Prep", "end", "End", "Layer 1 Up"]})

    result = normalize_machining_process(df)

    assert result["Machining_Process"].tolist() == ["Prep", "End", "End", "Layer 1 Up"]


def test_does_not_mutate_input_dataframe():
    df = pd.DataFrame({"Machining_Process": ["end"]})

    normalize_machining_process(df)

    assert df["Machining_Process"].tolist() == ["end"]
