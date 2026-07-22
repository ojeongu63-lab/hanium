import pandas as pd
import pytest

from preprocessing.scaling import fit_scaler, scaler_to_dict, transform_features


def test_fit_scaler_and_transform_standardizes_train():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})

    scaler = fit_scaler(df, ["a", "b"])
    transformed = transform_features(df, ["a", "b"], scaler)

    assert transformed["a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_transform_features_applies_train_scaler_without_refitting():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    other = pd.DataFrame({"a": [10.0]})

    scaler = fit_scaler(train, ["a"])
    transformed_other = transform_features(other, ["a"], scaler)

    expected = (10.0 - train["a"].mean()) / train["a"].std(ddof=0)
    assert transformed_other["a"].iloc[0] == pytest.approx(expected)


def test_scaler_to_dict_returns_mean_and_std_per_column():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    scaler = fit_scaler(df, ["a"])

    result = scaler_to_dict(scaler, ["a"])

    assert result["a"]["mean"] == pytest.approx(2.0)
    assert result["a"]["std"] == pytest.approx(df["a"].std(ddof=0))
