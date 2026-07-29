import numpy as np
import pandas as pd
import pytest
import torch

from lstm_ae.model import LSTMAutoencoder
from serving.inference import (
    predict_experiment,
    rank_feature_contributions,
    scale_features,
    score_to_label,
    validate_columns,
)

FEATURE_COLUMNS = ["f0", "f1", "f2"]


def _scaler_dict():
    return {col: {"mean": 0.0, "std": 1.0} for col in FEATURE_COLUMNS}


def _feature_baseline():
    return {
        "mean": {col: 0.5 for col in FEATURE_COLUMNS},
        "std": {col: 0.1 for col in FEATURE_COLUMNS},
    }


def _raw_df(rows: int) -> pd.DataFrame:
    data = {col: np.random.randn(rows).astype(np.float32) for col in FEATURE_COLUMNS}
    return pd.DataFrame(data)


def test_validate_columns_reports_missing():
    df = pd.DataFrame({"f0": [1.0], "f1": [2.0]})
    assert validate_columns(df, FEATURE_COLUMNS) == ["f2"]


def test_validate_columns_empty_when_all_present():
    assert validate_columns(_raw_df(5), FEATURE_COLUMNS) == []


def test_scale_features_applies_standardization():
    df = pd.DataFrame({"f0": [10.0], "f1": [0.0], "f2": [0.0]})
    scaler_dict = {
        "f0": {"mean": 5.0, "std": 2.0},
        "f1": {"mean": 0.0, "std": 1.0},
        "f2": {"mean": 0.0, "std": 1.0},
    }
    scaled = scale_features(df, FEATURE_COLUMNS, scaler_dict)
    assert scaled.loc[0, "f0"] == pytest.approx(2.5)


def test_score_to_label_bad_when_above_threshold():
    assert score_to_label(score=5.0, threshold=1.0) == (1, "bad")


def test_score_to_label_good_when_at_or_below_threshold():
    assert score_to_label(score=1.0, threshold=1.0) == (0, "good")
    assert score_to_label(score=0.5, threshold=1.0) == (0, "good")


def test_predict_experiment_raises_on_missing_columns():
    df = pd.DataFrame({"f0": [1.0] * 10, "f1": [1.0] * 10})
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)
    with pytest.raises(ValueError, match="missing required columns"):
        predict_experiment(
            df=df, model=model, feature_columns=FEATURE_COLUMNS,
            scaler_dict=_scaler_dict(), window_size=6, threshold=1.0, method="mean",
            feature_baseline=_feature_baseline(),
        )


def test_predict_experiment_raises_on_too_short_experiment():
    df = _raw_df(5)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)
    with pytest.raises(ValueError, match="needs at least"):
        predict_experiment(
            df=df, model=model, feature_columns=FEATURE_COLUMNS,
            scaler_dict=_scaler_dict(), window_size=6, threshold=1.0, method="mean",
            feature_baseline=_feature_baseline(),
        )


def test_predict_experiment_returns_expected_shape():
    torch.manual_seed(0)
    np.random.seed(0)
    df = _raw_df(20)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)

    result = predict_experiment(
        df=df, model=model, feature_columns=FEATURE_COLUMNS,
        scaler_dict=_scaler_dict(), window_size=6, threshold=1.0, method="mean",
        feature_baseline=_feature_baseline(),
    )

    assert set(result.keys()) == {
        "predicted_label", "predicted_label_text", "score", "threshold", "method",
        "feature_contributions",
    }
    assert result["predicted_label"] in (0, 1)
    assert result["predicted_label_text"] in ("good", "bad")
    assert result["method"] == "mean"
    assert result["threshold"] == 1.0
    assert (result["predicted_label"] == 1) == (result["score"] > 1.0)

    contributions = result["feature_contributions"]
    assert {c["feature"] for c in contributions} == set(FEATURE_COLUMNS)
    z_scores = [c["z_score"] for c in contributions]
    assert z_scores == sorted(z_scores, reverse=True)


def test_predict_experiment_forwards_exclude_from_ranking():
    torch.manual_seed(0)
    np.random.seed(0)
    df = _raw_df(20)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)

    result = predict_experiment(
        df=df, model=model, feature_columns=FEATURE_COLUMNS,
        scaler_dict=_scaler_dict(), window_size=6, threshold=1.0, method="mean",
        feature_baseline=_feature_baseline(), exclude_from_ranking=["f1"],
    )

    assert {c["feature"] for c in result["feature_contributions"]} == {"f0", "f2"}


def test_rank_feature_contributions_sorts_descending_by_z_score():
    # baseline mean=0.5,std=0.1 for all three -> z-scores: f0=-4, f1=0, f2=-2
    result = rank_feature_contributions(
        feature_errors=np.array([0.1, 0.5, 0.3]),
        feature_columns=["f0", "f1", "f2"],
        feature_baseline={
            "mean": {"f0": 0.5, "f1": 0.5, "f2": 0.5},
            "std": {"f0": 0.1, "f1": 0.1, "f2": 0.1},
        },
    )

    assert [r["feature"] for r in result] == ["f1", "f2", "f0"]
    assert result[0]["z_score"] == pytest.approx(0.0)
    assert result[-1]["z_score"] == pytest.approx(-4.0)
    assert result[0]["error"] == pytest.approx(0.5)


def test_rank_feature_contributions_excludes_given_columns():
    result = rank_feature_contributions(
        feature_errors=np.array([0.1, 0.5, 0.3]),
        feature_columns=["f0", "f1", "f2"],
        feature_baseline={
            "mean": {"f0": 0.5, "f1": 0.5, "f2": 0.5},
            "std": {"f0": 0.1, "f1": 0.1, "f2": 0.1},
        },
        exclude_from_ranking=["f1"],
    )

    assert {r["feature"] for r in result} == {"f0", "f2"}
