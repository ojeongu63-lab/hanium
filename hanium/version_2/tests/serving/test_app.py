import io

import numpy as np
import pandas as pd
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from lstm_ae.model import LSTMAutoencoder
from preprocessing.columns import FEATURE_COLUMNS
from serving.app import ModelState, app, get_model_state


def _fake_state(window_size: int = 6, threshold: float = 1.0) -> ModelState:
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=len(FEATURE_COLUMNS), hidden_size=4, latent_dim=2)
    scaler_dict = {col: {"mean": 0.0, "std": 1.0} for col in FEATURE_COLUMNS}
    return ModelState(
        model=model,
        scaler_dict=scaler_dict,
        thresholds={"mean": threshold, "max": threshold, "p95": threshold},
        window_size=window_size,
        model_version="1",
        mlflow_run_id="fake-run-id",
    )


def _raw_csv_bytes(rows: int) -> bytes:
    data = {col: np.random.randn(rows).astype(np.float32) for col in FEATURE_COLUMNS}
    return pd.DataFrame(data).to_csv(index=False).encode()


def test_health_returns_model_version_when_loaded():
    app.dependency_overrides[get_model_state] = _fake_state
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model_version": "1", "mlflow_run_id": "fake-run-id",
    }
    app.dependency_overrides.clear()


def test_health_returns_503_when_not_loaded():
    def _raise():
        raise HTTPException(status_code=503, detail="not loaded")

    app.dependency_overrides[get_model_state] = _raise
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    app.dependency_overrides.clear()


def test_predict_returns_prediction_for_valid_csv():
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "mean"
    assert body["predicted_label"] in (0, 1)
    assert body["model_version"] == "1"
    assert body["mlflow_run_id"] == "fake-run-id"
    app.dependency_overrides.clear()


def test_predict_returns_400_for_too_short_experiment():
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(3)), "text/csv")},
    )

    assert response.status_code == 400
    assert "needs at least" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_predict_returns_400_for_missing_columns():
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    csv_bytes = pd.DataFrame({"only_one_column": [1.0] * 20}).to_csv(index=False).encode()
    response = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 400
    assert "missing required columns" in response.json()["detail"]
    app.dependency_overrides.clear()
