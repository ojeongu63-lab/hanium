import io

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from lstm_ae.model import LSTMAutoencoder
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
from serving.app import ModelState, app, get_model_state


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _fake_state(window_size: int = 6, threshold: float = 1.0) -> ModelState:
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=len(FEATURE_COLUMNS), hidden_size=4, latent_dim=2)
    scaler_dict = {col: {"mean": 0.0, "std": 1.0} for col in FEATURE_COLUMNS}
    feature_baseline = {
        "mean": {col: 0.5 for col in FEATURE_COLUMNS},
        "std": {col: 0.1 for col in FEATURE_COLUMNS},
    }
    return ModelState(
        model=model,
        scaler_dict=scaler_dict,
        thresholds={"mean": threshold, "max": threshold, "p95": threshold},
        window_size=window_size,
        model_version="1",
        mlflow_run_id="fake-run-id",
        feature_baseline=feature_baseline,
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


def test_health_returns_503_when_not_loaded():
    def _raise():
        raise HTTPException(status_code=503, detail="not loaded")

    app.dependency_overrides[get_model_state] = _raise
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503


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
    assert {c["feature"] for c in body["feature_contributions"]} == (
        set(FEATURE_COLUMNS) - set(SETUP_CONSTANT_COLUMNS)
    )
    assert all("z_score" in c for c in body["feature_contributions"])


def test_predict_returns_400_for_too_short_experiment():
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(3)), "text/csv")},
    )

    assert response.status_code == 400
    assert "needs at least" in response.json()["detail"]


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


def test_predict_returns_400_for_empty_file():
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )

    assert response.status_code == 400


def test_predict_response_includes_guide_field():
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "guide" in body
    if body["predicted_label_text"] == "good":
        assert body["guide"]["cause_estimate"] == "이상 없음"
    else:
        assert body["guide"] is None  # _fake_state는 rag_corpus 등을 안 채움


def test_predict_logs_request_for_drift_monitoring(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    )

    from monitoring.logging import get_recent_requests
    recent = get_recent_requests(10, tmp_path / "requests.db")
    assert len(recent) == 1
    assert set(recent[0]["feature_means"].keys()) == set(FEATURE_COLUMNS)


def test_drift_status_reports_insufficient_data_when_log_empty(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.get("/drift-status")

    assert response.status_code == 200
    body = response.json()
    assert body["sufficient_data"] is False
    assert "checked_at" in body


def test_drift_status_flags_after_enough_requests(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    monkeypatch.setattr(app_module, "DRIFT_WINDOW_SIZE", 2)
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    for _ in range(2):
        client.post(
            "/predict",
            files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
        )

    response = client.get("/drift-status")

    assert response.status_code == 200
    body = response.json()
    assert body["sufficient_data"] is True
    assert body["n_requests_logged"] == 2


def test_reload_model_swaps_state_on_success(monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "_state", _fake_state())
    new_state = _fake_state()
    new_state.model_version = "7"
    monkeypatch.setattr(app_module, "load_model_state", lambda: new_state)
    client = TestClient(app)

    response = client.post("/reload-model")

    assert response.status_code == 200
    assert response.json() == {"status": "reloaded", "model_version": "7"}
    assert app_module._state is new_state


def test_reload_model_keeps_previous_state_on_failure(monkeypatch):
    import serving.app as app_module

    previous = _fake_state()
    monkeypatch.setattr(app_module, "_state", previous)

    def _boom():
        raise RuntimeError("MLflow 접속 실패")

    monkeypatch.setattr(app_module, "load_model_state", _boom)
    client = TestClient(app)

    response = client.post("/reload-model")

    assert response.status_code == 500
    assert app_module._state is previous  # 교체 실패가 서빙 중단으로 번지지 않는다


def test_companion_json_falls_back_to_local_path(tmp_path, monkeypatch):
    import serving.app as app_module

    fallback = tmp_path / "scaler.json"
    fallback.write_text('{"from": "fallback"}')

    def _fail_download(**kwargs):
        raise RuntimeError("아티팩트 없음")

    monkeypatch.setattr(app_module.mlflow.artifacts, "download_artifacts", _fail_download)

    result = app_module.load_companion_json("run-without-artifact", "scaler.json", fallback)

    assert result == {"from": "fallback"}


def test_companion_json_prefers_mlflow_artifact(tmp_path, monkeypatch):
    import serving.app as app_module

    fallback = tmp_path / "scaler.json"
    fallback.write_text('{"from": "fallback"}')
    artifact = tmp_path / "downloaded.json"
    artifact.write_text('{"from": "artifact"}')

    monkeypatch.setattr(
        app_module.mlflow.artifacts, "download_artifacts", lambda **kwargs: str(artifact)
    )

    result = app_module.load_companion_json("run-with-artifact", "scaler.json", fallback)

    assert result == {"from": "artifact"}
