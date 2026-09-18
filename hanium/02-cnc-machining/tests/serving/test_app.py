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


@pytest.fixture(autouse=True)
def _isolate_monitoring_state(tmp_path, monkeypatch):
    """실제 data/monitoring DB 에 쓰지 않게 테스트마다 격리한다. /start-shadow 는 전역을 직접 바꾸므로
    되돌려 두지 않으면 뒤 테스트의 /predict 가 섀도우를 실제 shadow.db 에 기록한다.
    테스트가 같은 값을 다시 monkeypatch 하면 그쪽이 이긴다."""
    import serving.app as app_module

    monkeypatch.setattr(app_module, "_shadow_state", None)
    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    monkeypatch.setattr(app_module, "SHADOW_DB", tmp_path / "shadow.db")


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


def test_predict_returns_prediction_for_valid_csv(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
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


def test_predict_response_includes_guide_field(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
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


def test_predict_response_includes_fault_field(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    )

    body = response.json()
    assert "fault" in body
    if body["predicted_label_text"] == "good":
        assert body["fault"]["verdict"] == "none"
    else:
        assert body["fault"] is None  # _fake_state는 rag_corpus를 안 채움
    assert {"predicted_label", "predicted_label_text", "score", "threshold", "method",
            "feature_contributions", "model_version", "mlflow_run_id", "guide"} <= set(body)


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


def test_start_shadow_loads_candidate_state(monkeypatch):
    import serving.app as app_module

    candidate = _fake_state()
    candidate.model_version = "99"
    monkeypatch.setattr(app_module, "load_candidate_state", lambda version: candidate)
    client = TestClient(app)

    response = client.post("/start-shadow", json={"model_version": "99"})

    assert response.status_code == 200
    assert response.json() == {"status": "shadow_started", "candidate_version": "99"}
    assert app_module._shadow_state is candidate


def test_stop_shadow_clears_state(monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "_shadow_state", _fake_state())
    client = TestClient(app)

    response = client.post("/stop-shadow")

    assert response.status_code == 200
    assert response.json() == {"status": "shadow_stopped"}
    assert app_module._shadow_state is None


def test_predict_logs_shadow_prediction_when_shadow_active(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    monkeypatch.setattr(app_module, "SHADOW_DB", tmp_path / "shadow.db")
    monkeypatch.setattr(app_module, "_shadow_state", _fake_state(window_size=6))
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("day37_0.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    )

    assert response.status_code == 200
    assert "candidate_label" not in response.json()

    from monitoring.shadow_log import get_shadow_predictions
    recorded = get_shadow_predictions(["day37_0"], tmp_path / "shadow.db")
    assert "day37_0" in recorded
    assert recorded["day37_0"]["champion_label"] == response.json()["predicted_label_text"]


def test_predict_succeeds_even_if_shadow_inference_fails(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    monkeypatch.setattr(app_module, "SHADOW_DB", tmp_path / "shadow.db")

    class _BrokenShadow:
        model = None
        scaler_dict = {}
        window_size = 6
        thresholds = {"mean": 1.0}
        feature_baseline = {}

    monkeypatch.setattr(app_module, "_shadow_state", _BrokenShadow())
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("day37_0.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    )

    assert response.status_code == 200


def test_predict_response_includes_versions_object(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    body = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    ).json()

    assert body["versions"] == {"playbook": None, "corpus": None, "chat_model": None}

    state = _fake_state(window_size=6)
    state.rag_versions = {"playbook": "abcd1234", "built_at": "2026-09-03T12:05+09:00", "chunks": 42}
    state.openai_client = object()  # rag_corpus가 없으므로 실제 호출은 일어나지 않는다
    app.dependency_overrides[get_model_state] = lambda: state
    np.random.seed(0)
    body = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    ).json()

    assert body["versions"] == {
        "playbook": "abcd1234", "corpus": "2026-09-03T12:05+09:00", "chat_model": "gpt-4o-mini",
    }


def test_demo_routes_serve_page_and_inputs(tmp_path, monkeypatch):
    import serving.app as app_module

    index = tmp_path / "index.html"
    index.write_text('<html><script id="demo-data" type="application/json">{}</script></html>')
    csv = tmp_path / "tool_wear.csv"
    csv.write_text("a,b\n1,2\n")
    monkeypatch.setattr(app_module, "DEMO_INDEX", index)
    monkeypatch.setattr(app_module, "DEMO_INPUTS", {"tool_wear": csv, "missing": tmp_path / "nope.csv"})
    client = TestClient(app)

    page = client.get("/demo")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "demo-data" in page.text

    got = client.get("/demo/inputs/tool_wear")
    assert got.status_code == 200 and got.text.startswith("a,b")
    assert client.get("/demo/inputs/unknown").status_code == 404
    assert client.get("/demo/inputs/missing").status_code == 404

    monkeypatch.setattr(app_module, "DEMO_INDEX", tmp_path / "absent.html")
    assert client.get("/demo").status_code == 404


def test_demo_timeline_route_generates_batch_csv(monkeypatch):
    import serving.app as app_module

    calls = []

    def fake_generate(day, index, scenario):
        calls.append((day, index, scenario))
        return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    monkeypatch.setattr(app_module, "_generate_timeline_batch", fake_generate)
    client = TestClient(app)

    got = client.get("/demo/timeline/tool_wear/21/0")
    assert got.status_code == 200
    assert got.headers["content-type"].startswith("text/csv")
    assert got.text.splitlines()[0] == "a,b"
    assert calls == [(21, 0, "tool_wear")]

    assert client.get("/demo/timeline/nope/21/0").status_code == 404
    assert client.get("/demo/timeline/tool_wear/0/0").status_code == 404
    assert client.get("/demo/timeline/tool_wear/21/5").status_code == 404


def test_demo_timeline_route_404_when_dataset_missing(monkeypatch):
    import serving.app as app_module

    def missing(day, index, scenario):
        raise FileNotFoundError("experiment_01.csv")

    monkeypatch.setattr(app_module, "_generate_timeline_batch", missing)
    client = TestClient(app)

    got = client.get("/demo/timeline/temperature/3/1")
    assert got.status_code == 404
    assert "데이터셋" in got.json()["detail"]


def test_predict_does_not_block_other_requests(tmp_path, monkeypatch):
    """추론이 도는 동안 /health 가 응답해야 한다. 추론(과 RAG 의 LLM 호출)은 동기 코드라
    이벤트 루프 안에서 실행되면 서버 전체가 멈춘다(09-15 실측: predict 4.4초 동안 /health 3.9초 대기).
    타이밍이 아니라 순서로 검증한다 — /health 가 답한 뒤에야 추론을 풀어 준다."""
    import threading

    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    monkeypatch.setattr(app_module, "_state", None)
    monkeypatch.setattr(app_module, "_shadow_state", None)  # 앞 테스트(/start-shadow)가 남긴 전역을 격리
    monkeypatch.setattr(app_module, "load_model_state", lambda: _fake_state(window_size=6))
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)

    entered = threading.Event()   # predict 가 추론에 들어갔다
    release = threading.Event()   # /health 가 답한 뒤에 풀어 준다
    released_in_time = []
    real_predict = app_module.predict_experiment

    def stuck_predict(*args, **kwargs):
        entered.set()
        released_in_time.append(release.wait(timeout=5))
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(app_module, "predict_experiment", stuck_predict)

    responses = {}
    with TestClient(app) as client:  # 두 요청이 같은 이벤트 루프를 타야 한다
        worker = threading.Thread(
            target=lambda: responses.update(
                predict=client.post(
                    "/predict",
                    files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
                )
            )
        )
        worker.start()
        assert entered.wait(timeout=5), "predict 가 추론에 진입하지 못함"
        responses["health"] = client.get("/health")
        release.set()
        worker.join(timeout=10)

    assert responses["health"].status_code == 200
    assert responses["predict"].status_code == 200
    assert released_in_time == [True], "추론이 도는 동안 /health 가 막혔다(이벤트 루프 블로킹)"
