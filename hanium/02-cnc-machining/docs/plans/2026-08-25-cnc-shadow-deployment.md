# 섀도우 배포 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 게이트(G1+G2) 통과 후 즉시 100% 승격하던 걸, candidate를 실트래픽에 병행 투입(응답엔 영향 없음)해 미래 데이터로 한 번 더 검증하는 섀도우 단계로 바꾼다.

**Architecture:** 서빙 앱이 candidate 모델을 추가로 로드해 `/predict`마다 champion과 나란히(응답에는 안 나타나게) 추론하고 결과를 새 SQLite 테이블에 기록한다. 워커가 라벨 20건이 새로 도착할 때마다 이 기록을 라벨과 매칭해 champion·candidate 정확도를 재비교하고, candidate가 이기면 그제서야 정식 승격한다.

**Tech Stack:** 기존과 동일 — FastAPI, MLflow, SQLite, httpx2(워커 HTTP 클라이언트).

**Spec:** `02-cnc-machining/docs/specs/2026-08-25-cnc-shadow-deployment-design.md`

## Global Constraints

- `GATE_SAMPLE_SIZE = 20` — 섀도우 종료 조건(새 라벨 도착 건수)도 이 값 재사용. `monitoring/drift_worker.py`에 이미 정의됨.
- `COOLDOWN_DAYS = 5`, `CONSECUTIVE_K = 3` — 트리거 로직 불변.
- `REGISTERED_MODEL_NAME`, `CHAMPION_ALIAS` (`src/lstm_ae/tracking.py`) — 불변.
- 섀도우 DB 경로: `data/monitoring/shadow.db`.
- 시나리오 검증은 `--days 55`로 재현(라벨 지연 7일 + 섀도우 20건 확보 기간).
- `src/retraining/{trigger,runner,promotion}.py`, `src/monitoring/labels.py`, `monitoring/simulate_timeline.py`, `monitoring/sweep_drift_constants.py`는 변경하지 않는다.

---

### Task 1: 섀도우 예측 로그

**Files:**
- Create: `02-cnc-machining/src/monitoring/shadow_log.py`
- Test: `02-cnc-machining/tests/monitoring/test_shadow_log.py`

**Interfaces:**
- Produces: `record_shadow_prediction(batch_id: str, champion_label: str, candidate_label: str, db_path: Path) -> None`, `get_shadow_predictions(batch_ids: list[str], db_path: Path) -> dict[str, dict]` (반환값: `{batch_id: {"champion_label": str, "candidate_label": str}}`, 없는 batch_id는 결과 dict에서 빠짐)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
from monitoring.shadow_log import get_shadow_predictions, record_shadow_prediction


def test_record_and_retrieve_roundtrip(tmp_path):
    db_path = tmp_path / "shadow.db"
    record_shadow_prediction("day37_0", "good", "bad", db_path)
    record_shadow_prediction("day37_1", "bad", "bad", db_path)

    result = get_shadow_predictions(["day37_0", "day37_1"], db_path)

    assert result["day37_0"] == {"champion_label": "good", "candidate_label": "bad"}
    assert result["day37_1"] == {"champion_label": "bad", "candidate_label": "bad"}


def test_missing_batch_ids_are_omitted(tmp_path):
    db_path = tmp_path / "shadow.db"
    record_shadow_prediction("day37_0", "good", "good", db_path)

    result = get_shadow_predictions(["day37_0", "day37_1"], db_path)

    assert list(result.keys()) == ["day37_0"]


def test_missing_db_returns_empty_dict(tmp_path):
    assert get_shadow_predictions(["day37_0"], tmp_path / "nope.db") == {}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/monitoring/test_shadow_log.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'monitoring.shadow_log'`

- [ ] **Step 3: 구현**

```python
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    champion_label TEXT NOT NULL,
    candidate_label TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def record_shadow_prediction(
    batch_id: str, champion_label: str, candidate_label: str, db_path: Path
) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            "INSERT INTO shadow_predictions "
            "(batch_id, champion_label, candidate_label, timestamp) VALUES (?, ?, ?, ?)",
            (batch_id, champion_label, candidate_label, datetime.now(timezone.utc).isoformat()),
        )
    conn.close()


def get_shadow_predictions(batch_ids: list[str], db_path: Path) -> dict[str, dict]:
    if not Path(db_path).exists() or not batch_ids:
        return {}
    conn = _connect(db_path)
    placeholders = ",".join("?" for _ in batch_ids)
    rows = conn.execute(
        f"SELECT batch_id, champion_label, candidate_label FROM shadow_predictions "
        f"WHERE batch_id IN ({placeholders})",
        batch_ids,
    ).fetchall()
    conn.close()
    return {r[0]: {"champion_label": r[1], "candidate_label": r[2]} for r in rows}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/monitoring/test_shadow_log.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
cd /home/sure/project/hanium
git add 02-cnc-machining/src/monitoring/shadow_log.py 02-cnc-machining/tests/monitoring/test_shadow_log.py
git commit -m "feat: add shadow prediction log store"
```

---

### Task 2: 섀도우 최종 판정 함수

**Files:**
- Modify: `02-cnc-machining/src/retraining/gate.py`
- Test: `02-cnc-machining/tests/retraining/test_gate.py`

**Interfaces:**
- Consumes: 없음 (순수 함수, 다른 모듈에 의존 안 함)
- Produces: `evaluate_shadow(candidate_accuracy: float, champion_accuracy: float) -> dict` — 반환 키: `"decision"` (`"promoted"` | `"rejected"`), `"accuracy_delta"` (float)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/retraining/test_gate.py` 파일 상단 import에 `evaluate_shadow` 추가하고, 파일 끝에 추가:

```python
def test_shadow_promotes_when_candidate_better():
    verdict = evaluate_shadow(candidate_accuracy=0.9, champion_accuracy=0.7)
    assert verdict["decision"] == "promoted"
    assert verdict["accuracy_delta"] == pytest.approx(0.2)


def test_shadow_rejects_when_candidate_not_better():
    verdict = evaluate_shadow(candidate_accuracy=0.7, champion_accuracy=0.7)
    assert verdict["decision"] == "rejected"
    assert verdict["accuracy_delta"] == pytest.approx(0.0)
```

(파일 상단에 이미 `import pytest`가 없다면 추가할 것 — `pytest.approx`를 쓰므로.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/retraining/test_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_shadow'`

- [ ] **Step 3: 구현**

`src/retraining/gate.py` 파일 끝에 추가:

```python
def evaluate_shadow(candidate_accuracy: float, champion_accuracy: float) -> dict:
    """섀도우 기간 종료 후 최종 판정. G1은 트리거 시점에 이미 확인했고
    원본 eval은 시간이 지나도 안 바뀌므로 재확인하지 않는다 — G2에
    해당하는 정확도 비교만 반복한다."""
    promoted = candidate_accuracy > champion_accuracy
    return {
        "decision": "promoted" if promoted else "rejected",
        "accuracy_delta": candidate_accuracy - champion_accuracy,
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/retraining/test_gate.py -v`
Expected: PASS (전체 gate 테스트 포함 통과)

- [ ] **Step 5: 커밋**

```bash
cd /home/sure/project/hanium
git add 02-cnc-machining/src/retraining/gate.py 02-cnc-machining/tests/retraining/test_gate.py
git commit -m "feat: add shadow verdict evaluation"
```

---

### Task 3: 서빙 앱 — candidate 로더 + 섀도우 시작/종료 엔드포인트

**Files:**
- Modify: `02-cnc-machining/src/serving/app.py`
- Test: `02-cnc-machining/tests/serving/test_app.py`

**Interfaces:**
- Consumes: 없음 (기존 `ModelState`, `load_companion_json`, `load_rag_state` 재사용)
- Produces: `load_candidate_state(version: str) -> ModelState`, module global `_shadow_state: ModelState | None`, `SHADOW_DB: Path` 상수, `POST /start-shadow`(body `{"model_version": str}`, 반환 `{"status": "shadow_started", "candidate_version": str}`), `POST /stop-shadow`(반환 `{"status": "shadow_stopped"}`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/serving/test_app.py` 파일 끝에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/serving/test_app.py -v -k shadow`
Expected: FAIL — `404` (엔드포인트 없음) 또는 `AttributeError: module 'serving.app' has no attribute 'load_candidate_state'`

- [ ] **Step 3: 구현**

`src/serving/app.py`에서 `DB_PATH = ROOT / "data" / "monitoring" / "requests.db"` 아래에 추가:

```python
SHADOW_DB = ROOT / "data" / "monitoring" / "shadow.db"
```

`_state: ModelState | None = None` 아래에 추가:

```python
_shadow_state: ModelState | None = None
```

기존 `load_model_state()` 함수 전체를 아래로 교체(로직은 동일, `_build_model_state` 헬퍼로 분리하고 `load_candidate_state`를 추가하는 리팩터링):

```python
def _build_model_state(mv, run, model, include_rag: bool) -> ModelState:
    thresholds = {
        method: run.data.metrics[f"{method}_threshold"] for method in ["mean", "max", "p95"]
    }
    window_size = int(run.data.params["window_size"])
    scaler_dict = load_companion_json(
        mv.run_id, "scaler.json", ROOT / "data" / "processed" / "scaler.json"
    )
    feature_baseline = load_companion_json(
        mv.run_id, "feature_baseline.json", ROOT / "data" / "model" / "feature_baseline.json"
    )
    rag_corpus, rag_index, openai_client = (
        load_rag_state() if include_rag else (None, None, None)
    )
    return ModelState(
        model=model,
        scaler_dict=scaler_dict,
        thresholds=thresholds,
        window_size=window_size,
        model_version=str(mv.version),
        mlflow_run_id=mv.run_id,
        feature_baseline=feature_baseline,
        rag_corpus=rag_corpus,
        rag_index=rag_index,
        openai_client=openai_client,
    )


def load_model_state() -> ModelState:
    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    run = client.get_run(mv.run_id)
    return _build_model_state(mv, run, model, include_rag=True)


def load_candidate_state(version: str) -> ModelState:
    """섀도우 후보를 champion alias 없이 특정 버전으로 직접 로드한다 —
    승격 전이라 champion alias는 아직 candidate를 안 가리킨다. RAG는
    섀도우 추론(로그 기록용)에는 필요 없어 안 채운다."""
    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version(REGISTERED_MODEL_NAME, version)
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    run = client.get_run(mv.run_id)
    return _build_model_state(mv, run, model, include_rag=False)
```

`/reload-model` 엔드포인트 바로 아래(파일 끝)에 추가:

```python
@app.post("/start-shadow")
def start_shadow(payload: dict) -> dict:
    global _shadow_state
    _shadow_state = load_candidate_state(payload["model_version"])
    return {"status": "shadow_started", "candidate_version": _shadow_state.model_version}


@app.post("/stop-shadow")
def stop_shadow() -> dict:
    global _shadow_state
    _shadow_state = None
    return {"status": "shadow_stopped"}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/serving/test_app.py -v`
Expected: PASS (전체 test_app.py 통과, 기존 테스트도 회귀 없어야 함)

- [ ] **Step 5: 커밋**

```bash
cd /home/sure/project/hanium
git add 02-cnc-machining/src/serving/app.py 02-cnc-machining/tests/serving/test_app.py
git commit -m "feat: add candidate model loader and shadow start/stop endpoints"
```

---

### Task 4: `/predict`에 섀도우 병행 추론 추가

**Files:**
- Modify: `02-cnc-machining/src/serving/app.py`
- Test: `02-cnc-machining/tests/serving/test_app.py`

**Interfaces:**
- Consumes: Task 1의 `record_shadow_prediction`, Task 3의 `_shadow_state`/`SHADOW_DB`
- Produces: `/predict`가 `_shadow_state`가 설정돼 있을 때 `shadow.db`에 예측을 남김. 응답 바디는 변경 없음(섀도우 정보 노출 안 함). 섀도우 추론 실패는 `/predict` 응답에 영향 없음.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/serving/test_app.py` 파일 끝에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/serving/test_app.py -v -k shadow_prediction`
Expected: FAIL — `shadow.db`에 아무것도 기록 안 됨(`assert "day37_0" in recorded` 실패)

- [ ] **Step 3: 구현**

`src/serving/app.py` 상단 import에 추가:

```python
from monitoring.shadow_log import record_shadow_prediction
```

`/predict` 핸들러의 `log_request(...)` 호출 바로 다음 줄에 추가(즉 `try` 블록 안, `except ValueError` 이전):

```python
        if _shadow_state is not None:
            _record_shadow_if_possible(df, method, result["predicted_label_text"], file.filename)
```

`/predict` 함수 정의 바로 다음(파일 내 아무 위치, 예: `/predict` 함수 뒤)에 헬퍼 추가:

```python
def _record_shadow_if_possible(df, method, champion_label, filename) -> None:
    """섀도우 후보 추론이 실패해도 사용자 응답에는 영향을 주면 안 된다."""
    try:
        shadow_result = predict_experiment(
            df=df,
            model=_shadow_state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=_shadow_state.scaler_dict,
            window_size=_shadow_state.window_size,
            threshold=_shadow_state.thresholds[method],
            method=method,
            feature_baseline=_shadow_state.feature_baseline,
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
        )
        batch_id = Path(filename).stem
        record_shadow_prediction(
            batch_id, champion_label, shadow_result["predicted_label_text"], SHADOW_DB
        )
    except Exception as exc:
        print(f"섀도우 추론 실패(무시하고 계속): {exc}")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/serving/test_app.py -v`
Expected: PASS (전체 통과, 기존 테스트 회귀 없음)

- [ ] **Step 5: 전체 스위트 회귀 확인**

Run: `cd 02-cnc-machining && uv run pytest -q`
Expected: PASS, 이전보다 테스트 개수 늘어남(Task 1~4에서 추가한 테스트 반영)

- [ ] **Step 6: 커밋**

```bash
cd /home/sure/project/hanium
git add 02-cnc-machining/src/serving/app.py 02-cnc-machining/tests/serving/test_app.py
git commit -m "feat: run candidate inference alongside champion when shadow is active"
```

---

### Task 5: 워커 — 섀도우 시작·감시·승격 로직

**Files:**
- Modify: `02-cnc-machining/monitoring/drift_worker.py`

**Interfaces:**
- Consumes: Task 2의 `evaluate_shadow`, Task 1의 `get_shadow_predictions`, Task 3의 `POST /start-shadow`/`POST /stop-shadow`, 기존 `get_arrived_labels`(`src/monitoring/labels.py`), 기존 `accuracy_from_pairs`(`src/retraining/gate.py`), 기존 `swap_with_rollback`/`promote_to_champion`/`verify_serving_contract`
- Produces: `ShadowState` dataclass(필드: `candidate_version: str`, `run_id: str`, `retrain_dir: str`, `missed: int`, `start_day: int`, `labels_seen_at_start: int`), `WorkerState.shadow: ShadowState | None = None`. `tick()`이 반환하는 `action`에 새 값 추가: `"shadow_started"`, `"shadow_pending"`, `"shadow_rejected"` (기존 `"none"`/`"rejected"`/`"promoted"`는 의미 유지).

이 파일은 HTTP(`httpx2`)와 MLflow에 강하게 의존해 pytest 유닛 테스트 대상이
아니다(기존 관례 — "판정 로직은 src/retraining/에 있고 워커는 호출만
한다"). 이 태스크는 import 성공과 문법 검증으로 확인하고, 실제 동작은
Task 6의 실측으로 검증한다.

- [ ] **Step 1: `ShadowState` 추가, `WorkerState`에 필드 추가**

`monitoring/drift_worker.py`의 `@dataclass class WorkerState:` 바로 위에 추가:

```python
@dataclass
class ShadowState:
    candidate_version: str
    run_id: str
    retrain_dir: str
    missed: int                   # 트리거 시점 G1 놓친 개수 — 승격 확정 시 champion_missed 갱신용
    start_day: int
    labels_seen_at_start: int     # 섀도우 시작 시점까지 이미 도착해 있던 라벨 수
```

`WorkerState`에 필드 추가:

```python
@dataclass
class WorkerState:
    flag_history: list[bool] = field(default_factory=list)
    cooldown_remaining: int = 0
    champion_missed: int = 1
    champion_accuracy: float = 0.0
    shadow: ShadowState | None = None
```

파일 상단 import에서 `from retraining.gate import evaluate_gate` 를 아래로 교체:

```python
from retraining.gate import evaluate_gate, evaluate_shadow  # noqa: E402
```

- [ ] **Step 2: `tick()`을 섀도우 분기 포함하도록 교체**

기존 `tick()` 함수 전체를 아래로 교체:

```python
def tick(client, state: WorkerState, current_day: int, scenario: str) -> dict:
    status = client.get("/drift-status").json()
    flagged = is_drift_flagged(status)
    state.flag_history.append(flagged)
    ratio = status.get("output_drift", {}).get("ratio_to_threshold", 0.0)

    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1

    if state.shadow is not None:
        action = _check_shadow(client, state, current_day, scenario)
        return {"ratio": ratio, "flagged": flagged, "action": action}

    if not should_retrain(state.flag_history, CONSECUTIVE_K, state.cooldown_remaining):
        return {"ratio": ratio, "flagged": flagged, "action": "none"}

    print(f"  [Day {current_day}] 트리거 발동 — 재학습 시작", flush=True)
    result = run_retraining(
        timeline_dir=ROOT / "data" / "timeline" / scenario,
        labels_db=LABELS_DB,
        current_day=current_day,
        root=ROOT,
    )
    state.cooldown_remaining = COOLDOWN_DAYS

    action = _decide_and_start_shadow(client, state, result, current_day, scenario)
    return {"ratio": ratio, "flagged": flagged, "action": action}
```

- [ ] **Step 3: `_decide_and_promote`를 `_decide_and_start_shadow`로 교체**

기존 `_decide_and_promote` 함수 전체(승격 로직 포함)를 삭제하고 아래로 교체
— 게이트 통과 시 즉시 승격하던 뒷부분을 `_start_shadow` 호출로 바꾼다:

```python
def _decide_and_start_shadow(client, state, result, current_day, scenario) -> str:
    mlflow_client = MlflowClient()
    run = mlflow_client.get_run(result["run_id"])

    # 계약 확인을 파일 교체보다 앞에 둔다 — 위반이면 롤백할 것 자체가 생기지 않는다.
    missing = verify_serving_contract(run.data.metrics, run.data.params)
    if missing:
        _tag(mlflow_client, result["run_id"], scenario, current_day,
             decision="rejected", reason=f"서빙 계약 미충족: {missing}")
        print(f"  거부 — 서빙 계약 미충족: {missing}", flush=True)
        return "rejected"

    champion_accuracy, retrained_accuracy, sample_size = _gate_accuracies(
        result, current_day, scenario
    )
    verdict = evaluate_gate(
        retrained_missed=result["missed"],
        champion_missed=state.champion_missed,
        retrained_accuracy=retrained_accuracy,
        champion_accuracy=champion_accuracy,
    )
    _tag(mlflow_client, result["run_id"], scenario, current_day,
         decision=verdict["decision"], reason=verdict["reject_reason"],
         extra={"gate_g1_missed": verdict["g1_missed"],
                "gate_g2_accuracy_delta": verdict["g2_accuracy_delta"],
                "gate_g2_sample_size": sample_size})

    print(f"  게이트: G1 놓침={verdict['g1_missed']}건 (champion {state.champion_missed}건, "
          f"허용 {state.champion_missed + 1}건) / "
          f"G2 {retrained_accuracy:.2f} vs {champion_accuracy:.2f} "
          f"(표본 {sample_size}건)", flush=True)

    if verdict["decision"] == "rejected":
        print(f"  거부 — {verdict['reject_reason']}  (champion 유지, 사람 확인 필요)", flush=True)
        return "rejected"

    _start_shadow(client, state, result, current_day)
    return "shadow_started"


def _start_shadow(client, state, result, current_day) -> None:
    from monitoring.labels import get_arrived_labels

    client.post(
        "/start-shadow", json={"model_version": str(result["model_version"])}
    ).raise_for_status()
    labels_seen = len(get_arrived_labels(current_day, LABELS_DB))
    state.shadow = ShadowState(
        candidate_version=str(result["model_version"]),
        run_id=result["run_id"],
        retrain_dir=str(result["retrain_dir"]),
        missed=result["missed"],
        start_day=current_day,
        labels_seen_at_start=labels_seen,
    )
    print(
        f"  섀도우 시작 — version {result['model_version']} "
        f"(라벨 {GATE_SAMPLE_SIZE}건 도착까지 관찰)",
        flush=True,
    )
```

- [ ] **Step 4: 섀도우 종료 판정 함수 추가**

파일에서 `_gate_accuracies` 함수 바로 다음에 추가:

```python
def _check_shadow(client, state, current_day, scenario) -> str:
    """섀도우가 끝났는지 확인하고, 끝났으면 최종 판정까지 수행한다."""
    from monitoring.labels import get_arrived_labels
    from monitoring.shadow_log import get_shadow_predictions
    from retraining.gate import accuracy_from_pairs

    arrived = get_arrived_labels(current_day, LABELS_DB)
    new_labels = arrived[state.shadow.labels_seen_at_start :]
    if len(new_labels) < GATE_SAMPLE_SIZE:
        return "shadow_pending"

    sample = new_labels[:GATE_SAMPLE_SIZE]
    batch_ids = [r["batch_id"] for r in sample]
    predictions = get_shadow_predictions(batch_ids, SHADOW_DB)

    matched = [b for b in batch_ids if b in predictions]
    if len(matched) < GATE_SAMPLE_SIZE:
        # /predict 가 이 배치들을 아직 다 처리하지 못했을 수 있다 — 다음 tick에 재시도.
        return "shadow_pending"

    label_by_batch = {r["batch_id"]: r["label"] for r in sample}
    truths = [label_by_batch[b] for b in matched]
    champion_preds = [predictions[b]["champion_label"] for b in matched]
    candidate_preds = [predictions[b]["candidate_label"] for b in matched]

    champion_accuracy = accuracy_from_pairs(truths, champion_preds)
    candidate_accuracy = accuracy_from_pairs(truths, candidate_preds)
    verdict = evaluate_shadow(candidate_accuracy, champion_accuracy)

    mlflow_client = MlflowClient()
    _tag(mlflow_client, state.shadow.run_id, scenario, current_day,
         decision=f"shadow_{verdict['decision']}", reason="",
         extra={"shadow_accuracy_delta": verdict["accuracy_delta"],
                "shadow_candidate_accuracy": candidate_accuracy,
                "shadow_champion_accuracy": champion_accuracy})

    print(f"  섀도우 종료 — candidate {candidate_accuracy:.2f} vs champion "
          f"{champion_accuracy:.2f} → {verdict['decision']}", flush=True)

    if verdict["decision"] == "promoted":
        _promote_shadow(client, state)
        result_action = "promoted"
    else:
        client.post("/stop-shadow")
        print("  섀도우 거부 — champion 유지, 사람 확인 필요", flush=True)
        result_action = "shadow_rejected"

    state.shadow = None
    return result_action


def _promote_shadow(client, state) -> None:
    mlflow_client = MlflowClient()
    previous_version = mlflow_client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS
    ).version

    def _promote() -> None:
        promote_to_champion(state.shadow.candidate_version)
        client.post("/reload-model").raise_for_status()

    def _verify() -> None:
        health = client.get("/health").json()
        if health["model_version"] != state.shadow.candidate_version:
            raise RuntimeError(f"리로드 후 버전 불일치: {health['model_version']}")

    try:
        swap_with_rollback(
            state.shadow.retrain_dir, MODEL_DIR, SCALER_PATH, BACKUP_ROOT,
            promote=_promote, verify=_verify,
        )
    except Exception as exc:
        print(f"  승격 실패, 롤백 중: {exc}", flush=True)
        promote_to_champion(previous_version)
        client.post("/reload-model")
        client.post("/stop-shadow")
        raise

    client.post("/stop-shadow")
    state.champion_missed = state.shadow.missed
    print(f"  승격 완료 — version {state.shadow.candidate_version}", flush=True)
```

`SHADOW_DB` 상수를 파일 상단(`BACKUP_ROOT = ROOT / "data" / "model_backup"` 근처)에 추가:

```python
SHADOW_DB = ROOT / "data" / "monitoring" / "shadow.db"
```

- [ ] **Step 5: import 성공 및 문법 검증**

Run:
```bash
cd 02-cnc-machining
uv run python -m py_compile monitoring/drift_worker.py && echo "syntax OK"
uv run python -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'monitoring')
import drift_worker
print('ShadowState:', drift_worker.ShadowState)
print('tick:', drift_worker.tick)
print('_start_shadow:', drift_worker._start_shadow)
print('_check_shadow:', drift_worker._check_shadow)
print('_promote_shadow:', drift_worker._promote_shadow)
"
```
Expected: `syntax OK` 출력, 이어서 각 이름이 정상 출력(에러 없음)

- [ ] **Step 6: 전체 테스트 스위트 회귀 확인**

Run: `cd 02-cnc-machining && uv run pytest -q`
Expected: PASS (이 태스크는 `drift_worker.py`만 바꿨으므로 테스트 개수는 Task 4 이후와 동일해야 함)

- [ ] **Step 7: 커밋**

```bash
cd /home/sure/project/hanium
git add 02-cnc-machining/monitoring/drift_worker.py
git commit -m "feat: gate now starts a shadow instead of promoting immediately"
```

---

### Task 6: 시나리오 A 재현 (--days 55)으로 섀도우 매커니즘 검증

**Files:** 없음(코드 변경 없음, 실행 및 관찰만)

**Interfaces:**
- Consumes: Task 1~5에서 만든 모든 것, 기존 `monitoring/simulate_timeline.py`(변경 없음)

**주의 — `--days`와 `TOTAL_DAYS`가 별개다**: `simulate_timeline.py`의
`progress_for(day)`는 모듈 상수 `TOTAL_DAYS = 40`으로 정규화한다
(`(day - DRIFT_START_DAY) / (TOTAL_DAYS - DRIFT_START_DAY)`). `--days 55`로
돌리면 Day 41~55는 `progress_for`가 1.0을 넘는 값을 내며 변형 강도가
Day 40 시점보다 계속 세진다. 이건 의도적으로 코드를 안 고치는 것이다
(`simulate_timeline.py`는 Global Constraints에 따라 변경 금지) — 관찰된
값을 그대로 기록하고, 예상과 다르면 **조정해서 통과시키지 않는다**.

공유 서버이므로 시작 전 `who`/`top`으로 부하를 확인하고 전부 `nice -n 19`로
띄운다.

- [ ] **Step 1: 서버·워커·feeder를 위한 사전 정리**

`data/monitoring/labels.db`, `data/monitoring/requests.db`가 이전 시나리오
실행 잔여물을 담고 있으면 비운다(백업 후):

```bash
cd 02-cnc-machining
who; uptime
cp data/monitoring/labels.db data/monitoring/labels.db.bak-pre-shadow 2>/dev/null
cp data/monitoring/requests.db data/monitoring/requests.db.bak-pre-shadow 2>/dev/null
rm -f data/monitoring/labels.db data/monitoring/requests.db data/monitoring/shadow.db
rm -rf data/timeline/temperature
```

- [ ] **Step 2: 서버 기동**

```bash
nice -n 19 uv run uvicorn src.serving.app:app --app-dir . --host 127.0.0.1 --port 8000
```
(별도 터미널/백그라운드로 띄우고 `/health`가 200을 줄 때까지 대기)

- [ ] **Step 3: 워커 기동**

```bash
nice -n 19 uv run python monitoring/drift_worker.py temperature --base-url http://127.0.0.1:8000 --poll-interval 2
```

- [ ] **Step 4: feeder로 55일 재현**

```bash
nice -n 19 uv run python monitoring/simulate_timeline.py temperature --days 55 --serve-url http://127.0.0.1:8000
```

- [ ] **Step 5: 워커 로그에서 확인할 것**

- `action=shadow_started`가 한 번 이상 찍히는지 (게이트 통과 후 섀도우 진입)
- 그 이후 며칠간 `action=shadow_pending`이 찍히다가, 라벨 20건이 채워지면
  `action=promoted` 또는 `action=shadow_rejected`로 종결되는지
- 섀도우 진행 중에는 새로운 `트리거 발동` 로그가 안 찍히는지(중복 트리거 억제 확인)

- [ ] **Step 6: 결과를 `tasks/todo.md`에 기록**

실측된 섀도우 시작 Day, 종료 Day, 최종 판정(승격/거부), champion/candidate
정확도 값을 있는 그대로 적는다. 예상과 다르게 나와도(예: 섀도우가 거부로
끝나거나, Day 55 안에 끝나지 않으면) 그 사실 그대로 적고 원인을 분석한다
— 값을 바꿔서 원하는 결과로 맞추지 않는다.

- [ ] **Step 7: 뒷정리**

```bash
kill <워커 PID> <서버 PID>  # 잔여 프로세스 종료
```

`data/monitoring/labels.db`/`requests.db`를 Step 1의 백업에서 복원. champion이
실제로 바뀌었다면(승격됐다면) 이전 세션들의 관례대로 `promote_model.py`
+ `restore_backup()`으로 되돌릴지, 새 champion을 유지할지는 사용자에게
확인 후 결정한다 — 자동으로 되돌리지 않는다.

- [ ] **Step 8: 커밋**

```bash
cd /home/sure/project/hanium
git add tasks/todo.md
git commit -m "docs: record shadow deployment scenario A reproduction"
```

---

## 자체 검토 결과

**스펙 커버리지**: Part A(섀도우 로그) → Task 1. Part B(서빙 앱) → Task 3,4.
Part C(워커) → Task 5, `ShadowState`에 스펙에 없던 `retrain_dir`/`missed`
필드를 추가로 채움(승격 확정 시 `swap_with_rollback`과 `champion_missed`
갱신에 필수 — 계획 작성 중 발견한 스펙 누락, 이 계획이 정본). Part D(검증
방법) → Task 6, `--days 55`와 `TOTAL_DAYS` 불일치를 주의사항으로 명시.
알려진 한계(인메모리 상태) → 코드에 추가 안전장치 없이 그대로 두기로
스펙에서 이미 결정됨, 별도 태스크 불필요.

**플레이스홀더 스캔**: 없음 — 모든 스텝에 실제 코드/명령어 포함.

**타입 일관성**: `evaluate_shadow(candidate_accuracy, champion_accuracy)`
(Task 2)를 `_check_shadow`(Task 5)가 정확히 이 인자 순서로 호출.
`get_shadow_predictions`가 반환하는 `{"champion_label":.., "candidate_label":..}`
키 이름을 `_check_shadow`와 `/predict`(Task 4의 `record_shadow_prediction`
호출 인자 순서 `champion_label, candidate_label`)가 일관되게 사용.
`ShadowState` 필드명(`candidate_version`, `run_id`, `retrain_dir`, `missed`,
`start_day`, `labels_seen_at_start`)이 Task 5의 `_start_shadow`/`_check_shadow`/
`_promote_shadow` 전체에서 동일하게 쓰임.
