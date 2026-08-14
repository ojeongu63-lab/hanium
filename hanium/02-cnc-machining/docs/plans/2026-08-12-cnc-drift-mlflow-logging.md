# CNC 드리프트 지표 MLflow 기록 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `GET /drift-status`가 계산한 드리프트 값을 champion 모델의 MLflow 학습 run에 metric으로 같이 기록해, MLflow UI에서 시간에 따른 드리프트 추이를 볼 수 있게 한다.

**Architecture:** `src/monitoring/mlflow_logging.py`에 `log_drift_metrics()` 순수 함수(주입 가능한 client 파라미터로 테스트 용이) 하나를 추가하고, `app.py`의 `/drift-status` 라우트에서 응답을 만들기 전에 한 줄 호출한다.

**Tech Stack:** 기존 `mlflow` 의존성 그대로, 새 패키지 없음.

## Global Constraints

- `sufficient_data=false`면 아무 것도 기록하지 않는다.
- MLflow 기록 실패는 내부에서 예외를 삼키고 로그만 남긴다 — `/drift-status` 응답은 항상 정상 반환.
- metric 이름: `drift_output_ratio_to_threshold`, `drift_input_flagged_count`, `drift_avg_score_recent` (기존 `mean_precision` 등과 충돌 없음).
- `mlflow.start_run()`으로 재개하지 않는다 — `MlflowClient().log_metric(run_id, ...)`를 직접 호출(스파이크로 완료된 run에도 잘 동작함을 이미 확인).
- 새 pytest 테스트 작성(`src/` 정식 코드).

---

## Task 1: `src/monitoring/mlflow_logging.py`

**Files:**
- Create: `02-cnc-machining/src/monitoring/mlflow_logging.py`
- Create: `02-cnc-machining/tests/monitoring/test_mlflow_logging.py`

**Interfaces:**
- Produces: `monitoring.mlflow_logging.log_drift_metrics(status: dict, run_id: str, client=None) -> None`
  (`status`는 `monitoring.drift.compute_drift_status()`가 만드는 dict — 최소
  `sufficient_data`, `output_drift.ratio_to_threshold`, `output_drift.avg_score_recent`,
  `input_drift.flagged_features` 키 필요)

- [ ] **Step 1: 실패하는 테스트 작성**

`02-cnc-machining/tests/monitoring/test_mlflow_logging.py`:
```python
from monitoring.mlflow_logging import log_drift_metrics


class _FakeMlflowClient:
    def __init__(self, raise_on_call: bool = False):
        self.logged = []
        self._raise_on_call = raise_on_call

    def log_metric(self, run_id, key, value):
        if self._raise_on_call:
            raise RuntimeError("mlflow 저장소에 연결할 수 없음")
        self.logged.append((run_id, key, value))


_SUFFICIENT_STATUS = {
    "sufficient_data": True,
    "output_drift": {"ratio_to_threshold": 1.2, "avg_score_recent": 0.9},
    "input_drift": {"flagged_features": [{"feature": "f0", "avg_scaled_mean": 3.0}]},
}


def test_log_drift_metrics_skips_when_insufficient_data():
    client = _FakeMlflowClient()

    log_drift_metrics({"sufficient_data": False}, "run123", client=client)

    assert client.logged == []


def test_log_drift_metrics_logs_three_metrics_when_sufficient():
    client = _FakeMlflowClient()

    log_drift_metrics(_SUFFICIENT_STATUS, "run123", client=client)

    logged_keys = {key for _, key, _ in client.logged}
    assert logged_keys == {
        "drift_output_ratio_to_threshold",
        "drift_input_flagged_count",
        "drift_avg_score_recent",
    }
    assert ("run123", "drift_input_flagged_count", 1) in client.logged
    assert ("run123", "drift_output_ratio_to_threshold", 1.2) in client.logged
    assert ("run123", "drift_avg_score_recent", 0.9) in client.logged


def test_log_drift_metrics_swallows_exceptions():
    client = _FakeMlflowClient(raise_on_call=True)

    # 예외가 밖으로 안 나가야 한다 - 호출 자체가 실패 없이 끝나면 통과
    log_drift_metrics(_SUFFICIENT_STATUS, "run123", client=client)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/monitoring/test_mlflow_logging.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: `mlflow_logging.py` 구현**

`02-cnc-machining/src/monitoring/mlflow_logging.py`:
```python
from mlflow.tracking import MlflowClient


def log_drift_metrics(status: dict, run_id: str, client=None) -> None:
    if not status["sufficient_data"]:
        return
    client = client or MlflowClient()
    try:
        client.log_metric(
            run_id,
            "drift_output_ratio_to_threshold",
            status["output_drift"]["ratio_to_threshold"],
        )
        client.log_metric(
            run_id,
            "drift_input_flagged_count",
            len(status["input_drift"]["flagged_features"]),
        )
        client.log_metric(
            run_id,
            "drift_avg_score_recent",
            status["output_drift"]["avg_score_recent"],
        )
    except Exception as exc:
        print(f"드리프트 metric MLflow 기록 실패: {exc}")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/monitoring/test_mlflow_logging.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add 02-cnc-machining/src/monitoring/mlflow_logging.py 02-cnc-machining/tests/monitoring/test_mlflow_logging.py
git commit -m "Add MLflow drift-metric logging with graceful failure handling"
```

## Task 2: `app.py` 통합 + 전체 검증

**Files:**
- Modify: `02-cnc-machining/src/serving/app.py`

**Interfaces:**
- Consumes: Task 1의 `log_drift_metrics(status, run_id, client=None)`

- [ ] **Step 1: `app.py` import 추가**

`from monitoring.drift import compute_drift_status` 줄을:
```python
from monitoring.drift import compute_drift_status
```
다음으로 교체:
```python
from monitoring.drift import compute_drift_status
from monitoring.mlflow_logging import log_drift_metrics
```

- [ ] **Step 2: `/drift-status` 라우트에 로깅 추가**

```python
@app.get("/drift-status")
def drift_status(state: ModelState = Depends(get_model_state)) -> dict:
    recent = get_recent_requests(DRIFT_WINDOW_SIZE, DB_PATH)
    status = compute_drift_status(
        recent, threshold=state.thresholds["mean"], window_size=DRIFT_WINDOW_SIZE
    )
    return {**status, "checked_at": datetime.now(timezone.utc).isoformat()}
```
다음으로 교체:
```python
@app.get("/drift-status")
def drift_status(state: ModelState = Depends(get_model_state)) -> dict:
    recent = get_recent_requests(DRIFT_WINDOW_SIZE, DB_PATH)
    status = compute_drift_status(
        recent, threshold=state.thresholds["mean"], window_size=DRIFT_WINDOW_SIZE
    )
    log_drift_metrics(status, state.mlflow_run_id)
    return {**status, "checked_at": datetime.now(timezone.utc).isoformat()}
```

- [ ] **Step 3: 문법 확인**

Run: `cd 02-cnc-machining && uv run python -m py_compile src/serving/app.py`
Expected: 에러 없이 종료

- [ ] **Step 4: 기존 드리프트 테스트가 여전히 통과하는지 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/serving/test_app.py -v -k drift`
Expected: PASS(3개) — `_fake_state()`의 `mlflow_run_id="fake-run-id"`로 실제
MLflow 저장소에 없는 run에 기록을 시도하면서 예외가 나겠지만,
`log_drift_metrics`가 내부에서 삼키므로 라우트 자체는 그대로 200 반환.

- [ ] **Step 5: 전체 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest -q`
Expected: 기존 94개 + 신규 3개 = 97개 전부 통과

- [ ] **Step 6: 실제 champion run에 실제로 기록되는지 수동 확인**

Run:
```bash
cd 02-cnc-machining
who && top -bn1 | head -6
nice -n 19 uv run uvicorn serving.app:app --port 8899 &
sleep 5
for i in $(seq 1 10); do
  curl -s -X POST "http://127.0.0.1:8899/predict" \
    -F "file=@synthetic/scenarios/tool_wear.csv" > /dev/null
done
curl -s "http://127.0.0.1:8899/drift-status" | python3 -m json.tool | head -5
kill %1
uv run python3 -c "
import sys; sys.path.insert(0, 'src')
from lstm_ae.tracking import configure_tracking
from mlflow.tracking import MlflowClient
configure_tracking()
client = MlflowClient()
run = client.get_run('56635df9337b486999f5ce135e1da466')
drift_metrics = {k: v for k, v in run.data.metrics.items() if k.startswith('drift_')}
print('champion run의 drift_ metric:', drift_metrics)
"
```
Expected: `drift_output_ratio_to_threshold`, `drift_input_flagged_count`,
`drift_avg_score_recent` 3개가 champion run(`56635df9...`)의 metrics에 실제로
찍혀있음(스파이크 때 남긴 `drift_spike_test=0.42`도 여전히 같이 보일 것 —
무해하니 그대로 둠).

- [ ] **Step 7: Commit**

```bash
git add 02-cnc-machining/src/serving/app.py
git commit -m "Log drift metrics to the champion MLflow run on every /drift-status check"
```

---

## Self-Review 완료 사항

- 스펙 커버리지: `log_drift_metrics()` 함수(Task 1), `/drift-status` 통합(Task 2),
  실패 시 예외 삼키기(Task 1 테스트 + Task 2 Step 4), 실제 MLflow 기록 확인(Task 2
  Step 6) 전부 매핑됨.
- 플레이스홀더 없음.
- 타입/시그니처 일관성: `log_drift_metrics(status, run_id, client=None)`이
  Task 1 정의와 Task 2의 `app.py` 호출부에서 동일.
