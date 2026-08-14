# CNC 드리프트 모니터링 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/predict` 요청을 누적 기록하고, 최근 요청들의 입력 피처 분포·출력 점수를 train 기준 아티팩트와 비교해 드리프트를 감지하는 `GET /drift-status` 엔드포인트를 추가한다.

**Architecture:** SQLite(표준 라이브러리)로 요청을 누적하는 `logging.py`, 그 누적값과 기존 train 아티팩트를 비교하는 순수함수 `drift.py`를 `src/monitoring/`에 만들고, `serving/app.py`가 `/predict` 끝에서 로그를 남기고 `/drift-status`로 조회 가능하게 한다. 검증은 합성 진폭 시퀀스로 실제 앱(`TestClient`)에 반복 요청을 흘려보내며 확인한다.

**Tech Stack:** Python (uv run), `sqlite3`(표준 라이브러리, 새 의존성 없음), 기존 FastAPI/pytest.

## Global Constraints

- 새 의존성 추가 없음(`sqlite3`는 표준 라이브러리).
- `predict_experiment()`(`serving/inference.py`)의 반환 스키마는 수정하지 않는다 — 기존 클라이언트 호환성 유지. 피처 평균은 `app.py`에서 `scale_features()`를 한 번 더 호출해 별도로 구한다.
- 입력 드리프트 판정 기준: 최근 `window_size`(기본 10)개 요청의 피처별 스케일링된 평균의 절댓값이 **2.0**을 초과.
- 출력 드리프트 판정 기준: 최근 `window_size`개 요청의 `score` 평균이 `threshold`의 **0.8배**를 초과.
- 자동 재학습/재승격/알림 발송은 하지 않는다 — 조회 가능한 엔드포인트까지만.
- `data/monitoring/`은 `.gitignore` 대상(`data/` 전체 관례).
- `src/`에 들어가는 정식 코드라 pytest 단위테스트를 작성한다(RAG와 동일 관례). `monitoring/simulate_drift.py`(검증 스크립트)는 `src/` 밖, 테스트 없음(`loocv`/`synthetic`과 동일 관례).

---

## File Structure

- Create: `02-cnc-machining/src/monitoring/__init__.py`, `logging.py`, `drift.py`
- Create: `02-cnc-machining/tests/monitoring/test_logging.py`, `test_drift.py`
- Modify: `02-cnc-machining/src/serving/app.py`
- Modify: `02-cnc-machining/tests/serving/test_app.py`
- Create: `02-cnc-machining/monitoring/simulate_drift.py`
- Create: `02-cnc-machining/monitoring/.gitignore`(비어있음 방지용 — 실제로는 `data/monitoring/`이 산출물 위치라 이 폴더 자체엔 불필요, 아래 참고)

## Task 1: `src/monitoring/logging.py` — 요청 로그 누적

**Files:**
- Create: `02-cnc-machining/src/monitoring/__init__.py`(빈 파일)
- Create: `02-cnc-machining/src/monitoring/logging.py`
- Create: `02-cnc-machining/tests/monitoring/test_logging.py`

**Interfaces:**
- Produces: `monitoring.logging.log_request(feature_means: dict[str, float], score: float, predicted_label_text: str, db_path: Path) -> None`,
  `monitoring.logging.get_recent_requests(n: int, db_path: Path) -> list[dict]`
  (반환 각 원소: `{"timestamp": str, "feature_means": dict, "score": float, "predicted_label_text": str}`, 최신순 정렬)

**주의**: `02-cnc-machining/tests/monitoring/`에는 `__init__.py`를 만들지 않는다 — 이전
LOOCV 작업에서 `tests/rag/__init__.py`를 만들었다가 pytest의 `--import-mode=importlib`가
`rag` 패키지를 `src/rag` 대신 `tests/rag`로 잘못 resolve하는 버그를 겪었다
(다른 `tests/*` 하위 디렉토리도 전부 `__init__.py` 없음 — 그 관례를 따른다).

- [ ] **Step 1: `__init__.py` 생성**

`02-cnc-machining/src/monitoring/__init__.py`: 빈 파일.

- [ ] **Step 2: 실패하는 테스트 작성**

`02-cnc-machining/tests/monitoring/test_logging.py`:
```python
from monitoring.logging import get_recent_requests, log_request


def test_log_and_retrieve_roundtrip(tmp_path):
    db_path = tmp_path / "requests.db"
    log_request({"f0": 0.1, "f1": -0.2}, score=0.5, predicted_label_text="good", db_path=db_path)
    log_request({"f0": 0.3, "f1": -0.1}, score=0.9, predicted_label_text="bad", db_path=db_path)

    recent = get_recent_requests(n=10, db_path=db_path)

    assert len(recent) == 2
    assert recent[0]["score"] == 0.9  # 최신이 먼저
    assert recent[0]["predicted_label_text"] == "bad"
    assert recent[1]["feature_means"] == {"f0": 0.1, "f1": -0.2}


def test_get_recent_requests_respects_limit(tmp_path):
    db_path = tmp_path / "requests.db"
    for i in range(5):
        log_request({"f0": float(i)}, score=float(i), predicted_label_text="good", db_path=db_path)

    recent = get_recent_requests(n=3, db_path=db_path)

    assert len(recent) == 3
    assert recent[0]["score"] == 4.0


def test_get_recent_requests_empty_db_returns_empty_list(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    assert get_recent_requests(n=10, db_path=db_path) == []
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/monitoring/test_logging.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'monitoring.logging'`)

- [ ] **Step 4: `logging.py` 구현**

`02-cnc-machining/src/monitoring/logging.py`:
```python
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predict_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            feature_means_json TEXT NOT NULL,
            score REAL NOT NULL,
            predicted_label_text TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_request(
    feature_means: dict[str, float],
    score: float,
    predicted_label_text: str,
    db_path: Path,
) -> None:
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO predict_log (timestamp, feature_means_json, score, predicted_label_text) "
        "VALUES (?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(feature_means),
            score,
            predicted_label_text,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_requests(n: int, db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, feature_means_json, score, predicted_label_text "
        "FROM predict_log ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [
        {
            "timestamp": row["timestamp"],
            "feature_means": json.loads(row["feature_means_json"]),
            "score": row["score"],
            "predicted_label_text": row["predicted_label_text"],
        }
        for row in rows
    ]
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/monitoring/test_logging.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add 02-cnc-machining/src/monitoring/__init__.py 02-cnc-machining/src/monitoring/logging.py \
        02-cnc-machining/tests/monitoring/test_logging.py
git commit -m "Add SQLite-backed request logging for drift monitoring"
```

## Task 2: `src/monitoring/drift.py` — 드리프트 계산

**Files:**
- Create: `02-cnc-machining/src/monitoring/drift.py`
- Create: `02-cnc-machining/tests/monitoring/test_drift.py`

**Interfaces:**
- Consumes: Task 1의 `get_recent_requests()`가 만드는 형태의 dict 리스트(`feature_means`, `score`, `predicted_label_text`)
- Produces: `monitoring.drift.compute_drift_status(recent_requests: list[dict], threshold: float, window_size: int = 10) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`02-cnc-machining/tests/monitoring/test_drift.py`:
```python
from monitoring.drift import compute_drift_status


def _request(feature_means: dict, score: float, label: str = "good") -> dict:
    return {"feature_means": feature_means, "score": score, "predicted_label_text": label}


def test_insufficient_data_returns_early():
    recent = [_request({"f0": 0.0}, 0.5) for _ in range(3)]

    status = compute_drift_status(recent, threshold=1.0, window_size=10)

    assert status["sufficient_data"] is False
    assert status["n_requests_logged"] == 3


def test_no_drift_when_values_near_baseline():
    recent = [_request({"f0": 0.1, "f1": -0.1}, 0.3) for _ in range(10)]

    status = compute_drift_status(recent, threshold=1.0, window_size=10)

    assert status["sufficient_data"] is True
    assert status["input_drift"]["flagged_features"] == []
    assert status["output_drift"]["flagged"] is False


def test_input_drift_flags_feature_far_from_baseline():
    recent = [_request({"f0": 3.5, "f1": 0.0}, 0.3) for _ in range(10)]

    status = compute_drift_status(recent, threshold=1.0, window_size=10)

    flagged = status["input_drift"]["flagged_features"]
    assert len(flagged) == 1
    assert flagged[0]["feature"] == "f0"


def test_output_drift_flags_when_score_near_threshold():
    recent = [_request({"f0": 0.0}, 0.9) for _ in range(10)]

    status = compute_drift_status(recent, threshold=1.0, window_size=10)

    assert status["output_drift"]["flagged"] is True
    assert status["output_drift"]["ratio_to_threshold"] == 0.9
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/monitoring/test_drift.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: `drift.py` 구현**

`02-cnc-machining/src/monitoring/drift.py`:
```python
INPUT_DRIFT_Z_THRESHOLD = 2.0
OUTPUT_DRIFT_RATIO_THRESHOLD = 0.8


def compute_drift_status(
    recent_requests: list[dict], threshold: float, window_size: int = 10
) -> dict:
    n = len(recent_requests)
    if n < window_size:
        return {
            "n_requests_logged": n,
            "window_size": window_size,
            "sufficient_data": False,
        }

    window = recent_requests[:window_size]

    feature_names = window[0]["feature_means"].keys()
    avg_means = {
        feature: sum(r["feature_means"][feature] for r in window) / window_size
        for feature in feature_names
    }
    flagged_features = [
        {"feature": feature, "avg_scaled_mean": avg}
        for feature, avg in avg_means.items()
        if abs(avg) > INPUT_DRIFT_Z_THRESHOLD
    ]
    flagged_features.sort(key=lambda f: abs(f["avg_scaled_mean"]), reverse=True)

    avg_score = sum(r["score"] for r in window) / window_size
    ratio_to_threshold = avg_score / threshold

    return {
        "n_requests_logged": n,
        "window_size": window_size,
        "sufficient_data": True,
        "input_drift": {
            "flagged_features": flagged_features,
            "all_feature_avg_scaled_means": avg_means,
        },
        "output_drift": {
            "avg_score_recent": avg_score,
            "threshold": threshold,
            "ratio_to_threshold": ratio_to_threshold,
            "flagged": ratio_to_threshold > OUTPUT_DRIFT_RATIO_THRESHOLD,
        },
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/monitoring/test_drift.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add 02-cnc-machining/src/monitoring/drift.py 02-cnc-machining/tests/monitoring/test_drift.py
git commit -m "Add pure drift-detection logic"
```

## Task 3: `/predict` 로그 통합 + `GET /drift-status`

**Files:**
- Modify: `02-cnc-machining/src/serving/app.py`
- Modify: `02-cnc-machining/tests/serving/test_app.py`

**Interfaces:**
- Consumes: Task 1의 `log_request`, `get_recent_requests`; Task 2의 `compute_drift_status`;
  기존 `serving.inference.scale_features(df, feature_columns, scaler_dict) -> pd.DataFrame`
- Produces: `GET /drift-status` 응답(스펙의 JSON 스키마), `DB_PATH`/`DRIFT_WINDOW_SIZE` 모듈 상수(Task 4에서 재사용)

- [ ] **Step 1: 실패하는 테스트 작성**

`02-cnc-machining/tests/serving/test_app.py` 맨 아래에 추가:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/serving/test_app.py -v -k drift`
Expected: FAIL(`AttributeError: module 'serving.app' has no attribute 'DB_PATH'` 등)

- [ ] **Step 3: `app.py` 수정 — import 추가**

`02-cnc-machining/src/serving/app.py` 상단 import에 추가:
```python
from datetime import datetime, timezone
```
그리고 기존 `from serving.inference import predict_experiment` 줄을:
```python
from serving.inference import predict_experiment
```
다음으로 교체:
```python
from monitoring.drift import compute_drift_status
from monitoring.logging import get_recent_requests, log_request
from serving.inference import predict_experiment, scale_features
```

- [ ] **Step 4: 모듈 상수 추가**

`ROOT = Path(__file__).resolve().parent.parent.parent` 줄 바로 다음에 추가:
```python
DB_PATH = ROOT / "data" / "monitoring" / "requests.db"
DRIFT_WINDOW_SIZE = 10
```

- [ ] **Step 5: `/predict` 라우트에 로그 기록 추가**

`/predict` 함수 안의 다음 부분을:
```python
        result = predict_experiment(
            df=df,
            model=state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=state.scaler_dict,
            window_size=state.window_size,
            threshold=state.thresholds[method],
            method=method,
            feature_baseline=state.feature_baseline,
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
            rag_corpus=state.rag_corpus,
            rag_index=state.rag_index,
            openai_client=state.openai_client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```
다음으로 교체:
```python
        result = predict_experiment(
            df=df,
            model=state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=state.scaler_dict,
            window_size=state.window_size,
            threshold=state.thresholds[method],
            method=method,
            feature_baseline=state.feature_baseline,
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
            rag_corpus=state.rag_corpus,
            rag_index=state.rag_index,
            openai_client=state.openai_client,
        )
        scaled = scale_features(df, FEATURE_COLUMNS, state.scaler_dict)
        feature_means = scaled[FEATURE_COLUMNS].mean().to_dict()
        log_request(feature_means, result["score"], result["predicted_label_text"], DB_PATH)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

- [ ] **Step 6: `GET /drift-status` 엔드포인트 추가**

파일 끝에 추가:
```python
@app.get("/drift-status")
def drift_status(state: ModelState = Depends(get_model_state)) -> dict:
    recent = get_recent_requests(DRIFT_WINDOW_SIZE, DB_PATH)
    status = compute_drift_status(
        recent, threshold=state.thresholds["mean"], window_size=DRIFT_WINDOW_SIZE
    )
    return {**status, "checked_at": datetime.now(timezone.utc).isoformat()}
```

- [ ] **Step 7: 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/serving/test_app.py -v -k drift`
Expected: PASS (3 passed)

- [ ] **Step 8: 전체 테스트 통과 확인**

Run: `cd 02-cnc-machining && uv run pytest -q`
Expected: 기존 84개 + 신규(logging 3 + drift 4 + app 3 = 10개) 전부 통과

- [ ] **Step 9: Commit**

```bash
git add 02-cnc-machining/src/serving/app.py 02-cnc-machining/tests/serving/test_app.py
git commit -m "Integrate drift logging into /predict and add /drift-status endpoint"
```

## Task 4: `monitoring/simulate_drift.py` — 합성 점진적 드리프트 검증

**Files:**
- Create: `02-cnc-machining/monitoring/simulate_drift.py`

**Interfaces:**
- Consumes: `serving.app.app`, `serving.app.DRIFT_WINDOW_SIZE`; `preprocessing.columns.FEATURE_COLUMNS`

이 태스크는 정식 pytest 테스트를 만들지 않는다(`loocv`/`synthetic`과 동일 관례 —
실제 champion 모델을 로드해 돌리는 검증 스크립트).

- [ ] **Step 1: `simulate_drift.py` 작성**

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing.columns import FEATURE_COLUMNS
from serving.app import DRIFT_WINDOW_SIZE, app

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = (
    ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
)
AMPLITUDES = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]


def jitter(df: pd.DataFrame, amplitude: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    for col in FEATURE_COLUMNS:
        std = df[col].std()
        df[col] = df[col] + rng.normal(0, std * amplitude, size=len(df))
    return df


def main() -> None:
    base_df = pd.read_csv(DATASET_DIR / "experiment_01.csv")

    with TestClient(app) as client:
        seed = 0
        for amplitude in AMPLITUDES:
            print(f"=== amplitude={amplitude} ===", flush=True)
            for _ in range(DRIFT_WINDOW_SIZE):
                seed += 1
                perturbed = jitter(base_df, amplitude, seed)
                csv_bytes = perturbed.to_csv(index=False).encode()
                response = client.post(
                    "/predict",
                    files={"file": ("experiment.csv", csv_bytes, "text/csv")},
                )
                assert response.status_code == 200, response.text

            status = client.get("/drift-status").json()
            print(f"  sufficient_data={status['sufficient_data']}")
            if status["sufficient_data"]:
                top = sorted(
                    status["input_drift"]["all_feature_avg_scaled_means"].items(),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )[:3]
                print(f"  상위 3개 피처 평균 편차: {top}")
                print(f"  output_drift: {status['output_drift']}")
                print(
                    f"  flagged_features 개수: "
                    f"{len(status['input_drift']['flagged_features'])}"
                )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 문법 확인**

Run: `cd 02-cnc-machining && uv run python -m py_compile monitoring/simulate_drift.py`
Expected: 에러 없이 종료

- [ ] **Step 3: 서버 상태 확인 후 실행**

Run:
```bash
cd 02-cnc-machining
who && top -bn1 | head -6
nice -n 19 uv run python monitoring/simulate_drift.py
```
Expected: 진폭 6단계 전부 출력, assertion 에러 없음(모든 `/predict` 응답
200). 처음 몇 단계는 `sufficient_data=false`가 나올 수 있음(요청이 아직
`DRIFT_WINDOW_SIZE`개 안 쌓였을 때가 아니라 — 매 단계가 정확히
`DRIFT_WINDOW_SIZE`번씩 호출하므로 매 단계 끝나면 항상 `sufficient_data=true`여야
함. 만약 `false`가 나오면 버그이므로 조사 필요).

- [ ] **Step 4: 결과 확인 — 가정하지 않고 있는 그대로 기록**

진폭이 커질수록 `input_drift`의 편차나 `output_drift.ratio_to_threshold`가
실제로 커지는 추세인지, 어느 진폭부터 `flagged`가 `true`로 바뀌는지 관찰.
안 잡히면(진폭 0.10에서도 flag 없음) 임계값(`INPUT_DRIFT_Z_THRESHOLD=2.0`,
`OUTPUT_DRIFT_RATIO_THRESHOLD=0.8`)이 너무 느슨하다는 뜻이므로, 그것도
있는 그대로 사용자에게 보고한다(미리 좋아질 거라 가정하지 않음).

- [ ] **Step 5: Commit**

```bash
git add 02-cnc-machining/monitoring/simulate_drift.py
git commit -m "Add synthetic progressive-drift simulation script"
```

---

## Self-Review 완료 사항

- 스펙 커버리지: SQLite 로그(Task 1), 순수 드리프트 계산(Task 2), `/predict`
  통합+`/drift-status`(Task 3), 합성 시퀀스 검증(Task 4) 전부 매핑됨.
- 플레이스홀더 없음: 전 태스크 실행 가능한 완성 코드.
- 타입/시그니처 일관성: `compute_drift_status(recent_requests, threshold, window_size)`가
  Task 2 정의와 Task 3의 `/drift-status` 호출부에서 동일. `log_request`/`get_recent_requests`
  시그니처가 Task 1 정의와 Task 3의 `app.py` 통합부·테스트에서 동일.
  `DRIFT_WINDOW_SIZE`가 Task 3에서 정의되고 Task 4에서 그대로 import됨.
- `tests/monitoring/`에 `__init__.py`를 만들지 않도록 Task 1에 명시적으로
  경고를 남겨둠(이전 LOOCV 작업에서 겪은 실제 버그 재발 방지).
