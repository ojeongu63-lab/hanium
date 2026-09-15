# 재학습 루프 통합 테스트 + compose 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 실데이터 없이 `uv run pytest -m integration` 한 번으로 재학습 루프(트리거 → 재학습 → 게이트 →
섀도우 → 승격/거부)가 실제 서버·워커 프로세스 사이에서 끝까지 도는지 검증하고, 그 테스트를 CI에
넣고, compose로 스택을 띄울 수 있게 한다.

**Architecture:** `src/config.py` 하나가 데이터 루트와 루프 상수를 환경변수에서 읽는다(기본값 =
지금 값). 합성 데이터셋 생성기가 KAMP와 같은 폴더 구조를 임시 루트에 쓰고, 기존 스크립트 3개가
그 위에서 작은 champion을 만든다. pytest가 uvicorn·워커를 서브프로세스로 띄우고 feeder를 하루씩
동기화하며 승격 경로·거부 경로를 단언한다. CI에 job 2개, compose 파일 1개.

**Tech Stack:** Python 3.14, uv, pytest(importlib 모드), FastAPI/uvicorn, MLflow(sqlite), PyTorch CPU,
httpx2, docker compose, GitHub Actions.

**Spec:** `02-cnc-machining/docs/specs/2026-09-15-cnc-loop-integration-test-design.md`

## Global Constraints

- 모든 명령은 `02-cnc-machining/`에서 실행한다. 공유 서버(da20-suresoft)라 학습·통합 테스트 실행은
  `nice -n 19`를 붙이고 시작 전 `who`/`top`으로 부하를 본다.
- **환경변수가 없으면 동작 불변.** 매 태스크 끝에 `uv run pytest -q`가 기존 212개 + 새 테스트 전부
  통과해야 한다. 상수 기본값은 스펙 §1 표의 값 그대로: 10, 3, 5, 20, 40, 5, 10, 7, 21, 상한 없음,
  0.02, 0.02, 0.2, 3.65, 50.
- **모듈 속성 이름 유지.** 다른 파일과 테스트가 이 이름으로 접근한다:
  `serving.app.{DB_PATH, SHADOW_DB, DRIFT_WINDOW_SIZE, DATASET_DIR, TIMELINE_BATCHES_PER_DAY}`,
  `simulate_timeline.{DATASET_DIR, LABELS_DB, TOTAL_DAYS, BATCHES_PER_DAY, DRIFT_START_DAY,
  LABEL_DELAY_DAYS, WEAR_LABEL_FLIP_DAY, VIBRATION_LABEL_FLIP_DAY, POS_DRIFT, CUR_DRIFT, WEAR_RATE,
  VIBRATION_RATE, generate_batch, true_label, feed_day, PERTURBATIONS}`
  (`sweep_drift_constants.py`가 `st.POS_DRIFT = v`처럼 모듈 속성을 바꿔 쓰므로 변형 함수는 반드시
  모듈 전역을 읽어야 한다), `drift_worker.{LABELS_DB, REQUESTS_DB, MODEL_DIR, SCALER_PATH,
  BACKUP_ROOT, SHADOW_DB, COOLDOWN_DAYS, CONSECUTIVE_K, GATE_SAMPLE_SIZE, WorkerState, tick}`.
- **손대지 않는 것:** `src/retraining/gate.py`·`trigger.py`·`promotion.py`의 판정 로직,
  `src/monitoring/drift.py`, `demo/`, `rag/`, `synthetic/`, `loocv/`, `augmentation/`,
  `monitoring/simulate_drift.py`, `monitoring/sweep_drift_constants.py`.
- 실데이터 DB(`data/monitoring/*.db`)와 `data/timeline/`을 건드리는 스모크는 실행 전 옮겨 두고
  끝난 뒤 되돌린다. 프로세스 종료는 PID 파일로 한다(`tasks/lessons.md`: `pkill -f`는 자기 셸을 죽인다).
- 커밋 메시지는 기존 관례(`feat(...)`, `fix(...)`, `refactor:`, `test(...)`, `build:`, `ci:`, `docs:`).
  본문 끝에 두 줄:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` /
  `Claude-Session: https://claude.ai/code/session_016VSRNz3Sj91ak13Qep4ALG`
- 이 서버에는 docker가 없다. compose 파일은 YAML 파싱과 CI `docker-build` job으로 검증한다.
- git 저장소 루트는 `/home/sure/project`다. 워크플로 파일 경로는 저장소 루트 기준
  `.github/workflows/cnc-tests.yml`, 나머지 경로는 전부 `02-cnc-machining/` 기준이다.

---

## 파일 구조

| 파일 | 책임 | 태스크 |
|---|---|---|
| `src/config.py` (신규) | 환경변수 → 데이터 루트·루프 상수·진폭·epochs. import 시 한 번 읽는다 | 1 |
| `tests/test_config.py` (신규) | 기본값·파싱 + 서브프로세스 배선 테스트 3개 | 1·2·3·4 |
| `src/lstm_ae/tracking.py` | `MLFLOW_DIR`를 config에서 | 2 |
| `src/serving/app.py` | 경로 6곳·`DRIFT_WINDOW_SIZE`·`TIMELINE_BATCHES_PER_DAY`를 config에서 | 2 |
| `src/retraining/runner.py` | epochs를 config에서, `root` → `data_root` | 2 |
| `scripts/run_preprocessing.py`, `scripts/run_lstm_training.py` | 경로·epochs를 config에서 | 2 |
| `monitoring/simulate_timeline.py` | 경로·상수·진폭을 config에서, 진행도 상한, `--start-day` | 3 |
| `tests/monitoring/test_simulate_timeline.py` | 상한·`--start-day` 테스트 추가 | 3 |
| `monitoring/drift_worker.py` | 경로·상수를 config에서, `champion_missed`를 MLflow에서 | 4 |
| `tests/monitoring/test_drift_worker.py` (신규) | 순수 함수 테스트 | 4 |
| `pyproject.toml`, `uv.lock` | `httpx2` 본 의존성, `integration` 마커, 기본 실행 제외 | 5 |
| `tests/integration/fixture_dataset.py` (신규) | 합성 데이터셋 생성기 | 6 |
| `tests/integration/test_fixture_dataset.py` (신규) | 생성기 테스트 | 6 |
| `tests/integration/conftest.py` (신규) | 부트스트랩, `Loop` 하네스, `loop_factory` 픽스처 | 7 |
| `tests/integration/test_loop.py` (신규) | 하네스 스모크 1개 + 승격 경로 + 거부 경로 | 7·8·9 |
| `.github/workflows/cnc-tests.yml` | `loop-integration`, `docker-build` job | 10 |
| `docker-compose.yml` (신규), `README.md`, `docs/STRUCTURE.md` | 스택 정의와 문서 | 11 |
| 스펙 "실행 결과에 따른 정정" 절, `tasks/todo.md` | 실측 기록 | 12 |

---

### Task 1: `src/config.py` — 환경변수 설정 모듈

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`
- Modify: `../tasks/todo.md` (작업 절 추가)

**Interfaces:**
- Produces: 모듈 `config`의 상수 `PROJECT_ROOT: Path`, `DATA_ROOT: Path`, `DATASET_DIR: Path`,
  `EXPERIMENT_DIR: Path`, `DRIFT_WINDOW_SIZE: int`, `CONSECUTIVE_K: int`, `COOLDOWN_DAYS: int`,
  `GATE_SAMPLE_SIZE: int`, `TOTAL_DAYS: int`, `BATCHES_PER_DAY: int`, `DRIFT_START_DAY: int`,
  `LABEL_DELAY_DAYS: int`, `LABEL_FLIP_DAY: int`, `DRIFT_MAX_PROGRESS: float | None`,
  `POS_DRIFT: float`, `CUR_DRIFT: float`, `WEAR_RATE: float`, `VIBRATION_RATE: float`,
  `TRAIN_EPOCHS: int`. 이후 모든 태스크가 `from config import ...`로 쓴다.
  `src/`는 `.venv/.../cnc_preprocessing.pth`로 `sys.path`에 있어 `import config`가 바로 된다.

- [ ] **Step 1: `tasks/todo.md`에 작업 절 추가**

`../tasks/todo.md` 끝에 붙인다:

```markdown

## 재학습 루프 통합 테스트 + compose (2026-09-15)

스펙 `02-cnc-machining/docs/specs/2026-09-15-cnc-loop-integration-test-design.md`,
계획 `02-cnc-machining/docs/plans/2026-09-15-cnc-loop-integration-test.md`.
사용자 결정: 배관만 보장, pytest가 서버·워커를 서브프로세스로 기동, compose는 CI에서 빌드만.

- [ ] Task 1 `src/config.py` — 환경변수 설정 모듈
- [ ] Task 2 src 모듈·스크립트가 config를 읽도록
- [ ] Task 3 feeder — config, 진행도 상한, `--start-day`
- [ ] Task 4 워커 — config, champion 놓침 수를 MLflow에서
- [ ] Task 5 pyproject — httpx2 본 의존성, integration 마커
- [ ] Task 6 합성 데이터셋 생성기
- [ ] Task 7 통합 테스트 하네스 + 스모크
- [ ] Task 8 승격 경로 테스트
- [ ] Task 9 거부 경로 테스트
- [ ] Task 10 CI job 2개
- [ ] Task 11 docker-compose + README/STRUCTURE
- [ ] Task 12 실데이터 스모크, 스펙 정정 절, 리뷰
```

- [ ] **Step 2: 실패하는 테스트 작성** — `tests/test_config.py`

```python
"""src/config.py — 환경변수가 없으면 코드에 박혀 있던 값, 있으면 그 값. 파싱 실패는 즉시 죽는다."""
import importlib
import os

import pytest


def _reload(monkeypatch, **env):
    """CNC_* 환경변수를 전부 지운 뒤 주어진 것만 넣고 config를 다시 읽는다."""
    for key in list(os.environ):
        if key.startswith("CNC_"):
            monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import config

    return importlib.reload(config)


@pytest.fixture(autouse=True)
def _restore_config():
    """reload가 남긴 상태를 되돌린다 — 다른 테스트가 import config 를 다시 할 수 있다."""
    yield
    for key in list(os.environ):
        if key.startswith("CNC_"):
            os.environ.pop(key, None)
    import config

    importlib.reload(config)


def test_defaults_match_previous_hardcoded_values(monkeypatch):
    cfg = _reload(monkeypatch)

    assert cfg.DATA_ROOT == cfg.PROJECT_ROOT / "data"
    assert cfg.DATASET_DIR == cfg.DATA_ROOT / "dataset" / "CNC 비식별화 원본데이터_1209"
    assert cfg.EXPERIMENT_DIR == cfg.DATASET_DIR / "CNC Virtual Data set _v2"
    assert (cfg.DRIFT_WINDOW_SIZE, cfg.CONSECUTIVE_K, cfg.COOLDOWN_DAYS, cfg.GATE_SAMPLE_SIZE) == (
        10, 3, 5, 20,
    )
    assert (
        cfg.TOTAL_DAYS, cfg.BATCHES_PER_DAY, cfg.DRIFT_START_DAY, cfg.LABEL_DELAY_DAYS, cfg.LABEL_FLIP_DAY,
    ) == (40, 5, 10, 7, 21)
    assert cfg.DRIFT_MAX_PROGRESS is None
    assert (cfg.POS_DRIFT, cfg.CUR_DRIFT, cfg.WEAR_RATE, cfg.VIBRATION_RATE) == (0.02, 0.02, 0.2, 3.65)
    assert cfg.TRAIN_EPOCHS == 50


def test_env_overrides_are_parsed(monkeypatch, tmp_path):
    cfg = _reload(
        monkeypatch,
        CNC_DATA_ROOT=str(tmp_path),
        CNC_GATE_SAMPLE_SIZE="5",
        CNC_DRIFT_MAX_PROGRESS="1.0",
        CNC_POS_DRIFT="8",
    )

    assert cfg.DATA_ROOT == tmp_path
    assert cfg.DATASET_DIR == tmp_path / "dataset" / "CNC 비식별화 원본데이터_1209"
    assert cfg.GATE_SAMPLE_SIZE == 5 and isinstance(cfg.GATE_SAMPLE_SIZE, int)
    assert cfg.DRIFT_MAX_PROGRESS == 1.0
    assert cfg.POS_DRIFT == 8.0


def test_unparsable_value_fails_loudly(monkeypatch):
    with pytest.raises(ValueError, match="CNC_COOLDOWN_DAYS"):
        _reload(monkeypatch, CNC_COOLDOWN_DAYS="five")
```

- [ ] **Step 3: 실패 확인**

Run: `cd 02-cnc-machining && uv run pytest tests/test_config.py -q`
Expected: 3 failed — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 4: `src/config.py` 작성**

```python
"""실행 환경 설정 — 데이터 루트와 루프 상수를 환경변수에서 읽는다.

기본값은 이 모듈이 생기기 전에 각 파일에 박혀 있던 값과 같다. 환경변수가 없으면 동작은 이전과
같다. 통합 테스트가 임시 데이터 루트와 며칠짜리 루프로 서버·워커를 띄우기 위해 만들었다
(docs/specs/2026-09-15-cnc-loop-integration-test-design.md §1).

값이 있는데 숫자로 읽히지 않으면 import 시점에 ValueError로 죽는다 — 조용히 기본값으로
떨어지면 오타 하나가 40일 루프를 엉뚱하게 돌린다.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 02-cnc-machining/


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: 정수가 아닙니다") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: 실수가 아닙니다") from exc


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return _env_float(name, 0.0)


# 데이터 위치. 원본 CSV는 인덱스 폴더 아래 한 단계 더 들어간다.
DATA_ROOT = Path(os.environ.get("CNC_DATA_ROOT") or PROJECT_ROOT / "data")
DATASET_DIR = DATA_ROOT / "dataset" / "CNC 비식별화 원본데이터_1209"  # 인덱스 train.csv
EXPERIMENT_DIR = DATASET_DIR / "CNC Virtual Data set _v2"  # experiment_XX.csv

# 서빙 — 드리프트 판정에 쓰는 최근 요청 수
DRIFT_WINDOW_SIZE = _env_int("CNC_DRIFT_WINDOW_SIZE", 10)

# 워커 — 트리거·게이트 (근거는 monitoring/drift_worker.py 주석)
CONSECUTIVE_K = _env_int("CNC_CONSECUTIVE_K", 3)
COOLDOWN_DAYS = _env_int("CNC_COOLDOWN_DAYS", 5)
GATE_SAMPLE_SIZE = _env_int("CNC_GATE_SAMPLE_SIZE", 20)

# feeder — 가상 운영 타임라인 (근거는 monitoring/simulate_timeline.py 주석)
TOTAL_DAYS = _env_int("CNC_TOTAL_DAYS", 40)
BATCHES_PER_DAY = _env_int("CNC_BATCHES_PER_DAY", 5)
DRIFT_START_DAY = _env_int("CNC_DRIFT_START_DAY", 10)
LABEL_DELAY_DAYS = _env_int("CNC_LABEL_DELAY_DAYS", 7)
LABEL_FLIP_DAY = _env_int("CNC_LABEL_FLIP_DAY", 21)  # 고장 시나리오에서 QC 불합격이 시작되는 날
DRIFT_MAX_PROGRESS = _env_optional_float("CNC_DRIFT_MAX_PROGRESS")  # None이면 상한 없음
POS_DRIFT = _env_float("CNC_POS_DRIFT", 0.02)
CUR_DRIFT = _env_float("CNC_CUR_DRIFT", 0.02)
WEAR_RATE = _env_float("CNC_WEAR_RATE", 0.2)
VIBRATION_RATE = _env_float("CNC_VIBRATION_RATE", 3.65)

# 학습·재학습 epochs (나머지 하이퍼파라미터는 바꾸지 않는다)
TRAIN_EPOCHS = _env_int("CNC_TRAIN_EPOCHS", 50)
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/test_config.py -q`
Expected: 3 passed

- [ ] **Step 6: 전체 스위트**

Run: `uv run pytest -q`
Expected: 215 passed

- [ ] **Step 7: 커밋**

```bash
cd /home/sure/project/hanium
git add 02-cnc-machining/src/config.py 02-cnc-machining/tests/test_config.py tasks/todo.md
git commit -m "feat(config): read data root and loop constants from CNC_* environment variables"
```

---

### Task 2: src 모듈과 스크립트가 config를 읽도록

**Files:**
- Modify: `src/lstm_ae/tracking.py:19-20`
- Modify: `src/serving/app.py:30-41, 63-76, 104-109, 288`
- Modify: `src/retraining/runner.py:14-26, 110-131`
- Modify: `monitoring/drift_worker.py:96-101` (`run_retraining` 호출 인자 이름만)
- Modify: `scripts/run_preprocessing.py`, `scripts/run_lstm_training.py`
- Modify: `tests/test_config.py` (배선 테스트 추가)

**Interfaces:**
- Consumes: Task 1의 `config`.
- Produces: `retraining.runner.run_retraining(timeline_dir, labels_db, current_day, data_root, lookback_days=30)`
  — 인자 이름이 `root`에서 `data_root`로 바뀐다(`root / "data"`가 아니라 데이터 루트 자체를 받는다).

- [ ] **Step 1: 실패하는 배선 테스트 추가** — `tests/test_config.py` 끝에

```python
import json
import subprocess
import sys

SRC_WIRING = """
import json
import lstm_ae.tracking as tracking
import retraining.runner as runner
import serving.app as app
print(json.dumps({
    "mlflow_dir": str(tracking.MLFLOW_DIR),
    "db_path": str(app.DB_PATH),
    "shadow_db": str(app.SHADOW_DB),
    "dataset_dir": str(app.DATASET_DIR),
    "drift_window": app.DRIFT_WINDOW_SIZE,
    "batches_per_day": app.TIMELINE_BATCHES_PER_DAY,
    "epochs": runner.TRAINING_CONFIG["epochs"],
}))
"""


def _run_snippet(snippet: str, env: dict) -> dict:
    """config는 import 시 환경을 읽으므로, 배선은 새 인터프리터에서 확인한다."""
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        env={**os.environ, **env}, capture_output=True, text=True, check=True, timeout=180,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_src_modules_follow_config(tmp_path):
    got = _run_snippet(SRC_WIRING, {
        "CNC_DATA_ROOT": str(tmp_path),
        "CNC_DRIFT_WINDOW_SIZE": "4",
        "CNC_BATCHES_PER_DAY": "2",
        "CNC_TRAIN_EPOCHS": "1",
    })

    root = str(tmp_path)
    assert got["mlflow_dir"] == f"{root}/mlflow"
    assert got["db_path"] == f"{root}/monitoring/requests.db"
    assert got["shadow_db"] == f"{root}/monitoring/shadow.db"
    assert got["dataset_dir"] == f"{root}/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2"
    assert got["drift_window"] == 4
    assert got["batches_per_day"] == 2
    assert got["epochs"] == 1
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_config.py::test_src_modules_follow_config -q`
Expected: FAIL — `mlflow_dir`가 `.../02-cnc-machining/data/mlflow`(tmp가 아님)

- [ ] **Step 3: `src/lstm_ae/tracking.py`**

19~20행을 바꾼다:

```python
from config import DATA_ROOT

MLFLOW_DIR = DATA_ROOT / "mlflow"
```

`ROOT`는 이 파일에서 더 쓰지 않으므로 지운다. import 위치는 `from .plotting import (...)` 블록 위,
서드파티 import 아래(첫 번째 first-party import).

- [ ] **Step 4: `src/serving/app.py`**

import 블록의 first-party 첫 줄로 추가:

```python
from config import BATCHES_PER_DAY, DATA_ROOT, DRIFT_WINDOW_SIZE, EXPERIMENT_DIR, PROJECT_ROOT
```

30~41행의 상수 블록을 다음으로 교체(`DRIFT_WINDOW_SIZE = 10` 줄은 삭제 — import로 들어온다):

```python
ROOT = PROJECT_ROOT
DB_PATH = DATA_ROOT / "monitoring" / "requests.db"
SHADOW_DB = DATA_ROOT / "monitoring" / "shadow.db"
DEMO_INDEX = ROOT / "demo" / "index.html"
DATASET_DIR = EXPERIMENT_DIR
DEMO_INPUTS = {
    "tool_wear": ROOT / "synthetic" / "scenarios" / "tool_wear.csv",
    "feed_overload": ROOT / "synthetic" / "scenarios" / "feed_overload.csv",
    "vibration_backlash": ROOT / "synthetic" / "scenarios" / "vibration_backlash.csv",
    "experiment_07": DATASET_DIR / "experiment_07.csv",
    "experiment_12": DATASET_DIR / "experiment_12.csv",
}
```

`load_rag_state` 안의 세 경로:

```python
    corpus_path = DATA_ROOT / "rag" / "corpus.json"
    index_path = DATA_ROOT / "rag" / "corpus.index"
    ...
    meta_path = DATA_ROOT / "rag" / "corpus_meta.json"
```

`_build_model_state` 안의 폴백 두 곳:

```python
    scaler_dict = load_companion_json(
        mv.run_id, "scaler.json", DATA_ROOT / "processed" / "scaler.json"
    )
    feature_baseline = load_companion_json(
        mv.run_id, "feature_baseline.json", DATA_ROOT / "model" / "feature_baseline.json"
    )
```

288행: `TIMELINE_BATCHES_PER_DAY = BATCHES_PER_DAY`. `_generate_timeline_batch`의
`ROOT / "monitoring"`은 그대로(코드 폴더라 데이터 루트가 아니다).

- [ ] **Step 5: `src/retraining/runner.py`**

import에 `from config import TRAIN_EPOCHS` 추가(first-party 첫 줄). `TRAINING_CONFIG`의
`"epochs": 50` → `"epochs": TRAIN_EPOCHS`. 주석은 "epochs만 config(CNC_TRAIN_EPOCHS)에서 온다"를
한 줄 덧붙인다.

`run_retraining` 시그니처와 본문의 `root`:

```python
def run_retraining(
    timeline_dir: Path,
    labels_db: Path,
    current_day: int,
    data_root: Path,
    lookback_days: int = 30,
) -> dict:
    """라벨 도착분으로 재학습하고 MLflow에 새 run으로 기록한다. 승격은 하지 않는다.

    산출물은 <data_root>/retrain/<timestamp>/ 에 격리한다 — 게이트가 거부할 수도 있는데
    정본(<data_root>/model/, <data_root>/processed/scaler.json)을 먼저 덮어쓰면 champion과
    짝이 어긋난 상태로 남는다.
    """
```

```python
    retrain_dir = data_root / "retrain" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ...
    old_scaler_dict = json.loads((data_root / "processed" / "scaler.json").read_text())
    eval_old = pd.read_csv(data_root / "processed" / "eval.csv")
```

- [ ] **Step 6: `monitoring/drift_worker.py` 호출부만**

`tick()` 안의 `run_retraining(...)` 호출에서 `root=ROOT,` → `data_root=ROOT / "data",`.
(전체 배선은 Task 4. 여기서는 시그니처 변경을 따라가기만 한다.)

- [ ] **Step 7: `scripts/run_preprocessing.py`** 전체 교체

```python
import json

from config import DATA_ROOT, DATASET_DIR, EXPERIMENT_DIR
from preprocessing.pipeline import run_pipeline
from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)


def main() -> None:
    manifest = run_pipeline(
        experiment_index_path=str(DATASET_DIR / "train.csv"),
        experiment_dir=str(EXPERIMENT_DIR),
        output_dir=str(DATA_ROOT / "processed"),
        train_experiment_ids=TRAIN_EXPERIMENT_IDS,
        eval_good_experiment_ids=EVAL_GOOD_EXPERIMENT_IDS,
        eval_bad_experiment_ids=EVAL_BAD_EXPERIMENT_IDS,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: `scripts/run_lstm_training.py`**

`from pathlib import Path`와 `ROOT = ...` 줄을 지우고 `from config import DATA_ROOT, TRAIN_EPOCHS`를
first-party import 첫 줄에 넣는다. 그 아래에:

```python
PROCESSED_DIR = DATA_ROOT / "processed"
MODEL_DIR = DATA_ROOT / "model"
```

`TRAINING_CONFIG`의 `"epochs": 50` → `"epochs": TRAIN_EPOCHS`. 본문의 경로를 전부 바꾼다:

```python
    manifest = json.loads((PROCESSED_DIR / "manifest.json").read_text())
    ...
            train_csv_path=str(PROCESSED_DIR / "train.csv"),
            eval_csv_path=str(PROCESSED_DIR / "eval.csv"),
            ...
            output_dir=str(MODEL_DIR),
    ...
        experiment_scores = pd.read_csv(MODEL_DIR / "experiment_scores.csv")
        feature_error_scores = pd.read_csv(MODEL_DIR / "eval_feature_errors.csv")
        feature_baseline = json.loads((MODEL_DIR / "feature_baseline.json").read_text())
        timeline_errors = pd.read_csv(MODEL_DIR / "eval_timeline_errors.csv")
        reconstruction_overlay = pd.read_csv(MODEL_DIR / "eval_reconstruction_overlay.csv")
```

- [ ] **Step 9: 통과 확인 + 전체 스위트**

Run: `uv run pytest tests/test_config.py -q` → 4 passed
Run: `uv run pytest -q` → 216 passed
Run: `grep -rn 'ROOT / "data"' src scripts` → 출력 없음

- [ ] **Step 10: 커밋**

```bash
git add 02-cnc-machining/src/lstm_ae/tracking.py 02-cnc-machining/src/serving/app.py \
  02-cnc-machining/src/retraining/runner.py 02-cnc-machining/monitoring/drift_worker.py \
  02-cnc-machining/scripts/run_preprocessing.py 02-cnc-machining/scripts/run_lstm_training.py \
  02-cnc-machining/tests/test_config.py
git commit -m "refactor: derive data paths and training epochs from config"
```

---

### Task 3: feeder — config, 진행도 상한, `--start-day`

**Files:**
- Modify: `monitoring/simulate_timeline.py:21-70, 102-107, 192-232`
- Modify: `tests/monitoring/test_simulate_timeline.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: Task 1의 `config`.
- Produces: `simulate_timeline.progress_for(day) -> float`(상한 적용), CLI 옵션
  `--start-day N`(기본 1) — `range(start_day, days + 1)`을 보낸다. 모듈 전역 `DRIFT_MAX_PROGRESS`.

- [ ] **Step 1: 실패하는 테스트** — `tests/monitoring/test_simulate_timeline.py`

import 줄을 바꾸고(기존 `from simulate_timeline import apply_fixture_loosening` 유지) 추가:

```python
import simulate_timeline as st  # noqa: E402
```

파일 끝에:

```python
def test_progress_is_unbounded_without_cap(monkeypatch):
    monkeypatch.setattr(st, "DRIFT_START_DAY", 10)
    monkeypatch.setattr(st, "TOTAL_DAYS", 40)
    monkeypatch.setattr(st, "DRIFT_MAX_PROGRESS", None)

    assert st.progress_for(5) == 0.0
    assert st.progress_for(40) == pytest.approx(1.0)
    assert st.progress_for(70) == pytest.approx(2.0)  # 08-25 스펙의 알려진 한계 그대로


def test_progress_is_capped_when_configured(monkeypatch):
    monkeypatch.setattr(st, "DRIFT_START_DAY", 2)
    monkeypatch.setattr(st, "TOTAL_DAYS", 3)
    monkeypatch.setattr(st, "DRIFT_MAX_PROGRESS", 1.0)

    assert st.progress_for(2) == 0.0
    assert st.progress_for(3) == pytest.approx(1.0)
    assert st.progress_for(9) == pytest.approx(1.0)  # 계단


def test_serve_url_mode_feeds_only_requested_day_range(monkeypatch, tmp_path):
    fed = []
    monkeypatch.setattr(st, "feed_day", lambda client, day, scenario, out_dir: fed.append(day))
    monkeypatch.setattr(st, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "simulate_timeline.py", "temperature", "--serve-url", "http://127.0.0.1:1",
        "--start-day", "5", "--days", "6",
    ])

    st.main()

    assert fed == [5, 6]
    assert (tmp_path / "timeline" / "temperature").is_dir()
```

`tests/test_config.py` 끝에:

```python
MONITORING_DIR = str(Path(__file__).resolve().parent.parent / "monitoring")

FEEDER_WIRING = f"""
import json, sys
sys.path.insert(0, {MONITORING_DIR!r})
import simulate_timeline as st
print(json.dumps({{
    "labels_db": str(st.LABELS_DB),
    "dataset_dir": str(st.DATASET_DIR),
    "pos_drift": st.POS_DRIFT,
    "flip": st.WEAR_LABEL_FLIP_DAY,
    "progress_9": st.progress_for(9),
}}))
"""


def test_feeder_follows_config(tmp_path):
    got = _run_snippet(FEEDER_WIRING, {
        "CNC_DATA_ROOT": str(tmp_path),
        "CNC_POS_DRIFT": "8",
        "CNC_LABEL_FLIP_DAY": "3",
        "CNC_DRIFT_START_DAY": "2",
        "CNC_TOTAL_DAYS": "3",
        "CNC_DRIFT_MAX_PROGRESS": "1.0",
    })

    root = str(tmp_path)
    assert got["labels_db"] == f"{root}/monitoring/labels.db"
    assert got["dataset_dir"] == f"{root}/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2"
    assert got["pos_drift"] == 8.0
    assert got["flip"] == 3
    assert got["progress_9"] == 1.0
```

(`from pathlib import Path`를 `tests/test_config.py` 상단 import에 추가한다.)

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/monitoring/test_simulate_timeline.py tests/test_config.py::test_feeder_follows_config -q`
Expected: 4 failed — `AttributeError: ... has no attribute 'DRIFT_MAX_PROGRESS'`, `DATA_ROOT`,
`unrecognized arguments: --start-day`, 배선 값 불일치

- [ ] **Step 3: `monitoring/simulate_timeline.py` 상단 교체**

21~70행(`ROOT = ...`부터 `VIBRATION_RATE = 3.65`까지)을 다음으로 바꾼다. 08-19 스윕 기록 주석
블록(`# sweep_drift_constants.py 로 확정한 값...`부터 `# 출력하므로 조용히 틀리지는 않는다.`까지)은
그대로 두되 그 위에 한 줄을 붙인다:

```python
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    BATCHES_PER_DAY,
    CUR_DRIFT,
    DATA_ROOT,
    DRIFT_MAX_PROGRESS,
    DRIFT_START_DAY,
    EXPERIMENT_DIR,
    LABEL_DELAY_DAYS,
    LABEL_FLIP_DAY,
    POS_DRIFT,
    TOTAL_DAYS,
    VIBRATION_RATE,
    WEAR_RATE,
)
from monitoring.labels import record_label  # noqa: E402
from preprocessing.split import TRAIN_EXPERIMENT_IDS  # noqa: E402

# 경로·상수·진폭은 config.py(환경변수 CNC_*)에서 온다. 기본값은 이 파일에 박혀 있던 값과 같다.
# 변형 함수는 모듈 전역(POS_DRIFT 등)을 읽는다 — sweep_drift_constants.py 가 st.POS_DRIFT = v 로
# 바꿔 가며 부르기 때문에 config.POS_DRIFT 를 직접 참조하면 안 된다.
DATASET_DIR = EXPERIMENT_DIR
LABELS_DB = DATA_ROOT / "monitoring" / "labels.db"
WEAR_LABEL_FLIP_DAY = LABEL_FLIP_DAY       # 시나리오 B에서 QC 불합격이 시작되는 날
VIBRATION_LABEL_FLIP_DAY = LABEL_FLIP_DAY  # 고정구 풀림도 같은 날부터

# 아래 값들(POS_DRIFT=0.02, CUR_DRIFT=0.02, WEAR_RATE=0.2, VIBRATION_RATE=3.65)의 근거 —
# sweep_drift_constants.py 로 확정한 값. champion v1 (threshold 0.8566) 기준.
```

(기존 주석 블록의 첫 줄 `# sweep_drift_constants.py 로 확정한 값. champion v1 (threshold 0.8566) 기준.`은
위 마지막 줄로 대체되므로 지운다. `POS_DRIFT = 0.02` … `VIBRATION_RATE = 3.65` 네 줄과 그 옆 주석은
삭제한다.)

- [ ] **Step 4: `progress_for`**

```python
def progress_for(day: int) -> float:
    """변형 진행도. DRIFT_START_DAY 이전 0, TOTAL_DAYS 에 1. 상한이 없으면 그 뒤로도 계속 커진다
    (08-25 섀도우 스펙의 알려진 한계). CNC_DRIFT_MAX_PROGRESS 를 주면 거기서 멈춘다 — 통합 테스트는
    1.0 으로 계단 변형을 만든다."""
    progress = max(0.0, (day - DRIFT_START_DAY) / (TOTAL_DAYS - DRIFT_START_DAY))
    if DRIFT_MAX_PROGRESS is not None:
        progress = min(progress, DRIFT_MAX_PROGRESS)
    return progress
```

- [ ] **Step 5: `main()`**

`parser.add_argument("--days", ...)` 다음에:

```python
    parser.add_argument(
        "--start-day", type=int, default=1,
        help="이 날부터 보낸다(기본 1). 통합 테스트가 하루씩 끊어 보낼 때 쓴다: --start-day N --days N",
    )
```

`out_dir = ROOT / "data" / "timeline" / args.scenario` → `out_dir = DATA_ROOT / "timeline" / args.scenario`.
두 루프 모두 `for day in range(1, args.days + 1)` → `for day in range(args.start_day, args.days + 1)`.

- [ ] **Step 6: 통과 확인 + 전체 스위트**

Run: `uv run pytest tests/monitoring/test_simulate_timeline.py tests/test_config.py -q` → 10 passed
Run: `uv run pytest -q` → 220 passed
Run: `uv run python monitoring/sweep_drift_constants.py --help 2>&1 | head -3` 대신 import만 확인:
`uv run python -c "import sys; sys.path.insert(0,'monitoring'); import sweep_drift_constants"` → 에러 없음
(이 스크립트는 champion을 로드하므로 실행하지 않는다.)

- [ ] **Step 7: 커밋**

```bash
git add 02-cnc-machining/monitoring/simulate_timeline.py \
  02-cnc-machining/tests/monitoring/test_simulate_timeline.py 02-cnc-machining/tests/test_config.py
git commit -m "feat(monitoring): feeder reads config, caps drift progress, accepts --start-day"
```

---

### Task 4: 워커 — config, champion 놓침 수를 MLflow에서

**Files:**
- Modify: `monitoring/drift_worker.py:17-74, 96-101, 253-256, 373-420`
- Modify: `monitoring/simulate_timeline.py:219-221` (in-process 모드의 `WorkerState()` 호출)
- Create: `tests/monitoring/test_drift_worker.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: Task 1의 `config`, Task 2의 `run_retraining(..., data_root=...)`.
- Produces: `drift_worker.champion_missed_from_metrics(metrics: dict) -> int`,
  `drift_worker.load_champion_missed() -> int`, `WorkerState(champion_missed: int, ...)` —
  `champion_missed`가 필수 첫 인자가 된다.

- [ ] **Step 1: 실패하는 테스트** — `tests/monitoring/test_drift_worker.py`

```python
"""monitoring/drift_worker.py 의 순수 함수. 루프 전체는 tests/integration/ 이 실제 프로세스로 검증한다."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "monitoring"))

import drift_worker as dw  # noqa: E402


def test_champion_missed_from_metrics_reads_mean_fn():
    assert dw.champion_missed_from_metrics({"mean_fn": 1.0, "mean_recall": 0.909}) == 1


def test_champion_missed_from_metrics_fails_without_metric():
    with pytest.raises(KeyError, match="mean_fn"):
        dw.champion_missed_from_metrics({"mean_recall": 0.909})


def test_worker_state_requires_champion_missed():
    # 하드코딩 1 이 사라졌다 — 워커는 champion run 에서 읽은 값을 넘겨야 한다.
    with pytest.raises(TypeError):
        dw.WorkerState()
```

`tests/test_config.py` 끝에:

```python
WORKER_WIRING = f"""
import json, sys
sys.path.insert(0, {MONITORING_DIR!r})
import drift_worker as dw
print(json.dumps({{
    "labels_db": str(dw.LABELS_DB),
    "model_dir": str(dw.MODEL_DIR),
    "backup_root": str(dw.BACKUP_ROOT),
    "gate_sample": dw.GATE_SAMPLE_SIZE,
    "cooldown": dw.COOLDOWN_DAYS,
    "k": dw.CONSECUTIVE_K,
}}))
"""


def test_worker_follows_config(tmp_path):
    got = _run_snippet(WORKER_WIRING, {
        "CNC_DATA_ROOT": str(tmp_path),
        "CNC_GATE_SAMPLE_SIZE": "5",
        "CNC_COOLDOWN_DAYS": "2",
        "CNC_CONSECUTIVE_K": "2",
    })

    root = str(tmp_path)
    assert got["labels_db"] == f"{root}/monitoring/labels.db"
    assert got["model_dir"] == f"{root}/model"
    assert got["backup_root"] == f"{root}/model_backup"
    assert (got["gate_sample"], got["cooldown"], got["k"]) == (5, 2, 2)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/monitoring/test_drift_worker.py tests/test_config.py::test_worker_follows_config -q`
Expected: 4 failed — `AttributeError: champion_missed_from_metrics`, `WorkerState()`가 TypeError 없이
생성됨, 배선 값 불일치

- [ ] **Step 3: `monitoring/drift_worker.py` 상단**

`from lstm_ae.tracking import (...)`에 `configure_tracking`을 추가하고, 그 import 블록 위(첫
first-party import)로:

```python
from config import CONSECUTIVE_K, COOLDOWN_DAYS, DATA_ROOT, GATE_SAMPLE_SIZE  # noqa: E402
```

31~54행의 경로·상수 블록을 다음으로 교체. `GATE_SAMPLE_SIZE = 20` 위의 긴 주석(60으로 올려봤다가
되돌린 이유)은 그대로 두고, 그 블록의 첫 줄을 `# GATE_SAMPLE_SIZE(기본 20)의 근거 — config.py 에서
읽는다.`로 바꾼다:

```python
LABELS_DB = DATA_ROOT / "monitoring" / "labels.db"
REQUESTS_DB = DATA_ROOT / "monitoring" / "requests.db"
MODEL_DIR = DATA_ROOT / "model"
SCALER_PATH = DATA_ROOT / "processed" / "scaler.json"
BACKUP_ROOT = DATA_ROOT / "model_backup"
SHADOW_DB = DATA_ROOT / "monitoring" / "shadow.db"
# COOLDOWN_DAYS(5)·CONSECUTIVE_K(3)·GATE_SAMPLE_SIZE(20)는 config.py 에서 온다(환경변수 CNC_*).
```

`COOLDOWN_DAYS = 5`, `CONSECUTIVE_K = 3`, `GATE_SAMPLE_SIZE = 20` 세 줄은 삭제한다.

- [ ] **Step 4: `WorkerState`와 헬퍼**

```python
@dataclass
class WorkerState:
    champion_missed: int                    # champion run 의 mean_fn — load_champion_missed() 로 읽는다
    flag_history: list[bool] = field(default_factory=list)
    cooldown_remaining: int = 0
    champion_accuracy: float = 0.0          # 첫 게이트 평가 시 측정값으로 대체
    shadow: ShadowState | None = None
    rag_corpus: list[dict] | None = None
    openai_client: object | None = None


def champion_missed_from_metrics(metrics: dict) -> int:
    """G1 기준 — champion 이 원본 eval 에서 놓친 불량 수. build_run_metrics 가 모든 run 에
    mean_fn 으로 기록한다. 예전엔 1 로 박혀 있어 champion 이 바뀌면 기준이 틀어졌다."""
    if "mean_fn" not in metrics:
        raise KeyError("champion run 에 mean_fn 지표가 없습니다 — G1 기준을 정할 수 없습니다")
    return int(metrics["mean_fn"])


def load_champion_missed() -> int:
    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    return champion_missed_from_metrics(client.get_run(mv.run_id).data.metrics)
```

- [ ] **Step 5: 경로 호출부**

`tick()`:

```python
    result = run_retraining(
        timeline_dir=DATA_ROOT / "timeline" / scenario,
        labels_db=LABELS_DB,
        current_day=current_day,
        data_root=DATA_ROOT,
    )
```

`_gate_predictions()`: `timeline_dir = ROOT / "data" / "timeline" / scenario` →
`timeline_dir = DATA_ROOT / "timeline" / scenario`.

`main()`: `state = WorkerState()` →

```python
    state = WorkerState(champion_missed=load_champion_missed())
    print(f"champion G1 기준: 원본 eval 놓침 {state.champion_missed}건", flush=True)
```

`monitoring/simulate_timeline.py` in-process 모드:

```python
    from drift_worker import WorkerState, load_champion_missed, tick  # 같은 폴더

    state = WorkerState(champion_missed=load_champion_missed())
```

- [ ] **Step 6: 통과 확인 + 전체 스위트**

Run: `uv run pytest tests/monitoring/test_drift_worker.py tests/test_config.py -q` → 9 passed
Run: `uv run pytest -q` → 224 passed
Run: `grep -n 'ROOT / "data"' monitoring/drift_worker.py monitoring/simulate_timeline.py` → 출력 없음

- [ ] **Step 7: 커밋**

```bash
git add 02-cnc-machining/monitoring/drift_worker.py 02-cnc-machining/monitoring/simulate_timeline.py \
  02-cnc-machining/tests/monitoring/test_drift_worker.py 02-cnc-machining/tests/test_config.py
git commit -m "feat(monitoring): worker reads config and champion miss count from MLflow"
```

---

### Task 5: pyproject — httpx2 본 의존성, integration 마커

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (`uv lock`이 갱신)

**Interfaces:**
- Produces: `pytest.mark.integration` 마커. `uv run pytest`는 그 마커를 제외하고,
  `uv run pytest -m integration`은 그것만 돈다. `httpx2`가 `--no-dev` 설치에도 들어간다(compose 워커·feeder).

- [ ] **Step 1: `pyproject.toml` 수정**

`dependencies`에 `"httpx2>=2.9.1",`을 `"fastapi>=0.140.7",` 다음 줄에 넣고 `dev` 그룹에서 지운다:

```toml
[dependency-groups]
dev = [
    "pytest>=9.1.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--import-mode=importlib -m 'not integration'"
markers = [
    "integration: 서버·워커를 서브프로세스로 띄우는 루프 통합 테스트. uv run pytest -m integration 으로 따로 돈다",
]
```

- [ ] **Step 2: lock 갱신·동기화**

Run: `uv lock && uv sync`
Expected: `uv.lock`에서 httpx2가 dev 그룹이 아니라 본 의존성으로 이동. `git diff --stat uv.lock`에 변경 있음.

- [ ] **Step 3: 기본 실행이 마커를 제외하는지 확인**

명령줄 `-m`은 addopts의 `-m`을 덮어쓴다(계획 작성 중 pytest 9.1로 확인: `-m integration`을 주면
integration 테스트만 선택된다). 그러므로 addopts 방식으로 충분하다.

Run: `uv run pytest -q 2>&1 | tail -1` → `224 passed`
Run: `uv run pytest -m integration --collect-only -q 2>&1 | tail -1` →
`224 deselected` 또는 `no tests collected`(아직 integration 테스트가 없다)

- [ ] **Step 4: 워커가 `--no-dev` 환경에서도 import되는지**

Run: `uv run --no-dev python -c "import httpx2; print('ok')"`
Expected: `ok`

- [ ] **Step 5: 커밋**

```bash
git add 02-cnc-machining/pyproject.toml 02-cnc-machining/uv.lock
git commit -m "build: httpx2 as a runtime dependency; integration marker excluded from the default pytest run"
```

---

### Task 6: 합성 데이터셋 생성기

**Files:**
- Create: `tests/integration/fixture_dataset.py`
- Create: `tests/integration/test_fixture_dataset.py`

**Interfaces:**
- Produces: `fixture_dataset.write_dataset(dataset_dir: Path, seed: int = 0) -> None` — `dataset_dir`는
  인덱스 `train.csv`가 놓이는 폴더(= `config.DATASET_DIR`). 그 아래 `EXPERIMENT_SUBDIR`
  (`"CNC Virtual Data set _v2"`)에 `experiment_XX.csv` 22개. 상수 `ROWS = 150`, `NOISY_TRAIN_ID = 17`.

- [ ] **Step 1: 실패하는 테스트** — `tests/integration/test_fixture_dataset.py`

```python
"""합성 데이터셋이 KAMP 원본과 같은 모양이라 전처리·학습·feeder 가 코드 변경 없이 도는지."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib 모드라 형제 모듈이 경로에 없다

from fixture_dataset import EXPERIMENT_SUBDIR, ROWS, write_dataset  # noqa: E402
from preprocessing.columns import DEAD_SENSOR_COLUMNS, FEATURE_COLUMNS  # noqa: E402
from preprocessing.split import (  # noqa: E402
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)


def _read(dataset_dir: Path, experiment_id: int) -> pd.DataFrame:
    return pd.read_csv(dataset_dir / EXPERIMENT_SUBDIR / f"experiment_{experiment_id:02d}.csv")


def test_index_lists_the_split_ids_with_consistent_labels(tmp_path):
    write_dataset(tmp_path)

    index = pd.read_csv(tmp_path / "train.csv").set_index("No")

    assert sorted(index.index) == sorted(
        TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS + EVAL_BAD_EXPERIMENT_IDS
    )
    for experiment_id in TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS:
        assert index.loc[experiment_id, "passed_visual_inspection"] == "yes"
    for experiment_id in EVAL_BAD_EXPERIMENT_IDS:
        assert index.loc[experiment_id, "passed_visual_inspection"] == "no"
    assert (index["machining_finalized"] == "yes").all()


def test_experiment_files_have_kamp_shape(tmp_path):
    write_dataset(tmp_path)

    df = _read(tmp_path, 1)

    assert len(df) == ROWS
    assert len(df.columns) == 48
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    assert set(DEAD_SENSOR_COLUMNS) <= set(df.columns)
    assert {"Machining_Process", "M_sequence_number", "M_CURRENT_PROGRAM_NUMBER"} <= set(df.columns)
    assert not df.isna().any().any()


def test_generation_is_deterministic(tmp_path):
    write_dataset(tmp_path / "a", seed=0)
    write_dataset(tmp_path / "b", seed=0)

    for name in ["train.csv", f"{EXPERIMENT_SUBDIR}/experiment_17.csv"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_noisy_train_and_bad_experiments_differ_from_baseline(tmp_path):
    write_dataset(tmp_path)

    base, noisy, bad = _read(tmp_path, 1), _read(tmp_path, 17), _read(tmp_path, 4)

    # 17 은 임계값을 끌어올리는 잡음 큰 train 실험(스펙 §2), 4 는 eval_bad
    assert noisy["X_ActualPosition"].diff().std() > 2 * base["X_ActualPosition"].diff().std()
    assert bad["S_OutputCurrent"].mean() > 2 * base["S_OutputCurrent"].mean()
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/integration/test_fixture_dataset.py -q`
Expected: `ModuleNotFoundError: No module named 'fixture_dataset'`

- [ ] **Step 3: `tests/integration/fixture_dataset.py`**

```python
"""KAMP CNC 데이터셋과 같은 모양의 합성 데이터셋 — 통합 테스트용.

목적은 배관 검증이지 모델 품질이 아니다. 값의 의미는 없고 형식만 같다:
인덱스 train.csv(No, material, feedrate, clamp_pressure, tool_condition, machining_finalized,
passed_visual_inspection)와 experiment_XX.csv(48컬럼). 실험 번호는 preprocessing/split.py 의
22개 그대로라 전처리·학습·승격 스크립트가 코드 변경 없이 돈다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.columns import DEAD_SENSOR_COLUMNS, FEATURE_COLUMNS
from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)

EXPERIMENT_SUBDIR = "CNC Virtual Data set _v2"
ROWS = 150
# 임계값은 train 실험 8개 점수의 p95 라 사실상 최댓값이다. 8개가 비슷하면 정상 배치의
# 점수/임계값 비율이 1.0 근처가 되어 변형 전에도 출력 드리프트(비율 > 0.8)가 켜진다.
# 실험 하나의 잡음을 키워 임계값을 끌어올리면 나머지 7개는 0.5 안팎에 머문다(스펙 §2).
NOISY_TRAIN_ID = 17
NOISE_FACTOR = 3.0
BAD_SCALE_COLUMNS = ["S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback"]
BAD_FACTOR = 3.0
CONSTANT_COLUMNS = {"S_SystemInertia": 12.0, "M_CURRENT_FEEDRATE": 6.0}  # 실험당 상수(설정값)


def write_dataset(dataset_dir: Path, seed: int = 0) -> None:
    dataset_dir = Path(dataset_dir)
    experiment_dir = dataset_dir / EXPERIMENT_SUBDIR
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment_id in sorted(TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS + EVAL_BAD_EXPERIMENT_IDS):
        bad = experiment_id in EVAL_BAD_EXPERIMENT_IDS
        rows.append({
            "No": experiment_id,
            "material": "wax",
            "feedrate": int(CONSTANT_COLUMNS["M_CURRENT_FEEDRATE"]),
            "clamp_pressure": 4,
            "tool_condition": "unworn",
            "machining_finalized": "yes",
            "passed_visual_inspection": "no" if bad else "yes",
        })
        _write_experiment(experiment_dir / f"experiment_{experiment_id:02d}.csv", experiment_id, bad, seed)

    pd.DataFrame(rows).to_csv(dataset_dir / "train.csv", index=False)


def _write_experiment(path: Path, experiment_id: int, bad: bool, seed: int) -> None:
    rng = np.random.default_rng([seed, experiment_id])
    t = np.arange(ROWS)
    noise = 0.5 * (NOISE_FACTOR if experiment_id == NOISY_TRAIN_ID else 1.0)

    data: dict[str, np.ndarray] = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        if col in CONSTANT_COLUMNS:
            data[col] = np.full(ROWS, CONSTANT_COLUMNS[col])
            continue
        phase = rng.uniform(0.0, 2.0 * np.pi)
        base = 100.0 + 10.0 * (i % 7)  # 컬럼마다 다른 수준
        data[col] = base + 5.0 * np.sin(2.0 * np.pi * t / 50.0 + phase) + rng.normal(0.0, noise, ROWS)
    if bad:
        for col in BAD_SCALE_COLUMNS:
            data[col] = data[col] * BAD_FACTOR
    for col in DEAD_SENSOR_COLUMNS:
        data[col] = np.zeros(ROWS)
    data["Machining_Process"] = np.array(["Layer 1 Up"] * ROWS)
    data["M_sequence_number"] = t
    data["M_CURRENT_PROGRAM_NUMBER"] = np.ones(ROWS, dtype=int)

    pd.DataFrame(data).to_csv(path, index=False)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/integration/test_fixture_dataset.py -q` → 4 passed
Run: `uv run pytest -q` → 228 passed (이 파일은 마커가 없어 기본 실행에 포함된다 — 의도한 것.
생성기 자체는 가볍다)

- [ ] **Step 5: 커밋**

```bash
git add 02-cnc-machining/tests/integration/fixture_dataset.py 02-cnc-machining/tests/integration/test_fixture_dataset.py
git commit -m "test(integration): synthetic KAMP-shaped dataset generator"
```

---

### Task 7: 통합 테스트 하네스 + 스모크

**Files:**
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_loop.py`

**Interfaces:**
- Consumes: Task 6의 `write_dataset`, Task 3의 `--start-day`, Task 4의 워커 시작 로그.
- Produces: 픽스처 `loop_factory(scenario: str, extra_env: dict | None = None) -> Loop`.
  `Loop.run_days(n)`(하루씩 feed → 워커가 그 날을 처리할 때까지 대기), `Loop.days() ->
  list[tuple[int, float, bool, str]]`(day, ratio, flagged, action), `Loop.health() -> dict`,
  `Loop.data_root: Path`. 상수 `LOOP_ENV`. (테스트 모듈은 conftest 를 import 하지 않는다 —
  pytest 가 플러그인으로 이미 로드한 모듈을 다시 import 하면 사본이 생긴다.)

- [ ] **Step 1: 실패하는 스모크 테스트** — `tests/integration/test_loop.py`

```python
"""재학습 루프 통합 테스트 — 서버·워커는 실제 서브프로세스, feeder 는 하루씩 동기화.
스펙 docs/specs/2026-09-15-cnc-loop-integration-test-design.md §3. 실데이터 불필요.
실행: uv run pytest -m integration -q (공유 서버에서는 nice -n 19)."""
import pytest

pytestmark = pytest.mark.integration


def test_harness_bootstraps_champion_and_feeds_one_day(loop_factory):
    loop = loop_factory("temperature")

    assert loop.health()["model_version"] == "1"

    loop.run_days(1)

    days = loop.days()
    assert [d for d, *_ in days] == [1]
    assert days[0][3] == "none"
    assert (loop.data_root / "monitoring" / "labels.db").exists()
    assert (loop.data_root / "timeline" / "temperature" / "day01_0.csv").exists()
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest -m integration -q`
Expected: ERROR — `fixture 'loop_factory' not found`

- [ ] **Step 3: `tests/integration/conftest.py`**

```python
"""루프 통합 테스트 하네스.

서버(uvicorn)와 워커(monitoring/drift_worker.py)는 실제 서브프로세스로 띄운다 — 지금까지 찾은
루프 버그(날짜 스킵, 섀도우 시작일, feeder 페이스, 승격 직후 재트리거)가 전부 프로세스 경계에서
나왔기 때문이다. feeder(monitoring/simulate_timeline.py)는 테스트가 하루씩 실행하고, 워커 로그에
그 날이 찍힌 뒤에야 다음 날을 보낸다. 재학습에 몇 초가 걸리든 결과가 타이밍에 의존하지 않는다.
"""

import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib 모드라 형제 모듈이 경로에 없다

from fixture_dataset import write_dataset  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]  # 02-cnc-machining/
SCRIPTS = PROJECT / "scripts"
MONITORING = PROJECT / "monitoring"
DATASET_DIRNAME = "CNC 비식별화 원본데이터_1209"  # config.DATASET_DIR 의 마지막 요소

# 스펙 §3 "루프 상수". 40일 루프를 며칠로 줄인다. DRIFT_START_DAY=2, TOTAL_DAYS=3, 상한 1.0 이라
# Day 3 부터 변형이 계단으로 고정된다. TOTAL_DAYS 는 램프 기울기에만 쓰이고 며칠을 보낼지는
# 테스트가 정한다.
LOOP_ENV = {
    "CNC_BATCHES_PER_DAY": "5",
    "CNC_DRIFT_WINDOW_SIZE": "5",
    "CNC_CONSECUTIVE_K": "3",
    "CNC_COOLDOWN_DAYS": "2",
    "CNC_GATE_SAMPLE_SIZE": "5",
    "CNC_LABEL_DELAY_DAYS": "1",
    "CNC_DRIFT_START_DAY": "2",
    "CNC_TOTAL_DAYS": "3",
    "CNC_DRIFT_MAX_PROGRESS": "1.0",
    "CNC_LABEL_FLIP_DAY": "3",
    "CNC_TRAIN_EPOCHS": "2",
}
DAY_LINE = re.compile(r"^Day (\d{2})\s+score/threshold=([\d.]+)\s+flagged=(True|False)\s+action=(\w+)")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run(cmd: list[str], env: dict, timeout: int = 600) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise AssertionError(
            f"실패: {' '.join(cmd)}\n--- stdout\n{result.stdout[-4000:]}\n--- stderr\n{result.stderr[-4000:]}"
        )
    return result


def bootstrap(data_root: Path, extra_env: dict | None = None) -> dict:
    """합성 데이터셋 → 전처리 → 학습 → champion v1. 기존 스크립트 3개를 그대로 쓴다."""
    env = {**os.environ, **LOOP_ENV, "CNC_DATA_ROOT": str(data_root), "PYTHONUNBUFFERED": "1"}
    env.pop("OPENAI_API_KEY", None)  # 셸에 키가 있어도 통합 테스트는 LLM 을 부르지 않는다
    env.update(extra_env or {})
    write_dataset(data_root / "dataset" / DATASET_DIRNAME)
    run([sys.executable, str(SCRIPTS / "run_preprocessing.py")], env)
    run([sys.executable, str(SCRIPTS / "run_lstm_training.py")], env)
    run([sys.executable, str(SCRIPTS / "promote_model.py"), "1"], env)
    return env


@dataclass
class Loop:
    scenario: str
    data_root: Path
    env: dict
    port: int = field(default_factory=free_port)
    server: subprocess.Popen | None = None
    worker: subprocess.Popen | None = None
    _handles: list = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def server_log(self) -> Path:
        return self.data_root / "server.log"

    @property
    def worker_log(self) -> Path:
        return self.data_root / "worker.log"

    def _open(self, path: Path):
        handle = path.open("w")
        self._handles.append(handle)
        return handle

    def start_server(self) -> None:
        self.server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "serving.app:app", "--port", str(self.port)],
            cwd=PROJECT, env=self.env, stdout=self._open(self.server_log), stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if httpx2.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return
            except httpx2.HTTPError:
                pass
            if self.server.poll() is not None:
                break
            time.sleep(0.5)
        raise AssertionError(f"서버가 /health 200 을 내지 않음\n{self.server_log.read_text()[-4000:]}")

    def start_worker(self) -> None:
        self.worker = subprocess.Popen(
            [
                sys.executable, str(MONITORING / "drift_worker.py"), self.scenario,
                "--base-url", self.base_url, "--poll-interval", "0.2",
            ],
            cwd=PROJECT, env=self.env, stdout=self._open(self.worker_log), stderr=subprocess.STDOUT,
        )

    def feed_day(self, day: int) -> None:
        run(
            [
                sys.executable, str(MONITORING / "simulate_timeline.py"), self.scenario,
                "--serve-url", self.base_url, "--start-day", str(day), "--days", str(day),
            ],
            self.env,
        )

    def days(self) -> list[tuple[int, float, bool, str]]:
        out = []
        for line in self.worker_log.read_text().splitlines():
            m = DAY_LINE.match(line)
            if m:
                out.append((int(m[1]), float(m[2]), m[3] == "True", m[4]))
        return out

    def wait_for_day(self, day: int, timeout: int = 180) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(d == day for d, *_ in self.days()):
                return
            if self.worker.poll() is not None:
                raise AssertionError(
                    f"워커가 종료됨 (exit {self.worker.returncode})\n{self.worker_log.read_text()[-4000:]}"
                )
            time.sleep(0.2)
        raise AssertionError(f"워커가 {timeout}초 안에 Day {day:02d} 를 처리하지 않음\n{self.worker_log.read_text()[-4000:]}")

    def run_days(self, days: int) -> None:
        for day in range(1, days + 1):
            self.feed_day(day)
            self.wait_for_day(day)

    def health(self) -> dict:
        return httpx2.get(f"{self.base_url}/health", timeout=5).json()

    def stop(self) -> None:
        for proc in (self.worker, self.server):
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        for handle in self._handles:
            handle.close()


@pytest.fixture
def loop_factory(tmp_path):
    loops: list[Loop] = []

    def make(scenario: str, extra_env: dict | None = None) -> Loop:
        data_root = tmp_path / scenario
        env = bootstrap(data_root, extra_env)
        loop = Loop(scenario=scenario, data_root=data_root, env=env)
        loops.append(loop)
        loop.start_server()
        loop.start_worker()
        return loop

    yield make
    for loop in loops:
        loop.stop()
```

- [ ] **Step 4: 통과 확인**

Run: `who && top -bn1 | head -5` (부하 확인) 뒤
`nice -n 19 uv run pytest -m integration -q -s 2>&1 | tail -5`
Expected: `1 passed`. 소요 시간을 적어 둔다(부트스트랩 포함 30~60초 예상).
실패하면 `tmp_path/temperature/server.log`·`worker.log`가 AssertionError 본문에 붙어 나온다.

- [ ] **Step 5: 기본 실행에서 제외되는지**

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: `228 passed, 1 deselected`

- [ ] **Step 6: 커밋**

```bash
git add 02-cnc-machining/tests/integration/conftest.py 02-cnc-machining/tests/integration/test_loop.py
git commit -m "test(integration): loop harness — bootstrap a champion, spawn server and worker, feed one day"
```

---

### Task 8: 승격 경로 테스트

**Files:**
- Modify: `tests/integration/test_loop.py`

**Interfaces:**
- Consumes: Task 7의 `loop_factory`, `Loop.run_days/days/health/data_root`.
  MLflow 태그 이름은 `monitoring/drift_worker.py`의 `_tag`·`_g2_tags`가 정한다: `scenario`,
  `trigger_day`, `gate_decision`, `gate_reject_reason`, `gate_g1_missed`, `gate_g2_sample_size`,
  `gate_g2_n_good`, `gate_g2_n_bad`, `gate_g2_fa_delta`, `gate_g2_miss_delta`, 섀도우 종료 시
  같은 run 에 `gate_decision=shadow_promoted|shadow_rejected`(덮어씀)와 `shadow_n_good` 등.

- [ ] **Step 1: 실패하는 테스트** — `tests/integration/test_loop.py` 끝에

```python
import hashlib
from pathlib import Path

from mlflow.tracking import MlflowClient

from monitoring.shadow_log import get_shadow_predictions

MODEL_NAME = "cnc-lstm-ae"


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _mlflow(loop) -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{loop.data_root / 'mlflow' / 'mlflow.db'}")


def _scenario_runs(client: MlflowClient, scenario: str) -> list:
    experiment = client.get_experiment_by_name(MODEL_NAME)
    return client.search_runs([experiment.experiment_id], filter_string=f"tags.scenario = '{scenario}'")


def _all_batch_ids(days: int, batches_per_day: int = 5) -> list[str]:
    return [f"day{day:02d}_{index}" for day in range(1, days + 1) for index in range(batches_per_day)]


def test_shift_scenario_reaches_shadow_and_promotion(loop_factory):
    """라벨은 전부 정상, Day 3 부터 위치 3축이 8σ 계단 이동 → 트리거 → 재학습 → 게이트 통과 →
    섀도우 → 승격. 판정의 옳고 그름이 아니라 배관이 승격 끝까지 닿는지 본다(스펙 §3)."""
    loop = loop_factory("temperature", {"CNC_POS_DRIFT": "8", "CNC_CUR_DRIFT": "1.0"})
    model_before = _md5(loop.data_root / "model" / "model.pt")
    scaler_before = _md5(loop.data_root / "processed" / "scaler.json")

    loop.run_days(12)

    days = loop.days()
    assert [d for d, *_ in days] == list(range(1, 13)), "날짜가 빠지거나 순서가 어긋남"
    actions = [action for *_, action in days]
    first_action_day = next(d for d, *_, action in days if action != "none")
    assert first_action_day >= 5, f"연속 3회 전에 트리거: {days}"
    assert "shadow_started" in actions and "promoted" in actions, actions
    assert actions.index("shadow_started") < actions.index("promoted")

    client = _mlflow(loop)
    runs = _scenario_runs(client, "temperature")
    assert runs, "재학습 run 이 MLflow 에 없음"
    tags = [run.data.tags for run in runs]
    assert any(t.get("gate_decision") == "shadow_promoted" for t in tags), tags
    promoted = next(t for t in tags if t.get("gate_decision") == "shadow_promoted")
    assert {"gate_g2_n_good", "gate_g2_fa_delta", "shadow_n_good", "shadow_fa_delta"} <= set(promoted)

    champion = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    assert champion != "1"
    assert loop.health()["model_version"] == champion

    model_after = _md5(loop.data_root / "model" / "model.pt")
    assert model_after != model_before
    assert _md5(loop.data_root / "processed" / "scaler.json") != scaler_before
    retrain_models = {_md5(d / "model.pt") for d in (loop.data_root / "retrain").iterdir() if (d / "model.pt").exists()}
    assert model_after in retrain_models, "디스크 정본이 어느 재학습 산출물과도 같지 않음"
    assert list((loop.data_root / "model_backup").iterdir()), "승격 전 백업이 없음"

    shadow = get_shadow_predictions(_all_batch_ids(12), loop.data_root / "monitoring" / "shadow.db")
    assert len(shadow) >= 5, "섀도우가 관찰한 배치가 5건 미만"
```

- [ ] **Step 2: 실패 확인**

Run: `nice -n 19 uv run pytest -m integration -q -k shift 2>&1 | tail -5`
Expected: 이 시점의 코드로는 **통과할 수도 있다** — 배관은 이미 다 있고 이 태스크가 만드는 것은
테스트뿐이기 때문이다. 통과하면 그대로 다음 단계로 간다(테스트가 기존 동작을 문서화한다).
실패하면 AssertionError 본문의 워커 로그로 원인을 나눈다:
  - `first_action_day < 5`: 변형 전 구간에서 헛트리거. `fixture_dataset.NOISE_FACTOR`를 5.0으로 올린다.
  - 12일 안에 `promoted` 없음(거부만 반복, 사유 "개선 없음"): 후보 오탐이 champion 과 같다는 뜻.
    `CNC_POS_DRIFT`를 "20"으로 올린다.
  - `shadow_pending` 이 계속됨: 라벨 지연·GATE_SAMPLE_SIZE 배선 문제 — `LOOP_ENV` 가 워커·feeder 양쪽에
    전달되는지 `env` 를 확인한다.
  - 워커가 죽음: 로그의 traceback 이 진짜 버그다. 고치고 스펙 정정 절에 적는다.

- [ ] **Step 3: 통과 확인**

Run: `nice -n 19 uv run pytest -m integration -q 2>&1 | tail -3`
Expected: `2 passed`. 소요 시간 기록.

- [ ] **Step 4: 커밋**

```bash
git add 02-cnc-machining/tests/integration/test_loop.py 02-cnc-machining/tests/integration/fixture_dataset.py
git commit -m "test(integration): shift scenario reaches shadow and promotion"
```

---

### Task 9: 거부 경로 테스트

**Files:**
- Modify: `tests/integration/test_loop.py`

**Interfaces:**
- Consumes: Task 8의 `_mlflow`, `_scenario_runs`, `_md5`, `MODEL_NAME`.

- [ ] **Step 1: 테스트 추가** — `tests/integration/test_loop.py` 끝에

```python
def test_fault_scenario_is_rejected_and_keeps_champion(loop_factory):
    """Day 3 부터 스핀들 부하 계단 상승 + QC 불합격 라벨 → 트리거 → 재학습 → G2 창에 정상 라벨이
    없어 거부 → 원인 추정 태그. champion 과 정본 파일은 그대로(스펙 §3)."""
    loop = loop_factory("tool_wear", {"CNC_WEAR_RATE": "20"})
    model_before = _md5(loop.data_root / "model" / "model.pt")
    scaler_before = _md5(loop.data_root / "processed" / "scaler.json")

    loop.run_days(8)

    days = loop.days()
    assert [d for d, *_ in days] == list(range(1, 9))
    actions = [action for *_, action in days]
    assert "rejected" in actions, actions
    assert "shadow_started" not in actions and "promoted" not in actions

    client = _mlflow(loop)
    rejected = [r for r in _scenario_runs(client, "tool_wear") if r.data.tags.get("gate_decision") == "rejected"]
    assert rejected, "거부 run 이 없음"
    for run in rejected:
        tags = run.data.tags
        assert "정상 라벨 없음" in tags["gate_reject_reason"], tags["gate_reject_reason"]
        assert tags["estimated_cause"] in {"tool_wear", "vibration_backlash"}
        assert "recommended_action" in tags  # RAG 없음 → 빈 문자열이지만 키는 남는다

    assert client.get_model_version_by_alias(MODEL_NAME, "champion").version == "1"
    assert loop.health()["model_version"] == "1"
    assert _md5(loop.data_root / "model" / "model.pt") == model_before
    assert _md5(loop.data_root / "processed" / "scaler.json") == scaler_before
    assert not (loop.data_root / "model_backup").exists()
```

- [ ] **Step 2: 실행**

Run: `nice -n 19 uv run pytest -m integration -q -k fault 2>&1 | tail -5`
Expected: `1 passed`. Task 8과 같은 이유로 처음부터 통과할 수 있다. 실패 시:
  - `rejected` 가 없고 `none` 만: 드리프트가 안 잡힘. 워커 로그의 `score/threshold` 와 `flagged` 를 본다.
    `CNC_WEAR_RATE` 를 "50" 으로 올린다.
  - 사유가 "정상 라벨 없음" 이 아님: G2 창에 정상 라벨이 섞였다는 뜻 — `CNC_LABEL_FLIP_DAY=3`,
    `CNC_LABEL_DELAY_DAYS=1` 이 feeder 에 전달됐는지 `labels.db` 를 sqlite3 로 열어 본다.
  - 워커가 `collect_normal_batches` 의 "정상 라벨 배치가 없습니다" 로 죽음: Day 1·2 라벨이 good 이
    아니라는 뜻 — `true_label` 과 `LABEL_FLIP_DAY` 배선을 본다.

- [ ] **Step 3: 전체 통합 + 단위**

Run: `nice -n 19 uv run pytest -m integration -q 2>&1 | tail -3` → `3 passed`, 총 소요 시간 기록
Run: `uv run pytest -q 2>&1 | tail -1` → `228 passed, 3 deselected`

- [ ] **Step 4: 커밋**

```bash
git add 02-cnc-machining/tests/integration/test_loop.py
git commit -m "test(integration): fault scenario is rejected and champion stays"
```

---

### Task 10: CI — 통합 테스트 job과 이미지 빌드 job

**Files:**
- Modify: `/home/sure/project/.github/workflows/cnc-tests.yml` (저장소 루트)

**Interfaces:**
- Consumes: Task 5의 마커, Task 11의 `docker-compose.yml`(빌드 job 은 Task 11 뒤에 녹색이 된다 —
  이 태스크에서는 파일만 쓰고, 검증은 Task 11 커밋 후 push 에서 한다).

- [ ] **Step 1: 워크플로 교체**

```yaml
name: CNC tests

on:
  push:
    branches: [main]
    paths: ['hanium/02-cnc-machining/**', '.github/workflows/cnc-tests.yml']
  pull_request:
    paths: ['hanium/02-cnc-machining/**', '.github/workflows/cnc-tests.yml']
  workflow_dispatch: {}

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: hanium/02-cnc-machining
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d # v10.0.1
      - run: uv sync
      - run: uv run pytest -q

  loop-integration:
    # 서버·워커를 서브프로세스로 띄워 재학습 루프(트리거→재학습→게이트→섀도우→승격/거부)를 끝까지 돈다.
    # 실데이터 없이 합성 데이터셋으로 champion 을 만든다 — docs/specs/2026-09-15-cnc-loop-integration-test-design.md
    runs-on: ubuntu-latest
    timeout-minutes: 20
    defaults:
      run:
        working-directory: hanium/02-cnc-machining
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d # v10.0.1
      - run: uv sync
      - run: uv run pytest -m integration -q

  docker-build:
    # Dockerfile 이 지금까지 한 번도 자동 검증된 적이 없다. main push 와 수동 실행에서만 빌드한다.
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: hanium/02-cnc-machining
    steps:
      - uses: actions/checkout@v7
      - run: docker compose build
```

- [ ] **Step 2: YAML 파싱 확인**

Run: `uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('/home/sure/project/.github/workflows/cnc-tests.yml').read_text()); print('yaml ok')"`
Expected: `yaml ok` (`yaml` 은 mlflow 의존성으로 이미 설치돼 있다)

- [ ] **Step 3: 커밋**

```bash
cd /home/sure/project
git add .github/workflows/cnc-tests.yml
git commit -m "ci: run the loop integration tests and build the compose stack"
```

---

### Task 11: docker-compose + README/STRUCTURE

**Files:**
- Create: `docker-compose.yml`
- Modify: `README.md` §2-7 끝, §2-8
- Modify: `docs/STRUCTURE.md` §2 표

**Interfaces:**
- Consumes: Task 5(`httpx2` 가 `--no-dev` 이미지에 들어감), Task 3·4(워커·feeder 가 컨테이너 안
  `/app/data` = `PROJECT_ROOT / "data"` 를 그대로 씀 — 환경변수 불필요).

- [ ] **Step 1: `docker-compose.yml`**

```yaml
# 서빙 + 감시 워커 + (선택) feeder. README §2-7 의 세 터미널을 한 명령으로.
#   docker compose up                                  # 서빙 + 워커(temperature)
#   SCENARIO=tool_wear docker compose up               # 다른 시나리오
#   DAYS=40 PACE=15 docker compose --profile demo up   # feeder 까지 (가상 타임라인 주입)
# data/ 는 호스트 볼륨 — 재학습·백업·MLflow 기록이 호스트에 남는다. 시나리오를 바꾸기 전엔
# data/monitoring/{labels,requests,shadow}.db 와 data/timeline/<이전> 을 비운다(README §2-7).
services:
  serving:
    build: .
    image: cnc-serving
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]
    env_file:
      - path: .env
        required: false
    healthcheck:
      # /health 는 champion 이 로드돼야 200 — healthy = 모델 준비됨
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 5s
      timeout: 3s
      retries: 24

  worker:
    image: cnc-serving
    command: uv run --no-sync python monitoring/drift_worker.py ${SCENARIO:-temperature} --base-url http://serving:8000
    volumes: ["./data:/app/data"]
    env_file:
      - path: .env
        required: false
    depends_on:
      serving:
        condition: service_healthy

  feeder:
    image: cnc-serving
    profiles: ["demo"]
    command: uv run --no-sync python monitoring/simulate_timeline.py ${SCENARIO:-temperature} --serve-url http://serving:8000 --days ${DAYS:-40} --pace-seconds ${PACE:-15}
    volumes: ["./data:/app/data"]
    depends_on:
      serving:
        condition: service_healthy
```

- [ ] **Step 2: YAML 파싱 확인** (docker 가 없는 서버)

Run: `uv run python -c "import yaml; d = yaml.safe_load(open('docker-compose.yml')); print(sorted(d['services']))"`
Expected: `['feeder', 'serving', 'worker']`

- [ ] **Step 3: README §2-7 끝에 추가** (`- 진행 경과와 실측 결과...` 항목 다음)

````markdown

**docker compose로 같은 것을 한 명령으로** (docker 가 있는 PC — 이미지 안에 코드·의존성, `data/`는 볼륨):

```bash
cd 02-cnc-machining
SCENARIO=temperature DAYS=40 PACE=15 docker compose --profile demo up --abort-on-container-exit
```

- `serving`(8000 포트) → healthy 가 되면 `worker`, `feeder` 가 순서대로 뜬다. `--profile demo` 를
  빼면 feeder 없이 서빙 + 워커만 뜬다(실트래픽을 직접 넣을 때).
- 루프 상수·경로는 `CNC_*` 환경변수로 바꿀 수 있다(`src/config.py`). 기본값은 이 절의 설명과 같다.
  예: `CNC_DRIFT_MAX_PROGRESS=1.0` 을 주면 40일 이후 변형이 더 커지지 않는다(섀도우 스펙의 정정 절).
- 시나리오 전환 전 DB·timeline 비우기는 세 터미널 방식과 같다.
- 리눅스 호스트에서는 컨테이너가 만든 `data/` 파일이 root 소유가 된다. 거슬리면 서비스마다
  `user: "1000:1000"` 을 붙인다.
````

§2-8 첫 문단 뒤에 한 줄: `워커·feeder까지 함께 띄우려면 §2-7 끝의 compose 설명을 본다.`

- [ ] **Step 4: `docs/STRUCTURE.md` §2 표에 두 행 추가** (`| 공통 | tests/ | ...` 행을 고치고 한 행 추가)

```markdown
| 공통 | `tests/` | 단위 테스트 228개(`src/` 구조를 따름) + `tests/integration/` 루프 통합 테스트 3개(서버·워커를 실제 프로세스로 띄워 합성 데이터로 트리거→재학습→게이트→섀도우→승격/거부까지, `uv run pytest -m integration`) |
| 운영 | `docker-compose.yml` | 서빙 + 워커 + feeder 를 한 명령으로. `data/` 는 볼륨 |
```

§4 "알아 둘 것"에 한 줄: `- 데이터 루트와 루프 상수는 `CNC_*` 환경변수로 바꿀 수 있다(`src/config.py`). 없으면 기본값.`

- [ ] **Step 5: 단위 스위트 (문서만 바꿨지만 습관)**

Run: `uv run pytest -q 2>&1 | tail -1` → `228 passed, 3 deselected`

- [ ] **Step 6: 커밋**

```bash
cd /home/sure/project/hanium
git add 02-cnc-machining/docker-compose.yml 02-cnc-machining/README.md 02-cnc-machining/docs/STRUCTURE.md
git commit -m "feat(deploy): docker compose stack for serving, worker, and feeder"
```

---

### Task 12: 실데이터 스모크, 스펙 정정 절, 리뷰

**Files:**
- Modify: `docs/specs/2026-09-15-cnc-loop-integration-test-design.md` (정정 절 추가)
- Modify: `../tasks/todo.md` (체크·리뷰)
- Modify: `docs/specs/2026-08-25-cnc-shadow-deployment-design.md` (알려진 한계 절에 한 줄)

**Interfaces:** 없음(문서).

- [ ] **Step 1: 실데이터 경로 불변 스모크 — 환경변수 없이 세 터미널 3일**

스펙 검증 5번. Day 1~3 은 변형 없는 구간이라 트리거가 걸리지 않고 champion 도 안 바뀐다.
실행 전 원본 DB·timeline 을 옮겨 두고 끝나면 되돌린다. PID 파일로 종료한다.

```bash
cd /home/sure/project/hanium/02-cnc-machining
who && top -bn1 | head -5
KEEP=$(mktemp -d /home/sure/.claude/jobs/29ca99ae/tmp/keep.XXXX)
mkdir -p "$KEEP/monitoring" && mv data/monitoring/labels.db data/monitoring/requests.db data/monitoring/shadow.db "$KEEP/monitoring/" 2>/dev/null
[ -d data/timeline/temperature ] && mv data/timeline/temperature "$KEEP/timeline_temperature"
md5sum data/model/model.pt data/processed/scaler.json > "$KEEP/hashes.before"

nice -n 19 uv run uvicorn serving.app:app --port 8918 > "$KEEP/server.log" 2>&1 & echo $! > "$KEEP/server.pid"
for i in $(seq 1 60); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8918/health | grep -q 200 && break; sleep 1; done
nice -n 19 uv run python monitoring/drift_worker.py temperature --base-url http://127.0.0.1:8918 --poll-interval 2 > "$KEEP/worker.log" 2>&1 & echo $! > "$KEEP/worker.pid"
sleep 15   # 워커가 torch·mlflow 를 import 하고 champion G1 기준을 읽을 시간
nice -n 19 uv run python monitoring/simulate_timeline.py temperature --serve-url http://127.0.0.1:8918 --days 3
sleep 15
grep -E "^Day|champion G1" "$KEEP/worker.log"
```

Expected: `champion G1 기준: 원본 eval 놓침 1건`, `Day 01`·`Day 02`·`Day 03` 이 순서대로, 전부
`flagged=False action=none`. Day 1 은 요청이 드리프트 창(10개)에 못 미쳐 `ratio=0.00`, 그 뒤는
0.6~0.7 대(08-24 기록의 무변형 기준 0.72 근처).

정리(별도 명령으로):

```bash
cd /home/sure/project/hanium/02-cnc-machining
kill $(cat "$KEEP/worker.pid") $(cat "$KEEP/server.pid"); sleep 2
md5sum data/model/model.pt data/processed/scaler.json | diff - "$KEEP/hashes.before" && echo "정본 불변"
rm -f data/monitoring/labels.db data/monitoring/requests.db data/monitoring/shadow.db
rm -rf data/timeline/temperature
mv "$KEEP/monitoring/"*.db data/monitoring/ 2>/dev/null; [ -d "$KEEP/timeline_temperature" ] && mv "$KEEP/timeline_temperature" data/timeline/temperature
uv run python -c "import sqlite3; print('requests', sqlite3.connect('data/monitoring/requests.db').execute('select count(*) from predict_log').fetchone()[0])"
git -C /home/sure/project/hanium status --short
```

Expected: `정본 불변`, `requests 8`(09-04 상태), git status 에 data/ 관련 변경 없음(원래 git 밖).

- [ ] **Step 2: 스펙 "실행 결과에 따른 정정" 절**

`docs/specs/2026-09-15-cnc-loop-integration-test-design.md` 끝에 실측을 적는다. 표 형식:

```markdown
## 실행 결과에 따른 정정 (2026-09-1X)

| 항목 | 결과 |
|---|---|
| `uv run pytest -q` | N passed, 3 deselected, X초 |
| `uv run pytest -m integration -q` (이 서버, nice) | 3 passed, X초 (부트스트랩 3회 포함) |
| 승격 경로 | 트리거 Day N, 게이트 통과 Day N, 섀도우 시작 Day N, 승격 Day N (버전 v) |
| 거부 경로 | 트리거 Day N·N, 사유 "정상 라벨 없음", estimated_cause=... |
| 실데이터 3일 스모크(환경변수 없음) | Day 01~03 flagged=False, 정본 해시 불변, G1 기준 1건 |
| CI | (push 후 사용자가 확인 — 워크플로 URL) |
| compose | 이 서버엔 docker 없음. YAML 파싱만. 개인 PC 리허설 항목으로 남김 |

계획과 달랐던 점: (Task 8·9 에서 진폭·잡음을 바꿨으면 그 값과 이유. 워커·서버 버그를 고쳤으면 그 내용.)
```

`docs/specs/2026-08-25-cnc-shadow-deployment-design.md` 의 알려진 한계(`--days`가 `TOTAL_DAYS`를
넘으면 변형이 커진다) 항목 끝에 한 문장: `2026-09-15: CNC_DRIFT_MAX_PROGRESS=1.0 을 주면 상한이
걸린다(src/config.py). 기존 실행 명령은 그대로다.`

- [ ] **Step 3: `tasks/todo.md` 체크와 리뷰 절**

Task 1 에서 만든 절의 항목을 전부 `[x]` 로 바꾸고 아래에:

```markdown
### 리뷰

- 통합 테스트 3개, 소요 X초. 단위 N개. 커밋 <첫 해시>..<마지막 해시>.
- 실행 중 발견한 것: (Task 8·9 의 조정, 버그 수정, 테스트 격리 문제 등 실제로 있었던 것만)
- 배운 것: (있으면 tasks/lessons.md 에도)
- 남은 일: CI 결과 확인(push 후), 개인 PC 에서 `docker compose build` 와 `--profile demo` 리허설
```

- [ ] **Step 4: 커밋**

```bash
cd /home/sure/project/hanium
git add 02-cnc-machining/docs/specs/2026-09-15-cnc-loop-integration-test-design.md \
  02-cnc-machining/docs/specs/2026-08-25-cnc-shadow-deployment-design.md tasks/todo.md
git commit -m "docs: record loop integration test results and the drift progress cap"
```

push 는 사용자가 결정한다. push 뒤 GitHub Actions 세 job 이 녹색인지 확인하고 URL 을 정정 절에 적는다.
