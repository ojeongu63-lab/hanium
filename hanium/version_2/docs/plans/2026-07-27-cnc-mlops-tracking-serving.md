# CNC LSTM-AE 실험 추적 및 추론 서빙(MLOps) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 학습을 돌릴 때마다 MLflow(sqlite 백엔드)에 설정·지표·모델이 자동 기록되고, 사람이 검토 후 승격한 champion 모델을 FastAPI가 서빙해 원본 실험 CSV로 양품/불량을 판정하는 파이프라인을 만든다.

**Architecture:** `lstm_ae/tracking.py`가 MLflow 설정(sqlite URI, 실험/레지스트리 이름)과 param/metric 변환 로직을 한 곳에 모아, 학습 스크립트·승격 스크립트·서빙 앱이 모두 이 모듈을 통해서만 MLflow와 상호작용한다. 서빙(`src/serving/`)은 `lstm_ae`/`preprocessing`의 기존 함수(윈도잉·집계·피처 목록)를 재사용하고, FastAPI에 의존하지 않는 순수 함수(`inference.py`)와 라우팅(`app.py`)을 분리한다.

**Tech Stack:** Python 3.14, uv, PyTorch(CPU), MLflow 3.x(sqlite tracking+registry, 서버 프로세스 없음), FastAPI + uvicorn, pytest(`--import-mode=importlib`).

## Global Constraints

- MLflow tracking/registry URI: `sqlite:///{ROOT}/data/mlflow/mlflow.db` (`ROOT` = `version_2/`), 아티팩트 위치: `file://{ROOT}/data/mlflow/artifacts`. `data/`는 이미 `.gitignore` 대상.
- 실험(experiment) 이름 = 등록 모델(registered model) 이름 = `"cnc-lstm-ae"`, champion alias 이름 = `"champion"`. 이 세 상수는 `lstm_ae/tracking.py`에만 정의하고 다른 파일은 여기서 import해서 쓴다(중복 금지).
- `mlflow.pytorch.log_model(...)`은 반드시 `serialization_format="pickle"`을 명시한다 — 기본값 `pt2`(torch.export 기반)는 `nn.LSTM`을 포함한 모델에서 `Constraints violated (dynamic_dim)` 오류로 실패함을 스파이크로 확인했다.
- MLflow params에 리스트 값(실험 ID 목록 등)을 넣을 때는 `json.dumps(...)`로 문자열화한다(파이썬 `str()` repr에 의존하지 않음).
- FastAPI TestClient용 테스트 의존성은 `httpx`가 아니라 `httpx2`를 쓴다 — `httpx`는 최신 starlette에서 deprecated 경고를 낸다(스파이크로 확인).
- `FastAPI(lifespan=...)`가 등록된 앱에 대해 `TestClient(app)`을 `with` 없이 사용하면 lifespan(startup/shutdown)이 실행되지 않는다(스파이크로 확인) — 테스트에서 실제 MLflow 스토어를 건드리지 않기 위해 이 특성을 활용한다.
- 새 패키지 `src/serving/`은 기존 `src/preprocessing/`, `src/lstm_ae/`와 동일하게 패키지 루트에만 빈 `__init__.py`를 둔다. `tests/` 아래에는 `__init__.py`를 만들지 않는다(기존 컨벤션, `pyproject.toml`의 `--import-mode=importlib`로 불필요).
- 모든 신규/변경 코드는 `cd version_2 && uv run pytest tests/ -v`로 검증한다.

---

### Task 1: MLflow/FastAPI/uvicorn 의존성 추가

**Files:**
- Modify: `version_2/pyproject.toml`
- Test: `version_2/tests/test_mlops_dependencies.py`

**Interfaces:**
- Produces: `mlflow`, `mlflow.pytorch`, `fastapi`, `uvicorn` importable in the project's `uv` environment; `httpx2` importable in the dev/test environment.

- [ ] **Step 1: Write the failing test**

```python
# version_2/tests/test_mlops_dependencies.py
def test_mlflow_importable():
    import mlflow

    assert mlflow.__version__


def test_mlflow_pytorch_importable():
    import mlflow.pytorch  # noqa: F401


def test_fastapi_importable():
    import fastapi

    assert fastapi.__version__


def test_uvicorn_importable():
    import uvicorn

    assert uvicorn.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd version_2 && uv run pytest tests/test_mlops_dependencies.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlflow'` (dependency not declared yet)

- [ ] **Step 3: Add the dependencies**

Run:
```bash
cd version_2
uv add mlflow fastapi uvicorn
uv add --dev httpx2
```

This updates `[project].dependencies` and `[dependency-groups].dev` in `pyproject.toml` and refreshes `uv.lock`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd version_2 && uv run pytest tests/test_mlops_dependencies.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Regression check — existing suite still passes**

Run: `cd version_2 && uv run pytest tests/ -v`
Expected: All existing tests (preprocessing + lstm_ae) still PASS — new dependencies must not shift any resolved version (e.g. numpy/pandas/torch) in a breaking way.

- [ ] **Step 6: Commit**

```bash
cd version_2
git add pyproject.toml uv.lock tests/test_mlops_dependencies.py
git commit -m "Add mlflow, fastapi, uvicorn dependencies for MLOps tracking/serving"
```

---

### Task 2: `run_lstm_pipeline()`이 학습된 모델 객체를 반환하도록 수정

**Files:**
- Modify: `version_2/src/lstm_ae/pipeline.py:112-118`
- Modify: `version_2/tests/lstm_ae/test_pipeline.py:7`, `:88-90`

**Interfaces:**
- Consumes: none (existing `run_lstm_pipeline` signature unchanged)
- Produces: `run_lstm_pipeline(...)`'s return dict now includes `"model": <LSTMAutoencoder instance>`, used by Task 4's MLflow logging.

- [ ] **Step 1: Write the failing test**

In `version_2/tests/lstm_ae/test_pipeline.py`, add the import after line 7 (`from lstm_ae.pipeline import run_lstm_pipeline`):

```python
from lstm_ae.model import LSTMAutoencoder
```

Then add this assertion at the end of `test_run_lstm_pipeline_creates_expected_output_files` (after the existing line `assert summary["eval_windows"] == 30`):

```python
    assert isinstance(summary["model"], LSTMAutoencoder)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_pipeline.py -v`
Expected: FAIL — `KeyError: 'model'`

- [ ] **Step 3: Write minimal implementation**

In `version_2/src/lstm_ae/pipeline.py`, replace the final return statement (lines 112-118):

```python
    return {
        "train_windows": len(train_windows),
        "eval_windows": len(eval_windows),
        "final_train_loss": loss_history[-1],
        "thresholds": thresholds,
        "results": report,
    }
```

with:

```python
    return {
        "model": model,
        "train_windows": len(train_windows),
        "eval_windows": len(eval_windows),
        "final_train_loss": loss_history[-1],
        "thresholds": thresholds,
        "results": report,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd version_2
git add src/lstm_ae/pipeline.py tests/lstm_ae/test_pipeline.py
git commit -m "Return trained model object from run_lstm_pipeline"
```

---

### Task 3: `lstm_ae/tracking.py` — MLflow 설정 및 param/metric 변환 헬퍼

**Files:**
- Create: `version_2/src/lstm_ae/tracking.py`
- Test: `version_2/tests/lstm_ae/test_tracking.py`

**Interfaces:**
- Produces:
  - `EXPERIMENT_NAME: str = "cnc-lstm-ae"`, `REGISTERED_MODEL_NAME: str = "cnc-lstm-ae"`, `CHAMPION_ALIAS: str = "champion"`, `MLFLOW_DIR: Path` (module constants)
  - `configure_tracking(mlflow_dir: Path = MLFLOW_DIR) -> None` — sets the MLflow tracking URI to a local sqlite file under `mlflow_dir` and ensures the `cnc-lstm-ae` experiment exists with a local file artifact location. Idempotent.
  - `build_run_params(training_config: dict, manifest: dict) -> dict` — flattens training config + split experiment IDs (from a `preprocessing` `manifest.json`-shaped dict) into a single MLflow-params-ready dict.
  - `build_run_metrics(thresholds: dict, results: dict) -> dict` — flattens the `{mean,max,p95}` thresholds/results (same shape as `evaluation_report.json`) into a flat MLflow-metrics-ready dict.
- Consumed by: Task 4 (`run_lstm_training.py`), Task 5 (`promote_model.py`), Task 7 (`serving/app.py`).

- [ ] **Step 1: Write the failing tests**

```python
# version_2/tests/lstm_ae/test_tracking.py
import json

import mlflow

from lstm_ae.tracking import (
    EXPERIMENT_NAME,
    build_run_metrics,
    build_run_params,
    configure_tracking,
)


def test_configure_tracking_creates_db_and_experiment(tmp_path):
    configure_tracking(tmp_path)

    assert (tmp_path / "mlflow.db").exists()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None


def test_configure_tracking_is_idempotent(tmp_path):
    configure_tracking(tmp_path)
    configure_tracking(tmp_path)

    client = mlflow.tracking.MlflowClient()
    matches = [e for e in client.search_experiments() if e.name == EXPERIMENT_NAME]
    assert len(matches) == 1


def test_build_run_params_flattens_config_and_split():
    training_config = {"window_size": 20, "hidden_size": 64}
    manifest = {
        "experiment_split": {
            "train": {"experiment_ids": [1, 2, 3]},
            "eval_good": {"experiment_ids": [12, 18]},
            "eval_bad": {"experiment_ids": [4, 5]},
        }
    }

    params = build_run_params(training_config, manifest)

    assert params["window_size"] == 20
    assert params["hidden_size"] == 64
    assert json.loads(params["train_experiment_ids"]) == [1, 2, 3]
    assert json.loads(params["eval_good_experiment_ids"]) == [12, 18]
    assert json.loads(params["eval_bad_experiment_ids"]) == [4, 5]


def test_build_run_metrics_flattens_thresholds_and_results():
    thresholds = {"mean": 0.85, "max": 5.0, "p95": 2.8}
    results = {
        "mean": {"precision": 0.9, "recall": 0.9, "tp": 10, "fp": 1, "fn": 1, "tn": 2},
        "max": {"precision": 0.8, "recall": 0.9, "tp": 10, "fp": 2, "fn": 1, "tn": 1},
        "p95": {"precision": 0.9, "recall": 0.8, "tp": 9, "fp": 1, "fn": 2, "tn": 2},
    }

    metrics = build_run_metrics(thresholds, results)

    assert metrics["mean_threshold"] == 0.85
    assert metrics["mean_precision"] == 0.9
    assert metrics["mean_tp"] == 10
    assert metrics["max_fp"] == 2
    assert metrics["p95_tn"] == 2
    assert len(metrics) == 3 * 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.tracking'`

- [ ] **Step 3: Write minimal implementation**

```python
# version_2/src/lstm_ae/tracking.py
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent.parent
MLFLOW_DIR = ROOT / "data" / "mlflow"
EXPERIMENT_NAME = "cnc-lstm-ae"
REGISTERED_MODEL_NAME = "cnc-lstm-ae"
CHAMPION_ALIAS = "champion"


def configure_tracking(mlflow_dir: Path = MLFLOW_DIR) -> None:
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")

    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        artifacts_dir = mlflow_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        client.create_experiment(EXPERIMENT_NAME, artifact_location=f"file://{artifacts_dir}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def build_run_params(training_config: dict, manifest: dict) -> dict:
    split = manifest["experiment_split"]
    return {
        **training_config,
        "train_experiment_ids": json.dumps(split["train"]["experiment_ids"]),
        "eval_good_experiment_ids": json.dumps(split["eval_good"]["experiment_ids"]),
        "eval_bad_experiment_ids": json.dumps(split["eval_bad"]["experiment_ids"]),
    }


def build_run_metrics(thresholds: dict, results: dict) -> dict:
    metrics = {}
    for method in ["mean", "max", "p95"]:
        metrics[f"{method}_threshold"] = thresholds[method]
        r = results[method]
        metrics[f"{method}_precision"] = r["precision"]
        metrics[f"{method}_recall"] = r["recall"]
        metrics[f"{method}_tp"] = r["tp"]
        metrics[f"{method}_fp"] = r["fp"]
        metrics[f"{method}_fn"] = r["fn"]
        metrics[f"{method}_tn"] = r["tn"]
    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_tracking.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd version_2
git add src/lstm_ae/tracking.py tests/lstm_ae/test_tracking.py
git commit -m "Add MLflow tracking config and param/metric flattening helpers"
```

---

### Task 4: `run_lstm_training.py`에 MLflow 로깅 연동

**Files:**
- Modify: `version_2/scripts/run_lstm_training.py` (전체 재작성)

**Interfaces:**
- Consumes: `run_lstm_pipeline(...)` (Task 2, returns `"model"` key), `lstm_ae.tracking.{configure_tracking, build_run_params, build_run_metrics, REGISTERED_MODEL_NAME}` (Task 3)
- Produces: 실행할 때마다 MLflow에 새 run(params/metrics/모델 아티팩트+레지스트리 버전)이 기록됨. 이후 Task 5의 `promote_to_champion`이 승격할 대상.

이 태스크는 스펙(`docs/specs/2026-07-27-cnc-mlops-tracking-serving-design.md`, "5. 테스트 범위")에서 명시적으로 단위 테스트 대상에서 제외했다 — 실제 sqlite MLflow 스토어에 의존하는 통합 성격이라, 아래처럼 실제 실행 + MLflow 조회로 검증한다. 따라서 이 태스크는 표준 "실패하는 테스트 먼저" 패턴을 따르지 않는다.

- [ ] **Step 1: 전체 스크립트 재작성**

`version_2/scripts/run_lstm_training.py`의 기존 내용을 아래로 전체 교체한다:

```python
import json
from pathlib import Path

import mlflow
import mlflow.pytorch

from lstm_ae.pipeline import run_lstm_pipeline
from lstm_ae.tracking import (
    REGISTERED_MODEL_NAME,
    build_run_metrics,
    build_run_params,
    configure_tracking,
)
from preprocessing.columns import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent

TRAINING_CONFIG = {
    "window_size": 20,
    "hidden_size": 64,
    "latent_dim": 16,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "random_seed": 42,
    "threshold_percentile": 95.0,
}


def main() -> None:
    configure_tracking()
    manifest = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())

    with mlflow.start_run():
        mlflow.log_params(build_run_params(TRAINING_CONFIG, manifest))

        summary = run_lstm_pipeline(
            train_csv_path=str(ROOT / "data" / "processed" / "train.csv"),
            eval_csv_path=str(ROOT / "data" / "processed" / "eval.csv"),
            feature_columns=FEATURE_COLUMNS,
            output_dir=str(ROOT / "data" / "model"),
            **TRAINING_CONFIG,
        )

        mlflow.log_metrics(build_run_metrics(summary["thresholds"], summary["results"]))

        mlflow.pytorch.log_model(
            summary["model"],
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )

    print(f"train_windows: {summary['train_windows']}")
    print(f"eval_windows: {summary['eval_windows']}")
    print(f"final_train_loss: {summary['final_train_loss']:.6f}")
    print(f"thresholds: {summary['thresholds']}")
    for method in ["mean", "max", "p95"]:
        r = summary["results"][method]
        print(
            f"[{method}] precision={r['precision']:.4f} recall={r['recall']:.4f} "
            f"tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 회귀 확인 — 기존 lstm_ae 테스트가 여전히 통과하는지**

Run: `cd version_2 && uv run pytest tests/lstm_ae/ -v`
Expected: 모두 PASS (스크립트는 건드렸지만 `lstm_ae` 패키지 자체는 Task 2 이후 변경 없음)

- [ ] **Step 3: 실제 데이터로 실행해 MLflow 기록을 검증**

공유 서버 예절(CLAUDE.md) 준수 — 실행 전 `who`/`top`으로 부하 확인 후 `nice -n 19`로 실행:

Run: `cd version_2 && nice -n 19 uv run python scripts/run_lstm_training.py`
Expected: 기존과 동일한 콘솔 출력(train_windows/eval_windows/thresholds/precision/recall 등) + 에러 없이 종료.

이어서 아래 검증 스크립트로 MLflow에 실제로 기록됐는지 확인한다:

```bash
cd version_2 && uv run python -c "
from lstm_ae.tracking import configure_tracking, EXPERIMENT_NAME, REGISTERED_MODEL_NAME
import mlflow
from mlflow.tracking import MlflowClient

configure_tracking()
client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs([exp.experiment_id], order_by=['start_time DESC'], max_results=1)
run = runs[0]
print('run_id:', run.info.run_id)
print('params keys:', sorted(run.data.params.keys()))
print('mean_precision metric:', run.data.metrics.get('mean_precision'))
versions = client.search_model_versions(f\"name='{REGISTERED_MODEL_NAME}'\")
print('registered versions:', [v.version for v in versions])
"
```

Expected: `params keys`에 `window_size`, `train_experiment_ids` 등이 보이고, `mean_precision`이 이전에 확인한 값(약 0.909)과 일치하며, `registered versions`에 최소 1개 버전이 존재.

- [ ] **Step 4: Commit**

```bash
cd version_2
git add scripts/run_lstm_training.py
git commit -m "Log training runs to MLflow (params, metrics, registered model)"
```

---

### Task 5: Champion 승격 — `lstm_ae/tracking.py` 확장 + `promote_model.py`

**Files:**
- Modify: `version_2/src/lstm_ae/tracking.py` (append)
- Modify: `version_2/tests/lstm_ae/test_tracking.py` (append)
- Create: `version_2/scripts/promote_model.py`

**Interfaces:**
- Consumes: `configure_tracking`, `REGISTERED_MODEL_NAME`, `CHAMPION_ALIAS`, `MLFLOW_DIR` (Task 3)
- Produces: `promote_to_champion(version: str, mlflow_dir: Path = MLFLOW_DIR) -> None` — sets the `champion` alias on `cnc-lstm-ae`'s registry to the given version. Used by Task 7's serving app (via the alias, not the function directly) to know which model to load.

- [ ] **Step 1: Write the failing test**

Append to `version_2/tests/lstm_ae/test_tracking.py` (add these imports to the top alongside the existing ones, and the new test at the end):

```python
import mlflow.pytorch
import torch.nn as nn

from lstm_ae.tracking import promote_to_champion
```

```python
def test_promote_to_champion_sets_alias(tmp_path):
    configure_tracking(tmp_path)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, x):
            return self.lin(x)

    with mlflow.start_run():
        result = mlflow.pytorch.log_model(
            Tiny(),
            artifact_path="model",
            registered_model_name="cnc-lstm-ae",
            serialization_format="pickle",
        )

    promote_to_champion(result.registered_model_version, tmp_path)

    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias("cnc-lstm-ae", "champion")
    assert mv.version == result.registered_model_version
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_tracking.py::test_promote_to_champion_sets_alias -v`
Expected: FAIL — `ImportError: cannot import name 'promote_to_champion'`

- [ ] **Step 3: Write minimal implementation**

Append to `version_2/src/lstm_ae/tracking.py`:

```python
def promote_to_champion(version: str, mlflow_dir: Path = MLFLOW_DIR) -> None:
    configure_tracking(mlflow_dir)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_tracking.py -v`
Expected: PASS (all tests in the file, including the 4 from Task 3)

- [ ] **Step 5: `promote_model.py` CLI 작성**

```python
# version_2/scripts/promote_model.py
import argparse

from lstm_ae.tracking import promote_to_champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an MLflow model version to champion")
    parser.add_argument("version", help="Registered model version number to promote")
    args = parser.parse_args()
    promote_to_champion(args.version)
    print(f"promoted version {args.version} to champion")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 수동 확인 — Task 4에서 등록된 버전을 실제로 승격**

Run: `cd version_2 && uv run python scripts/promote_model.py 1`
Expected: `promoted version 1 to champion` 출력 (Task 4에서 등록된 버전 번호로 대체 — Step 3의 검증 스크립트 출력 `registered versions`를 참고)

- [ ] **Step 7: Commit**

```bash
cd version_2
git add src/lstm_ae/tracking.py tests/lstm_ae/test_tracking.py scripts/promote_model.py
git commit -m "Add champion alias promotion (tracking helper + CLI script)"
```

---

### Task 6: `serving/inference.py` — 순수 추론 로직

**Files:**
- Modify: `version_2/src/lstm_ae/pipeline.py:18`, `:60-61` (rename `_compute_window_errors` → `compute_window_errors`)
- Create: `version_2/src/serving/__init__.py` (빈 파일)
- Create: `version_2/src/serving/inference.py`
- Test: `version_2/tests/serving/test_inference.py`

**Interfaces:**
- Consumes: `lstm_ae.pipeline.compute_window_errors` (renamed public in this task), `lstm_ae.scoring.aggregate_window_errors_by_experiment`, `lstm_ae.sequencing.make_eval_windows` (기존)
- Produces:
  - `validate_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]`
  - `scale_features(df: pd.DataFrame, feature_columns: list[str], scaler_dict: dict) -> pd.DataFrame`
  - `score_to_label(score: float, threshold: float) -> tuple[int, str]`
  - `predict_experiment(df, model, feature_columns, scaler_dict, window_size, threshold, method) -> dict` (keys: `predicted_label`, `predicted_label_text`, `score`, `threshold`, `method`) — raises `ValueError` on missing columns or too-short experiment.
  - Consumed by Task 7's `serving/app.py`.

- [ ] **Step 1: `pipeline.py` 내부 함수 공개화 (선행 리팩터)**

`version_2/src/lstm_ae/pipeline.py`에서 `_compute_window_errors` 정의(18번 줄)와 두 호출부(60-61번 줄)의 이름을 `compute_window_errors`로 바꾼다:

```python
def compute_window_errors(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32)
        reconstructed = model(x).numpy()
    squared_errors = (reconstructed - windows) ** 2
    return squared_errors.reshape(len(windows), -1).mean(axis=1)
```

그리고 호출부:

```python
    train_window_errors = compute_window_errors(model, train_windows)
    eval_window_errors = compute_window_errors(model, eval_windows)
```

Run: `cd version_2 && uv run pytest tests/lstm_ae/ -v`
Expected: 모두 PASS (순수 이름 변경, 외부 동작 불변 — `tests/lstm_ae/test_pipeline.py`는 이 private 함수를 직접 참조하지 않음)

- [ ] **Step 2: Write the failing tests**

```python
# version_2/tests/serving/test_inference.py
import numpy as np
import pandas as pd
import pytest
import torch

from lstm_ae.model import LSTMAutoencoder
from serving.inference import (
    predict_experiment,
    scale_features,
    score_to_label,
    validate_columns,
)

FEATURE_COLUMNS = ["f0", "f1", "f2"]


def _scaler_dict():
    return {col: {"mean": 0.0, "std": 1.0} for col in FEATURE_COLUMNS}


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
        )


def test_predict_experiment_raises_on_too_short_experiment():
    df = _raw_df(5)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)
    with pytest.raises(ValueError, match="needs at least"):
        predict_experiment(
            df=df, model=model, feature_columns=FEATURE_COLUMNS,
            scaler_dict=_scaler_dict(), window_size=6, threshold=1.0, method="mean",
        )


def test_predict_experiment_returns_expected_shape():
    torch.manual_seed(0)
    np.random.seed(0)
    df = _raw_df(20)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)

    result = predict_experiment(
        df=df, model=model, feature_columns=FEATURE_COLUMNS,
        scaler_dict=_scaler_dict(), window_size=6, threshold=1.0, method="mean",
    )

    assert set(result.keys()) == {
        "predicted_label", "predicted_label_text", "score", "threshold", "method",
    }
    assert result["predicted_label"] in (0, 1)
    assert result["predicted_label_text"] in ("good", "bad")
    assert result["method"] == "mean"
    assert result["threshold"] == 1.0
    assert (result["predicted_label"] == 1) == (result["score"] > 1.0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd version_2 && uv run pytest tests/serving/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'serving'`

- [ ] **Step 4: Write minimal implementation**

`version_2/src/serving/__init__.py`은 완전히 빈 파일로 생성한다(내용 없음, `preprocessing`/`lstm_ae` 패키지 루트와 동일 컨벤션).

```python
# version_2/src/serving/inference.py
import pandas as pd
import torch

from lstm_ae.pipeline import compute_window_errors
from lstm_ae.scoring import aggregate_window_errors_by_experiment
from lstm_ae.sequencing import make_eval_windows

DEMO_EXPERIMENT_ID = 0


def validate_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [col for col in feature_columns if col not in df.columns]


def scale_features(
    df: pd.DataFrame, feature_columns: list[str], scaler_dict: dict
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_columns:
        mean = scaler_dict[col]["mean"]
        std = scaler_dict[col]["std"]
        df[col] = (df[col] - mean) / std
    return df


def score_to_label(score: float, threshold: float) -> tuple[int, str]:
    if score > threshold:
        return 1, "bad"
    return 0, "good"


def predict_experiment(
    df: pd.DataFrame,
    model: torch.nn.Module,
    feature_columns: list[str],
    scaler_dict: dict,
    window_size: int,
    threshold: float,
    method: str,
) -> dict:
    missing = validate_columns(df, feature_columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if len(df) < window_size:
        raise ValueError(
            f"experiment has {len(df)} rows, needs at least {window_size} rows "
            f"for window_size={window_size}"
        )

    scaled = scale_features(df, feature_columns, scaler_dict)
    scaled["experiment_id"] = DEMO_EXPERIMENT_ID

    windows, experiment_ids = make_eval_windows(scaled, feature_columns, window_size)
    window_errors = compute_window_errors(model, windows)
    experiment_scores = aggregate_window_errors_by_experiment(window_errors, experiment_ids)

    score = float(experiment_scores.loc[0, f"{method}_score"])
    predicted_label, predicted_label_text = score_to_label(score, threshold)

    return {
        "predicted_label": predicted_label,
        "predicted_label_text": predicted_label_text,
        "score": score,
        "threshold": threshold,
        "method": method,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd version_2 && uv run pytest tests/serving/test_inference.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: 패키지 임포트 가능 여부 확인 (uv_build 3번째 sibling 패키지)**

Run: `cd version_2 && uv run python -c "import serving; print('ok')"`
Expected: `ok` — `pyproject.toml`의 `[tool.uv.build-backend] module-name = "preprocessing"` 오버라이드가 `lstm_ae`처럼 `serving`도 막지 않는지 확인(이전에 `lstm_ae`로 검증된 것과 동일한 패턴).

- [ ] **Step 7: Commit**

```bash
cd version_2
git add src/lstm_ae/pipeline.py src/serving/__init__.py src/serving/inference.py tests/serving/test_inference.py
git commit -m "Add serving package with pure prediction logic (inference.py)"
```

---

### Task 7: `serving/app.py` — FastAPI 라우팅 + 전체 파이프라인 검증

**Files:**
- Create: `version_2/src/serving/app.py`
- Test: `version_2/tests/serving/test_app.py`

**Interfaces:**
- Consumes: `lstm_ae.tracking.{configure_tracking, REGISTERED_MODEL_NAME, CHAMPION_ALIAS}` (Task 3/5), `preprocessing.columns.FEATURE_COLUMNS` (기존), `serving.inference.predict_experiment` (Task 6)
- Produces: `app: FastAPI` with `GET /health`, `POST /predict`; `ModelState` dataclass; `get_model_state` FastAPI dependency (테스트에서 override 대상).

- [ ] **Step 1: Write the failing tests**

```python
# version_2/tests/serving/test_app.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd version_2 && uv run pytest tests/serving/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'serving.app'`

- [ ] **Step 3: Write minimal implementation**

```python
# version_2/src/serving/app.py
import io
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from mlflow.tracking import MlflowClient

from lstm_ae.tracking import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, configure_tracking
from preprocessing.columns import FEATURE_COLUMNS
from serving.inference import predict_experiment

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelState:
    model: torch.nn.Module
    scaler_dict: dict
    thresholds: dict
    window_size: int
    model_version: str
    mlflow_run_id: str


_state: ModelState | None = None


def load_model_state() -> ModelState:
    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    run = client.get_run(mv.run_id)
    thresholds = {
        method: run.data.metrics[f"{method}_threshold"] for method in ["mean", "max", "p95"]
    }
    window_size = int(run.data.params["window_size"])
    scaler_dict = json.loads((ROOT / "data" / "processed" / "scaler.json").read_text())
    return ModelState(
        model=model,
        scaler_dict=scaler_dict,
        thresholds=thresholds,
        window_size=window_size,
        model_version=mv.version,
        mlflow_run_id=mv.run_id,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    try:
        _state = load_model_state()
    except Exception as exc:
        print(
            "champion 모델을 로드하지 못했습니다 "
            f"(scripts/promote_model.py를 먼저 실행하세요): {exc}"
        )
        _state = None
    yield


app = FastAPI(lifespan=lifespan)


def get_model_state() -> ModelState:
    if _state is None:
        raise HTTPException(
            status_code=503,
            detail="champion 모델이 로드되지 않았습니다. scripts/promote_model.py를 먼저 실행하세요.",
        )
    return _state


@app.get("/health")
def health(state: ModelState = Depends(get_model_state)) -> dict:
    return {
        "status": "ok",
        "model_version": state.model_version,
        "mlflow_run_id": state.mlflow_run_id,
    }


@app.post("/predict")
async def predict(
    file: UploadFile,
    method: Literal["mean", "max", "p95"] = "mean",
    state: ModelState = Depends(get_model_state),
) -> dict:
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    try:
        result = predict_experiment(
            df=df,
            model=state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=state.scaler_dict,
            window_size=state.window_size,
            threshold=state.thresholds[method],
            method=method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **result,
        "model_version": state.model_version,
        "mlflow_run_id": state.mlflow_run_id,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd version_2 && uv run pytest tests/serving/test_app.py -v`
Expected: PASS (5 tests) — lifespan은 `with` 없이 쓰는 `TestClient(app)`에서 실행되지 않으므로 실제 MLflow 스토어를 건드리지 않는다(Global Constraints 참고).

- [ ] **Step 5: 전체 테스트 스위트 재확인**

Run: `cd version_2 && uv run pytest tests/ -v`
Expected: 모든 테스트(preprocessing + lstm_ae + serving + 의존성) PASS

- [ ] **Step 6: 전체 파이프라인 수동 검증 (스펙 "6. 검증 방법")**

서버 기동:

Run: `cd version_2 && nice -n 19 uv run uvicorn serving.app:app --reload &`
(백그라운드로 띄우고, 아래 확인 후 `kill %1`로 종료)

Health 확인:

Run: `curl -s http://127.0.0.1:8000/health`
Expected: `{"status":"ok","model_version":"1","mlflow_run_id":"..."}` (Task 5에서 승격한 버전과 일치)

eval 세트 실험(양품 1개, 불량 1개)으로 예측 확인 — `EVAL_GOOD_EXPERIMENT_IDS = [12, 18, 22]`, `EVAL_BAD_EXPERIMENT_IDS = [4, 5, 6, 7, 8, 9, 10, 16, 20, 21, 23]`(`src/preprocessing/split.py`)에서 각각 하나씩 골라(예: 12, 4), 원본 실험 CSV(경로에 공백이 있으니 따옴표 필수)를 업로드:

```bash
RAW_DIR="data/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2"
curl -s -X POST "http://127.0.0.1:8000/predict?method=mean" \
  -F "file=@$RAW_DIR/experiment_12.csv"   # 양품 실험(EVAL_GOOD_EXPERIMENT_IDS 중 하나)
curl -s -X POST "http://127.0.0.1:8000/predict?method=mean" \
  -F "file=@$RAW_DIR/experiment_04.csv"   # 불량 실험(EVAL_BAD_EXPERIMENT_IDS 중 하나)
```

Expected: 각각 `predicted_label_text`가 `"good"`/`"bad"`로, 실제 라벨과 일치.

에러 케이스 확인:

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" -F "file=@/dev/null"
```

Expected: HTTP 400, `missing required columns`를 포함한 에러 메시지.

서버 종료: `kill %1`

- [ ] **Step 7: Commit**

```bash
cd version_2
git add src/serving/app.py tests/serving/test_app.py
git commit -m "Add FastAPI serving app (/predict, /health) loading the champion model"
```
