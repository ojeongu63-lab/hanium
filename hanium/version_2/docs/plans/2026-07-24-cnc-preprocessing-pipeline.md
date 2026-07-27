# CNC 전처리 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `version_2/data/dataset/CNC 비식별화 원본데이터_1209`의 원본 25개 실험(`train.csv` +
`experiment_01~25.csv`)을 실험 단위로 리키지 없이 정제·분할해, 준지도 이상탐지와 지도학습
분류기 두 트랙이 공통으로 쓸 `train.csv`/`eval.csv`/`scaler.json`/`manifest.json`을
`version_2/data/processed/`에 만든다.

**Architecture:** hanium 루트의 `src/preprocessing/` 모듈 스타일(단일 책임 소파일 + 얇은
`pipeline.py` 오케스트레이터)을 `version_2/src/preprocessing/` 아래 독립적으로 복제한다.
루트 `src/`와는 코드 공유 없음(사용자 확인: "독립 구조"). 각 함수는 순수 함수로 작성해
합성 데이터로 유닛테스트하고, 실제 25개 실험 전체 실행은 `scripts/run_preprocessing.py`로
수동 검증한다.

**Tech Stack:** Python 3.14, pandas, numpy, scikit-learn(StandardScaler만), pytest,
uv(패키지/실행 관리).

## Global Constraints

- Python `>=3.14`, `uv_build` 백엔드 (hanium 루트 `pyproject.toml`과 동일 패턴).
- pytest 설정: `testpaths = ["tests"]`, `addopts = "--import-mode=importlib"` — 테스트
  파일에 `__init__.py` 불필요.
- `version_2/` 아래 코드만 사용. hanium 루트 `src/`를 import하지 않는다.
- 새 의존성은 numpy/pandas/scikit-learn(+dev: pytest)만 — torch는 이번 전처리 스펙에서
  불필요하므로 추가하지 않는다.
- **커밋 정책(CLAUDE.md)**: 사용자가 명시적으로 요청하기 전까지 `git commit`을 실행하지
  않는다. 각 태스크의 마지막 단계는 "커밋"이 아니라 "`git add`로 스테이징"이다 — 실제
  커밋은 사용자가 전체 플랜 완료 후 요청할 때 한 번에 한다.
- 입력 CSV 경로에 공백과 한글이 섞여 있다(`CNC 비식별화 원본데이터_1209`,
  `CNC Virtual Data set _v2`) — 코드에서 항상 `pathlib.Path`로 조합하고, f-string에
  직접 공백 포함 경로를 하드코딩할 때도 따옴표 처리에 주의한다.

---

### Task 1: 프로젝트 스캐폴딩 + `columns.py`

**Files:**
- Create: `version_2/pyproject.toml`
- Create: `version_2/src/preprocessing/__init__.py` (빈 파일)
- Create: `version_2/src/preprocessing/columns.py`
- Create: `version_2/tests/preprocessing/test_columns.py`

**Interfaces:**
- Produces: `FEATURE_COLUMNS: list[str]` (41개), `DEAD_SENSOR_COLUMNS: list[str]`(4개),
  `METADATA_EXCLUDED_COLUMNS: list[str]`(3개), `select_features(df: pd.DataFrame) -> pd.DataFrame`
  — Task 7(`pipeline.py`)이 이 네 가지를 그대로 가져다 쓴다.

- [ ] **Step 1: `version_2/pyproject.toml` 작성**

```toml
[project]
name = "cnc-preprocessing"
version = "0.1.0"
description = "CNC machine AI dataset: preprocessing pipeline"
authors = [
    { name = "jwoh", email = "ojeongu63@gmail.com" }
]
requires-python = ">=3.14"
dependencies = [
    "numpy>=2.5.1",
    "pandas>=3.0.3",
    "scikit-learn>=1.9.0",
]

[build-system]
requires = ["uv_build>=0.11.30,<0.12.0"]
build-backend = "uv_build"

[dependency-groups]
dev = [
    "pytest>=9.1.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--import-mode=importlib"
```

- [ ] **Step 2: 빈 `version_2/src/preprocessing/__init__.py` 생성**

- [ ] **Step 3: 실패하는 테스트 작성 — `version_2/tests/preprocessing/test_columns.py`**

```python
from preprocessing.columns import (
    DEAD_SENSOR_COLUMNS,
    FEATURE_COLUMNS,
    METADATA_EXCLUDED_COLUMNS,
    select_features,
)
import pandas as pd


def test_feature_columns_has_41_entries_and_no_overlap_with_dropped():
    assert len(FEATURE_COLUMNS) == 41
    assert len(set(FEATURE_COLUMNS)) == 41
    assert set(FEATURE_COLUMNS).isdisjoint(DEAD_SENSOR_COLUMNS)
    assert set(FEATURE_COLUMNS).isdisjoint(METADATA_EXCLUDED_COLUMNS)


def test_dead_and_metadata_excluded_columns_match_spec():
    assert DEAD_SENSOR_COLUMNS == [
        "Z_CurrentFeedback",
        "Z_DCBusVoltage",
        "Z_OutputCurrent",
        "Z_OutputVoltage",
    ]
    assert METADATA_EXCLUDED_COLUMNS == [
        "M_CURRENT_PROGRAM_NUMBER",
        "M_sequence_number",
        "Machining_Process",
    ]


def test_select_features_keeps_only_feature_columns_in_order():
    df = pd.DataFrame({col: [1.0] for col in FEATURE_COLUMNS + ["extra_col"]})

    result = select_features(df)

    assert list(result.columns) == FEATURE_COLUMNS
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_columns.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing'` (아직 `columns.py`가 없음)

- [ ] **Step 5: `version_2/src/preprocessing/columns.py` 구현**

```python
import pandas as pd

DEAD_SENSOR_COLUMNS = [
    "Z_CurrentFeedback",
    "Z_DCBusVoltage",
    "Z_OutputCurrent",
    "Z_OutputVoltage",
]

METADATA_EXCLUDED_COLUMNS = [
    "M_CURRENT_PROGRAM_NUMBER",
    "M_sequence_number",
    "Machining_Process",
]

FEATURE_COLUMNS = [
    "X_ActualPosition",
    "X_ActualVelocity",
    "X_ActualAcceleration",
    "X_SetPosition",
    "X_SetVelocity",
    "X_SetAcceleration",
    "X_CurrentFeedback",
    "X_DCBusVoltage",
    "X_OutputCurrent",
    "X_OutputVoltage",
    "X_OutputPower",
    "Y_ActualPosition",
    "Y_ActualVelocity",
    "Y_ActualAcceleration",
    "Y_SetPosition",
    "Y_SetVelocity",
    "Y_SetAcceleration",
    "Y_CurrentFeedback",
    "Y_DCBusVoltage",
    "Y_OutputCurrent",
    "Y_OutputVoltage",
    "Y_OutputPower",
    "Z_ActualPosition",
    "Z_ActualVelocity",
    "Z_ActualAcceleration",
    "Z_SetPosition",
    "Z_SetVelocity",
    "Z_SetAcceleration",
    "S_ActualPosition",
    "S_ActualVelocity",
    "S_ActualAcceleration",
    "S_SetPosition",
    "S_SetVelocity",
    "S_SetAcceleration",
    "S_CurrentFeedback",
    "S_DCBusVoltage",
    "S_OutputCurrent",
    "S_OutputVoltage",
    "S_OutputPower",
    "S_SystemInertia",
    "M_CURRENT_FEEDRATE",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_columns.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: 스테이징 (커밋 아님 — Global Constraints 참고)**

```bash
git add version_2/pyproject.toml version_2/src/preprocessing/__init__.py version_2/src/preprocessing/columns.py version_2/tests/preprocessing/test_columns.py
```

---

### Task 2: `labels.py` — 양품/불량 라벨 도출

**Files:**
- Create: `version_2/src/preprocessing/labels.py`
- Create: `version_2/tests/preprocessing/test_labels.py`

**Interfaces:**
- Consumes: 없음 (순수 pandas 함수)
- Produces: `add_labels(df: pd.DataFrame) -> pd.DataFrame` — `machining_finalized`,
  `passed_visual_inspection` 컬럼이 있는 DataFrame을 받아 `label` 컬럼(0=양품/1=불량)을
  추가해 반환. Task 7(`pipeline.py`)이 실험 메타데이터(`train.csv`)에 적용한다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/preprocessing/test_labels.py`**

```python
import pandas as pd

from preprocessing.labels import add_labels


def test_finalized_and_passed_is_good():
    df = pd.DataFrame({
        "machining_finalized": ["yes"],
        "passed_visual_inspection": ["yes"],
    })

    result = add_labels(df)

    assert result["label"].tolist() == [0]


def test_not_finalized_is_bad():
    df = pd.DataFrame({
        "machining_finalized": ["no"],
        "passed_visual_inspection": [None],
    })

    result = add_labels(df)

    assert result["label"].tolist() == [1]


def test_finalized_but_failed_visual_inspection_is_bad():
    df = pd.DataFrame({
        "machining_finalized": ["yes"],
        "passed_visual_inspection": ["no"],
    })

    result = add_labels(df)

    assert result["label"].tolist() == [1]


def test_mixed_rows_labeled_independently():
    df = pd.DataFrame({
        "machining_finalized": ["yes", "no", "yes"],
        "passed_visual_inspection": ["yes", None, "no"],
    })

    result = add_labels(df)

    assert result["label"].tolist() == [0, 1, 1]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.labels'`

- [ ] **Step 3: `version_2/src/preprocessing/labels.py` 구현**

```python
import pandas as pd


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_good = (df["machining_finalized"] == "yes") & (df["passed_visual_inspection"] == "yes")
    df["label"] = (~is_good).astype(int)
    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_labels.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 스테이징**

```bash
git add version_2/src/preprocessing/labels.py version_2/tests/preprocessing/test_labels.py
```

---

### Task 3: `cleaning.py` — `Machining_Process` 대소문자 정리

**Files:**
- Create: `version_2/src/preprocessing/cleaning.py`
- Create: `version_2/tests/preprocessing/test_cleaning.py`

**Interfaces:**
- Produces: `normalize_machining_process(df: pd.DataFrame) -> pd.DataFrame` — `Machining_Process`
  컬럼의 `"end"`를 `"End"`로 통일해 반환. Task 7이 실험별 시계열 로드 직후 호출한다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/preprocessing/test_cleaning.py`**

```python
import pandas as pd

from preprocessing.cleaning import normalize_machining_process


def test_lowercase_end_normalized_to_capitalized():
    df = pd.DataFrame({"Machining_Process": ["Prep", "end", "End", "Layer 1 Up"]})

    result = normalize_machining_process(df)

    assert result["Machining_Process"].tolist() == ["Prep", "End", "End", "Layer 1 Up"]


def test_does_not_mutate_input_dataframe():
    df = pd.DataFrame({"Machining_Process": ["end"]})

    normalize_machining_process(df)

    assert df["Machining_Process"].tolist() == ["end"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_cleaning.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.cleaning'`

- [ ] **Step 3: `version_2/src/preprocessing/cleaning.py` 구현**

```python
import pandas as pd


def normalize_machining_process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Machining_Process"] = df["Machining_Process"].replace({"end": "End"})
    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_cleaning.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 스테이징**

```bash
git add version_2/src/preprocessing/cleaning.py version_2/tests/preprocessing/test_cleaning.py
```

---

### Task 4: `split.py` — 실험 단위 train/eval 분할 상수

**Files:**
- Create: `version_2/src/preprocessing/split.py`
- Create: `version_2/tests/preprocessing/test_split.py`

**Interfaces:**
- Produces: `TRAIN_EXPERIMENT_IDS: list[int]`(8개), `EVAL_GOOD_EXPERIMENT_IDS: list[int]`(5개),
  `EVAL_BAD_EXPERIMENT_IDS: list[int]`(12개) — Task 7과 Task 8(`scripts/run_preprocessing.py`)이
  가져다 쓴다. 스펙 문서(`version_2/docs/specs/2026-07-24-cnc-preprocessing-pipeline-design.md`)
  의 "Train / Eval 분할" 표와 정확히 일치해야 한다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/preprocessing/test_split.py`**

```python
from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)


def test_split_counts_match_spec():
    assert len(TRAIN_EXPERIMENT_IDS) == 8
    assert len(EVAL_GOOD_EXPERIMENT_IDS) == 5
    assert len(EVAL_BAD_EXPERIMENT_IDS) == 12


def test_split_partitions_all_25_experiments_with_no_overlap():
    all_ids = TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS + EVAL_BAD_EXPERIMENT_IDS

    assert sorted(all_ids) == list(range(1, 26))
    assert len(set(all_ids)) == 25


def test_exact_experiment_ids_match_spec():
    assert TRAIN_EXPERIMENT_IDS == [1, 2, 3, 11, 13, 14, 15, 17]
    assert EVAL_GOOD_EXPERIMENT_IDS == [12, 18, 22, 24, 25]
    assert EVAL_BAD_EXPERIMENT_IDS == [4, 5, 6, 7, 8, 9, 10, 16, 19, 20, 21, 23]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.split'`

- [ ] **Step 3: `version_2/src/preprocessing/split.py` 구현**

```python
TRAIN_EXPERIMENT_IDS = [1, 2, 3, 11, 13, 14, 15, 17]
EVAL_GOOD_EXPERIMENT_IDS = [12, 18, 22, 24, 25]
EVAL_BAD_EXPERIMENT_IDS = [4, 5, 6, 7, 8, 9, 10, 16, 19, 20, 21, 23]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_split.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 스테이징**

```bash
git add version_2/src/preprocessing/split.py version_2/tests/preprocessing/test_split.py
```

---

### Task 5: `scaling.py` — StandardScaler 래퍼

**Files:**
- Create: `version_2/src/preprocessing/scaling.py`
- Create: `version_2/tests/preprocessing/test_scaling.py`

**Interfaces:**
- Produces: `fit_scaler(df, feature_columns) -> StandardScaler`,
  `transform_features(df, feature_columns, scaler) -> pd.DataFrame`,
  `scaler_to_dict(scaler, feature_columns) -> dict` — Task 7이 train에만 fit하고 train/eval
  둘 다 transform할 때 사용.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/preprocessing/test_scaling.py`**

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_scaling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.scaling'`

- [ ] **Step 3: `version_2/src/preprocessing/scaling.py` 구현**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(df: pd.DataFrame, feature_columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_columns])
    return scaler


def transform_features(
    df: pd.DataFrame, feature_columns: list[str], scaler: StandardScaler
) -> pd.DataFrame:
    df = df.copy()
    df[feature_columns] = scaler.transform(df[feature_columns])
    return df


def scaler_to_dict(scaler: StandardScaler, feature_columns: list[str]) -> dict:
    return {
        col: {"mean": float(mean), "std": float(std)}
        for col, mean, std in zip(feature_columns, scaler.mean_, scaler.scale_)
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_scaling.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 스테이징**

```bash
git add version_2/src/preprocessing/scaling.py version_2/tests/preprocessing/test_scaling.py
```

---

### Task 6: `manifest.py` — 재현성 기록

**Files:**
- Create: `version_2/src/preprocessing/manifest.py`
- Create: `version_2/tests/preprocessing/test_manifest.py`

**Interfaces:**
- Consumes: 없음
- Produces: `build_manifest(**kwargs) -> dict` (키워드 전용 인자, 아래 시그니처 그대로) —
  Task 7이 파이프라인 실행 끝에 호출해 `manifest.json`으로 저장한다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/preprocessing/test_manifest.py`**

```python
from preprocessing.manifest import build_manifest


def test_build_manifest_records_split_and_columns():
    manifest = build_manifest(
        total_rows=32048,
        train_rows=14654,
        eval_rows=17394,
        eval_good_rows=7991,
        eval_bad_rows=9403,
        train_experiment_ids=[1, 2, 3, 11, 13, 14, 15, 17],
        eval_good_experiment_ids=[12, 18, 22, 24, 25],
        eval_bad_experiment_ids=[4, 5, 6, 7, 8, 9, 10, 16, 19, 20, 21, 23],
        feature_columns=["a", "b"],
        dead_sensor_columns=["dead1"],
        metadata_excluded_columns=["meta1"],
    )

    assert manifest["total_rows"] == 32048
    assert manifest["experiment_split"]["train"] == {
        "experiment_ids": [1, 2, 3, 11, 13, 14, 15, 17],
        "rows": 14654,
    }
    assert manifest["experiment_split"]["eval_good"] == {
        "experiment_ids": [12, 18, 22, 24, 25],
        "rows": 7991,
    }
    assert manifest["experiment_split"]["eval_bad"] == {
        "experiment_ids": [4, 5, 6, 7, 8, 9, 10, 16, 19, 20, 21, 23],
        "rows": 9403,
    }
    assert manifest["eval_rows"] == 17394
    assert manifest["feature_columns"] == ["a", "b"]
    assert manifest["dropped_columns"] == {"dead_sensors": ["dead1"]}
    assert manifest["metadata_excluded_columns"] == ["meta1"]
    assert "processed_at" in manifest
    assert "source" in manifest
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.manifest'`

- [ ] **Step 3: `version_2/src/preprocessing/manifest.py` 구현**

```python
from datetime import datetime, timezone

SOURCE = "CNC 비식별화 원본데이터_1209 (train.csv + experiment_01~25.csv)"


def build_manifest(
    *,
    total_rows: int,
    train_rows: int,
    eval_rows: int,
    eval_good_rows: int,
    eval_bad_rows: int,
    train_experiment_ids: list[int],
    eval_good_experiment_ids: list[int],
    eval_bad_experiment_ids: list[int],
    feature_columns: list[str],
    dead_sensor_columns: list[str],
    metadata_excluded_columns: list[str],
) -> dict:
    return {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "total_rows": total_rows,
        "experiment_split": {
            "train": {"experiment_ids": train_experiment_ids, "rows": train_rows},
            "eval_good": {"experiment_ids": eval_good_experiment_ids, "rows": eval_good_rows},
            "eval_bad": {"experiment_ids": eval_bad_experiment_ids, "rows": eval_bad_rows},
        },
        "eval_rows": eval_rows,
        "feature_columns": feature_columns,
        "dropped_columns": {"dead_sensors": dead_sensor_columns},
        "metadata_excluded_columns": metadata_excluded_columns,
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_manifest.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 스테이징**

```bash
git add version_2/src/preprocessing/manifest.py version_2/tests/preprocessing/test_manifest.py
```

---

### Task 7: `pipeline.py` — 오케스트레이터 (통합 테스트 포함)

**Files:**
- Create: `version_2/src/preprocessing/pipeline.py`
- Create: `version_2/tests/preprocessing/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1의 `FEATURE_COLUMNS`/`DEAD_SENSOR_COLUMNS`/`METADATA_EXCLUDED_COLUMNS`/
  `select_features`, Task 2의 `add_labels`, Task 3의 `normalize_machining_process`, Task 5의
  `fit_scaler`/`transform_features`/`scaler_to_dict`, Task 6의 `build_manifest`.
- Produces: `run_pipeline(experiment_index_path, experiment_dir, output_dir,
  train_experiment_ids, eval_good_experiment_ids, eval_bad_experiment_ids) -> dict` —
  Task 8(`scripts/run_preprocessing.py`)이 실제 25개 실험 경로와 Task 4의 상수를 넘겨 호출한다.
  실험 ID 목록을 인자로 받게 해서(하드코딩하지 않음) 합성 데이터로 통합테스트 가능하게 한다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/preprocessing/test_pipeline.py`**

```python
import json

import pandas as pd

from preprocessing.columns import FEATURE_COLUMNS
from preprocessing.pipeline import run_pipeline


def _make_experiment_csv(path, n_rows, machining_process="Prep"):
    data = {col: [float(i % 5) + 1 for i in range(n_rows)] for col in FEATURE_COLUMNS}
    data["Machining_Process"] = [machining_process] * n_rows
    data["M_sequence_number"] = list(range(n_rows))
    data["M_CURRENT_PROGRAM_NUMBER"] = [0] * n_rows
    pd.DataFrame(data).to_csv(path, index=False)


def _make_index_csv(path):
    pd.DataFrame({
        "No": [1, 2, 3, 4],
        "material": ["aluminum"] * 4,
        "feedrate": [3, 6, 3, 6],
        "clamp_pressure": [4, 4, 3, 3],
        "tool_condition": ["unworn", "unworn", "unworn", "worn"],
        "machining_finalized": ["yes", "yes", "yes", "yes"],
        "passed_visual_inspection": ["yes", "yes", "yes", "no"],
    }).to_csv(path, index=False)


def test_run_pipeline_creates_expected_output_files(tmp_path):
    experiment_dir = tmp_path / "experiments"
    experiment_dir.mkdir()
    index_path = tmp_path / "train.csv"
    output_dir = tmp_path / "processed"

    _make_index_csv(index_path)
    _make_experiment_csv(experiment_dir / "experiment_01.csv", n_rows=10)
    _make_experiment_csv(experiment_dir / "experiment_02.csv", n_rows=10)
    _make_experiment_csv(experiment_dir / "experiment_03.csv", n_rows=8)
    _make_experiment_csv(experiment_dir / "experiment_04.csv", n_rows=6, machining_process="end")

    manifest = run_pipeline(
        experiment_index_path=str(index_path),
        experiment_dir=str(experiment_dir),
        output_dir=str(output_dir),
        train_experiment_ids=[1, 2],
        eval_good_experiment_ids=[3],
        eval_bad_experiment_ids=[4],
    )

    for name in ["train.csv", "eval.csv", "scaler.json", "manifest.json"]:
        assert (output_dir / name).exists()

    train_df = pd.read_csv(output_dir / "train.csv")
    assert list(train_df.columns) == FEATURE_COLUMNS + ["experiment_id"]
    assert len(train_df) == 20
    assert set(train_df["experiment_id"]) == {1, 2}

    eval_df = pd.read_csv(output_dir / "eval.csv")
    assert len(eval_df) == 14
    assert set(eval_df["experiment_id"]) == {3, 4}
    assert eval_df.loc[eval_df["experiment_id"] == 3, "label"].unique().tolist() == [0]
    assert eval_df.loc[eval_df["experiment_id"] == 4, "label"].unique().tolist() == [1]
    assert set(eval_df.loc[eval_df["experiment_id"] == 4, "Machining_Process"]) == {"End"}

    scaler_dict = json.loads((output_dir / "scaler.json").read_text())
    assert set(scaler_dict.keys()) == set(FEATURE_COLUMNS)

    assert manifest["train_rows"] == 20
    assert manifest["eval_rows"] == 14
    assert manifest["experiment_split"]["eval_bad"]["rows"] == 6


def test_run_pipeline_train_never_touches_eval_experiments(tmp_path):
    experiment_dir = tmp_path / "experiments"
    experiment_dir.mkdir()
    index_path = tmp_path / "train.csv"
    output_dir = tmp_path / "processed"

    _make_index_csv(index_path)
    for i, n in zip([1, 2, 3, 4], [10, 10, 8, 6]):
        _make_experiment_csv(experiment_dir / f"experiment_{i:02d}.csv", n_rows=n)

    run_pipeline(
        experiment_index_path=str(index_path),
        experiment_dir=str(experiment_dir),
        output_dir=str(output_dir),
        train_experiment_ids=[1, 2],
        eval_good_experiment_ids=[3],
        eval_bad_experiment_ids=[4],
    )

    train_df = pd.read_csv(output_dir / "train.csv")
    eval_df = pd.read_csv(output_dir / "eval.csv")

    assert set(train_df["experiment_id"]).isdisjoint(set(eval_df["experiment_id"]))
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.pipeline'`

- [ ] **Step 3: `version_2/src/preprocessing/pipeline.py` 구현**

```python
import json
from pathlib import Path

import pandas as pd

from .cleaning import normalize_machining_process
from .columns import (
    DEAD_SENSOR_COLUMNS,
    FEATURE_COLUMNS,
    METADATA_EXCLUDED_COLUMNS,
    select_features,
)
from .labels import add_labels
from .manifest import build_manifest
from .scaling import fit_scaler, scaler_to_dict, transform_features

EVAL_METADATA_COLUMNS = [
    "experiment_id",
    "label",
    "tool_condition",
    "feedrate",
    "clamp_pressure",
    "material",
    "Machining_Process",
    "M_sequence_number",
    "M_CURRENT_PROGRAM_NUMBER",
]


def _load_experiment(experiment_dir: str, experiment_id: int) -> pd.DataFrame:
    path = Path(experiment_dir) / f"experiment_{experiment_id:02d}.csv"
    df = pd.read_csv(path)
    df = normalize_machining_process(df)
    df["experiment_id"] = experiment_id
    return df


def run_pipeline(
    experiment_index_path: str,
    experiment_dir: str,
    output_dir: str,
    train_experiment_ids: list[int],
    eval_good_experiment_ids: list[int],
    eval_bad_experiment_ids: list[int],
) -> dict:
    index = pd.read_csv(experiment_index_path)
    index = add_labels(index)
    index_by_id = index.set_index("No")

    eval_experiment_ids = set(eval_good_experiment_ids) | set(eval_bad_experiment_ids)

    train_frames = []
    eval_frames = []
    for experiment_id in train_experiment_ids:
        train_frames.append(_load_experiment(experiment_dir, experiment_id))
    for experiment_id in sorted(eval_experiment_ids):
        ts = _load_experiment(experiment_dir, experiment_id)
        meta = index_by_id.loc[experiment_id]
        ts["label"] = int(meta["label"])
        ts["tool_condition"] = meta["tool_condition"]
        ts["feedrate"] = meta["feedrate"]
        ts["clamp_pressure"] = meta["clamp_pressure"]
        ts["material"] = meta["material"]
        eval_frames.append(ts)

    train_raw = pd.concat(train_frames, ignore_index=True)
    eval_raw = pd.concat(eval_frames, ignore_index=True)

    scaler = fit_scaler(train_raw, FEATURE_COLUMNS)
    train_scaled = transform_features(train_raw, FEATURE_COLUMNS, scaler)
    eval_scaled = transform_features(eval_raw, FEATURE_COLUMNS, scaler)

    train_out = pd.concat(
        [select_features(train_scaled), train_scaled[["experiment_id"]]], axis=1
    )
    eval_out = pd.concat(
        [select_features(eval_scaled), eval_scaled[EVAL_METADATA_COLUMNS]], axis=1
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(out_dir / "train.csv", index=False)
    eval_out.to_csv(out_dir / "eval.csv", index=False)

    scaler_dict = scaler_to_dict(scaler, FEATURE_COLUMNS)
    (out_dir / "scaler.json").write_text(json.dumps(scaler_dict, indent=2, ensure_ascii=False))

    eval_good_rows = int((eval_out["label"] == 0).sum())
    eval_bad_rows = int((eval_out["label"] == 1).sum())

    manifest = build_manifest(
        total_rows=len(train_out) + len(eval_out),
        train_rows=len(train_out),
        eval_rows=len(eval_out),
        eval_good_rows=eval_good_rows,
        eval_bad_rows=eval_bad_rows,
        train_experiment_ids=list(train_experiment_ids),
        eval_good_experiment_ids=list(eval_good_experiment_ids),
        eval_bad_experiment_ids=list(eval_bad_experiment_ids),
        feature_columns=FEATURE_COLUMNS,
        dead_sensor_columns=DEAD_SENSOR_COLUMNS,
        metadata_excluded_columns=METADATA_EXCLUDED_COLUMNS,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/preprocessing/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 전체 유닛테스트 스위트 통과 확인**

Run: `cd version_2 && uv run pytest -v`
Expected: 모든 테스트 PASS (Task 1~7 누적, 총 16개)

- [ ] **Step 6: 스테이징**

```bash
git add version_2/src/preprocessing/pipeline.py version_2/tests/preprocessing/test_pipeline.py
```

---

### Task 8: `scripts/run_preprocessing.py` — 실제 25개 실험 전체 실행 + 수동 검증

**Files:**
- Create: `version_2/scripts/run_preprocessing.py`

**Interfaces:**
- Consumes: Task 4의 `TRAIN_EXPERIMENT_IDS`/`EVAL_GOOD_EXPERIMENT_IDS`/`EVAL_BAD_EXPERIMENT_IDS`,
  Task 7의 `run_pipeline`.
- Produces: 없음 (스크립트, 실행하면 `version_2/data/processed/`에 4개 파일 생성)

- [ ] **Step 1: `version_2/scripts/run_preprocessing.py` 작성**

```python
import json
from pathlib import Path

from preprocessing.pipeline import run_pipeline
from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209"


def main() -> None:
    manifest = run_pipeline(
        experiment_index_path=str(DATASET_DIR / "train.csv"),
        experiment_dir=str(DATASET_DIR / "CNC Virtual Data set _v2"),
        output_dir=str(ROOT / "data" / "processed"),
        train_experiment_ids=TRAIN_EXPERIMENT_IDS,
        eval_good_experiment_ids=EVAL_GOOD_EXPERIMENT_IDS,
        eval_bad_experiment_ids=EVAL_BAD_EXPERIMENT_IDS,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

Run: `cd version_2 && uv run python scripts/run_preprocessing.py`

- [ ] **Step 3: 출력을 스펙 문서의 실측값과 대조 (수동 검증)**

`manifest.json`(또는 stdout)에서 아래 값이 스펙(`version_2/docs/specs/2026-07-24-cnc-preprocessing-pipeline-design.md`)과 정확히 일치하는지 확인한다:

```
total_rows == 32048
train_rows == 14654
eval_rows == 17394
experiment_split.eval_good.rows == 7991
experiment_split.eval_bad.rows == 9403
```

하나라도 다르면 원본 CSV 자체가 스펙 작성 시점과 달라졌거나(재다운로드 등) 구현에 버그가
있다는 뜻이므로, 다음 단계로 넘어가지 말고 원인을 확인한다.

- [ ] **Step 4: 산출 파일 존재 확인**

Run: `ls -la version_2/data/processed/`
Expected: `train.csv`, `eval.csv`, `scaler.json`, `manifest.json` 4개 파일 존재

- [ ] **Step 5: 스테이징**

```bash
git add version_2/scripts/run_preprocessing.py
```

(`version_2/data/processed/`는 루트 `.gitignore`의 `data` 패턴에 이미 걸려 자동 제외됨 —
확인됨: `git check-ignore -v version_2/data/dataset`)

---

## 완료 후 체크리스트

- [ ] `cd version_2 && uv run pytest -v` 전체 통과 (16개 유닛/통합 테스트)
- [ ] `scripts/run_preprocessing.py` 실행 결과가 스펙의 실측값(14654/17394/7991/9403)과 일치
- [ ] `git status`로 스테이징된 파일 확인 — **커밋은 사용자가 명시적으로 요청할 때까지 하지 않음**
