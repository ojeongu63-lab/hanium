# CN7 전처리 파이프라인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `data/dataset/cn7_labeled.csv`/`cn7_unlabeled.csv`를 입력으로 받아 `docs/superpowers/specs/2026-07-21-cn7-preprocessing-pipeline-design.md`에 설계된 정제 → 자가정제 → 스케일링 → 분할 파이프라인을 실행하고, LSTM-AE 학습에 바로 투입 가능한 `train.csv`/`eval.csv`/`scaler.json`/`removed_outliers.csv`/`manifest.json`을 `data/processed/`에 생성한다.

**Architecture:** `src/preprocessing/` 아래 단계별 순수 함수 모듈(로더 / 완전중복제거 / 피처컬럼선택 / 라벨인코딩 / 자가정제 / 스케일링 / manifest)을 각각 TDD로 구현하고, `pipeline.py`가 이들을 조합해 5개 산출물을 생성한다. `scripts/run_preprocessing.py`가 실제 데이터로 파이프라인을 실행하는 엔트리포인트다.

**Tech Stack:** Python 3.14(서버에 이미 설치됨), uv(패키지/venv 관리), pandas, numpy, scikit-learn(IsolationForest, StandardScaler), pytest.

## Global Constraints

- `StandardScaler`는 자가정제 후 train(unlabeled)에만 `fit`하고, eval(labeled)에는 동일 스케일러로 `transform`만 적용한다 — 재fit 금지(리키지 방지).
- `IsolationForest`는 자가정제 전용 도구이며 `contamination=0.01`(1%)로 고정한다. 최종 LSTM-AE 모델과는 별개다(이 계획 범위 밖).
- eval 데이터는 어떤 단계에서도 학습(`fit`)에 사용하지 않는다.
- validation split은 두지 않는다 — train/eval 2분할만 존재한다.
- labeled 데이터의 완전 중복행(전체 컬럼 값이 동일한 행)은 제거한다 — 이번 세션에서 실측(82%, 6,736→3,974행) 후 사용자 승인됨. unlabeled는 중복이 0건이라 대상이 아니다.
- `Reason == '초기허용불량'`인 행은 물리적 불량이 아니므로 정상(`label=0`)으로 재분류한다.
- 파이썬 패키지 관리는 uv로 한다 — `pip install --break-system-packages` 금지.
- `data/`는 이미 `.gitignore` 처리되어 있으므로 `data/processed/` 산출물은 git에 커밋하지 않는다.

---

## 배경 메모 (실측으로 확정된 수치)

계획 작성 중 스펙에 없던 이슈를 발견해 사용자와 확인했다:

- `cn7_labeled.csv` 6,736행 중 5,524행(82%)이 완전 중복(정확히 2번씩 반복, 3번 이상 없음). 중복 제거 후 3,974행.
- 중복 제거 후 최종 라벨 분포: **정상 3,956 / 가스 13 / 미성형 5** (총 불량 18건). 스펙 문서의 "정상 6,717/불량 19"는 중복 포함 수치이므로 이 계획에서는 갱신된 수치를 기준으로 한다.
- `cn7_unlabeled.csv`(35,239행)는 중복 0건 — 대상 아님.
- 피처 컬럼(모델 입력)은 정확히 24개이며 순서 고정: `Injection_Time, Filling_Time, Plasticizing_Time, Cycle_Time, Clamp_Close_Time, Cushion_Position, Plasticizing_Position, Clamp_Open_Position, Max_Injection_Speed, Max_Screw_RPM, Average_Screw_RPM, Max_Injection_Pressure, Max_Switch_Over_Pressure, Max_Back_Pressure, Average_Back_Pressure, Barrel_Temperature_1~6, Hopper_Temperature, Mold_Temperature_3, Mold_Temperature_4`.
- `IsolationForest(contamination=0.01, random_state=42)`를 unlabeled 35,239행에 실측 적용한 결과: **353개 제거 → train 34,886행**. (Task 9의 실행 검증 기대값으로 사용)

---

### Task 1: 프로젝트 셋업 + 데이터 로더

**Files:**
- Create: `pyproject.toml` (uv init으로 생성)
- Create: `src/preprocessing/__init__.py`
- Create: `src/preprocessing/data_io.py`
- Test: `tests/preprocessing/test_data_io.py`

**Interfaces:**
- Produces: `load_csv(path: str) -> pd.DataFrame` — `_id` 컬럼을 문자열로 강제 로드(MongoDB ObjectId 형태 문자열이 숫자로 오인되는 것 방지).

- [ ] **Step 1: uv 프로젝트 초기화**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/sure/project/hanium
uv init --lib --no-readme --name preprocessing
```

Expected: `Initialized project `preprocessing`` 출력, `pyproject.toml`과 `src/preprocessing/__init__.py`, `src/preprocessing/py.typed`, `.python-version` 생성됨.

- [ ] **Step 2: 의존성 추가**

```bash
uv add pandas numpy scikit-learn
uv add --dev pytest
```

Expected: `pyproject.toml`의 `dependencies`에 pandas/numpy/scikit-learn이, `[dependency-groups] dev`에 pytest가 추가되고 `.venv`, `uv.lock`이 생성됨.

- [ ] **Step 3: pytest 탐색 경로 설정**

`pyproject.toml`에 아래 섹션을 추가한다(파일 끝에):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: 자동 생성된 스텁 정리**

`src/preprocessing/__init__.py` 내용을 비운다(빈 파일로):

```python
```

- [ ] **Step 5: 실패하는 테스트 작성**

`tests/preprocessing/test_data_io.py`:

```python
from preprocessing.data_io import load_csv


def test_load_csv_preserves_id_as_string(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("_id,value\n0001,1.5\n0002,2.5\n")

    df = load_csv(str(csv_path))

    assert df["_id"].tolist() == ["0001", "0002"]
    assert df["value"].tolist() == [1.5, 2.5]
```

- [ ] **Step 6: 테스트 실패 확인**

```bash
uv run pytest tests/preprocessing/test_data_io.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.data_io'`

- [ ] **Step 7: 최소 구현**

`src/preprocessing/data_io.py`:

```python
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"_id": str})
```

- [ ] **Step 8: 테스트 통과 확인**

```bash
uv run pytest tests/preprocessing/test_data_io.py -v
```

Expected: PASS

- [ ] **Step 9: 커밋**

```bash
git add pyproject.toml uv.lock .python-version src/preprocessing/__init__.py src/preprocessing/py.typed src/preprocessing/data_io.py tests/preprocessing/test_data_io.py
git commit -m "$(cat <<'EOF'
Set up uv project and add CSV loader

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 완전 중복행 제거

**Files:**
- Create: `src/preprocessing/dedup.py`
- Test: `tests/preprocessing/test_dedup.py`

**Interfaces:**
- Produces: `remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame` — 모든 컬럼 값이 동일한 행을 첫 번째만 남기고 제거, 인덱스 리셋.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/preprocessing/test_dedup.py`:

```python
import pandas as pd

from preprocessing.dedup import remove_exact_duplicates


def test_removes_fully_identical_rows():
    df = pd.DataFrame({
        "_id": ["a", "b", "a"],
        "value": [1, 2, 1],
    })

    result = remove_exact_duplicates(df)

    assert len(result) == 2
    assert result["_id"].tolist() == ["a", "b"]


def test_keeps_rows_with_same_id_but_different_values():
    df = pd.DataFrame({
        "_id": ["a", "a"],
        "value": [1, 2],
    })

    result = remove_exact_duplicates(df)

    assert len(result) == 2
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/preprocessing/test_dedup.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.dedup'`

- [ ] **Step 3: 최소 구현**

`src/preprocessing/dedup.py`:

```python
import pandas as pd


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/preprocessing/test_dedup.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing/dedup.py tests/preprocessing/test_dedup.py
git commit -m "$(cat <<'EOF'
Add exact-duplicate row removal

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 피처 컬럼 화이트리스트

**Files:**
- Create: `src/preprocessing/columns.py`
- Test: `tests/preprocessing/test_columns.py`

**Interfaces:**
- Produces: `FEATURE_COLUMNS: list[str]`(24개, 순서 고정), `DEAD_SENSOR_COLUMNS: list[str]`(10개), `DISABLED_SENSOR_COLUMNS: list[str]`(2개), `select_features(df: pd.DataFrame) -> pd.DataFrame`.
- 다음 태스크(라벨 인코딩, 자가정제, 스케일링, manifest, pipeline)가 모두 `FEATURE_COLUMNS`를 import해 사용한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/preprocessing/test_columns.py`:

```python
import pandas as pd

from preprocessing.columns import (
    DEAD_SENSOR_COLUMNS,
    DISABLED_SENSOR_COLUMNS,
    FEATURE_COLUMNS,
    select_features,
)


def test_feature_columns_has_24_entries():
    assert len(FEATURE_COLUMNS) == 24


def test_dropped_sensor_columns_are_not_in_feature_columns():
    for col in DEAD_SENSOR_COLUMNS + DISABLED_SENSOR_COLUMNS:
        assert col not in FEATURE_COLUMNS


def test_select_features_returns_only_whitelisted_columns_in_order():
    data = {col: [0.0] for col in FEATURE_COLUMNS}
    data["_id"] = ["x"]
    data["PassOrFail"] = ["Y"]
    data["Mold_Temperature_1"] = [0.0]
    df = pd.DataFrame(data)

    result = select_features(df)

    assert list(result.columns) == FEATURE_COLUMNS
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/preprocessing/test_columns.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.columns'`

- [ ] **Step 3: 최소 구현**

`src/preprocessing/columns.py`:

```python
import pandas as pd

DEAD_SENSOR_COLUMNS = [
    "Mold_Temperature_1",
    "Mold_Temperature_2",
    "Mold_Temperature_5",
    "Mold_Temperature_6",
    "Mold_Temperature_7",
    "Mold_Temperature_8",
    "Mold_Temperature_9",
    "Mold_Temperature_10",
    "Mold_Temperature_11",
    "Mold_Temperature_12",
]

DISABLED_SENSOR_COLUMNS = ["Switch_Over_Position", "Barrel_Temperature_7"]

FEATURE_COLUMNS = [
    "Injection_Time",
    "Filling_Time",
    "Plasticizing_Time",
    "Cycle_Time",
    "Clamp_Close_Time",
    "Cushion_Position",
    "Plasticizing_Position",
    "Clamp_Open_Position",
    "Max_Injection_Speed",
    "Max_Screw_RPM",
    "Average_Screw_RPM",
    "Max_Injection_Pressure",
    "Max_Switch_Over_Pressure",
    "Max_Back_Pressure",
    "Average_Back_Pressure",
    "Barrel_Temperature_1",
    "Barrel_Temperature_2",
    "Barrel_Temperature_3",
    "Barrel_Temperature_4",
    "Barrel_Temperature_5",
    "Barrel_Temperature_6",
    "Hopper_Temperature",
    "Mold_Temperature_3",
    "Mold_Temperature_4",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/preprocessing/test_columns.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing/columns.py tests/preprocessing/test_columns.py
git commit -m "$(cat <<'EOF'
Add feature column whitelist

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: 라벨 인코딩

**Files:**
- Create: `src/preprocessing/labels.py`
- Test: `tests/preprocessing/test_labels.py`

**Interfaces:**
- Consumes: 없음(순수 함수, `PassOrFail`/`Reason` 컬럼만 있으면 됨)
- Produces: `encode_labels(df: pd.DataFrame) -> pd.DataFrame` — 원본 컬럼 유지한 채 `label`(0/1) 컬럼 추가.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/preprocessing/test_labels.py`:

```python
import pandas as pd

from preprocessing.labels import encode_labels


def test_encode_labels_maps_pass_fail_to_binary():
    df = pd.DataFrame({
        "PassOrFail": ["Y", "N", "N"],
        "Reason": [None, "가스", "미성형"],
    })

    result = encode_labels(df)

    assert result["label"].tolist() == [0, 1, 1]


def test_encode_labels_reclassifies_initial_startup_defect_as_normal():
    df = pd.DataFrame({
        "PassOrFail": ["N", "N"],
        "Reason": ["초기허용불량", "가스"],
    })

    result = encode_labels(df)

    assert result["label"].tolist() == [0, 1]
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/preprocessing/test_labels.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.labels'`

- [ ] **Step 3: 최소 구현**

`src/preprocessing/labels.py`:

```python
import pandas as pd

INITIAL_STARTUP_DEFECT_REASON = "초기허용불량"


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    label = (df["PassOrFail"] == "N").astype(int)
    label = label.where(df["Reason"] != INITIAL_STARTUP_DEFECT_REASON, 0)
    df["label"] = label
    return df
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/preprocessing/test_labels.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing/labels.py tests/preprocessing/test_labels.py
git commit -m "$(cat <<'EOF'
Add PassOrFail/Reason label encoding

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 자가정제 (IsolationForest)

**Files:**
- Create: `src/preprocessing/outliers.py`
- Test: `tests/preprocessing/test_outliers.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`는 호출자가 리스트로 전달(이 모듈은 `columns.py`에 의존하지 않음 — 순수하게 재사용 가능하도록).
- Produces: `remove_outliers(df, feature_columns, contamination=0.01, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame]` — `(정제된 df, 제거된 행의 [_id, outlier_score] df)`. `df`에는 반드시 `_id` 컬럼이 있어야 한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/preprocessing/test_outliers.py`:

```python
import numpy as np
import pandas as pd

from preprocessing.outliers import remove_outliers


def test_remove_outliers_flags_extreme_row():
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    df = pd.DataFrame(normal, columns=["a", "b"])
    df["_id"] = [f"id{i}" for i in range(len(df))]
    df.loc[0, ["a", "b"]] = [100.0, 100.0]

    cleaned, removed = remove_outliers(
        df, feature_columns=["a", "b"], contamination=0.01, random_state=42
    )

    assert "id0" in removed["_id"].tolist()
    assert "id0" not in cleaned["_id"].tolist()
    assert len(cleaned) + len(removed) == len(df)
    assert list(removed.columns) == ["_id", "outlier_score"]
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/preprocessing/test_outliers.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.outliers'`

- [ ] **Step 3: 최소 구현**

`src/preprocessing/outliers.py`:

```python
import pandas as pd
from sklearn.ensemble import IsolationForest


def remove_outliers(
    df: pd.DataFrame,
    feature_columns: list[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    X = df[feature_columns]
    predictions = model.fit_predict(X)
    scores = model.score_samples(X)

    is_outlier = predictions == -1
    cleaned = df.loc[~is_outlier].reset_index(drop=True)
    removed = pd.DataFrame({
        "_id": df.loc[is_outlier, "_id"].values,
        "outlier_score": scores[is_outlier],
    }).reset_index(drop=True)
    return cleaned, removed
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/preprocessing/test_outliers.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing/outliers.py tests/preprocessing/test_outliers.py
git commit -m "$(cat <<'EOF'
Add IsolationForest-based self-cleaning step

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 스케일링

**Files:**
- Create: `src/preprocessing/scaling.py`
- Test: `tests/preprocessing/test_scaling.py`

**Interfaces:**
- Produces:
  - `fit_scaler(df: pd.DataFrame, feature_columns: list[str]) -> StandardScaler`
  - `transform_features(df: pd.DataFrame, feature_columns: list[str], scaler: StandardScaler) -> pd.DataFrame`
  - `scaler_to_dict(scaler: StandardScaler, feature_columns: list[str]) -> dict[str, dict[str, float]]` — `{col: {"mean": ..., "std": ...}}`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/preprocessing/test_scaling.py`:

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

```bash
uv run pytest tests/preprocessing/test_scaling.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.scaling'`

- [ ] **Step 3: 최소 구현**

`src/preprocessing/scaling.py`:

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

```bash
uv run pytest tests/preprocessing/test_scaling.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing/scaling.py tests/preprocessing/test_scaling.py
git commit -m "$(cat <<'EOF'
Add StandardScaler fit/transform helpers

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: manifest 빌더

**Files:**
- Create: `src/preprocessing/manifest.py`
- Test: `tests/preprocessing/test_manifest.py`

**Interfaces:**
- Consumes: 없음(모든 값을 키워드 인자로 받는 순수 함수).
- Produces: `build_manifest(**kwargs) -> dict` — JSON 직렬화 가능한 dict.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/preprocessing/test_manifest.py`:

```python
from preprocessing.manifest import build_manifest


def test_build_manifest_records_row_counts_and_columns():
    manifest = build_manifest(
        raw_labeled_rows=6736,
        labeled_rows_after_dedup=3974,
        raw_unlabeled_rows=35239,
        train_rows_after_cleaning=34886,
        removed_outlier_rows=353,
        contamination=0.01,
        feature_columns=["a", "b"],
        dead_sensor_columns=["dead1"],
        disabled_sensor_columns=["disabled1"],
        eval_label_counts={"normal": 3956, "gas": 13, "misform": 5},
    )

    assert manifest["labeled"]["duplicates_removed"] == 2762
    assert manifest["unlabeled"]["outliers_removed"] == 353
    assert manifest["feature_columns"] == ["a", "b"]
    assert manifest["dropped_columns"] == {
        "dead_sensors": ["dead1"],
        "disabled_after_eval_window": ["disabled1"],
    }
    assert manifest["eval_label_counts"] == {"normal": 3956, "gas": 13, "misform": 5}
    assert "processed_at" in manifest
    assert "filter_condition" in manifest
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/preprocessing/test_manifest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.manifest'`

- [ ] **Step 3: 최소 구현**

`src/preprocessing/manifest.py`:

```python
from datetime import datetime, timezone

FILTER_CONDITION = "PART_NAME LIKE 'CN7%' AND EQUIP_NAME == '650톤-우진2호기'"


def build_manifest(
    *,
    raw_labeled_rows: int,
    labeled_rows_after_dedup: int,
    raw_unlabeled_rows: int,
    train_rows_after_cleaning: int,
    removed_outlier_rows: int,
    contamination: float,
    feature_columns: list[str],
    dead_sensor_columns: list[str],
    disabled_sensor_columns: list[str],
    eval_label_counts: dict[str, int],
) -> dict:
    return {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "filter_condition": FILTER_CONDITION,
        "labeled": {
            "raw_rows": raw_labeled_rows,
            "rows_after_dedup": labeled_rows_after_dedup,
            "duplicates_removed": raw_labeled_rows - labeled_rows_after_dedup,
        },
        "unlabeled": {
            "raw_rows": raw_unlabeled_rows,
            "rows_after_self_cleaning": train_rows_after_cleaning,
            "outliers_removed": removed_outlier_rows,
            "contamination": contamination,
        },
        "feature_columns": feature_columns,
        "dropped_columns": {
            "dead_sensors": dead_sensor_columns,
            "disabled_after_eval_window": disabled_sensor_columns,
        },
        "eval_label_counts": eval_label_counts,
    }
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/preprocessing/test_manifest.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing/manifest.py tests/preprocessing/test_manifest.py
git commit -m "$(cat <<'EOF'
Add manifest builder for reproducibility metadata

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: 파이프라인 오케스트레이션

**Files:**
- Create: `src/preprocessing/pipeline.py`
- Test: `tests/preprocessing/test_pipeline.py`

**Interfaces:**
- Consumes: `load_csv`(Task 1), `remove_exact_duplicates`(Task 2), `FEATURE_COLUMNS`/`DEAD_SENSOR_COLUMNS`/`DISABLED_SENSOR_COLUMNS`/`select_features`(Task 3), `encode_labels`(Task 4), `remove_outliers`(Task 5), `fit_scaler`/`transform_features`/`scaler_to_dict`(Task 6), `build_manifest`(Task 7).
- Produces: `run_pipeline(labeled_path: str, unlabeled_path: str, output_dir: str) -> dict` — `output_dir`에 5개 파일을 쓰고 manifest dict를 반환.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/preprocessing/test_pipeline.py`:

```python
import json

import pandas as pd

from preprocessing.columns import FEATURE_COLUMNS
from preprocessing.pipeline import run_pipeline


def _make_unlabeled_csv(path, n_rows):
    data = {col: [float(i % 5) for i in range(n_rows)] for col in FEATURE_COLUMNS}
    data["_id"] = [f"u{i}" for i in range(n_rows)]
    data["TimeStamp"] = [f"2020-01-01 00:{i // 60:02d}:{i % 60:02d}" for i in range(n_rows)]
    pd.DataFrame(data).to_csv(path, index=False)


def _make_labeled_csv(path):
    n_normal = 20
    n_rows = n_normal + 2
    data = {col: [float(i % 5) for i in range(n_rows)] for col in FEATURE_COLUMNS}
    data["_id"] = [f"l{i}" for i in range(n_rows)]
    data["TimeStamp"] = [f"2020-02-01 00:{i // 60:02d}:{i % 60:02d}" for i in range(n_rows)]
    data["PassOrFail"] = ["Y"] * n_normal + ["N", "N"]
    data["Reason"] = [None] * n_normal + ["가스", "미성형"]
    pd.DataFrame(data).to_csv(path, index=False)


def test_run_pipeline_creates_expected_output_files(tmp_path):
    labeled_path = tmp_path / "labeled.csv"
    unlabeled_path = tmp_path / "unlabeled.csv"
    output_dir = tmp_path / "processed"

    _make_labeled_csv(labeled_path)
    _make_unlabeled_csv(unlabeled_path, n_rows=200)

    manifest = run_pipeline(str(labeled_path), str(unlabeled_path), str(output_dir))

    for name in ["train.csv", "eval.csv", "scaler.json", "removed_outliers.csv", "manifest.json"]:
        assert (output_dir / name).exists()

    train_df = pd.read_csv(output_dir / "train.csv")
    assert list(train_df.columns) == FEATURE_COLUMNS

    eval_df = pd.read_csv(output_dir / "eval.csv")
    assert "label" in eval_df.columns
    assert "TimeStamp" in eval_df.columns
    assert eval_df["label"].sum() == 2

    scaler_dict = json.loads((output_dir / "scaler.json").read_text())
    assert set(scaler_dict.keys()) == set(FEATURE_COLUMNS)

    assert manifest["eval_label_counts"]["gas"] == 1
    assert manifest["eval_label_counts"]["misform"] == 1
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/preprocessing/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'preprocessing.pipeline'`

- [ ] **Step 3: 최소 구현**

`src/preprocessing/pipeline.py`:

```python
import json
from pathlib import Path

import pandas as pd

from .columns import (
    DEAD_SENSOR_COLUMNS,
    DISABLED_SENSOR_COLUMNS,
    FEATURE_COLUMNS,
    select_features,
)
from .data_io import load_csv
from .dedup import remove_exact_duplicates
from .labels import encode_labels
from .manifest import build_manifest
from .outliers import remove_outliers
from .scaling import fit_scaler, scaler_to_dict, transform_features

CONTAMINATION = 0.01
RANDOM_STATE = 42
EVAL_METADATA_COLUMNS = ["PassOrFail", "Reason", "TimeStamp", "label"]


def run_pipeline(labeled_path: str, unlabeled_path: str, output_dir: str) -> dict:
    raw_labeled = load_csv(labeled_path)
    raw_unlabeled = load_csv(unlabeled_path)

    labeled = remove_exact_duplicates(raw_labeled)
    labeled = encode_labels(labeled)

    train_clean, removed = remove_outliers(
        raw_unlabeled, FEATURE_COLUMNS, contamination=CONTAMINATION, random_state=RANDOM_STATE
    )

    scaler = fit_scaler(train_clean, FEATURE_COLUMNS)
    train_scaled = transform_features(train_clean, FEATURE_COLUMNS, scaler)
    eval_scaled = transform_features(labeled, FEATURE_COLUMNS, scaler)

    train_out = select_features(train_scaled)
    eval_out = pd.concat(
        [select_features(eval_scaled), eval_scaled[EVAL_METADATA_COLUMNS]], axis=1
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out.to_csv(out_dir / "train.csv", index=False)
    eval_out.to_csv(out_dir / "eval.csv", index=False)
    removed.to_csv(out_dir / "removed_outliers.csv", index=False)

    scaler_dict = scaler_to_dict(scaler, FEATURE_COLUMNS)
    (out_dir / "scaler.json").write_text(json.dumps(scaler_dict, indent=2, ensure_ascii=False))

    eval_label_counts = {
        "normal": int((eval_out["label"] == 0).sum()),
        "gas": int(((eval_out["label"] == 1) & (eval_out["Reason"] == "가스")).sum()),
        "misform": int(((eval_out["label"] == 1) & (eval_out["Reason"] == "미성형")).sum()),
    }
    manifest = build_manifest(
        raw_labeled_rows=len(raw_labeled),
        labeled_rows_after_dedup=len(labeled),
        raw_unlabeled_rows=len(raw_unlabeled),
        train_rows_after_cleaning=len(train_clean),
        removed_outlier_rows=len(removed),
        contamination=CONTAMINATION,
        feature_columns=FEATURE_COLUMNS,
        dead_sensor_columns=DEAD_SENSOR_COLUMNS,
        disabled_sensor_columns=DISABLED_SENSOR_COLUMNS,
        eval_label_counts=eval_label_counts,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return manifest
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/preprocessing/test_pipeline.py -v
```

Expected: PASS

- [ ] **Step 5: 전체 테스트 스위트 실행(회귀 확인)**

```bash
uv run pytest tests/ -v
```

Expected: 모든 테스트 PASS (Task 1~8에서 작성한 테스트 전부)

- [ ] **Step 6: 커밋**

```bash
git add src/preprocessing/pipeline.py tests/preprocessing/test_pipeline.py
git commit -m "$(cat <<'EOF'
Wire preprocessing pipeline stages together

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: 엔트리 스크립트 + 실제 데이터 실행 검증

**Files:**
- Create: `scripts/run_preprocessing.py`

**Interfaces:**
- Consumes: `run_pipeline`(Task 8).
- 이 태스크는 pytest 테스트를 추가하지 않는다 — 실제(gitignore된) 데이터 파일에 의존하는 산출물 검증이라 자동화된 회귀 테스트로 두기보다 수동 실행으로 확인한다.

- [ ] **Step 1: 엔트리 스크립트 작성**

`scripts/run_preprocessing.py`:

```python
from pathlib import Path

from preprocessing.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    manifest = run_pipeline(
        labeled_path=str(ROOT / "data" / "dataset" / "cn7_labeled.csv"),
        unlabeled_path=str(ROOT / "data" / "dataset" / "cn7_unlabeled.csv"),
        output_dir=str(ROOT / "data" / "processed"),
    )
    print(f"labeled: {manifest['labeled']}")
    print(f"unlabeled: {manifest['unlabeled']}")
    print(f"eval_label_counts: {manifest['eval_label_counts']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실제 데이터로 실행**

```bash
uv run python scripts/run_preprocessing.py
```

Expected (계획 작성 시 실측으로 확정된 값 — 랜덤시드 42 고정이므로 재현되어야 함):

```
labeled: {'raw_rows': 6736, 'rows_after_dedup': 3974, 'duplicates_removed': 2762}
unlabeled: {'raw_rows': 35239, 'rows_after_self_cleaning': 34886, 'outliers_removed': 353, 'contamination': 0.01}
eval_label_counts: {'normal': 3956, 'gas': 13, 'misform': 5}
```

- [ ] **Step 3: 산출물 확인**

```bash
ls -la data/processed/
wc -l data/processed/train.csv data/processed/eval.csv data/processed/removed_outliers.csv
```

Expected: `train.csv`(34,887줄=34,886행+헤더), `eval.csv`(3,975줄=3,974행+헤더), `removed_outliers.csv`(354줄=353행+헤더), `scaler.json`, `manifest.json` 모두 존재.

- [ ] **Step 4: 커밋**

```bash
git add scripts/run_preprocessing.py
git commit -m "$(cat <<'EOF'
Add preprocessing pipeline entry script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

(`data/processed/`는 `.gitignore`의 `data/` 규칙에 포함되므로 커밋 대상이 아니다.)
