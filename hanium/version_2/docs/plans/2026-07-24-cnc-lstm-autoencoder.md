# CNC LSTM 오토인코더 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `version_2/data/processed/{train,eval}.csv`(전처리 완료된 CNC 데이터)를 입력으로 받아
LSTM 오토인코더를 학습하고, 실험 단위(윈도우 오차 → 실험 단위 mean/max/95%ile 집계 →
train 분포 기반 임계값)로 양품/불량을 판정해 precision/recall을 산출하는
`version_2/src/lstm_ae/` 패키지를 만든다.

**Architecture:** hanium 루트의 `src/lstm_ae/` 모듈 스타일(단일 책임 소파일 + 얇은
`pipeline.py` 오케스트레이터)을 `version_2/src/lstm_ae/` 아래 독립적으로 복제한다.
`model.py`/`training.py`는 CN7과 아키텍처가 완전히 동일해 거의 그대로 옮기고,
`sequencing.py`(실험 경계를 넘지 않는 윈도우 구성)와 `scoring.py`(실험 단위 집계/평가)는
CNC 데이터 구조(샷 단위 라벨이 아니라 실험 단위 라벨)에 맞게 새로 설계한다.

**Tech Stack:** Python 3.14, PyTorch(CPU), pandas, numpy, pytest, uv.

## Global Constraints

- Python `>=3.14`, `uv_build` 백엔드. `version_2/pyproject.toml`은 이미 존재(전처리
  플랜에서 생성, `[tool.uv.build-backend] module-name = "preprocessing"` 포함) — Task 1에서
  `torch` 의존성만 추가하고 나머지는 건드리지 않는다. **주의**: `module-name` override가
  있어도 `src/` 아래 두 번째 패키지(`lstm_ae`)는 정상적으로 별도 import 가능함을 이미
  확인했다(`import lstm_ae; import preprocessing` 둘 다 성공, override는 discovery를
  단일 패키지로 제한하지 않음).
- pytest 설정은 기존과 동일: `testpaths = ["tests"]`, `addopts = "--import-mode=importlib"`.
- `version_2/` 아래 코드만 사용. hanium 루트 `src/`를 import하지 않는다(패턴만 참고, 코드
  공유 없음 — 사용자 확인된 "독립 구조" 원칙 유지).
- 새 패키지는 `lstm_ae` (`preprocessing`과 나란히 `version_2/src/` 아래).
- **커밋 정책(CLAUDE.md)**: 사용자가 명시적으로 요청하기 전까지 `git commit`을 실행하지
  않는다. 각 태스크의 마지막 단계는 "커밋"이 아니라 "`git add`로 스테이징"이다.
- 하이퍼파라미터 초기값(스펙에서 확정): `window_size=20`, `hidden_size=64`,
  `latent_dim=16`, `epochs=50`, `batch_size=64`, `learning_rate=1e-3`, `random_seed=42`,
  `threshold_percentile=95.0`.
- **모델 학습(Task 6)의 결과 수치(loss, precision/recall)는 사전에 알 수 없다** — 이건
  전처리 플랜과 달리 확정적 계산이 아니라 실제 학습 결과다. Task 6의 검증은 "정확한 수치
  일치"가 아니라 "구조적으로 말이 되는지"(NaN 없음, loss 감소 추세, tp+fn/tn+fp 합계가
  라벨 개수와 일치 등)로 한다.

---

### Task 1: `pyproject.toml`에 torch 추가 + `model.py`

**Files:**
- Modify: `version_2/pyproject.toml`
- Create: `version_2/src/lstm_ae/__init__.py` (빈 파일)
- Create: `version_2/src/lstm_ae/model.py`
- Create: `version_2/tests/lstm_ae/test_model.py`

**Interfaces:**
- Produces: `LSTMAutoencoder(num_features, hidden_size=64, latent_dim=16)` (nn.Module,
  `forward(x) -> Tensor` 같은 shape) — Task 2(`training.py`)와 Task 5(`pipeline.py`)가
  가져다 쓴다.

- [ ] **Step 1: `version_2/pyproject.toml`에 torch 의존성 추가**

기존 파일의 `dependencies` 리스트에 `"torch>=2.13.0"`을 추가하고, 파일 끝에 아래 두
테이블을 추가한다(hanium 루트 `pyproject.toml`과 동일한 CPU 전용 인덱스 설정):

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

기존 `[tool.uv.build-backend]`(`module-name = "preprocessing"`)와 다른 기존 내용은
그대로 둔다. 수정 후 `dependencies`는 `numpy`, `pandas`, `scikit-learn`, `torch` 4개가
된다.

- [ ] **Step 2: 빈 `version_2/src/lstm_ae/__init__.py` 생성**

- [ ] **Step 3: 실패하는 테스트 작성 — `version_2/tests/lstm_ae/test_model.py`**

```python
import torch

from lstm_ae.model import LSTMAutoencoder


def test_forward_pass_preserves_input_shape():
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=41, hidden_size=8, latent_dim=4)
    x = torch.randn(5, 20, 41)

    output = model(x)

    assert output.shape == x.shape


def test_forward_pass_works_for_different_batch_and_seq_sizes():
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=3, hidden_size=8, latent_dim=4)
    x = torch.randn(2, 7, 3)

    output = model(x)

    assert output.shape == (2, 7, 3)
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `cd version_2 && uv sync && uv run pytest tests/lstm_ae/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.model'` (`uv sync`는 torch
설치 때문에 다소 걸릴 수 있음, 수 분 이내)

- [ ] **Step 5: `version_2/src/lstm_ae/model.py` 구현**

```python
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=num_features, hidden_size=hidden_size, batch_first=True
        )
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim, hidden_size=hidden_size, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.to_latent(h_n[-1])
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(repeated)
        return self.output_layer(decoded)
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: 기존 preprocessing 테스트 스위트도 여전히 통과하는지 확인 (torch 추가로
  인한 회귀 없는지)**

Run: `cd version_2 && uv run pytest -v`
Expected: 21개(기존 19 + 신규 2) 전부 PASS

- [ ] **Step 8: 스테이징 (커밋 아님 — Global Constraints 참고)**

```bash
git add version_2/pyproject.toml version_2/uv.lock version_2/src/lstm_ae/__init__.py version_2/src/lstm_ae/model.py version_2/tests/lstm_ae/test_model.py
```

(`uv.lock`은 torch 추가로 내용이 바뀌므로 이번엔 스테이징 대상에 포함 — 이전 전처리
플랜에서는 lock 파일을 커밋하지 않았지만, 의존성이 실제로 바뀌는 시점이라 재현성을 위해
포함한다.)

---

### Task 2: `training.py`

**Files:**
- Create: `version_2/src/lstm_ae/training.py`
- Create: `version_2/tests/lstm_ae/test_training.py`

**Interfaces:**
- Consumes: Task 1의 `LSTMAutoencoder`
- Produces: `train_autoencoder(model, train_windows, epochs=50, batch_size=64,
  learning_rate=1e-3) -> list[float]` — Task 5(`pipeline.py`)가 가져다 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/lstm_ae/test_training.py`**

```python
import numpy as np
import torch

from lstm_ae.model import LSTMAutoencoder
from lstm_ae.training import train_autoencoder


def test_train_autoencoder_reduces_loss_on_easy_synthetic_target():
    torch.manual_seed(0)
    train_windows = np.zeros((20, 4, 3), dtype=np.float32)
    model = LSTMAutoencoder(num_features=3, hidden_size=8, latent_dim=4)

    losses = train_autoencoder(
        model, train_windows, epochs=20, batch_size=4, learning_rate=1e-2
    )

    assert len(losses) == 20
    assert all(np.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_training.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.training'`

- [ ] **Step 3: `version_2/src/lstm_ae/training.py` 구현**

```python
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_autoencoder(
    model: nn.Module,
    train_windows: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> list[float]:
    x = torch.tensor(train_windows, dtype=torch.float32)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        epoch_loss = total_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")
    return epoch_losses
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_training.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 스테이징**

```bash
git add version_2/src/lstm_ae/training.py version_2/tests/lstm_ae/test_training.py
```

---

### Task 3: `sequencing.py` — 실험 경계를 넘지 않는 윈도우 구성

**Files:**
- Create: `version_2/src/lstm_ae/sequencing.py`
- Create: `version_2/tests/lstm_ae/test_sequencing.py`

**Interfaces:**
- Consumes: 없음 (순수 pandas/numpy 함수)
- Produces:
  - `make_train_windows(df, feature_columns, window_size) -> tuple[np.ndarray, np.ndarray]`
    — `(windows, experiment_ids)`. `windows.shape = (num_windows, window_size,
    len(feature_columns))`, `experiment_ids.shape = (num_windows,)`이며
    `experiment_ids[i]`는 `windows[i]`가 속한 실험의 `experiment_id`.
  - `make_eval_windows(df, feature_columns, window_size) -> tuple[np.ndarray, np.ndarray]`
    — 같은 반환 형태, 중첩 슬라이딩.
  - 둘 다 `df`에 `experiment_id` 컬럼이 있다고 가정한다(전처리 플랜의 `train.csv`/`eval.csv`
    스키마). Task 5(`pipeline.py`)가 가져다 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/lstm_ae/test_sequencing.py`**

```python
import numpy as np
import pandas as pd

from lstm_ae.sequencing import make_eval_windows, make_train_windows


def _make_df(experiment_rows: dict[int, int], num_features: int) -> pd.DataFrame:
    frames = []
    for experiment_id, num_rows in experiment_rows.items():
        data = {
            f"f{i}": np.arange(num_rows, dtype=np.float32) * 10 + i
            for i in range(num_features)
        }
        frame = pd.DataFrame(data)
        frame["experiment_id"] = experiment_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_make_train_windows_respects_experiment_boundaries_and_drops_remainder():
    # experiment 1: 7 rows, window_size=3 -> 2 windows (rows 0-2, 3-5), row 6 dropped
    # experiment 2: 5 rows, window_size=3 -> 1 window (rows 0-2 of exp2), rows 3-4 dropped
    df = _make_df({1: 7, 2: 5}, num_features=2)
    feature_columns = ["f0", "f1"]

    windows, experiment_ids = make_train_windows(df, feature_columns, window_size=3)

    assert windows.shape == (3, 3, 2)
    assert experiment_ids.tolist() == [1, 1, 2]

    exp1_values = df.loc[df["experiment_id"] == 1, feature_columns].to_numpy(dtype=np.float32)
    exp2_values = df.loc[df["experiment_id"] == 2, feature_columns].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(windows[0], exp1_values[0:3])
    np.testing.assert_array_equal(windows[1], exp1_values[3:6])
    np.testing.assert_array_equal(windows[2], exp2_values[0:3])


def test_make_train_windows_never_mixes_two_experiments_in_one_window():
    # experiment 1 has 4 rows (values 0,10,20,30), experiment 2 has 4 rows (values
    # 1000,1010,1020,1030) so a boundary-crossing window would be immediately obvious.
    df = pd.concat(
        [
            pd.DataFrame({"f0": [0.0, 10.0, 20.0, 30.0], "experiment_id": 1}),
            pd.DataFrame({"f0": [1000.0, 1010.0, 1020.0, 1030.0], "experiment_id": 2}),
        ],
        ignore_index=True,
    )

    windows, experiment_ids = make_train_windows(df, ["f0"], window_size=4)

    assert windows.shape == (2, 4, 1)
    assert experiment_ids.tolist() == [1, 2]
    assert windows[0].max() < 1000.0  # window 0 is entirely experiment 1
    assert windows[1].min() >= 1000.0  # window 1 is entirely experiment 2


def test_make_eval_windows_is_overlapping_within_each_experiment():
    # experiment 1: 5 rows, window_size=3 -> 3 windows; experiment 2: 4 rows -> 2 windows
    df = _make_df({1: 5, 2: 4}, num_features=1)

    windows, experiment_ids = make_eval_windows(df, ["f0"], window_size=3)

    assert windows.shape == (5, 3, 1)
    assert experiment_ids.tolist() == [1, 1, 1, 2, 2]

    exp1_values = df.loc[df["experiment_id"] == 1, ["f0"]].to_numpy(dtype=np.float32)
    exp2_values = df.loc[df["experiment_id"] == 2, ["f0"]].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(windows[0], exp1_values[0:3])
    np.testing.assert_array_equal(windows[1], exp1_values[1:4])
    np.testing.assert_array_equal(windows[2], exp1_values[2:5])
    np.testing.assert_array_equal(windows[3], exp2_values[0:3])
    np.testing.assert_array_equal(windows[4], exp2_values[1:4])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_sequencing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.sequencing'`

- [ ] **Step 3: `version_2/src/lstm_ae/sequencing.py` 구현**

```python
import numpy as np
import pandas as pd


def make_train_windows(
    df: pd.DataFrame, feature_columns: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    all_windows = []
    all_experiment_ids = []
    for experiment_id, group in df.groupby("experiment_id", sort=True):
        values = group[feature_columns].to_numpy(dtype=np.float32)
        num_windows = len(values) // window_size
        trimmed = values[: num_windows * window_size]
        windows = trimmed.reshape(num_windows, window_size, len(feature_columns))
        all_windows.append(windows)
        all_experiment_ids.extend([experiment_id] * num_windows)
    return np.concatenate(all_windows, axis=0), np.array(all_experiment_ids)


def make_eval_windows(
    df: pd.DataFrame, feature_columns: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    all_windows = []
    all_experiment_ids = []
    for experiment_id, group in df.groupby("experiment_id", sort=True):
        values = group[feature_columns].to_numpy(dtype=np.float32)
        num_windows = len(values) - window_size + 1
        windows = np.stack([values[i : i + window_size] for i in range(num_windows)])
        all_windows.append(windows)
        all_experiment_ids.extend([experiment_id] * num_windows)
    return np.concatenate(all_windows, axis=0), np.array(all_experiment_ids)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_sequencing.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 스테이징**

```bash
git add version_2/src/lstm_ae/sequencing.py version_2/tests/lstm_ae/test_sequencing.py
```

---

### Task 4: `scoring.py` — 실험 단위 집계 & 임계값 & 평가

**Files:**
- Create: `version_2/src/lstm_ae/scoring.py`
- Create: `version_2/tests/lstm_ae/test_scoring.py`

**Interfaces:**
- Consumes: 없음 (순수 pandas/numpy 함수 — 모델 자체는 다루지 않는다, 윈도우 오차 배열은
  이미 계산되어 들어온다는 전제)
- Produces:
  - `aggregate_window_errors_by_experiment(window_errors: np.ndarray, experiment_ids:
    np.ndarray) -> pd.DataFrame` — 컬럼 `experiment_id, mean_score, max_score, p95_score`,
    실험당 한 행.
  - `compute_thresholds(train_experiment_scores: pd.DataFrame, percentile: float = 95.0)
    -> dict` — `{"mean": float, "max": float, "p95": float}`.
  - `evaluate_experiment_predictions(eval_experiment_scores: pd.DataFrame, labels:
    pd.Series, thresholds: dict) -> dict` — `labels`는 `experiment_id`로 인덱싱된
    Series(0=양품/1=불량). 반환값은 `{"mean": {...}, "max": {...}, "p95": {...}}`이며
    각 값은 `{"precision", "recall", "tp", "fp", "fn", "tn"}`.
  - Task 5(`pipeline.py`)가 이 세 함수를 순서대로 가져다 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/lstm_ae/test_scoring.py`**

```python
import numpy as np
import pandas as pd
import pytest

from lstm_ae.scoring import (
    aggregate_window_errors_by_experiment,
    compute_thresholds,
    evaluate_experiment_predictions,
)


def test_aggregate_window_errors_by_experiment_computes_mean_max_p95():
    # experiment 1: window errors [1, 2, 3, 4, 5] (mean=3, max=5, p95=4.8)
    # experiment 2: window errors [10, 20] (mean=15, max=20, p95=19.5)
    window_errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])
    experiment_ids = np.array([1, 1, 1, 1, 1, 2, 2])

    result = aggregate_window_errors_by_experiment(window_errors, experiment_ids)

    result = result.set_index("experiment_id")
    assert result.loc[1, "mean_score"] == pytest.approx(3.0)
    assert result.loc[1, "max_score"] == pytest.approx(5.0)
    assert result.loc[1, "p95_score"] == pytest.approx(np.percentile([1, 2, 3, 4, 5], 95))
    assert result.loc[2, "mean_score"] == pytest.approx(15.0)
    assert result.loc[2, "max_score"] == pytest.approx(20.0)


def test_compute_thresholds_is_percentile_of_train_experiment_scores():
    train_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3, 4],
        "mean_score": [10.0, 20.0, 30.0, 40.0],
        "max_score": [100.0, 200.0, 300.0, 400.0],
        "p95_score": [1.0, 2.0, 3.0, 4.0],
    })

    thresholds = compute_thresholds(train_scores, percentile=90)

    assert thresholds["mean"] == pytest.approx(np.percentile([10, 20, 30, 40], 90))
    assert thresholds["max"] == pytest.approx(np.percentile([100, 200, 300, 400], 90))
    assert thresholds["p95"] == pytest.approx(np.percentile([1, 2, 3, 4], 90))


def test_evaluate_experiment_predictions_computes_precision_recall_per_method():
    eval_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3, 4],
        "mean_score": [1.0, 5.0, 1.0, 5.0],
        "max_score": [1.0, 5.0, 1.0, 5.0],
        "p95_score": [1.0, 5.0, 1.0, 5.0],
    })
    labels = pd.Series({1: 0, 2: 1, 3: 1, 4: 0})  # exp2 correctly flagged, exp3 missed, exp4 false alarm
    thresholds = {"mean": 2.0, "max": 2.0, "p95": 2.0}

    result = evaluate_experiment_predictions(eval_scores, labels, thresholds)

    for method in ["mean", "max", "p95"]:
        assert result[method]["tp"] == 1  # exp2
        assert result[method]["fp"] == 1  # exp4
        assert result[method]["fn"] == 1  # exp3
        assert result[method]["tn"] == 1  # exp1
        assert result[method]["precision"] == pytest.approx(0.5)
        assert result[method]["recall"] == pytest.approx(0.5)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.scoring'`

- [ ] **Step 3: `version_2/src/lstm_ae/scoring.py` 구현**

```python
import numpy as np
import pandas as pd


def aggregate_window_errors_by_experiment(
    window_errors: np.ndarray, experiment_ids: np.ndarray
) -> pd.DataFrame:
    df = pd.DataFrame({"experiment_id": experiment_ids, "window_error": window_errors})
    grouped = df.groupby("experiment_id")["window_error"]
    result = pd.DataFrame({
        "mean_score": grouped.mean(),
        "max_score": grouped.max(),
        "p95_score": grouped.quantile(0.95),
    })
    return result.reset_index()


def compute_thresholds(
    train_experiment_scores: pd.DataFrame, percentile: float = 95.0
) -> dict:
    return {
        "mean": float(np.percentile(train_experiment_scores["mean_score"], percentile)),
        "max": float(np.percentile(train_experiment_scores["max_score"], percentile)),
        "p95": float(np.percentile(train_experiment_scores["p95_score"], percentile)),
    }


def evaluate_experiment_predictions(
    eval_experiment_scores: pd.DataFrame, labels: pd.Series, thresholds: dict
) -> dict:
    results = {}
    for method in ["mean", "max", "p95"]:
        scores = eval_experiment_scores.set_index("experiment_id")[f"{method}_score"]
        predictions = (scores > thresholds[method]).astype(int)
        aligned_labels = labels.loc[scores.index]
        tp = int(((predictions == 1) & (aligned_labels == 1)).sum())
        fp = int(((predictions == 1) & (aligned_labels == 0)).sum())
        fn = int(((predictions == 0) & (aligned_labels == 1)).sum())
        tn = int(((predictions == 0) & (aligned_labels == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[method] = {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return results
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_scoring.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 스테이징**

```bash
git add version_2/src/lstm_ae/scoring.py version_2/tests/lstm_ae/test_scoring.py
```

---

### Task 5: `pipeline.py` — 오케스트레이터 (통합 테스트 포함)

**Files:**
- Create: `version_2/src/lstm_ae/pipeline.py`
- Create: `version_2/tests/lstm_ae/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1의 `LSTMAutoencoder`, Task 2의 `train_autoencoder`, Task 3의
  `make_train_windows`/`make_eval_windows`, Task 4의
  `aggregate_window_errors_by_experiment`/`compute_thresholds`/
  `evaluate_experiment_predictions`.
- Produces: `run_lstm_pipeline(train_csv_path, eval_csv_path, feature_columns, output_dir,
  window_size=20, hidden_size=64, latent_dim=16, epochs=50, batch_size=64,
  learning_rate=1e-3, random_seed=42, threshold_percentile=95.0) -> dict` —
  Task 6(`scripts/run_lstm_training.py`)가 실제 데이터로 호출한다.

- [ ] **Step 1: 실패하는 테스트 작성 — `version_2/tests/lstm_ae/test_pipeline.py`**

```python
import json

import numpy as np
import pandas as pd
import torch

from lstm_ae.pipeline import run_lstm_pipeline

FEATURE_COLUMNS = ["f0", "f1", "f2"]


def _make_train_csv(path):
    # 2 experiments, 30 rows each -> with window_size=6, 5 windows per experiment
    frames = []
    for experiment_id in [101, 102]:
        data = {col: np.random.randn(30).astype(np.float32) for col in FEATURE_COLUMNS}
        frame = pd.DataFrame(data)
        frame["experiment_id"] = experiment_id
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _make_eval_csv(path):
    # 2 experiments: 201 (good, label=0), 202 (bad, label=1), 20 rows each
    frames = []
    for experiment_id, label in [(201, 0), (202, 1)]:
        data = {col: np.random.randn(20).astype(np.float32) for col in FEATURE_COLUMNS}
        frame = pd.DataFrame(data)
        frame["experiment_id"] = experiment_id
        frame["label"] = label
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def test_run_lstm_pipeline_creates_expected_output_files(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    output_dir = tmp_path / "model"

    _make_train_csv(train_path)
    _make_eval_csv(eval_path)

    summary = run_lstm_pipeline(
        train_csv_path=str(train_path),
        eval_csv_path=str(eval_path),
        feature_columns=FEATURE_COLUMNS,
        output_dir=str(output_dir),
        window_size=6,
        hidden_size=4,
        latent_dim=2,
        epochs=2,
        batch_size=4,
    )

    for name in [
        "model.pt",
        "training_config.json",
        "train_window_errors.csv",
        "eval_window_errors.csv",
        "experiment_scores.csv",
        "evaluation_report.json",
    ]:
        assert (output_dir / name).exists()

    train_errors = pd.read_csv(output_dir / "train_window_errors.csv")
    assert set(train_errors.columns) == {"experiment_id", "window_error"}
    assert len(train_errors) == 10  # 2 experiments x 5 non-overlapping windows each

    eval_errors = pd.read_csv(output_dir / "eval_window_errors.csv")
    assert len(eval_errors) == 30  # 2 experiments x (20-6+1)=15 overlapping windows each

    experiment_scores = pd.read_csv(output_dir / "experiment_scores.csv")
    assert len(experiment_scores) == 2  # one row per eval experiment
    assert set(experiment_scores["experiment_id"]) == {201, 202}
    assert set(experiment_scores.loc[experiment_scores["experiment_id"] == 201, "label"]) == {0}
    assert set(experiment_scores.loc[experiment_scores["experiment_id"] == 202, "label"]) == {1}

    report = json.loads((output_dir / "evaluation_report.json").read_text())
    assert set(report.keys()) == {"thresholds", "results"}
    assert set(report["thresholds"].keys()) == {"mean", "max", "p95"}
    for method in ["mean", "max", "p95"]:
        assert set(report["results"][method].keys()) >= {"precision", "recall", "tp", "fp", "fn", "tn"}
        assert report["results"][method]["tp"] + report["results"][method]["fn"] == 1  # 1 bad experiment
        assert report["results"][method]["tn"] + report["results"][method]["fp"] == 1  # 1 good experiment

    assert "train_windows" in summary
    assert summary["train_windows"] == 10
    assert summary["eval_windows"] == 30
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.pipeline'`

- [ ] **Step 3: `version_2/src/lstm_ae/pipeline.py` 구현**

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import LSTMAutoencoder
from .scoring import (
    aggregate_window_errors_by_experiment,
    compute_thresholds,
    evaluate_experiment_predictions,
)
from .sequencing import make_eval_windows, make_train_windows
from .training import train_autoencoder


def _compute_window_errors(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32)
        reconstructed = model(x).numpy()
    squared_errors = (reconstructed - windows) ** 2
    return squared_errors.reshape(len(windows), -1).mean(axis=1)


def run_lstm_pipeline(
    train_csv_path: str,
    eval_csv_path: str,
    feature_columns: list[str],
    output_dir: str,
    window_size: int = 20,
    hidden_size: int = 64,
    latent_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    threshold_percentile: float = 95.0,
) -> dict:
    torch.manual_seed(random_seed)

    train_df = pd.read_csv(train_csv_path)
    eval_df = pd.read_csv(eval_csv_path)

    train_windows, train_experiment_ids = make_train_windows(
        train_df, feature_columns, window_size
    )
    eval_windows, eval_experiment_ids = make_eval_windows(
        eval_df, feature_columns, window_size
    )

    model = LSTMAutoencoder(
        num_features=len(feature_columns), hidden_size=hidden_size, latent_dim=latent_dim
    )
    loss_history = train_autoencoder(
        model, train_windows, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )

    train_window_errors = _compute_window_errors(model, train_windows)
    eval_window_errors = _compute_window_errors(model, eval_windows)

    train_experiment_scores = aggregate_window_errors_by_experiment(
        train_window_errors, train_experiment_ids
    )
    eval_experiment_scores = aggregate_window_errors_by_experiment(
        eval_window_errors, eval_experiment_ids
    )

    thresholds = compute_thresholds(train_experiment_scores, percentile=threshold_percentile)

    labels = eval_df.groupby("experiment_id")["label"].first()
    report = evaluate_experiment_predictions(eval_experiment_scores, labels, thresholds)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")

    training_config = {
        "window_size": window_size,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "random_seed": random_seed,
        "threshold_percentile": threshold_percentile,
    }
    (out_dir / "training_config.json").write_text(json.dumps(training_config, indent=2))

    pd.DataFrame(
        {"experiment_id": train_experiment_ids, "window_error": train_window_errors}
    ).to_csv(out_dir / "train_window_errors.csv", index=False)
    pd.DataFrame(
        {"experiment_id": eval_experiment_ids, "window_error": eval_window_errors}
    ).to_csv(out_dir / "eval_window_errors.csv", index=False)

    experiment_scores = eval_experiment_scores.merge(
        labels.rename("label"), on="experiment_id"
    )
    for method in ["mean", "max", "p95"]:
        experiment_scores[f"{method}_exceeds_threshold"] = (
            experiment_scores[f"{method}_score"] > thresholds[method]
        )
    experiment_scores.to_csv(out_dir / "experiment_scores.csv", index=False)

    (out_dir / "evaluation_report.json").write_text(
        json.dumps({"thresholds": thresholds, "results": report}, indent=2)
    )

    return {
        "train_windows": len(train_windows),
        "eval_windows": len(eval_windows),
        "final_train_loss": loss_history[-1],
        "thresholds": thresholds,
        "results": report,
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/lstm_ae/test_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 전체 유닛테스트 스위트 통과 확인**

Run: `cd version_2 && uv run pytest -v`
Expected: 모든 테스트 PASS (Task 1~5 lstm_ae 누적 8개 + 기존 preprocessing 19개 = 27개)

- [ ] **Step 6: 스테이징**

```bash
git add version_2/src/lstm_ae/pipeline.py version_2/tests/lstm_ae/test_pipeline.py
```

---

### Task 6: `scripts/run_lstm_training.py` — 실제 데이터 전체 학습·평가 + 수동 검증

**Files:**
- Create: `version_2/scripts/run_lstm_training.py`

**Interfaces:**
- Consumes: Task 5의 `run_lstm_pipeline`, `preprocessing.columns.FEATURE_COLUMNS`(기존
  전처리 패키지, `version_2/src/preprocessing/columns.py` — import만 하고 수정하지 않음).
- Produces: 없음 (스크립트, 실행하면 `version_2/data/model/`에 6개 파일 생성)

- [ ] **Step 1: `version_2/scripts/run_lstm_training.py` 작성**

```python
import json
from pathlib import Path

from lstm_ae.pipeline import run_lstm_pipeline
from preprocessing.columns import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    summary = run_lstm_pipeline(
        train_csv_path=str(ROOT / "data" / "processed" / "train.csv"),
        eval_csv_path=str(ROOT / "data" / "processed" / "eval.csv"),
        feature_columns=FEATURE_COLUMNS,
        output_dir=str(ROOT / "data" / "model"),
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

- [ ] **Step 2: 실행 (CPU 학습, 729개 train 윈도우/50 epoch — CN7 때보다 훨씬 적은 데이터라
  1분 이내 예상. 공유 서버 예절상 `nice -n 19` 사용)**

Run: `cd version_2 && nice -n 19 uv run python scripts/run_lstm_training.py`

- [ ] **Step 3: 출력을 구조적으로 검증 (정확한 수치가 아니라 "말이 되는지" 확인 —
  Global Constraints 참고)**

`data/model/evaluation_report.json`(또는 stdout)에서 아래를 확인한다:

```
- final_train_loss가 유한한 값(NaN/inf 아님)
- mean/max/p95 세 방식 각각: precision, recall이 [0,1] 범위, NaN 아님
- mean/max/p95 각각: tp+fn == 12 (불량 실험 12개), tn+fp == 5 (양품 실험 5개)
- threshold 3개(mean/max/p95)가 서로 다른 스케일의 값(당연히 max용 임계값이 가장 큼)
```

하나라도 어긋나면(특히 tp+fn/tn+fp 합계 불일치) 구현 버그이므로 다음 단계로 넘어가지
말고 원인을 확인한다. precision/recall 수치 자체가 낮거나 세 방식이 서로 크게 다른 건
버그가 아니라 스펙에 이미 기록한 한계(표본 8개로 임계값을 정함, window_size 근거 약함)일
수 있으니 정상적인 결과로 받아들인다.

- [ ] **Step 4: 산출 파일 존재 확인**

Run: `ls -la version_2/data/model/`
Expected: `model.pt`, `training_config.json`, `train_window_errors.csv`,
`eval_window_errors.csv`, `experiment_scores.csv`, `evaluation_report.json` 6개 파일 존재

- [ ] **Step 5: 스테이징**

```bash
git add version_2/scripts/run_lstm_training.py
```

(`version_2/data/model/`은 루트 `.gitignore`의 `data` 패턴에 걸려 자동 제외됨)

---

## 완료 후 체크리스트

- [ ] `cd version_2 && uv run pytest -v` 전체 통과 (기존 19 + 신규 8 = 27개)
- [ ] `scripts/run_lstm_training.py` 실행 결과가 구조적으로 유효함(Task 6 Step 3 기준)
- [ ] `git status`로 스테이징된 파일 확인 — **커밋은 사용자가 명시적으로 요청할 때까지
  하지 않음**
