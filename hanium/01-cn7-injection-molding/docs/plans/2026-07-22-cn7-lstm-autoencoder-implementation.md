# CN7 LSTM 오토인코더 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/specs/2026-07-22-cn7-lstm-autoencoder-design.md`에 설계된
LSTM 오토인코더(시퀀스 구성 → 모델 → 학습 → 샷별 오차/임계값 → 평가)를 PyTorch로
구현하고, 실제 `data/processed/train.csv`/`eval.csv`에 적용해 `data/model/`에 5개
산출물(모델, 학습설정, train/eval 오차, 임계값, 평가 리포트)을 생성한다.

**Architecture:** `src/lstm_ae/` 아래 단계별 순수 함수/모듈(시퀀스 구성 / 모델 / 샷별
오차·임계값·평가 / 학습 루프)을 각각 TDD로 구현하고, `pipeline.py`가 이들을 조합한다.
`scripts/run_lstm_training.py`가 실제 데이터로 파이프라인을 실행하는 엔트리포인트다.
기존 `src/preprocessing/` 패키지와 같은 uv 프로젝트(`src/` 레이아웃, 여러 패키지 자동
인식됨 — 확인됨) 안에 나란히 추가한다.

**Tech Stack:** Python 3.14, uv, PyTorch(CPU 전용 wheel), numpy, pandas, scikit-learn
(기존 의존성), pytest.

## Global Constraints

- PyTorch는 **CPU 전용 wheel**(`torch==*+cpu`)을 설치한다 — 이 서버는 GPU가 없으므로
  기본 `uv add torch`가 받아오는 CUDA 빌드(수백MB~1GB의 nvidia-* 패키지, triton 등)를
  받으면 안 된다. `pytorch-cpu` 인덱스를 명시적으로 지정한다(아래 Task 1에서 검증된
  방법대로).
- 시퀀스 길이는 **N=12**로 고정한다.
- **train은 비중첩(stride=12)**, **eval은 중첩 슬라이딩(stride=1) + 샷별 평균집계**로
  시퀀스를 만든다. eval 시퀀스 구성이 train과 다른 것은 의도된 설계이지 실수가 아니다.
- 모델: Encoder-Decoder LSTM-AE, hidden_size=64, latent_dim=16, 1층 LSTM. 이 값들은
  **초기값이며 실제 데이터 적용 후 조정 가능**(스펙에 명시, 사용자 확인).
- 손실 함수: MSE. Optimizer: Adam, lr=1e-3(초기값). Batch size=64(초기값).
  Epochs=50(초기값). 별도 validation split 없음 — train loss 추이만 관찰.
- 임계값은 **train 샷별 오차의 평균+3표준편차**로 정하고, eval에는 적용만 한다(절대
  eval로 재fit/재계산하지 않는다 — 리키지 방지, 전처리 스펙과 동일한 원칙).
- 산출물은 `data/model/`에 저장한다 — 이미 `.gitignore`(`data`, 슬래시 없음)로 제외됨.
- 파이썬 패키지 관리는 uv로 한다.
- 마지막 태스크(실제 데이터 실행)는 신경망 학습의 확률적 특성상 **정확한 숫자를
  미리 예측해 대조하지 않는다** — loss가 감소하는지, 불량 샷의 평균 오차가 정상 샷보다
  높은지 등 방향성/정성적 확인 위주로 검증한다(이전 전처리 계획의 Task 9와 달리, 이번엔
  IsolationForest처럼 완전히 결정론적인 재현이 보장되지 않음).

---

### Task 1: 프로젝트 셋업(PyTorch CPU 전용) + 시퀀스 구성

**Files:**
- Modify: `pyproject.toml` (CPU 전용 torch 인덱스 설정 + `uv add torch`)
- Create: `src/lstm_ae/__init__.py`
- Create: `src/lstm_ae/sequencing.py`
- Test: `tests/lstm_ae/test_sequencing.py`

**Interfaces:**
- Produces:
  - `make_train_windows(df: pd.DataFrame, feature_columns: list[str], window_size: int) -> np.ndarray` — shape `(num_windows, window_size, num_features)`, 비중첩.
  - `make_eval_windows(df: pd.DataFrame, feature_columns: list[str], window_size: int) -> np.ndarray` — shape `(num_shots - window_size + 1, window_size, num_features)`, stride=1 중첩.

- [ ] **Step 1: pyproject.toml에 PyTorch CPU 전용 인덱스 추가**

`pyproject.toml` 끝에 추가:

```toml

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

- [ ] **Step 2: torch 설치**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv add torch
```

Expected: `Installed ... torch==2.13.0+cpu ...` 같은 줄이 보여야 한다(정확한 버전은
다를 수 있으나 **반드시 `+cpu` 접미사**가 붙어야 하고, `nvidia-*`/`triton` 패키지가
설치 목록에 나오면 안 된다).

- [ ] **Step 3: CPU 전용 설치 검증**

```bash
uv run python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:
```
2.13.0+cpu
False
```
(버전 문자열은 다를 수 있으나 `+cpu`로 끝나야 하고, `cuda.is_available()`는 반드시
`False`여야 한다.)

- [ ] **Step 4: 패키지 디렉토리 생성**

```bash
mkdir -p src/lstm_ae tests/lstm_ae
touch src/lstm_ae/__init__.py
```

- [ ] **Step 5: 실패하는 테스트 작성**

`tests/lstm_ae/test_sequencing.py`:

```python
import numpy as np
import pandas as pd

from lstm_ae.sequencing import make_eval_windows, make_train_windows


def _make_df(num_rows: int, num_features: int) -> pd.DataFrame:
    data = {
        f"f{i}": np.arange(num_rows, dtype=np.float32) * 10 + i
        for i in range(num_features)
    }
    return pd.DataFrame(data)


def test_make_train_windows_is_non_overlapping_and_drops_remainder():
    df = _make_df(num_rows=8, num_features=2)
    feature_columns = ["f0", "f1"]

    windows = make_train_windows(df, feature_columns, window_size=3)

    assert windows.shape == (2, 3, 2)
    # window 0 = rows 0..2, window 1 = rows 3..5 (rows 6,7 dropped as remainder)
    expected_window0 = df[feature_columns].to_numpy(dtype=np.float32)[0:3]
    expected_window1 = df[feature_columns].to_numpy(dtype=np.float32)[3:6]
    np.testing.assert_array_equal(windows[0], expected_window0)
    np.testing.assert_array_equal(windows[1], expected_window1)


def test_make_eval_windows_is_overlapping_stride_one():
    df = _make_df(num_rows=8, num_features=2)
    feature_columns = ["f0", "f1"]

    windows = make_eval_windows(df, feature_columns, window_size=3)

    assert windows.shape == (6, 3, 2)
    values = df[feature_columns].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(windows[0], values[0:3])
    np.testing.assert_array_equal(windows[1], values[1:4])
    np.testing.assert_array_equal(windows[5], values[5:8])
```

- [ ] **Step 6: 테스트 실패 확인**

```bash
uv run pytest tests/lstm_ae/test_sequencing.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.sequencing'`

- [ ] **Step 7: 최소 구현**

`src/lstm_ae/sequencing.py`:

```python
import numpy as np
import pandas as pd


def make_train_windows(
    df: pd.DataFrame, feature_columns: list[str], window_size: int
) -> np.ndarray:
    values = df[feature_columns].to_numpy(dtype=np.float32)
    num_windows = len(values) // window_size
    trimmed = values[: num_windows * window_size]
    return trimmed.reshape(num_windows, window_size, len(feature_columns))


def make_eval_windows(
    df: pd.DataFrame, feature_columns: list[str], window_size: int
) -> np.ndarray:
    values = df[feature_columns].to_numpy(dtype=np.float32)
    num_windows = len(values) - window_size + 1
    return np.stack([values[i : i + window_size] for i in range(num_windows)])
```

- [ ] **Step 8: 테스트 통과 확인**

```bash
uv run pytest tests/lstm_ae/test_sequencing.py -v
```

Expected: PASS

- [ ] **Step 9: 커밋**

```bash
git add pyproject.toml uv.lock src/lstm_ae/__init__.py src/lstm_ae/sequencing.py tests/lstm_ae/test_sequencing.py
git commit -m "$(cat <<'EOF'
Set up PyTorch (CPU) and add sequence windowing

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 모델 아키텍처 (Encoder-Decoder LSTM-AE)

**Files:**
- Create: `src/lstm_ae/model.py`
- Test: `tests/lstm_ae/test_model.py`

**Interfaces:**
- Produces: `LSTMAutoencoder(nn.Module)` — `__init__(self, num_features: int, hidden_size: int = 64, latent_dim: int = 16)`, `forward(self, x: torch.Tensor) -> torch.Tensor`에서 입력과 동일한 shape `(batch, seq_len, num_features)`을 반환.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/lstm_ae/test_model.py`:

```python
import torch

from lstm_ae.model import LSTMAutoencoder


def test_forward_pass_preserves_input_shape():
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=24, hidden_size=8, latent_dim=4)
    x = torch.randn(5, 12, 24)

    output = model(x)

    assert output.shape == x.shape


def test_forward_pass_works_for_different_batch_and_seq_sizes():
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=3, hidden_size=8, latent_dim=4)
    x = torch.randn(2, 7, 3)

    output = model(x)

    assert output.shape == (2, 7, 3)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/lstm_ae/test_model.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.model'`

- [ ] **Step 3: 최소 구현**

`src/lstm_ae/model.py`:

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

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/lstm_ae/test_model.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/lstm_ae/model.py tests/lstm_ae/test_model.py
git commit -m "$(cat <<'EOF'
Add LSTM encoder-decoder autoencoder model

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 샷별 복원오차, 임계값, 평가

**Files:**
- Create: `src/lstm_ae/scoring.py`
- Test: `tests/lstm_ae/test_scoring.py`

**Interfaces:**
- Produces:
  - `flatten_train_shot_errors(squared_errors: np.ndarray) -> np.ndarray` — `(num_windows, window_size, num_features)` → `(num_windows*window_size, num_features)`, 원래 샷 순서 보존.
  - `aggregate_eval_shot_errors(squared_errors: np.ndarray) -> np.ndarray` — `(num_windows, window_size, num_features)`(stride=1 결과) → `(num_shots, num_features)`, 샷이 속한 모든 윈도우의 평균.
  - `compute_threshold(train_shot_errors: np.ndarray) -> float` — `(num_shots, num_features)` → 변수 평균의 (평균 + 3*표준편차).
  - `evaluate_predictions(eval_shot_errors: np.ndarray, threshold: float, labels: np.ndarray) -> dict` — `{"precision", "recall", "tp", "fp", "fn", "tn"}`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/lstm_ae/test_scoring.py`:

```python
import numpy as np
import pytest

from lstm_ae.scoring import (
    aggregate_eval_shot_errors,
    compute_threshold,
    evaluate_predictions,
    flatten_train_shot_errors,
)


def test_flatten_train_shot_errors_preserves_row_major_order():
    squared_errors = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )  # (2 windows, 3 window_size, 1 feature)

    result = flatten_train_shot_errors(squared_errors)

    assert result.shape == (6, 1)
    np.testing.assert_array_equal(result.flatten(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_aggregate_eval_shot_errors_averages_overlapping_windows():
    # window_size=2, 3 windows -> 4 shots. 1 feature for simplicity.
    # window0 covers shots [0,1] = [1,2]; window1 covers shots [1,2] = [3,4];
    # window2 covers shots [2,3] = [5,6]
    squared_errors = np.array(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
        ]
    )

    result = aggregate_eval_shot_errors(squared_errors)

    assert result.shape == (4, 1)
    # shot0: only window0 pos0 -> 1.0
    # shot1: window0 pos1 (2.0) + window1 pos0 (3.0) -> mean 2.5
    # shot2: window1 pos1 (4.0) + window2 pos0 (5.0) -> mean 4.5
    # shot3: only window2 pos1 -> 6.0
    np.testing.assert_allclose(result.flatten(), [1.0, 2.5, 4.5, 6.0])


def test_compute_threshold_is_mean_plus_three_std():
    # per-shot scalar error (mean over features) will be exactly [1.0, 2.0, 3.0]
    # population mean=2.0, population std=sqrt(2/3)=0.816496..., so
    # threshold = 2.0 + 3*0.816496... = 4.449489...  (independently hand-computed,
    # not re-derived from the same formula the implementation uses)
    train_shot_errors = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    threshold = compute_threshold(train_shot_errors)

    assert threshold == pytest.approx(4.449489742783178)


def test_evaluate_predictions_computes_precision_recall_and_confusion_counts():
    # scalar error (mean over the single feature) = [0.5, 5.0, 0.5, 5.0]
    eval_shot_errors = np.array([[0.5], [5.0], [0.5], [5.0]])
    labels = np.array([0, 1, 1, 0])  # shot1 correctly flagged, shot2 missed, shot3 false alarm
    threshold = 2.0

    result = evaluate_predictions(eval_shot_errors, threshold, labels)

    assert result["tp"] == 1  # shot1: predicted 1, label 1
    assert result["fp"] == 1  # shot3: predicted 1, label 0
    assert result["fn"] == 1  # shot2: predicted 0, label 1
    assert result["tn"] == 1  # shot0: predicted 0, label 0
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/lstm_ae/test_scoring.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.scoring'`

- [ ] **Step 3: 최소 구현**

`src/lstm_ae/scoring.py`:

```python
import numpy as np


def flatten_train_shot_errors(squared_errors: np.ndarray) -> np.ndarray:
    num_windows, window_size, num_features = squared_errors.shape
    return squared_errors.reshape(num_windows * window_size, num_features)


def aggregate_eval_shot_errors(squared_errors: np.ndarray) -> np.ndarray:
    num_windows, window_size, num_features = squared_errors.shape
    num_shots = num_windows + window_size - 1
    sums = np.zeros((num_shots, num_features))
    counts = np.zeros(num_shots)
    for w in range(num_windows):
        for pos in range(window_size):
            shot_idx = w + pos
            sums[shot_idx] += squared_errors[w, pos]
            counts[shot_idx] += 1
    return sums / counts[:, None]


def compute_threshold(train_shot_errors: np.ndarray) -> float:
    per_shot_scalar = train_shot_errors.mean(axis=1)
    return float(per_shot_scalar.mean() + 3 * per_shot_scalar.std())


def evaluate_predictions(
    eval_shot_errors: np.ndarray, threshold: float, labels: np.ndarray
) -> dict:
    scalar_scores = eval_shot_errors.mean(axis=1)
    predictions = (scalar_scores > threshold).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/lstm_ae/test_scoring.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/lstm_ae/scoring.py tests/lstm_ae/test_scoring.py
git commit -m "$(cat <<'EOF'
Add shot-level error aggregation, threshold, and evaluation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: 학습 루프

**Files:**
- Create: `src/lstm_ae/training.py`
- Test: `tests/lstm_ae/test_training.py`

**Interfaces:**
- Consumes: `LSTMAutoencoder`(Task 2)
- Produces: `train_autoencoder(model: nn.Module, train_windows: np.ndarray, epochs: int = 50, batch_size: int = 64, learning_rate: float = 1e-3) -> list[float]` — epoch별 평균 loss 리스트를 반환.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/lstm_ae/test_training.py`:

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

```bash
uv run pytest tests/lstm_ae/test_training.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.training'`

- [ ] **Step 3: 최소 구현**

`src/lstm_ae/training.py`:

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

```bash
uv run pytest tests/lstm_ae/test_training.py -v
```

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/lstm_ae/training.py tests/lstm_ae/test_training.py
git commit -m "$(cat <<'EOF'
Add autoencoder training loop

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 파이프라인 오케스트레이션

**Files:**
- Create: `src/lstm_ae/pipeline.py`
- Test: `tests/lstm_ae/test_pipeline.py`

**Interfaces:**
- Consumes: `make_train_windows`/`make_eval_windows`(Task 1), `LSTMAutoencoder`(Task 2),
  `flatten_train_shot_errors`/`aggregate_eval_shot_errors`/`compute_threshold`/
  `evaluate_predictions`(Task 3), `train_autoencoder`(Task 4).
- Produces: `run_lstm_pipeline(train_csv_path: str, eval_csv_path: str, feature_columns: list[str], output_dir: str, window_size: int = 12, hidden_size: int = 64, latent_dim: int = 16, epochs: int = 50, batch_size: int = 64, learning_rate: float = 1e-3, random_seed: int = 42) -> dict` —
  `output_dir`에 6개 파일(`model.pt`, `training_config.json`,
  `train_reconstruction_errors.csv`, `eval_reconstruction_errors.csv`,
  `threshold.json`, `evaluation_report.json`)을 쓰고 요약 dict를 반환.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/lstm_ae/test_pipeline.py`:

```python
import json

import numpy as np
import pandas as pd
import torch

from lstm_ae.pipeline import run_lstm_pipeline

FEATURE_COLUMNS = ["f0", "f1", "f2"]


def _make_train_csv(path, n_rows):
    data = {col: np.random.randn(n_rows).astype(np.float32) for col in FEATURE_COLUMNS}
    pd.DataFrame(data).to_csv(path, index=False)


def _make_eval_csv(path, n_rows):
    data = {col: np.random.randn(n_rows).astype(np.float32) for col in FEATURE_COLUMNS}
    data["PassOrFail"] = ["Y"] * (n_rows - 2) + ["N", "N"]
    data["Reason"] = [None] * (n_rows - 2) + ["가스", "미성형"]
    data["TimeStamp"] = [f"2020-01-01 00:00:{i:02d}" for i in range(n_rows)]
    data["label"] = [0] * (n_rows - 2) + [1, 1]
    pd.DataFrame(data).to_csv(path, index=False)


def test_run_lstm_pipeline_creates_expected_output_files(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    output_dir = tmp_path / "model"

    _make_train_csv(train_path, n_rows=60)
    _make_eval_csv(eval_path, n_rows=30)

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
        "train_reconstruction_errors.csv",
        "eval_reconstruction_errors.csv",
        "threshold.json",
        "evaluation_report.json",
    ]:
        assert (output_dir / name).exists()

    train_errors = pd.read_csv(output_dir / "train_reconstruction_errors.csv")
    assert "shot_error" in train_errors.columns
    assert all(f"error_{c}" in train_errors.columns for c in FEATURE_COLUMNS)
    assert len(train_errors) == 60 // 6 * 6  # non-overlapping, remainder dropped

    eval_errors = pd.read_csv(output_dir / "eval_reconstruction_errors.csv")
    assert len(eval_errors) == 30  # overlapping covers every eval shot
    assert "label" in eval_errors.columns
    assert "TimeStamp" in eval_errors.columns

    threshold = json.loads((output_dir / "threshold.json").read_text())
    assert "threshold" in threshold

    report = json.loads((output_dir / "evaluation_report.json").read_text())
    assert set(report.keys()) >= {"precision", "recall", "tp", "fp", "fn", "tn"}

    assert "final_train_loss" in summary
    assert summary["train_shots"] == 60 // 6 * 6
    assert summary["eval_shots"] == 30
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/lstm_ae/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lstm_ae.pipeline'`

- [ ] **Step 3: 최소 구현**

`src/lstm_ae/pipeline.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import LSTMAutoencoder
from .scoring import (
    aggregate_eval_shot_errors,
    compute_threshold,
    evaluate_predictions,
    flatten_train_shot_errors,
)
from .sequencing import make_eval_windows, make_train_windows
from .training import train_autoencoder


def _compute_squared_errors(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32)
        reconstructed = model(x).numpy()
    return (reconstructed - windows) ** 2


def run_lstm_pipeline(
    train_csv_path: str,
    eval_csv_path: str,
    feature_columns: list[str],
    output_dir: str,
    window_size: int = 12,
    hidden_size: int = 64,
    latent_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
) -> dict:
    torch.manual_seed(random_seed)

    train_df = pd.read_csv(train_csv_path)
    eval_df = pd.read_csv(eval_csv_path)

    train_windows = make_train_windows(train_df, feature_columns, window_size)
    eval_windows = make_eval_windows(eval_df, feature_columns, window_size)

    model = LSTMAutoencoder(
        num_features=len(feature_columns), hidden_size=hidden_size, latent_dim=latent_dim
    )
    loss_history = train_autoencoder(
        model, train_windows, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )

    train_squared_errors = _compute_squared_errors(model, train_windows)
    eval_squared_errors = _compute_squared_errors(model, eval_windows)

    train_shot_errors = flatten_train_shot_errors(train_squared_errors)
    eval_shot_errors = aggregate_eval_shot_errors(eval_squared_errors)

    threshold = compute_threshold(train_shot_errors)
    labels = eval_df["label"].to_numpy()
    report = evaluate_predictions(eval_shot_errors, threshold, labels)

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
    }
    (out_dir / "training_config.json").write_text(json.dumps(training_config, indent=2))

    error_columns = ["shot_error"] + [f"error_{c}" for c in feature_columns]

    train_out = pd.DataFrame(
        np.column_stack([train_shot_errors.mean(axis=1), train_shot_errors]),
        columns=error_columns,
    )
    train_out.to_csv(out_dir / "train_reconstruction_errors.csv", index=False)

    eval_out = pd.DataFrame(
        np.column_stack([eval_shot_errors.mean(axis=1), eval_shot_errors]),
        columns=error_columns,
    )
    for col in ["PassOrFail", "Reason", "TimeStamp", "label"]:
        eval_out[col] = eval_df[col].to_numpy()
    eval_out.to_csv(out_dir / "eval_reconstruction_errors.csv", index=False)

    (out_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2))

    return {
        "train_shots": len(train_shot_errors),
        "eval_shots": len(eval_shot_errors),
        "final_train_loss": loss_history[-1],
        "threshold": threshold,
        **report,
    }
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/lstm_ae/test_pipeline.py -v
```

Expected: PASS

- [ ] **Step 5: 전체 테스트 스위트 실행(회귀 확인)**

```bash
uv run pytest tests/ -v
```

Expected: 모든 테스트 PASS (기존 `tests/preprocessing/` 14개 + `tests/lstm_ae/` 새 테스트 전부)

- [ ] **Step 6: 커밋**

```bash
git add src/lstm_ae/pipeline.py tests/lstm_ae/test_pipeline.py
git commit -m "$(cat <<'EOF'
Wire LSTM autoencoder pipeline stages together

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 엔트리 스크립트 + 실제 데이터 실행

**Files:**
- Create: `scripts/run_lstm_training.py`

**Interfaces:**
- Consumes: `run_lstm_pipeline`(Task 5), `preprocessing.columns.FEATURE_COLUMNS`(기존
  전처리 패키지 — 24개 피처 컬럼 목록의 단일 출처를 재사용).
- 이 태스크는 pytest 테스트를 추가하지 않는다. 신경망 학습은 확률적이라 정확한 숫자를
  미리 정해 대조할 수 없다 — loss 추이·불량/정상 그룹 간 오차 차이 등 방향성을
  확인하는 수동 실행 검증이다.

- [ ] **Step 1: 엔트리 스크립트 작성**

`scripts/run_lstm_training.py`:

```python
from pathlib import Path

import pandas as pd

from lstm_ae.pipeline import run_lstm_pipeline
from preprocessing.columns import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    output_dir = ROOT / "data" / "model"
    summary = run_lstm_pipeline(
        train_csv_path=str(ROOT / "data" / "processed" / "train.csv"),
        eval_csv_path=str(ROOT / "data" / "processed" / "eval.csv"),
        feature_columns=FEATURE_COLUMNS,
        output_dir=str(output_dir),
    )
    print(f"train_shots: {summary['train_shots']}")
    print(f"eval_shots: {summary['eval_shots']}")
    print(f"final_train_loss: {summary['final_train_loss']:.6f}")
    print(f"threshold: {summary['threshold']:.6f}")
    print(f"precision: {summary['precision']:.4f}  recall: {summary['recall']:.4f}")
    print(f"tp={summary['tp']} fp={summary['fp']} fn={summary['fn']} tn={summary['tn']}")

    eval_errors = pd.read_csv(output_dir / "eval_reconstruction_errors.csv")
    mean_normal = eval_errors.loc[eval_errors["label"] == 0, "shot_error"].mean()
    mean_defect = eval_errors.loc[eval_errors["label"] == 1, "shot_error"].mean()
    print(f"mean shot_error (label=0, 정상): {mean_normal:.6f}")
    print(f"mean shot_error (label=1, 불량): {mean_defect:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실제 데이터로 실행**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run python scripts/run_lstm_training.py
```

50 epoch 학습이 진행되며 epoch별 loss가 출력된다(CPU 환경이라 몇 분 정도 소요될 수
있음). 실행이 끝나면 다음을 확인한다:

- **loss 추이**: epoch이 진행되며 loss가 감소하는 추세인지 (첫 epoch loss보다 마지막
  epoch loss가 낮아야 함 — 그렇지 않으면 학습률/epoch 수 조정이 필요할 수 있음, 이
  시점에 사용자와 논의).
- **정상/불량 그룹 오차 차이**: `mean shot_error (label=1, 불량)`이
  `mean shot_error (label=0, 정상)`보다 뚜렷하게 높은지 확인한다(방향이 맞다는 정성적
  신호 — 수치가 낮더라도 방향이 맞으면 임계값/윈도우 크기 등을 조정해볼 가치가 있고,
  방향 자체가 틀리면 설계를 재검토해야 한다).
- **precision/recall**: 스펙에 이미 기록된 한계(불량 18건뿐)를 감안해 절대적인 수치보다
  방향성 위주로 판단한다.

이 결과를 사용자에게 보고하고, 하이퍼파라미터 조정이 필요한지 논의한다.

- [ ] **Step 3: 산출물 확인**

```bash
ls -la data/model/
wc -l data/model/train_reconstruction_errors.csv data/model/eval_reconstruction_errors.csv
```

Expected: `train_reconstruction_errors.csv`(train_shots+1행), `eval_reconstruction_errors.csv`
(3,975행=3,974행+헤더), `model.pt`, `training_config.json`, `threshold.json`,
`evaluation_report.json` 모두 존재.

- [ ] **Step 4: 커밋**

```bash
git add scripts/run_lstm_training.py
git commit -m "$(cat <<'EOF'
Add LSTM training entry script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

(`data/model/`은 `.gitignore`의 `data` 규칙에 포함되므로 커밋 대상이 아니다.)
