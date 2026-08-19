# CNC 드리프트 트리거 기반 자동 재학습 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 드리프트가 지속되면 자동으로 재학습이 발동되고, 승격 게이트를 통과한 경우에만 champion이 교체되며, 돌고 있는 서빙이 새 모델을 집어드는 닫힌 루프를 만든다.

**Architecture:** 서빙(`src/serving/app.py`)과 분리된 감시 워커가 `GET /drift-status`를 폴링한다. 트리거 조건을 만족하면 재학습 러너가 QC 라벨이 도착한 정상 배치로 새 모델을 학습하고, 게이트가 두 조건(원본 eval recall 회귀 없음 / 라벨 도착 구간 정확도 개선)을 평가한다. 통과 시에만 백업→파일교체→alias교체→리로드 순으로 승격하며, 어느 단계든 실패하면 롤백한다.

**Tech Stack:** Python 3.14, PyTorch, MLflow 3.x, FastAPI, sqlite3(표준 라이브러리), pandas, scikit-learn. 의존성 추가 없음.

**Spec:** `docs/specs/2026-08-19-cnc-drift-triggered-retraining-design.md`

## Global Constraints

- 실행은 전부 `02-cnc-machining/`에서 `uv run`으로 한다. 새 의존성을 추가하지 않는다.
- **기존 테스트 100개가 계속 통과해야 한다.** `src/serving/app.py`를 수정하므로 매 커밋마다 확인한다.
- `src/lstm_ae/`, `src/preprocessing/`, `src/monitoring/drift.py`, `src/monitoring/logging.py`는 **수정하지 않는다.** 함수를 호출·재사용만 한다.
- 서빙 계약: champion run에 `mean_threshold`, `max_threshold`, `p95_threshold` metric과 `window_size` param이 반드시 있어야 한다(`src/serving/app.py:68-71`이 직접 읽는다). 없으면 승격 후 모델 로드가 `KeyError`로 죽는다.
- 고정 분할 상수(`src/preprocessing/split.py`): train `[1, 2, 3, 11, 13, 14, 15, 17]`, eval good `[12, 18, 22]`, eval bad `[4, 5, 6, 7, 8, 9, 10, 16, 20, 21, 23]`. 게이트 기준셋이므로 변경 금지.
- 게이트 임계: G1 recall 허용 하락폭 **0.10**(불량 11개 기준 1건까지 허용), G2는 정확도 **엄격 개선**.
- 타임라인: 40일 × 하루 5배치 = 200배치. 변형 시작 Day 11. 라벨 지연 7일. 시나리오 B 라벨 전환 Day 21.
- 합성 변형 폭은 `synthetic/real_anomaly_reference.json`의 실측 대역 안에 가둔다. 목표 `score/threshold`: 시나리오 A는 Day 40에 1.5~2.0, 시나리오 B는 3.0.
- 테스트는 기존 관례를 따른다: 평문 pytest 함수, `tmp_path` fixture, 앱 테스트는 `app.dependency_overrides` + autouse 정리 fixture. `conftest.py`는 없다.
- 커밋 메시지는 한글 없이 conventional commit 형식(기존 이력과 동일).

---

### Task 1: QC 라벨 저장소

지연 도착하는 QC 검사 결과를 담는다. 기존 `predict_log` 스키마는 건드리지 않고 별도 테이블로 둔다 — 실제 운영에서도 검사 결과는 다른 시스템에서 온다.

**Files:**
- Create: `src/monitoring/labels.py`
- Test: `tests/monitoring/test_labels.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `record_label(batch_id: str, produced_day: int, arrived_day: int, label: str, db_path: Path) -> None`
  - `get_arrived_labels(current_day: int, db_path: Path) -> list[dict]` — 각 dict는 `{"batch_id", "produced_day", "arrived_day", "label"}`, `produced_day` 오름차순

- [ ] **Step 1: Write the failing test**

`tests/monitoring/test_labels.py`:

```python
from monitoring.labels import get_arrived_labels, record_label


def test_record_and_retrieve_roundtrip(tmp_path):
    db_path = tmp_path / "labels.db"
    record_label("day01_0", produced_day=1, arrived_day=8, label="good", db_path=db_path)
    record_label("day02_0", produced_day=2, arrived_day=9, label="bad", db_path=db_path)

    arrived = get_arrived_labels(current_day=9, db_path=db_path)

    assert len(arrived) == 2
    assert arrived[0]["batch_id"] == "day01_0"  # produced_day 오름차순
    assert arrived[1]["label"] == "bad"


def test_future_labels_are_not_returned(tmp_path):
    db_path = tmp_path / "labels.db"
    record_label("day01_0", produced_day=1, arrived_day=8, label="good", db_path=db_path)
    record_label("day05_0", produced_day=5, arrived_day=12, label="good", db_path=db_path)

    arrived = get_arrived_labels(current_day=8, db_path=db_path)

    assert len(arrived) == 1
    assert arrived[0]["batch_id"] == "day01_0"


def test_missing_db_returns_empty_list(tmp_path):
    assert get_arrived_labels(current_day=99, db_path=tmp_path / "nope.db") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitoring/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'monitoring.labels'`

- [ ] **Step 3: Write minimal implementation**

`src/monitoring/labels.py`:

```python
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qc_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    produced_day INTEGER NOT NULL,
    arrived_day INTEGER NOT NULL,
    label TEXT NOT NULL
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def record_label(
    batch_id: str, produced_day: int, arrived_day: int, label: str, db_path: Path
) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            "INSERT INTO qc_labels (batch_id, produced_day, arrived_day, label) "
            "VALUES (?, ?, ?, ?)",
            (batch_id, produced_day, arrived_day, label),
        )
    conn.close()


def get_arrived_labels(current_day: int, db_path: Path) -> list[dict]:
    if not Path(db_path).exists():
        return []
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT batch_id, produced_day, arrived_day, label FROM qc_labels "
        "WHERE arrived_day <= ? ORDER BY produced_day, id",
        (current_day,),
    ).fetchall()
    conn.close()
    return [
        {"batch_id": r[0], "produced_day": r[1], "arrived_day": r[2], "label": r[3]}
        for r in rows
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/monitoring/test_labels.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: 103 passed (기존 100 + 신규 3)

- [ ] **Step 6: Commit**

```bash
git add src/monitoring/labels.py tests/monitoring/test_labels.py
git commit -m "feat: add delayed QC label store for retraining"
```

---

### Task 2: 재학습 트리거

드리프트가 **지속**될 때만 발동한다. 단발 노이즈로 재학습이 도는 것을 막는다.

**Files:**
- Create: `src/retraining/__init__.py` (빈 파일)
- Create: `src/retraining/trigger.py`
- Test: `tests/retraining/test_trigger.py`

**Interfaces:**
- Consumes: `GET /drift-status` 응답 형태 (`src/monitoring/drift.py`의 `compute_drift_status()` 반환값)
- Produces:
  - `is_drift_flagged(status: dict) -> bool`
  - `should_retrain(flag_history: list[bool], consecutive_k: int = 3, cooldown_remaining: int = 0) -> bool`

- [ ] **Step 1: Write the failing test**

`tests/retraining/test_trigger.py`:

```python
from retraining.trigger import is_drift_flagged, should_retrain


def _status(sufficient=True, output_flagged=False, input_flagged=()):
    return {
        "sufficient_data": sufficient,
        "output_drift": {"flagged": output_flagged},
        "input_drift": {"flagged_features": list(input_flagged)},
    }


def test_not_flagged_when_data_insufficient():
    assert is_drift_flagged(_status(sufficient=False, output_flagged=True)) is False


def test_flagged_on_output_drift():
    assert is_drift_flagged(_status(output_flagged=True)) is True


def test_flagged_on_input_drift():
    assert is_drift_flagged(_status(input_flagged=[{"feature": "X_OutputPower"}])) is True


def test_not_flagged_when_clean():
    assert is_drift_flagged(_status()) is False


def test_no_retrain_before_k_consecutive():
    assert should_retrain([True, True], consecutive_k=3) is False


def test_retrain_after_k_consecutive():
    assert should_retrain([False, True, True, True], consecutive_k=3) is True


def test_no_retrain_when_streak_broken():
    assert should_retrain([True, True, False], consecutive_k=3) is False


def test_no_retrain_during_cooldown():
    assert should_retrain([True, True, True], consecutive_k=3, cooldown_remaining=2) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retraining/test_trigger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retraining'`

- [ ] **Step 3: Write minimal implementation**

`src/retraining/__init__.py` — 빈 파일로 생성.

`src/retraining/trigger.py`:

```python
def is_drift_flagged(status: dict) -> bool:
    """한 번의 /drift-status 조회가 '드리프트 있음'인지 판정한다."""
    if not status.get("sufficient_data"):
        return False
    output_flagged = bool(status["output_drift"]["flagged"])
    input_flagged = bool(status["input_drift"]["flagged_features"])
    return output_flagged or input_flagged


def should_retrain(
    flag_history: list[bool],
    consecutive_k: int = 3,
    cooldown_remaining: int = 0,
) -> bool:
    """연속 consecutive_k회 드리프트가 잡히면 재학습을 발동한다.

    쿨다운 중에는 발동하지 않는다 — 재학습 직후에는 요청 로그에 옛 데이터가
    남아 있어 즉시 재발동해 버린다.
    """
    if cooldown_remaining > 0:
        return False
    if len(flag_history) < consecutive_k:
        return False
    return all(flag_history[-consecutive_k:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retraining/test_trigger.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: 111 passed

- [ ] **Step 6: Commit**

```bash
git add src/retraining/ tests/retraining/
git commit -m "feat: add drift retraining trigger with hysteresis"
```

---

### Task 3: 승격 게이트

두 조건의 AND. G1은 "승격하면 안 되는 경우"를 거르고, G2는 "승격할 이유"를 댄다.

**Files:**
- Create: `src/retraining/gate.py`
- Test: `tests/retraining/test_gate.py`

**Interfaces:**
- Consumes: 없음 (순수 함수)
- Produces:
  - `evaluate_gate(retrained_recall: float, champion_recall: float, retrained_accuracy: float, champion_accuracy: float, recall_tolerance: float = 0.10) -> dict`
  - 반환 dict 키: `decision`(`"promoted"` | `"rejected"`), `g1_pass`, `g2_pass`, `g1_recall`, `g2_accuracy_delta`, `reject_reason`

- [ ] **Step 1: Write the failing test**

`tests/retraining/test_gate.py`:

```python
from retraining.gate import evaluate_gate

CHAMPION_RECALL = 10 / 11  # 0.9091 — 현 champion 실측


def test_promoted_when_both_conditions_pass():
    result = evaluate_gate(
        retrained_recall=CHAMPION_RECALL,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["decision"] == "promoted"
    assert result["g1_pass"] is True
    assert result["g2_pass"] is True
    assert result["reject_reason"] == ""


def test_g1_boundary_one_extra_miss_passes():
    # 9/11 = 0.8182, 허용선 0.9091 - 0.10 = 0.8091 → 통과
    result = evaluate_gate(
        retrained_recall=9 / 11,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["g1_pass"] is True
    assert result["decision"] == "promoted"


def test_g1_boundary_two_extra_misses_rejects():
    # 8/11 = 0.7273 < 0.8091 → 거부
    result = evaluate_gate(
        retrained_recall=8 / 11,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["g1_pass"] is False
    assert result["decision"] == "rejected"
    assert "G1" in result["reject_reason"]


def test_g2_rejects_when_no_improvement():
    result = evaluate_gate(
        retrained_recall=CHAMPION_RECALL,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.70,
        champion_accuracy=0.70,
    )

    assert result["g2_pass"] is False
    assert result["decision"] == "rejected"
    assert "G2" in result["reject_reason"]


def test_reject_reason_lists_both_violations():
    result = evaluate_gate(
        retrained_recall=0.10,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.10,
        champion_accuracy=0.70,
    )

    assert "G1" in result["reject_reason"]
    assert "G2" in result["reject_reason"]


def test_accuracy_delta_is_reported():
    result = evaluate_gate(
        retrained_recall=CHAMPION_RECALL,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.85,
        champion_accuracy=0.70,
    )

    assert result["g2_accuracy_delta"] == 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retraining/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retraining.gate'`

- [ ] **Step 3: Write minimal implementation**

`src/retraining/gate.py`:

```python
RECALL_TOLERANCE = 0.10


def evaluate_gate(
    retrained_recall: float,
    champion_recall: float,
    retrained_accuracy: float,
    champion_accuracy: float,
    recall_tolerance: float = RECALL_TOLERANCE,
) -> dict:
    """승격 여부를 두 조건의 AND로 판정한다.

    G1 (안전): 원본 실측 eval셋에서 불량 검출력이 champion 대비 유지되는가.
      허용 하락폭 0.10은 불량 11개 기준 1건(0.0909)까지만 봐준다는 뜻이다.
      precision을 보지 않는 이유는, 센서 좌표계가 이동한 환경에서 새 모델을
      옛 좌표계 eval에 적용하면 precision이 좌표계 차이 때문에 떨어지기 때문이다.
    G2 (근거): 라벨이 도착한 최근 구간에서 실제로 나아졌는가.
      G1만으로는 모든 것을 불량이라 판정하는 모델도 recall 1.0으로 통과한다.
    """
    g1_pass = retrained_recall >= champion_recall - recall_tolerance
    g2_pass = retrained_accuracy > champion_accuracy

    reasons = []
    if not g1_pass:
        reasons.append(
            f"G1 recall 회귀: {retrained_recall:.4f} < "
            f"{champion_recall - recall_tolerance:.4f}"
        )
    if not g2_pass:
        reasons.append(
            f"G2 개선 없음: {retrained_accuracy:.4f} <= {champion_accuracy:.4f}"
        )

    return {
        "decision": "promoted" if (g1_pass and g2_pass) else "rejected",
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "g1_recall": retrained_recall,
        "g2_accuracy_delta": retrained_accuracy - champion_accuracy,
        "reject_reason": "; ".join(reasons),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retraining/test_gate.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: 117 passed

- [ ] **Step 6: Commit**

```bash
git add src/retraining/gate.py tests/retraining/test_gate.py
git commit -m "feat: add promotion gate with recall regression guard"
```

---

### Task 4: 서빙 계약 검증 + 승격 파일 조작

승격에서 가장 위험한 부분이다. 모델(MLflow alias)과 동반 파일(디스크)이 서로 다른 저장소에 있어, 중간에 실패하면 짝이 어긋난다.

**Files:**
- Create: `src/retraining/promotion.py`
- Test: `tests/retraining/test_promotion.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `verify_serving_contract(metrics: dict, params: dict) -> list[str]` — 빠진 키 이름 목록. 빈 리스트면 계약 충족
  - `backup_artifacts(model_dir: Path, scaler_path: Path, backup_root: Path) -> Path` — 만들어진 백업 디렉터리
  - `install_artifacts(retrain_dir: Path, model_dir: Path, scaler_path: Path) -> None`
  - `restore_backup(backup_dir: Path, model_dir: Path, scaler_path: Path) -> None`
  - `swap_with_rollback(retrain_dir: Path, model_dir: Path, scaler_path: Path, backup_root: Path, promote: Callable[[], None], verify: Callable[[], None]) -> Path`

**`swap_with_rollback`을 `src/`에 두는 이유:** 이것이 이 작업에서 가장 위험한
로직인데, 워커 스크립트(`monitoring/`)에 두면 pytest가 import할 수 없어
테스트가 불가능하다(`.pth` 설정상 `src/`만 import된다). 실패 주입 테스트가
반드시 필요한 코드이므로 `src/retraining/`에 둔다. 워커는 호출만 한다.

- [ ] **Step 1: Write the failing test**

`tests/retraining/test_promotion.py`:

```python
import json

import pytest

from retraining.promotion import (
    backup_artifacts,
    install_artifacts,
    restore_backup,
    swap_with_rollback,
    verify_serving_contract,
)


def _full_metrics():
    return {"mean_threshold": 0.85, "max_threshold": 1.2, "p95_threshold": 1.0}


def test_contract_satisfied_returns_empty_list():
    assert verify_serving_contract(_full_metrics(), {"window_size": "20"}) == []


def test_contract_detects_missing_metric():
    metrics = _full_metrics()
    del metrics["mean_threshold"]

    missing = verify_serving_contract(metrics, {"window_size": "20"})

    assert missing == ["mean_threshold"]


def test_contract_detects_missing_param():
    missing = verify_serving_contract(_full_metrics(), {})

    assert missing == ["window_size"]


def _make_current(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("OLD_MODEL")
    (model_dir / "feature_baseline.json").write_text(json.dumps({"mean": {}, "std": {}}))
    scaler_path = tmp_path / "processed" / "scaler.json"
    scaler_path.parent.mkdir()
    scaler_path.write_text(json.dumps({"old": True}))
    return model_dir, scaler_path


def test_backup_then_install_then_restore_round_trip(tmp_path):
    model_dir, scaler_path = _make_current(tmp_path)
    retrain_dir = tmp_path / "retrain"
    retrain_dir.mkdir()
    (retrain_dir / "model.pt").write_text("NEW_MODEL")
    (retrain_dir / "feature_baseline.json").write_text(json.dumps({"mean": {}, "std": {}}))
    (retrain_dir / "scaler.json").write_text(json.dumps({"old": False}))
    (retrain_dir / "train.csv").write_text("should_not_be_installed")

    backup_dir = backup_artifacts(model_dir, scaler_path, tmp_path / "backup")
    install_artifacts(retrain_dir, model_dir, scaler_path)

    assert (model_dir / "model.pt").read_text() == "NEW_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": False}
    assert not (model_dir / "train.csv").exists()  # 학습 입력은 옮기지 않는다

    restore_backup(backup_dir, model_dir, scaler_path)

    assert (model_dir / "model.pt").read_text() == "OLD_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": True}


def _make_retrain(tmp_path):
    retrain_dir = tmp_path / "retrain"
    retrain_dir.mkdir()
    (retrain_dir / "model.pt").write_text("NEW_MODEL")
    (retrain_dir / "feature_baseline.json").write_text(json.dumps({"mean": {}, "std": {}}))
    (retrain_dir / "scaler.json").write_text(json.dumps({"old": False}))
    return retrain_dir


def test_swap_with_rollback_keeps_new_artifacts_on_success(tmp_path):
    model_dir, scaler_path = _make_current(tmp_path)
    retrain_dir = _make_retrain(tmp_path)

    swap_with_rollback(
        retrain_dir, model_dir, scaler_path, tmp_path / "backup",
        promote=lambda: None, verify=lambda: None,
    )

    assert (model_dir / "model.pt").read_text() == "NEW_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": False}


def test_swap_with_rollback_restores_when_promote_fails(tmp_path):
    model_dir, scaler_path = _make_current(tmp_path)
    retrain_dir = _make_retrain(tmp_path)

    def _promote_boom():
        raise RuntimeError("alias 교체 실패")

    with pytest.raises(RuntimeError, match="alias 교체 실패"):
        swap_with_rollback(
            retrain_dir, model_dir, scaler_path, tmp_path / "backup",
            promote=_promote_boom, verify=lambda: None,
        )

    # 정본이 원래대로 복원돼야 한다
    assert (model_dir / "model.pt").read_text() == "OLD_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": True}


def test_swap_with_rollback_restores_when_verify_fails(tmp_path):
    model_dir, scaler_path = _make_current(tmp_path)
    retrain_dir = _make_retrain(tmp_path)

    def _verify_boom():
        raise RuntimeError("리로드 후 버전 불일치")

    with pytest.raises(RuntimeError, match="버전 불일치"):
        swap_with_rollback(
            retrain_dir, model_dir, scaler_path, tmp_path / "backup",
            promote=lambda: None, verify=_verify_boom,
        )

    assert (model_dir / "model.pt").read_text() == "OLD_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retraining/test_promotion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retraining.promotion'`

- [ ] **Step 3: Write minimal implementation**

`src/retraining/promotion.py`:

```python
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

# 서빙이 champion run에서 직접 읽는 값들 (src/serving/app.py:68-71).
# 이게 없으면 승격은 성공하는데 모델 로드가 KeyError로 죽는다.
REQUIRED_METRICS = ["mean_threshold", "max_threshold", "p95_threshold"]
REQUIRED_PARAMS = ["window_size"]

# 학습 입력이라 정본 자리로 옮기지 않는 파일들
NOT_INSTALLED = {"train.csv", "eval.csv", "scaler.json"}


def verify_serving_contract(metrics: dict, params: dict) -> list[str]:
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    missing += [key for key in REQUIRED_PARAMS if key not in params]
    return missing


def backup_artifacts(model_dir: Path, scaler_path: Path, backup_root: Path) -> Path:
    backup_dir = Path(backup_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, backup_dir / "model", dirs_exist_ok=True)
    shutil.copy2(scaler_path, backup_dir / "scaler.json")
    return backup_dir


def install_artifacts(retrain_dir: Path, model_dir: Path, scaler_path: Path) -> None:
    for item in Path(retrain_dir).iterdir():
        if item.is_dir() or item.name in NOT_INSTALLED:
            continue
        shutil.copy2(item, Path(model_dir) / item.name)
    shutil.copy2(Path(retrain_dir) / "scaler.json", scaler_path)


def restore_backup(backup_dir: Path, model_dir: Path, scaler_path: Path) -> None:
    shutil.rmtree(model_dir, ignore_errors=True)
    shutil.copytree(Path(backup_dir) / "model", model_dir)
    shutil.copy2(Path(backup_dir) / "scaler.json", scaler_path)


def swap_with_rollback(
    retrain_dir: Path,
    model_dir: Path,
    scaler_path: Path,
    backup_root: Path,
    promote: Callable[[], None],
    verify: Callable[[], None],
) -> Path:
    """백업 → 파일 교체 → alias 교체 → 검증. 어느 단계든 실패하면 정본을 되돌린다.

    모델(MLflow alias)과 동반 파일(디스크)이 서로 다른 저장소에 있어, 중간에
    실패하면 짝이 어긋난 상태로 남는다. 그 상태는 에러 없이 조용히 틀린
    스케일로 추론하므로 반드시 롤백해야 한다.

    promote/verify를 콜러블로 받는 이유는 MLflow와 HTTP 호출을 이 함수에서
    떼어내 실패 주입 테스트를 가능하게 하기 위함이다.
    """
    backup_dir = backup_artifacts(model_dir, scaler_path, backup_root)
    try:
        install_artifacts(retrain_dir, model_dir, scaler_path)
        promote()
        verify()
    except Exception:
        restore_backup(backup_dir, model_dir, scaler_path)
        raise
    return backup_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retraining/test_promotion.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: 124 passed

- [ ] **Step 6: Commit**

```bash
git add src/retraining/promotion.py tests/retraining/test_promotion.py
git commit -m "feat: add serving contract check and artifact swap with rollback"
```

---

### Task 5: 서빙 모델 리로드 엔드포인트 (결함 ① 수정)

승격해도 돌고 있는 서버가 옛 모델을 계속 쓰는 문제를 고친다. 루프의 마지막 고리다.

**Files:**
- Modify: `src/serving/app.py` (`/drift-status` 라우트 뒤에 추가)
- Test: `tests/serving/test_app.py` (파일 끝에 추가)

**Interfaces:**
- Consumes: 기존 `load_model_state()`
- Produces: `POST /reload-model` → `{"status": "reloaded", "model_version": "<버전>"}`, 실패 시 500

**주의:** 기존 테스트가 `TestClient(app)`을 `with` 없이 쓰고 있어 lifespan이 실행되지 않는다. 따라서 `monkeypatch.setattr(app_module, "_state", ...)`로 모듈 전역을 직접 세팅하면 그대로 유지된다.

- [ ] **Step 1: Write the failing test**

`tests/serving/test_app.py` 끝에 추가:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/serving/test_app.py -k reload -v`
Expected: FAIL — 404 (라우트 없음)

- [ ] **Step 3: Write minimal implementation**

`src/serving/app.py`의 `/drift-status` 라우트 뒤에 추가:

```python
@app.post("/reload-model")
def reload_model() -> dict:
    """champion alias가 바뀐 뒤 돌고 있는 서버가 새 모델을 집어들게 한다.

    로드에 실패하면 기존 상태를 유지한다 — 교체 실패가 서빙 중단으로
    번지면 안 된다.
    """
    global _state
    previous = _state
    try:
        _state = load_model_state()
    except Exception as exc:
        _state = previous
        raise HTTPException(status_code=500, detail=f"모델 리로드 실패, 기존 모델 유지: {exc}")
    return {"status": "reloaded", "model_version": _state.model_version}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/serving/test_app.py -v`
Expected: 기존 앱 테스트 전부 + 신규 2개 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: 126 passed

- [ ] **Step 6: Commit**

```bash
git add src/serving/app.py tests/serving/test_app.py
git commit -m "feat: add model reload endpoint for automated promotion"
```

---

### Task 6: 동반 아티팩트 버전 결합 (결함 ② 수정)

`scaler.json`과 `feature_baseline.json`이 모델과 따로 노는 문제를 고친다. MLflow run 아티팩트를 우선 읽고, 없으면 기존 고정 경로로 폴백한다. **폴백이 필수인 이유: 현재 champion run에는 이 아티팩트가 없다.** 폴백이 없으면 기존 서빙과 테스트가 전부 깨진다.

**Files:**
- Modify: `src/serving/app.py:62-87` (`load_model_state()`)
- Test: `tests/serving/test_app.py` (파일 끝에 추가)

**Interfaces:**
- Consumes: 없음
- Produces: `load_companion_json(run_id: str, name: str, fallback: Path) -> dict` — `src/serving/app.py`의 모듈 레벨 함수

- [ ] **Step 1: Write the failing test**

`tests/serving/test_app.py` 끝에 추가:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/serving/test_app.py -k companion -v`
Expected: FAIL — `AttributeError: module 'serving.app' has no attribute 'load_companion_json'`

- [ ] **Step 3: Write minimal implementation**

`src/serving/app.py` 상단 import에 추가:

```python
import mlflow.artifacts
```

`load_model_state()` 위에 함수 추가:

```python
def load_companion_json(run_id: str, name: str, fallback: Path) -> dict:
    """모델 run에 붙은 동반 아티팩트를 우선 읽고, 없으면 고정 경로로 폴백한다.

    폴백이 필요한 이유: 최초 학습으로 만들어진 기존 champion run에는 이
    아티팩트가 없다. 자동 재학습으로 만들어진 run만 갖고 있다.
    """
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"companion/{name}"
        )
        return json.loads(Path(local_path).read_text())
    except Exception:
        return json.loads(fallback.read_text())
```

`load_model_state()` 안의 두 줄을 교체 — 기존:

```python
    scaler_dict = json.loads((ROOT / "data" / "processed" / "scaler.json").read_text())
    feature_baseline = json.loads((ROOT / "data" / "model" / "feature_baseline.json").read_text())
```

새로:

```python
    scaler_dict = load_companion_json(
        mv.run_id, "scaler.json", ROOT / "data" / "processed" / "scaler.json"
    )
    feature_baseline = load_companion_json(
        mv.run_id, "feature_baseline.json", ROOT / "data" / "model" / "feature_baseline.json"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/serving/test_app.py -v`
Expected: 전부 passed

- [ ] **Step 5: 실제 champion으로 폴백이 동작하는지 확인**

Run:
```bash
uv run python -c "
from serving.app import load_model_state
s = load_model_state()
print('model_version:', s.model_version)
print('scaler keys:', len(s.scaler_dict))
print('baseline keys:', list(s.feature_baseline))
"
```
Expected: 예외 없이 로드되고 `scaler keys: 41`. 현재 champion run에는 companion 아티팩트가 없으므로 **폴백 경로를 타는 것이 정상**이다.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest`
Expected: 128 passed

- [ ] **Step 7: Commit**

```bash
git add src/serving/app.py tests/serving/test_app.py
git commit -m "fix: version companion artifacts with model, fallback to local paths"
```

---

### Task 7: 재학습 데이터 구성

라벨이 도착한 정상 배치만 모아 새 train셋과 새 scaler를 만들고, eval셋을 새 scaler 좌표계로 옮긴다.

**eval을 원본 데이터셋에서 다시 만들지 않고 재스케일링하는 이유:** `data/processed/eval.csv`에는 이미 `label`, `experiment_id` 등 메타 컬럼이 붙어 있다. 옛 scaler로 역변환한 뒤 새 scaler로 재변환하면 **라벨과 실험 구성이 그대로 보존**되어 게이트 기준셋의 불변성이 보장되고, 라벨 부여 로직을 다시 구현할 필요도 없다.

**Files:**
- Create: `src/retraining/runner.py`
- Test: `tests/retraining/test_runner.py`

**Interfaces:**
- Consumes: `get_arrived_labels()` (Task 1)
- Produces:
  - `collect_normal_batches(arrived_labels: list[dict], timeline_dir: Path, current_day: int, lookback_days: int = 30) -> pd.DataFrame` — `experiment_id` 컬럼이 배치마다 다르게 붙은 raw 프레임
  - `rescale_eval(eval_scaled: pd.DataFrame, old_scaler: dict, new_scaler: dict, feature_columns: list[str]) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

`tests/retraining/test_runner.py`:

```python
import numpy as np
import pandas as pd
import pytest

from retraining.runner import collect_normal_batches, rescale_eval


def _write_batch(timeline_dir, batch_id, value):
    timeline_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"X_OutputPower": [value, value + 1.0]}).to_csv(
        timeline_dir / f"{batch_id}.csv", index=False
    )


def _label(batch_id, produced_day, label="good"):
    return {
        "batch_id": batch_id,
        "produced_day": produced_day,
        "arrived_day": produced_day + 7,
        "label": label,
    }


def test_collects_only_good_labels(tmp_path):
    _write_batch(tmp_path, "day05_0", 1.0)
    _write_batch(tmp_path, "day06_0", 2.0)
    labels = [_label("day05_0", 5), _label("day06_0", 6, label="bad")]

    result = collect_normal_batches(labels, tmp_path, current_day=20, lookback_days=30)

    assert len(result) == 2  # 배치 1개 × 2행
    assert result["experiment_id"].nunique() == 1


def test_respects_lookback_window(tmp_path):
    _write_batch(tmp_path, "day02_0", 1.0)
    _write_batch(tmp_path, "day25_0", 2.0)
    labels = [_label("day02_0", 2), _label("day25_0", 25)]

    result = collect_normal_batches(labels, tmp_path, current_day=30, lookback_days=10)

    assert result["experiment_id"].nunique() == 1
    assert result["X_OutputPower"].iloc[0] == 2.0


def test_each_batch_gets_distinct_experiment_id(tmp_path):
    _write_batch(tmp_path, "day05_0", 1.0)
    _write_batch(tmp_path, "day05_1", 2.0)
    labels = [_label("day05_0", 5), _label("day05_1", 5)]

    result = collect_normal_batches(labels, tmp_path, current_day=20, lookback_days=30)

    assert result["experiment_id"].nunique() == 2


def test_raises_when_no_usable_batches(tmp_path):
    labels = [_label("day05_0", 5, label="bad")]

    with pytest.raises(ValueError, match="정상 라벨 배치가 없습니다"):
        collect_normal_batches(labels, tmp_path, current_day=20, lookback_days=30)


def test_rescale_eval_is_identity_when_scalers_match():
    columns = ["X_OutputPower"]
    scaler = {"X_OutputPower": {"mean": 2.0, "std": 4.0}}
    eval_df = pd.DataFrame({"X_OutputPower": [0.5, -0.5], "label": [0, 1]})

    result = rescale_eval(eval_df, scaler, scaler, columns)

    np.testing.assert_allclose(result["X_OutputPower"], [0.5, -0.5])


def test_rescale_eval_moves_to_new_coordinates_and_keeps_labels():
    columns = ["X_OutputPower"]
    old = {"X_OutputPower": {"mean": 0.0, "std": 1.0}}
    new = {"X_OutputPower": {"mean": 10.0, "std": 2.0}}
    eval_df = pd.DataFrame({"X_OutputPower": [10.0, 12.0], "label": [0, 1]})

    result = rescale_eval(eval_df, old, new, columns)

    # raw = 10, 12  →  (raw - 10) / 2 = 0, 1
    np.testing.assert_allclose(result["X_OutputPower"], [0.0, 1.0])
    assert result["label"].tolist() == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retraining/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retraining.runner'`

- [ ] **Step 3: Write minimal implementation**

`src/retraining/runner.py`:

```python
from pathlib import Path

import pandas as pd


def collect_normal_batches(
    arrived_labels: list[dict],
    timeline_dir: Path,
    current_day: int,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """라벨이 도착했고 정상인 배치만 모아 raw 프레임으로 합친다.

    LSTM-AE는 정상 데이터만 학습하므로 불량 라벨은 제외한다.
    배치 하나가 가공 1회분이므로, 배치마다 다른 experiment_id를 부여한다
    (lstm_ae 파이프라인이 experiment_id로 윈도우를 그룹핑한다).
    """
    cutoff = current_day - lookback_days
    frames = []
    for record in arrived_labels:
        if record["label"] != "good" or record["produced_day"] < cutoff:
            continue
        csv_path = Path(timeline_dir) / f"{record['batch_id']}.csv"
        frames.append(pd.read_csv(csv_path).assign(experiment_id=len(frames)))

    if not frames:
        raise ValueError("재학습에 쓸 정상 라벨 배치가 없습니다")
    return pd.concat(frames, ignore_index=True)


def rescale_eval(
    eval_scaled: pd.DataFrame,
    old_scaler: dict,
    new_scaler: dict,
    feature_columns: list[str],
) -> pd.DataFrame:
    """옛 scaler로 스케일된 eval을 원값으로 되돌린 뒤 새 scaler로 다시 스케일한다.

    라벨·실험 구성 메타 컬럼은 손대지 않으므로 게이트 기준셋이 불변으로 유지된다.
    """
    out = eval_scaled.copy()
    for col in feature_columns:
        raw = out[col] * old_scaler[col]["std"] + old_scaler[col]["mean"]
        out[col] = (raw - new_scaler[col]["mean"]) / new_scaler[col]["std"]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retraining/test_runner.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: 134 passed

- [ ] **Step 6: Commit**

```bash
git add src/retraining/runner.py tests/retraining/test_runner.py
git commit -m "feat: build retraining dataset from labeled normal batches"
```

---

### Task 8: 재학습 실행 + MLflow 계약 충족

Task 7의 데이터 구성을 실제 학습으로 잇는다. 여기서 **서빙 계약(metric 3개 + param 1개)을 반드시 남긴다.**

**Files:**
- Modify: `src/retraining/runner.py` (함수 추가)
- Test: `tests/retraining/test_runner.py` (추가)

**Interfaces:**
- Consumes: `collect_normal_batches`, `rescale_eval` (Task 7), `run_lstm_pipeline`, `fit_scaler`, `scaler_to_dict`, `build_run_metrics`
- Produces: `run_retraining(timeline_dir: Path, labels_db: Path, current_day: int, root: Path) -> dict` — 반환 dict 키: `run_id`, `model_version`, `retrain_dir`, `recall`, `thresholds`

**`build_run_params()`를 쓰지 않는 이유:** 이 함수는 전처리 manifest의 `experiment_split`(train/eval 실험 ID 목록)을 요구하는데(`src/lstm_ae/tracking.py:96-103`), 재학습의 학습 데이터는 실험 ID가 아니라 날짜 배치라 그 개념이 성립하지 않는다. 러너가 자체 params를 만든다.

- [ ] **Step 1: Write the failing test**

`tests/retraining/test_runner.py` 끝에 추가:

```python
def test_retrain_params_include_window_size_for_serving_contract():
    from retraining.runner import build_retrain_params

    params = build_retrain_params(batch_days="11-21", batch_count=55)

    # src/serving/app.py:71 이 이 param을 직접 읽는다. 없으면 승격 후 로드가 죽는다.
    assert "window_size" in params
    assert params["source"] == "auto_retrain"
    assert params["retrain_batch_days"] == "11-21"
    assert params["retrain_batch_count"] == 55
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retraining/test_runner.py -k contract -v`
Expected: FAIL — `ImportError: cannot import name 'build_retrain_params'`

- [ ] **Step 3: Write minimal implementation**

`src/retraining/runner.py`에 추가 (상단 import 보강):

```python
import json
from datetime import datetime

import mlflow
import mlflow.pytorch

from lstm_ae.pipeline import run_lstm_pipeline
from lstm_ae.tracking import REGISTERED_MODEL_NAME, build_run_metrics, configure_tracking
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
from preprocessing.scaling import fit_scaler, scaler_to_dict

# scripts/run_lstm_training.py 의 TRAINING_CONFIG 와 동일하게 유지한다.
# 재학습은 모델 구조·하이퍼파라미터를 바꾸지 않는다 — 바뀌는 것은 데이터와 scaler뿐이다.
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


def build_retrain_params(batch_days: str, batch_count: int) -> dict:
    """재학습 run의 params. window_size 가 반드시 들어가야 서빙이 로드할 수 있다."""
    return {
        **TRAINING_CONFIG,
        "source": "auto_retrain",
        "retrain_batch_days": batch_days,
        "retrain_batch_count": batch_count,
    }


def run_retraining(
    timeline_dir: Path,
    labels_db: Path,
    current_day: int,
    root: Path,
    lookback_days: int = 30,
) -> dict:
    """라벨 도착분으로 재학습하고 MLflow에 새 run으로 기록한다. 승격은 하지 않는다."""
    from monitoring.labels import get_arrived_labels

    arrived = get_arrived_labels(current_day, labels_db)
    train_raw = collect_normal_batches(arrived, timeline_dir, current_day, lookback_days)

    retrain_dir = root / "data" / "retrain" / datetime.now().strftime("%Y%m%d_%H%M%S")
    retrain_dir.mkdir(parents=True, exist_ok=True)

    scaler = fit_scaler(train_raw, FEATURE_COLUMNS)
    new_scaler_dict = scaler_to_dict(scaler, FEATURE_COLUMNS)
    (retrain_dir / "scaler.json").write_text(
        json.dumps(new_scaler_dict, indent=2, ensure_ascii=False)
    )

    train_scaled = train_raw.copy()
    train_scaled[FEATURE_COLUMNS] = scaler.transform(train_raw[FEATURE_COLUMNS])
    train_scaled[FEATURE_COLUMNS + ["experiment_id"]].to_csv(
        retrain_dir / "train.csv", index=False
    )

    old_scaler_dict = json.loads((root / "data" / "processed" / "scaler.json").read_text())
    eval_old = pd.read_csv(root / "data" / "processed" / "eval.csv")
    rescale_eval(eval_old, old_scaler_dict, new_scaler_dict, FEATURE_COLUMNS).to_csv(
        retrain_dir / "eval.csv", index=False
    )

    used_days = sorted({r["produced_day"] for r in arrived if r["label"] == "good"})
    batch_days = f"{used_days[0]}-{used_days[-1]}" if used_days else "none"

    configure_tracking()
    with mlflow.start_run() as active:
        mlflow.log_params(
            build_retrain_params(batch_days, train_raw["experiment_id"].nunique())
        )
        summary = run_lstm_pipeline(
            train_csv_path=str(retrain_dir / "train.csv"),
            eval_csv_path=str(retrain_dir / "eval.csv"),
            feature_columns=FEATURE_COLUMNS,
            output_dir=str(retrain_dir),
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
            **TRAINING_CONFIG,
        )
        # 서빙 계약: {mean,max,p95}_threshold 를 만든다
        mlflow.log_metrics(build_run_metrics(summary["thresholds"], summary["results"]))
        for name in ["scaler.json", "feature_baseline.json"]:
            mlflow.log_artifact(str(retrain_dir / name), artifact_path="companion")
        model_info = mlflow.pytorch.log_model(
            summary["model"],
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )
        run_id = active.info.run_id

    return {
        "run_id": run_id,
        "model_version": model_info.registered_model_version,
        "retrain_dir": retrain_dir,
        "recall": summary["results"]["mean"]["recall"],
        "thresholds": summary["thresholds"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retraining/test_runner.py -v`
Expected: 7 passed

- [ ] **Step 5: `model_info.registered_model_version` 속성명 확인**

MLflow 3.x에서 `log_model()` 반환 객체의 등록 버전 속성명을 실제로 확인한다:

```bash
uv run python -c "
import mlflow.models
print([a for a in dir(mlflow.models.model.ModelInfo) if 'version' in a.lower()])
"
```
Expected: `registered_model_version`이 목록에 있어야 한다. 다르면 `run_retraining`의 반환문을 실제 속성명으로 고친다.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest`
Expected: 135 passed

- [ ] **Step 7: Commit**

```bash
git add src/retraining/runner.py tests/retraining/test_runner.py
git commit -m "feat: run retraining with serving contract metrics and companion artifacts"
```

---

### Task 9: 타임라인 스트림 생성기

**Files:**
- Create: `monitoring/simulate_timeline.py`

**Interfaces:**
- Consumes: `src/monitoring/labels.py`의 `record_label`
- Produces: `data/timeline/<scenario>/day{NN}_{i}.csv` 배치 파일, `data/monitoring/labels.db`의 라벨 레코드
- 모듈 함수: `apply_temperature(df, progress)`, `apply_tool_wear(df, progress)`, `generate_batch(day, index, scenario, dataset_dir)`

기존 관례대로 `src/`에 넣지 않는다(`loocv/`, `synthetic/`, `monitoring/`과 동일). `Path(__file__).parent.parent`로 루트를 역산하므로 파일 깊이를 바꾸지 않는다.

- [ ] **Step 1: 스크립트 작성**

`monitoring/simulate_timeline.py`:

```python
"""가상 운영 타임라인을 만들어 /predict 로 흘려보낸다.

보유 데이터에는 시간축이 없다(실험 25개는 서로 순서가 없는 독립 샘플).
이 스크립트는 train 실험 8개를 재료로 "날짜가 지날수록 조금씩 더 틀어진"
스트림을 만들어, 드리프트가 서서히 심해지는 상황을 재현한다.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from monitoring.labels import record_label  # noqa: E402
from preprocessing.split import TRAIN_EXPERIMENT_IDS  # noqa: E402

# monitoring/simulate_drift.py 와 동일한 경로 — 원본 CSV 는 두 단계 더 깊다.
DATASET_DIR = (
    ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
)
LABELS_DB = ROOT / "data" / "monitoring" / "labels.db"

TOTAL_DAYS = 40
BATCHES_PER_DAY = 5
DRIFT_START_DAY = 10          # Day 1~10 은 변형 없는 baseline 구간
LABEL_DELAY_DAYS = 7
WEAR_LABEL_FLIP_DAY = 21      # 시나리오 B에서 QC 불합격이 시작되는 날

# Task 10 의 스윕으로 확정한다. 그 전까지는 자리값이며 스윕 후 반드시 교체한다.
POS_DRIFT = 0.0
CUR_DRIFT = 0.0
WEAR_RATE = 0.0

TEMP_POSITION_COLUMNS = ["X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition"]
TEMP_CURRENT_COLUMNS = [
    "X_OutputCurrent", "Y_OutputCurrent", "X_OutputPower", "Y_OutputPower",
]
WEAR_COLUMNS = [
    "S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback",
    "X_OutputPower", "Y_OutputPower",
]


def progress_for(day: int) -> float:
    return max(0.0, (day - DRIFT_START_DAY) / (TOTAL_DAYS - DRIFT_START_DAY))


def apply_temperature(df: pd.DataFrame, progress: float) -> pd.DataFrame:
    """온도 상승: 열변위로 Actual 위치가 지령 대비 벌어지고, 서보 권선 저항이
    올라가 같은 토크에 전류·파워가 더 든다. SetPosition 계열은 건드리지 않는다."""
    out = df.copy()
    for col in TEMP_POSITION_COLUMNS:
        out[col] = out[col] + out[col].std() * POS_DRIFT * progress
    for col in TEMP_CURRENT_COLUMNS:
        out[col] = out[col] * (1.0 + CUR_DRIFT * progress)
    return out


def apply_tool_wear(df: pd.DataFrame, progress: float) -> pd.DataFrame:
    """공구마모: 절삭이 진행될수록 주축 부하가 선형으로 커진다
    (synthetic/generate_synthetic.py 의 tool_wear 패턴과 동일한 램프)."""
    out = df.copy()
    ramp = 1.0 + (WEAR_RATE * progress) * (pd.Series(range(len(out))) / max(len(out) - 1, 1))
    for col in WEAR_COLUMNS:
        out[col] = out[col] * ramp.to_numpy()
    return out


PERTURBATIONS = {"temperature": apply_temperature, "tool_wear": apply_tool_wear}


def true_label(scenario: str, day: int) -> str:
    """제품이 실제로 불량이냐. 온도는 제품 품질을 바꾸지 않는다."""
    if scenario == "temperature":
        return "good"
    return "bad" if day >= WEAR_LABEL_FLIP_DAY else "good"


def generate_batch(day: int, index: int, scenario: str) -> pd.DataFrame:
    experiment_id = TRAIN_EXPERIMENT_IDS[
        (day * BATCHES_PER_DAY + index) % len(TRAIN_EXPERIMENT_IDS)
    ]
    df = pd.read_csv(DATASET_DIR / f"experiment_{experiment_id:02d}.csv")
    return PERTURBATIONS[scenario](df, progress_for(day))


def main() -> None:
    parser = argparse.ArgumentParser(description="가상 운영 타임라인 생성 및 주입")
    parser.add_argument("scenario", choices=list(PERTURBATIONS))
    parser.add_argument("--days", type=int, default=TOTAL_DAYS)
    args = parser.parse_args()

    from fastapi.testclient import TestClient
    from serving.app import app

    from drift_worker import WorkerState, tick  # 같은 폴더

    out_dir = ROOT / "data" / "timeline" / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    state = WorkerState()

    for day in range(1, args.days + 1):
        for index in range(BATCHES_PER_DAY):
            batch_id = f"day{day:02d}_{index}"
            batch = generate_batch(day, index, args.scenario)
            csv_path = out_dir / f"{batch_id}.csv"
            batch.to_csv(csv_path, index=False)

            with csv_path.open("rb") as fh:
                response = client.post("/predict", files={"file": (csv_path.name, fh, "text/csv")})
            response.raise_for_status()

            record_label(
                batch_id=batch_id,
                produced_day=day,
                arrived_day=day + LABEL_DELAY_DAYS,
                label=true_label(args.scenario, day),
                db_path=LABELS_DB,
            )

        result = tick(client, state, current_day=day, scenario=args.scenario)
        print(
            f"Day {day:02d}  score/threshold={result['ratio']:.2f}  "
            f"flagged={result['flagged']}  action={result['action']}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: perturbation 함수만 단독 확인**

Run:
```bash
cd /home/sure/project/hanium/02-cnc-machining && uv run python -c "
import sys; sys.path.insert(0, 'monitoring'); sys.path.insert(0, 'src')
import simulate_timeline as st
import pandas as pd
df = pd.read_csv('data/dataset/experiment_01.csv')
st.CUR_DRIFT = 0.05; st.POS_DRIFT = 0.1
out = st.apply_temperature(df, 1.0)
print('전류 배율:', (out['X_OutputCurrent'].mean() / df['X_OutputCurrent'].mean()).round(4))
print('SetPosition 불변:', out['X_SetPosition'].equals(df['X_SetPosition']))
print('progress Day1/Day40:', st.progress_for(1), st.progress_for(40))
"
```
Expected: 전류 배율 ≈ 1.05, `SetPosition 불변: True`, `progress Day1/Day40: 0.0 1.0`

- [ ] **Step 3: Commit**

```bash
git add monitoring/simulate_timeline.py
git commit -m "feat: add virtual operating timeline stream generator"
```

---

### Task 10: 변형 상수 스윕

`POS_DRIFT`, `CUR_DRIFT`, `WEAR_RATE`를 실측 대역에 맞춰 확정한다. Task 9의 자리값 `0.0`을 반드시 교체한다.

**Files:**
- Create: `monitoring/sweep_drift_constants.py`
- Modify: `monitoring/simulate_timeline.py` (상수 3개 확정값으로 교체 + 스윕 결과 주석)

- [ ] **Step 1: 스윕 스크립트 작성**

`monitoring/sweep_drift_constants.py`:

```python
"""Day 40 시점의 변형 폭이 실측 대역 안에 들어오게 상수를 정한다.

목표 score/threshold — temperature: 1.5~2.0, tool_wear: 3.0.
실측 근거는 synthetic/real_anomaly_reference.json (GOOD 0.43~1.30, BAD 1.00~3.79).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "monitoring"))

import simulate_timeline as st  # noqa: E402
from serving.app import load_model_state  # noqa: E402
from serving.inference import predict_experiment  # noqa: E402
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS  # noqa: E402

GRID = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]


def ratio_for(df: pd.DataFrame, state) -> float:
    result = predict_experiment(
        df, state.model, state.scaler_dict, state.thresholds, state.window_size,
        FEATURE_COLUMNS, state.feature_baseline, SETUP_CONSTANT_COLUMNS, method="mean",
    )
    return result["score"] / state.thresholds["mean"]


def main() -> None:
    state = load_model_state()
    base = pd.read_csv(st.DATASET_DIR / "experiment_01.csv")

    print("=== temperature (POS_DRIFT = CUR_DRIFT = v) ===")
    for v in GRID:
        st.POS_DRIFT, st.CUR_DRIFT = v, v
        print(f"  v={v:<5} ratio={ratio_for(st.apply_temperature(base, 1.0), state):.2f}")

    print("=== tool_wear (WEAR_RATE = v) ===")
    for v in GRID:
        st.WEAR_RATE = v
        print(f"  v={v:<5} ratio={ratio_for(st.apply_tool_wear(base, 1.0), state):.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스윕 실행**

Run: `cd /home/sure/project/hanium/02-cnc-machining && nice -n 19 uv run python monitoring/sweep_drift_constants.py`
Expected: 두 표가 출력된다. 공유 서버이므로 `nice -n 19`를 반드시 붙인다.

- [ ] **Step 3: 상수 확정**

`monitoring/simulate_timeline.py`의 세 상수를 목표 비율에 가장 가까운 값으로 교체한다:
- `POS_DRIFT`, `CUR_DRIFT` → temperature 표에서 ratio가 **1.5~2.0**에 가장 가까운 `v`
- `WEAR_RATE` → tool_wear 표에서 ratio가 **3.0**에 가장 가까운 `v`

교체와 함께 스윕 결과 표 전체를 주석으로 남긴다. 모델이 재학습되면 이 상수가 낡는데, `simulate_timeline.py`가 매일 실제 비율을 출력하므로 조용히 틀리지는 않는다.

**목표 대역에 닿는 값이 하나도 없으면 여기서 멈추고 보고한다.** 그리드를 넓힐지 목표 대역을 조정할지는 사용자와 논의할 사안이며, 임의로 대역을 넓혀 통과시키지 않는다.

- [ ] **Step 4: 확정값 확인**

Run:
```bash
cd /home/sure/project/hanium/02-cnc-machining && uv run python -c "
import sys; sys.path.insert(0, 'monitoring')
import simulate_timeline as st
print(st.POS_DRIFT, st.CUR_DRIFT, st.WEAR_RATE)
assert st.WEAR_RATE > 0 and st.CUR_DRIFT > 0, '자리값 0.0 이 남아 있다'
"
```
Expected: 세 값 모두 0이 아니다.

- [ ] **Step 5: Commit**

```bash
git add monitoring/sweep_drift_constants.py monitoring/simulate_timeline.py
git commit -m "feat: calibrate drift constants to measured anomaly bands"
```

---

### Task 11: 감시 워커

Task 2~8을 순서대로 엮는다. 판정 로직은 전부 `src/retraining/`에 있고 워커는 호출만 한다.

**Files:**
- Create: `monitoring/drift_worker.py`

**Interfaces:**
- Consumes: `trigger`, `gate`, `promotion`, `runner` 전체
- Produces: `WorkerState` 클래스, `tick(client, state, current_day, scenario) -> dict` — 반환 dict 키: `ratio`, `flagged`, `action`(`"none"` | `"promoted"` | `"rejected"`)

- [ ] **Step 1: 워커 작성**

`monitoring/drift_worker.py`:

```python
"""드리프트 감시 워커 — 폴링, 트리거, 재학습, 게이트, 승격을 엮는다.

서빙과 별도 루프로 도는 구조라 학습이 추론 응답을 지연시키지 않는다.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from lstm_ae.tracking import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, promote_to_champion  # noqa: E402
from retraining.gate import evaluate_gate  # noqa: E402
from retraining.promotion import swap_with_rollback, verify_serving_contract  # noqa: E402
from retraining.runner import run_retraining  # noqa: E402
from retraining.trigger import is_drift_flagged, should_retrain  # noqa: E402

LABELS_DB = ROOT / "data" / "monitoring" / "labels.db"
MODEL_DIR = ROOT / "data" / "model"
SCALER_PATH = ROOT / "data" / "processed" / "scaler.json"
BACKUP_ROOT = ROOT / "data" / "model_backup"
COOLDOWN_DAYS = 5


@dataclass
class WorkerState:
    flag_history: list[bool] = field(default_factory=list)
    cooldown_remaining: int = 0
    champion_recall: float = 10 / 11        # 현 champion 실측
    champion_accuracy: float = 0.0          # 첫 게이트 평가 시 측정값으로 대체


def tick(client, state: WorkerState, current_day: int, scenario: str) -> dict:
    status = client.get("/drift-status").json()
    flagged = is_drift_flagged(status)
    state.flag_history.append(flagged)
    ratio = status.get("output_drift", {}).get("ratio_to_threshold", 0.0)

    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1

    if not should_retrain(state.flag_history, 3, state.cooldown_remaining):
        return {"ratio": ratio, "flagged": flagged, "action": "none"}

    print(f"  [Day {current_day}] 트리거 발동 — 재학습 시작")
    result = run_retraining(
        timeline_dir=ROOT / "data" / "timeline" / scenario,
        labels_db=LABELS_DB,
        current_day=current_day,
        root=ROOT,
    )
    state.cooldown_remaining = COOLDOWN_DAYS

    decision = _decide_and_promote(client, state, result, current_day, scenario)
    return {"ratio": ratio, "flagged": flagged, "action": decision}


def _decide_and_promote(client, state, result, current_day, scenario) -> str:
    mlflow_client = MlflowClient()
    run = mlflow_client.get_run(result["run_id"])

    # 계약 확인을 파일 교체보다 앞에 둔다 — 위반이면 롤백할 것 자체가 생기지 않는다.
    missing = verify_serving_contract(run.data.metrics, run.data.params)
    if missing:
        _tag_rejection(mlflow_client, result["run_id"], f"서빙 계약 미충족: {missing}", scenario, current_day)
        print(f"  거부 — 서빙 계약 미충족: {missing}")
        return "rejected"

    verdict = evaluate_gate(
        retrained_recall=result["recall"],
        champion_recall=state.champion_recall,
        retrained_accuracy=_accuracy_on_labeled(result, current_day),
        champion_accuracy=state.champion_accuracy,
    )
    for key, value in verdict.items():
        mlflow_client.set_tag(result["run_id"], f"gate_{key}", value)
    mlflow_client.set_tag(result["run_id"], "scenario", scenario)
    mlflow_client.set_tag(result["run_id"], "trigger_day", current_day)

    if verdict["decision"] == "rejected":
        print(f"  거부 — {verdict['reject_reason']}  (champion 유지, 사람 확인 필요)")
        return "rejected"

    previous_version = mlflow_client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS
    ).version

    def _promote() -> None:
        promote_to_champion(result["model_version"])
        client.post("/reload-model").raise_for_status()

    def _verify() -> None:
        health = client.get("/health").json()
        if health["model_version"] != str(result["model_version"]):
            raise RuntimeError(f"리로드 후 버전 불일치: {health['model_version']}")

    try:
        swap_with_rollback(
            result["retrain_dir"], MODEL_DIR, SCALER_PATH, BACKUP_ROOT,
            promote=_promote, verify=_verify,
        )
    except Exception as exc:
        # 파일은 swap_with_rollback 이 되돌렸다. alias 와 서빙 상태만 마저 되돌린다.
        print(f"  승격 실패, 롤백 중: {exc}")
        promote_to_champion(previous_version)
        client.post("/reload-model")
        raise
    print(f"  승격 완료 — version {result['model_version']}")
    return "promoted"


def _accuracy_on_labeled(result: dict, current_day: int) -> float:
    """라벨 도착 구간의 정확도. G2 의 입력이다."""
    from monitoring.labels import get_arrived_labels
    from monitoring.logging import get_recent_requests

    arrived = {r["batch_id"]: r["label"] for r in get_arrived_labels(current_day, LABELS_DB)}
    if not arrived:
        return 0.0
    recent = get_recent_requests(n=len(arrived), db_path=ROOT / "data" / "monitoring" / "requests.db")
    if not recent:
        return 0.0
    truths = list(arrived.values())[-len(recent):]
    hits = sum(1 for truth, req in zip(truths, reversed(recent))
               if truth == req["predicted_label_text"])
    return hits / len(recent)


def _tag_rejection(mlflow_client, run_id, reason, scenario, day) -> None:
    mlflow_client.set_tag(run_id, "gate_decision", "rejected")
    mlflow_client.set_tag(run_id, "gate_reject_reason", reason)
    mlflow_client.set_tag(run_id, "scenario", scenario)
    mlflow_client.set_tag(run_id, "trigger_day", day)
```

- [ ] **Step 2: import 확인**

Run:
```bash
cd /home/sure/project/hanium/02-cnc-machining && uv run python -c "
import sys; sys.path.insert(0, 'monitoring')
import drift_worker
print('ok', drift_worker.COOLDOWN_DAYS)
"
```
Expected: `ok 5`

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest`
Expected: 135 passed (워커는 pytest 대상이 아니지만 회귀 확인)

- [ ] **Step 4: Commit**

```bash
git add monitoring/drift_worker.py
git commit -m "feat: add drift monitoring worker closing the retraining loop"
```

---

### Task 12: 시나리오 A 전체 실행 — 승격 경로 검증

**Files:**
- 없음 (실행과 확인만)

- [ ] **Step 1: 초기 상태 정리**

Run:
```bash
cd /home/sure/project/hanium/02-cnc-machining
rm -f data/monitoring/requests.db data/monitoring/labels.db
rm -rf data/timeline/temperature
uv run python -c "
from serving.app import load_model_state
print('champion:', load_model_state().model_version)
"
```
Expected: 현재 champion 버전이 출력된다. 이 값을 기록해 둔다.

- [ ] **Step 2: 시나리오 A 실행**

Run: `nice -n 19 uv run python monitoring/simulate_timeline.py temperature 2>&1 | tee /tmp/claude-1000/-home-sure-project-hanium/41947d8d-11fc-488f-b556-b21d412221ed/scratchpad/timeline_temperature.log`
Expected: Day 1~40의 `score/threshold`가 단조 증가하고, 어느 시점에 `flagged=True`가 3일 연속 나온 뒤 `action=promoted`가 찍힌다.

- [ ] **Step 3: 결과 확인**

다음을 로그에서 확인해 기록한다:
- 트리거가 발동한 날
- 게이트 G1 recall과 G2 정확도 차
- 승격 후 `score/threshold`가 실제로 떨어졌는지 (오탐 소멸)

Run:
```bash
cd /home/sure/project/hanium/02-cnc-machining && uv run python -c "
from mlflow.tracking import MlflowClient
from lstm_ae.tracking import configure_tracking
configure_tracking()
for run in MlflowClient().search_runs(['1'], order_by=['start_time DESC'], max_results=3):
    print(run.info.run_id[:8], {k: v for k, v in run.data.tags.items() if k.startswith('gate') or k == 'scenario'})
"
```
Expected: 최신 run에 `gate_decision=promoted`, `scenario=temperature` 태그가 있다.

- [ ] **Step 4: 기대와 다르면 보고하고 멈춘다**

승격이 일어나지 않거나 오탐이 줄지 않으면 **값을 조정해 통과시키지 않는다.** 관측된 수치를 그대로 사용자에게 보고하고 게이트 기준 재조정 여부를 논의한다.

- [ ] **Step 5: `tasks/todo.md` 리뷰 절에 결과 기록**

`tasks/todo.md`의 "리뷰 — 시나리오 A" 절에 다음을 채운다: 트리거 발동일, G1 recall,
G2 정확도 차, 승격된 모델 버전, 승격 전후 `score/threshold`. 계획과 달랐던 점이
있으면 함께 적는다.

- [ ] **Step 6: Commit**

```bash
git add tasks/todo.md
git commit -m "docs: record scenario A promotion run results"
```

---

### Task 13: 시나리오 B 전체 실행 — 거부 경로 검증 + 결함 ② 수정 확인

**Files:**
- 없음 (실행과 확인만)

- [ ] **Step 1: champion을 시나리오 A 이전 상태로 되돌린다**

Task 12 Step 1에서 기록한 원래 버전으로 승격을 되돌린다:

```bash
cd /home/sure/project/hanium/02-cnc-machining
uv run python scripts/promote_model.py <Task12에서_기록한_원래_버전>
rm -f data/monitoring/requests.db data/monitoring/labels.db
rm -rf data/timeline/tool_wear
```

- [ ] **Step 2: 정본 파일 해시를 기록한다**

Run:
```bash
cd /home/sure/project/hanium/02-cnc-machining && md5sum data/processed/scaler.json data/model/model.pt data/model/feature_baseline.json
```
이 값을 기록해 둔다. Step 4에서 대조한다.

- [ ] **Step 3: 시나리오 B 실행**

Run: `nice -n 19 uv run python monitoring/simulate_timeline.py tool_wear 2>&1 | tee /tmp/claude-1000/-home-sure-project-hanium/41947d8d-11fc-488f-b556-b21d412221ed/scratchpad/timeline_tool_wear.log`
Expected: 트리거가 발동한 뒤 `action=rejected`가 찍히고, 거부 사유에 `G1`이 포함된다.

- [ ] **Step 4: 거부 경로가 정본을 건드리지 않았는지 확인 (결함 ② 수정 검증)**

Run:
```bash
cd /home/sure/project/hanium/02-cnc-machining && md5sum data/processed/scaler.json data/model/model.pt data/model/feature_baseline.json
uv run python -c "
from serving.app import load_model_state
s = load_model_state()
print('champion 정상 로드:', s.model_version, len(s.scaler_dict), '피처')
"
```
Expected: 해시 3개가 Step 2와 **완전히 동일**하고, champion이 정상 로드된다. 이것이 결함 ② 수정의 핵심 검증이다 — 거부된 재학습이 champion의 동반 파일을 덮어쓰지 않았다.

- [ ] **Step 5: 거부된 run이 MLflow에 남아 있는지 확인**

Run:
```bash
cd /home/sure/project/hanium/02-cnc-machining && uv run python -c "
from mlflow.tracking import MlflowClient
from lstm_ae.tracking import configure_tracking
configure_tracking()
for run in MlflowClient().search_runs(['1'], order_by=['start_time DESC'], max_results=3):
    tags = run.data.tags
    if tags.get('gate_decision') == 'rejected':
        print('거부 run:', run.info.run_id[:8], tags.get('gate_reject_reason'))
"
```
Expected: 거부 사유가 남아 있다. 이것이 데모의 핵심 증거다.

- [ ] **Step 6: 시나리오 B가 게이트를 통과해 버린 경우**

그렇다면 Day 11~20 구간의 마모가 약해 recall이 충분히 안 떨어진 것이다. `WEAR_RATE`나 `WEAR_LABEL_FLIP_DAY`를 조정해야 하는데, 이는 **"데모가 성립하도록 값을 맞추는" 행위**이므로:
1. 조정 전 관측값(재학습 모델의 recall, G1 허용선)을 먼저 기록한다
2. 조정 사실과 근거를 스펙의 "남은 리스크" 절에 추가한다
3. 사용자에게 보고한 뒤 진행한다

- [ ] **Step 7: 전체 테스트 최종 확인**

Run: `uv run pytest`
Expected: 135 passed

- [ ] **Step 8: `tasks/todo.md` 리뷰 절에 결과 기록**

"리뷰 — 시나리오 B" 절에 채운다: 트리거 발동일, 재학습 모델의 recall과 G1 허용선,
거부 사유, Step 2/4의 해시 대조 결과. 그리고 `docs/STRUCTURE.md`의 디렉터리 트리에
`src/retraining/`과 신규 스크립트 2개를 추가한다.

- [ ] **Step 9: Commit**

```bash
git add tasks/todo.md docs/STRUCTURE.md
git commit -m "docs: record scenario B rejection run and update structure doc"
```

---

## 완료 기준

- [ ] `uv run pytest` 135개 통과
- [ ] 시나리오 A가 **승격**으로, 시나리오 B가 **거부**로 끝난다
- [ ] 거부 경로 실행 후 `data/processed/scaler.json`과 `data/model/`의 해시가 불변
- [ ] MLflow에 두 시나리오의 run이 `gate_decision` 태그와 함께 남아 있다
- [ ] `docs/STRUCTURE.md`에 `src/retraining/`과 신규 스크립트 2개가 반영된다
