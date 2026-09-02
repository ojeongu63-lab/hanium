# 두 방향 승격 게이트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** G2와 섀도우 판정을 정확도 하나가 아니라 오탐·놓침 건수의 클래스별 비교로 바꿔, 정답이 한 클래스뿐인 창에서 "더 자주 불량이라 하는" 후보가 승격되는 사각지대를 닫는다.

**Architecture:** 판정은 `src/retraining/gate.py`의 순수 함수 `evaluate_two_sided`가 전담하고, `monitoring/drift_worker.py`는 (정답, champion 판정, 후보 판정) 세 리스트를 모아 넘기기만 한다. 정상 라벨이 없는 창은 거부로 처리하고 기존 거부 경로(원인 추정 + RAG 조치 + MLflow 태그)를 그대로 탄다. 재학습 파이프라인·서빙·트리거는 건드리지 않는다.

**Tech Stack:** Python 3.14, uv, pytest, MLflow(태그), FastAPI(TestClient·httpx2). PyTorch CPU 빌드.

**Spec:** `docs/specs/2026-09-02-cnc-two-sided-gate-design.md` — Part A·B가 구현 범위, Part C(홀드아웃 보정)는 보류.

## Global Constraints

- 작업 디렉터리는 `02-cnc-machining/`. 테스트는 `uv run pytest -q`(현재 165개 통과가 기준선).
- 무거운 실행(재학습·라이브 재현)은 `who`/`top` 확인 후 `nice -n 19`.
- 라벨 문자열은 `"good"`/`"bad"`, 판정 문자열은 `"promoted"`/`"rejected"` 그대로.
- G1(원본 eval 놓친 개수, 허용 champion+1), `GATE_SAMPLE_SIZE = 20`, `COOLDOWN_DAYS = 5`, `CONSECUTIVE_K = 3`은 변경 금지.
- 변경 금지 파일: `scripts/run_lstm_training.py`, `src/serving/*`, `src/retraining/{promotion,trigger,runner}.py`, `src/lstm_ae/pipeline.py`, `monitoring/simulate_timeline.py`.
- 커밋은 main에 직접, 메시지 끝에 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`와 `Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz`.
- 스크립트는 `Path(__file__).parent.parent`로 루트를 역산한다 — 파일 깊이를 바꾸지 않는다.

---

## 파일 구조

| 파일 | 책임 | 이번 변경 |
|---|---|---|
| `src/retraining/gate.py` | 승격 판정 순수 함수 | `evaluate_two_sided` 신규, `evaluate_gate` 시그니처 변경, `evaluate_shadow`·`accuracy_from_pairs` 제거 |
| `tests/retraining/test_gate.py` | 위 함수 단위 테스트 | 두 방향 규칙 8개 추가, 정확도·섀도우 테스트 5개 제거, `evaluate_gate` 테스트 7개 개정 |
| `monitoring/drift_worker.py` | 감시 루프 오케스트레이션 | `_gate_accuracies` → `_gate_predictions`, 게이트·섀도우 판정을 `evaluate_two_sided`로, 로그·태그 헬퍼 2개 |
| `docs/STRUCTURE.md`, `README.md` | 독자용 설명 | G2 설명 문구 |
| `docs/specs/2026-09-02-cnc-two-sided-gate-design.md` | 설계 | "실행 결과에 따른 정정" 절(Task 4) |
| `../tasks/todo.md` | 작업 기록 | 체크리스트 + 리뷰 절(Task 4) |

---

### Task 1: `evaluate_two_sided` — 두 방향 판정 순수 함수

**Files:**
- Modify: `src/retraining/gate.py` (파일 끝에 추가)
- Test: `tests/retraining/test_gate.py` (파일 끝에 추가)

**Interfaces:**
- Consumes: 없음
- Produces: `evaluate_two_sided(truths: list[str], champion_preds: list[str], candidate_preds: list[str]) -> dict` — 키 `n_good`, `n_bad`, `champion_false_alarms`, `candidate_false_alarms`, `champion_misses`, `candidate_misses`, `decision`("promoted"|"rejected"), `reject_reason`(str, promoted면 ""). Task 2·3이 이 딕셔너리를 그대로 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/retraining/test_gate.py` 맨 위 import에 `evaluate_two_sided`를 추가하고 파일 끝에 붙인다:

```python
from retraining.gate import accuracy_from_pairs, evaluate_gate, evaluate_shadow, evaluate_two_sided
```

```python
# ---- evaluate_two_sided: 오탐·놓침을 따로 세는 두 방향 규칙 -------------------

def test_all_good_window_promotes_when_false_alarms_drop():
    truths = ["good"] * 4
    g2 = evaluate_two_sided(
        truths,
        ["bad", "bad", "good", "good"],   # champion 오탐 2
        ["bad", "good", "good", "good"],  # 후보 오탐 1
    )
    assert g2["n_good"] == 4 and g2["n_bad"] == 0
    assert g2["champion_false_alarms"] == 2 and g2["candidate_false_alarms"] == 1
    assert g2["decision"] == "promoted"
    assert g2["reject_reason"] == ""


def test_all_good_window_rejects_when_false_alarms_equal():
    truths = ["good"] * 3
    g2 = evaluate_two_sided(truths, ["bad", "good", "good"], ["good", "bad", "good"])
    assert g2["decision"] == "rejected"
    assert "개선 없음" in g2["reject_reason"]


def test_all_bad_window_rejects_as_unmeasurable():
    # 09-02 fixture_loosening Day 34 — 후보가 놓침을 줄여도 오탐을 잴 수 없으면 거부.
    truths = ["bad"] * 3
    g2 = evaluate_two_sided(truths, ["good", "good", "good"], ["bad", "bad", "bad"])
    assert g2["n_good"] == 0
    assert g2["decision"] == "rejected"
    assert "정상 라벨 없음" in g2["reject_reason"]


def test_mixed_window_rejects_trading_misses_for_false_alarms():
    # 09-02 fixture Day 29 형태: 후보가 놓침은 줄이고 오탐은 늘림.
    truths = ["good", "good", "bad", "bad"]
    champion = ["good", "good", "good", "bad"]   # 오탐 0, 놓침 1
    candidate = ["bad", "good", "bad", "bad"]    # 오탐 1, 놓침 0
    g2 = evaluate_two_sided(truths, champion, candidate)
    assert g2["decision"] == "rejected"
    assert "오탐 회귀" in g2["reject_reason"]


def test_mixed_window_rejects_trading_false_alarms_for_misses():
    # 09-02 tool_wear Day 30 형태: 후보가 오탐은 줄이고 놓침은 늘림.
    truths = ["good", "good", "bad", "bad"]
    champion = ["bad", "good", "bad", "bad"]     # 오탐 1, 놓침 0
    candidate = ["good", "good", "good", "bad"]  # 오탐 0, 놓침 1
    g2 = evaluate_two_sided(truths, champion, candidate)
    assert g2["decision"] == "rejected"
    assert "놓침 회귀" in g2["reject_reason"]


def test_mixed_window_promotes_when_no_regression_and_one_side_improves():
    truths = ["good", "good", "bad", "bad"]
    champion = ["bad", "good", "good", "bad"]    # 오탐 1, 놓침 1
    candidate = ["good", "good", "good", "bad"]  # 오탐 0, 놓침 1
    g2 = evaluate_two_sided(truths, champion, candidate)
    assert g2["decision"] == "promoted"
    assert g2["reject_reason"] == ""


def test_empty_window_rejects_as_unmeasurable():
    g2 = evaluate_two_sided([], [], [])
    assert g2["decision"] == "rejected"
    assert "정상 라벨 없음" in g2["reject_reason"]


def test_two_sided_rejects_length_mismatch():
    with pytest.raises(ValueError, match="길이가 다릅니다"):
        evaluate_two_sided(["good", "bad"], ["good"], ["good", "bad"])
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `uv run pytest tests/retraining/test_gate.py -q`
Expected: ImportError — `cannot import name 'evaluate_two_sided'`.

- [ ] **Step 3: 최소 구현**

`src/retraining/gate.py` 파일 끝에 추가:

```python
def evaluate_two_sided(
    truths: list[str], champion_preds: list[str], candidate_preds: list[str]
) -> dict:
    """라벨 창을 정상/불량으로 나눠 오탐(정상→bad)과 놓침(불량→good)을 두 모델
    각각 센 뒤 승격 여부를 낸다.

    정확도 하나로 비교하면 창에 한 클래스만 있을 때 "더 자주 불량이라 하는"
    모델이 무조건 이긴다(09-02 fixture_loosening Day 34에서 실제로 통과됨).
    그래서 두 건수를 따로 보고, 한쪽을 다른 쪽과 맞바꾸는 후보와 오탐을 아예
    잴 수 없는 창(정상 라벨 0건)은 통과시키지 않는다. 정상 라벨만 있으면
    놓침 쪽은 G1이 맡는다 — temperature 시나리오의 기존 경로.

    빈 창은 정상 라벨 0건이므로 거부. 세 리스트 길이가 다르면 ValueError.
    """
    if not (len(truths) == len(champion_preds) == len(candidate_preds)):
        raise ValueError(
            f"라벨 {len(truths)}개, champion 판정 {len(champion_preds)}개, "
            f"후보 판정 {len(candidate_preds)}개의 길이가 다릅니다"
        )
    good = [i for i, t in enumerate(truths) if t == "good"]
    bad = [i for i, t in enumerate(truths) if t == "bad"]

    def false_alarms(preds: list[str]) -> int:
        return sum(1 for i in good if preds[i] == "bad")

    def misses(preds: list[str]) -> int:
        return sum(1 for i in bad if preds[i] == "good")

    counts = {
        "n_good": len(good),
        "n_bad": len(bad),
        "champion_false_alarms": false_alarms(champion_preds),
        "candidate_false_alarms": false_alarms(candidate_preds),
        "champion_misses": misses(champion_preds),
        "candidate_misses": misses(candidate_preds),
    }
    if counts["n_good"] == 0:
        return {
            **counts,
            "decision": "rejected",
            "reject_reason": "G2 판정 불가: 창에 정상 라벨 없음(오탐 회귀 확인 불가)",
        }

    fa_ok = counts["candidate_false_alarms"] <= counts["champion_false_alarms"]
    miss_ok = counts["n_bad"] == 0 or counts["candidate_misses"] <= counts["champion_misses"]
    improved = counts["candidate_false_alarms"] < counts["champion_false_alarms"] or (
        counts["n_bad"] > 0 and counts["candidate_misses"] < counts["champion_misses"]
    )

    reasons = []
    if not fa_ok:
        reasons.append(
            f"G2 오탐 회귀: 후보 {counts['candidate_false_alarms']}건 > "
            f"champion {counts['champion_false_alarms']}건 (정상 {counts['n_good']}건 중)"
        )
    if not miss_ok:
        reasons.append(
            f"G2 놓침 회귀: 후보 {counts['candidate_misses']}건 > "
            f"champion {counts['champion_misses']}건 (불량 {counts['n_bad']}건 중)"
        )
    if fa_ok and miss_ok and not improved:
        reasons.append("G2 개선 없음: 오탐·놓침 모두 champion과 동일")

    promoted = fa_ok and miss_ok and improved
    return {
        **counts,
        "decision": "promoted" if promoted else "rejected",
        "reject_reason": "; ".join(reasons),
    }
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `uv run pytest tests/retraining/test_gate.py -q`
Expected: 20 passed (기존 12 + 신규 8).

- [ ] **Step 5: 커밋**

```bash
git add src/retraining/gate.py tests/retraining/test_gate.py
git commit -m "feat: add evaluate_two_sided gate rule counting false alarms and misses separately"
```

---

### Task 2: `evaluate_gate`가 두 방향 결과를 받도록, 정확도 함수 제거

**Files:**
- Modify: `src/retraining/gate.py:4-64` (`evaluate_gate` 본문, `accuracy_from_pairs`·`evaluate_shadow` 삭제)
- Test: `tests/retraining/test_gate.py:1-124` (기존 테스트 개정)

**Interfaces:**
- Consumes: Task 1의 `evaluate_two_sided` 반환 딕셔너리.
- Produces: `evaluate_gate(retrained_missed: int, champion_missed: int, g2: dict, extra_misses_allowed: int = EXTRA_MISSES_ALLOWED) -> dict` — 키 `decision`, `g1_pass`, `g2_pass`, `g1_missed`, `g2`(입력 딕셔너리 그대로), `reject_reason`. `accuracy_from_pairs`·`evaluate_shadow`는 더 이상 존재하지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/retraining/test_gate.py`의 1~124행(import부터 `test_shadow_rejects_when_candidate_not_better`까지)을 아래로 교체한다. Task 1에서 추가한 두 방향 테스트 8개는 그 아래에 그대로 둔다.

```python
import pytest

from retraining.gate import evaluate_gate, evaluate_two_sided

# 현 champion 은 eval 불량 11개 중 10개를 잡는다 → 1건 놓침.
CHAMPION_MISSED = 1

# 정상 20건 창에서 후보가 오탐을 5 → 2로 줄인 두 방향 결과 (통과).
PROMOTED_G2 = {
    "n_good": 20, "n_bad": 0,
    "champion_false_alarms": 5, "candidate_false_alarms": 2,
    "champion_misses": 0, "candidate_misses": 0,
    "decision": "promoted", "reject_reason": "",
}
# 같은 창에서 오탐이 그대로인 결과 (거부).
REJECTED_G2 = {
    **PROMOTED_G2,
    "candidate_false_alarms": 5,
    "decision": "rejected",
    "reject_reason": "G2 개선 없음: 오탐·놓침 모두 champion과 동일",
}


def test_promoted_when_both_conditions_pass():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED, champion_missed=CHAMPION_MISSED, g2=PROMOTED_G2
    )

    assert result["decision"] == "promoted"
    assert result["g1_pass"] is True
    assert result["g2_pass"] is True
    assert result["reject_reason"] == ""


def test_g1_boundary_one_extra_miss_passes():
    # champion 1건 놓침 + 허용 1건 = 2건까지 통과
    result = evaluate_gate(retrained_missed=2, champion_missed=CHAMPION_MISSED, g2=PROMOTED_G2)

    assert result["g1_pass"] is True
    assert result["decision"] == "promoted"


def test_g1_boundary_two_extra_misses_rejects():
    result = evaluate_gate(retrained_missed=3, champion_missed=CHAMPION_MISSED, g2=PROMOTED_G2)

    assert result["g1_pass"] is False
    assert result["decision"] == "rejected"
    assert "G1" in result["reject_reason"]


def test_g1_passes_when_model_catches_everything():
    # 모든 것을 불량이라 판정하는 모델은 놓친 개수 0 이라 G1 을 통과한다.
    # 이것이 G2 가 반드시 필요한 이유다 — 실제 실행에서 벌어진 상황이기도 하다.
    g2 = evaluate_two_sided(["good"] * 4, ["good"] * 4, ["bad"] * 4)  # 후보 오탐 4 vs 0
    result = evaluate_gate(retrained_missed=0, champion_missed=CHAMPION_MISSED, g2=g2)

    assert result["g1_pass"] is True
    assert result["g2_pass"] is False
    assert result["decision"] == "rejected"


def test_g2_rejects_when_no_improvement():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED, champion_missed=CHAMPION_MISSED, g2=REJECTED_G2
    )

    assert result["g2_pass"] is False
    assert result["decision"] == "rejected"
    assert "G2" in result["reject_reason"]


def test_reject_reason_lists_both_violations():
    result = evaluate_gate(retrained_missed=9, champion_missed=CHAMPION_MISSED, g2=REJECTED_G2)

    assert "G1" in result["reject_reason"]
    assert "G2" in result["reject_reason"]


def test_g2_result_is_passed_through():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED, champion_missed=CHAMPION_MISSED, g2=PROMOTED_G2
    )

    assert result["g2"] is PROMOTED_G2
    assert "g2_accuracy_delta" not in result


def test_removed_accuracy_helpers_are_gone():
    import retraining.gate as gate

    assert not hasattr(gate, "accuracy_from_pairs")
    assert not hasattr(gate, "evaluate_shadow")
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `uv run pytest tests/retraining/test_gate.py -q`
Expected: `evaluate_gate` 테스트가 `TypeError: unexpected keyword argument 'g2'`로 실패, `test_removed_accuracy_helpers_are_gone` 실패.

- [ ] **Step 3: 최소 구현**

`src/retraining/gate.py`의 `evaluate_gate`를 아래로 교체하고, `accuracy_from_pairs`와 `evaluate_shadow` 함수를 삭제한다(`evaluate_two_sided`는 그대로 둔다):

```python
EXTRA_MISSES_ALLOWED = 1


def evaluate_gate(
    retrained_missed: int,
    champion_missed: int,
    g2: dict,
    extra_misses_allowed: int = EXTRA_MISSES_ALLOWED,
) -> dict:
    """승격 여부를 두 조건의 AND로 판정한다.

    G1 (안전): 원본 실측 eval셋에서 불량 검출력이 champion 대비 유지되는가.
      **놓친 개수(fn)로 비교한다.** eval 불량이 11개뿐이라 recall 소수값은
      실험 1개당 0.0909씩 뚝뚝 끊긴다 — 소수점 임계값은 그 눈금을 가리는
      가짜 정밀도라, 기준을 개수로 직접 표현한다.
      precision을 보지 않는 이유는, 센서 좌표계가 이동한 환경에서 새 모델을
      옛 좌표계 eval에 적용하면 precision이 좌표계 차이 때문에 떨어지기 때문이다.
    G2 (근거): 라벨이 도착한 최근 구간에서 실제로 나아졌는가.
      `evaluate_two_sided()`의 결과를 받는다 — 오탐과 놓침을 따로 세어 어느
      쪽도 후퇴하지 않고 한쪽은 나아져야 하며, 정상 라벨이 없는 창은 통과
      시키지 않는다. G1만으로는 모든 것을 불량이라 판정하는 모델도 놓친
      개수 0으로 통과한다. 실제로 두 시나리오 모두에서 판정을 내린 것은 G2였다.
    """
    g1_pass = retrained_missed <= champion_missed + extra_misses_allowed
    g2_pass = g2["decision"] == "promoted"

    reasons = []
    if not g1_pass:
        reasons.append(
            f"G1 불량 검출 회귀: {retrained_missed}건 놓침 > "
            f"허용 {champion_missed + extra_misses_allowed}건"
        )
    if not g2_pass:
        reasons.append(g2["reject_reason"])

    return {
        "decision": "promoted" if (g1_pass and g2_pass) else "rejected",
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "g1_missed": retrained_missed,
        "g2": g2,
        "reject_reason": "; ".join(reasons),
    }
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `uv run pytest tests/retraining/test_gate.py -q`
Expected: 16 passed (개정 8 + 두 방향 8). 이 시점에 `uv run pytest -q` 전체를 돌리면 `monitoring/drift_worker.py`는 테스트에서 import되지 않으므로 여전히 통과한다(워커는 Task 3에서 고친다).

- [ ] **Step 5: 커밋**

```bash
git add src/retraining/gate.py tests/retraining/test_gate.py
git commit -m "refactor: evaluate_gate takes two-sided G2 result, drop accuracy helpers"
```

---

### Task 3: 워커 통합 — 판정을 `evaluate_two_sided`로, 로그·태그 교체, 문서 갱신

**Files:**
- Modify: `monitoring/drift_worker.py:24` (import), `:109-163` (`_decide_and_start_shadow`), `:211-270` (`_gate_accuracies`), `:272-330` (`_check_shadow`)
- Modify: `docs/STRUCTURE.md` (gate.py 설명 3줄), `README.md` §2-7 (G2 설명 문장)
- Test: 단위 테스트 없음(워커는 테스트에서 import되지 않음). 임포트 확인 + 전체 테스트 + Task 4 라이브 재현으로 검증.

**Interfaces:**
- Consumes: Task 1 `evaluate_two_sided`, Task 2 `evaluate_gate(retrained_missed, champion_missed, g2)`.
- Produces: `_gate_predictions(result, current_day, scenario) -> tuple[list[str], list[str], list[str], list[list[dict]]]` (truths, champion_preds, candidate_preds, champion_contributions), `_describe_g2(g2) -> str`, `_g2_tags(prefix, g2) -> dict`. MLflow 태그 `gate_g2_n_good`, `gate_g2_n_bad`, `gate_g2_fa_delta`, `gate_g2_miss_delta`, `shadow_n_good`, `shadow_n_bad`, `shadow_fa_delta`, `shadow_miss_delta`.

- [ ] **Step 1: import 교체**

24행을:

```python
from retraining.gate import evaluate_gate, evaluate_two_sided  # noqa: E402
```

- [ ] **Step 2: 헬퍼 2개 추가**

`_predict_labels` 정의 바로 앞(193행 부근)에 추가:

```python
def _describe_g2(g2: dict) -> str:
    """콘솔 로그용 한 줄. 정상/불량이 0건인 쪽은 건수만 찍는다."""
    good = f"정상 {g2['n_good']}건"
    if g2["n_good"]:
        good += f" — 오탐 후보 {g2['candidate_false_alarms']} vs champion {g2['champion_false_alarms']}"
    bad = f"불량 {g2['n_bad']}건"
    if g2["n_bad"]:
        bad += f" — 놓침 후보 {g2['candidate_misses']} vs champion {g2['champion_misses']}"
    return f"{good} · {bad}"


def _g2_tags(prefix: str, g2: dict) -> dict:
    """MLflow 태그. delta 는 후보 − champion (음수가 개선)."""
    return {
        f"{prefix}_n_good": g2["n_good"],
        f"{prefix}_n_bad": g2["n_bad"],
        f"{prefix}_fa_delta": g2["candidate_false_alarms"] - g2["champion_false_alarms"],
        f"{prefix}_miss_delta": g2["candidate_misses"] - g2["champion_misses"],
    }
```

- [ ] **Step 3: `_gate_accuracies` → `_gate_predictions`**

211~270행의 함수를 통째로 아래로 교체한다(본문 대부분 동일, 정확도 계산만 사라지고 리스트를 돌려준다):

```python
def _gate_predictions(
    result: dict, current_day: int, scenario: str
) -> tuple[list[str], list[str], list[str], list[list[dict]]]:
    """G2 입력 — 라벨 도착 구간의 (정답, champion 판정, 재학습 모델 판정)과
    champion의 배치별 feature_contributions(원인 추정용). 판정 자체는
    retraining.gate 가 한다.

    두 모델을 같은 배치에 대고 직접 돌려 비교한다. champion 판정을 predict_log
    에서 꺼내오지 않는 이유는 배치 식별자가 로그에 없어 짝을 맞출 수 없기
    때문이고, /predict 를 다시 부르지 않는 이유는 _predict_labels 주석과 같다.
    """
    import json

    import torch

    from lstm_ae.model import LSTMAutoencoder
    from monitoring.labels import get_arrived_labels
    from preprocessing.columns import FEATURE_COLUMNS
    from retraining.runner import TRAINING_CONFIG
    from serving.app import load_model_state

    arrived = get_arrived_labels(current_day, LABELS_DB)[-GATE_SAMPLE_SIZE:]
    if not arrived:
        return [], [], [], []

    timeline_dir = ROOT / "data" / "timeline" / scenario
    batch_paths = [timeline_dir / f"{r['batch_id']}.csv" for r in arrived]
    truths = [r["label"] for r in arrived]

    champion = load_model_state()
    champion_results = _predict_labels(
        batch_paths, champion.model, champion.scaler_dict,
        champion.thresholds["mean"], champion.feature_baseline, champion.window_size,
    )
    champion_preds = [r["predicted_label_text"] for r in champion_results]
    champion_contributions = [r["feature_contributions"] for r in champion_results]

    retrain_dir = Path(result["retrain_dir"])
    model = LSTMAutoencoder(
        num_features=len(FEATURE_COLUMNS),
        hidden_size=TRAINING_CONFIG["hidden_size"],
        latent_dim=TRAINING_CONFIG["latent_dim"],
    )
    model.load_state_dict(torch.load(retrain_dir / "model.pt"))
    model.eval()
    retrained_results = _predict_labels(
        batch_paths,
        model,
        json.loads((retrain_dir / "scaler.json").read_text()),
        result["thresholds"]["mean"],
        json.loads((retrain_dir / "feature_baseline.json").read_text()),
        TRAINING_CONFIG["window_size"],
    )
    retrained_preds = [r["predicted_label_text"] for r in retrained_results]

    return truths, champion_preds, retrained_preds, champion_contributions
```

- [ ] **Step 4: `_decide_and_start_shadow`의 게이트 구간 교체**

121~141행(`champion_accuracy, retrained_accuracy, ... = _gate_accuracies(` 부터 게이트 `print`까지)을 아래로 교체한다. 그 아래 `if verdict["decision"] == "rejected":` 분기와 섀도우 시작은 그대로 둔다.

```python
    truths, champion_preds, retrained_preds, champion_contributions = _gate_predictions(
        result, current_day, scenario
    )
    g2 = evaluate_two_sided(truths, champion_preds, retrained_preds)
    verdict = evaluate_gate(
        retrained_missed=result["missed"],
        champion_missed=state.champion_missed,
        g2=g2,
    )

    extra_tags = {
        "gate_g1_missed": verdict["g1_missed"],
        "gate_g2_sample_size": len(truths),
        **_g2_tags("gate_g2", g2),
    }

    print(f"  게이트: G1 놓침={verdict['g1_missed']}건 (champion {state.champion_missed}건, "
          f"허용 {state.champion_missed + 1}건) / G2 {_describe_g2(g2)}", flush=True)
```

- [ ] **Step 5: `_check_shadow`의 판정 구간 교체**

276행의 `from retraining.gate import accuracy_from_pairs` 줄을 지우고, 301~313행(`champion_accuracy = ...` 부터 `섀도우 종료` print까지)을 아래로 교체한다:

```python
    verdict = evaluate_two_sided(truths, champion_preds, candidate_preds)

    mlflow_client = MlflowClient()
    _tag(mlflow_client, state.shadow.run_id, scenario, current_day,
         decision=f"shadow_{verdict['decision']}", reason=verdict["reject_reason"],
         extra=_g2_tags("shadow", verdict))

    print(f"  섀도우 종료 — {_describe_g2(verdict)} → {verdict['decision']}", flush=True)
```

- [ ] **Step 6: 임포트·문법 확인 + 전체 테스트**

Run (약 7초, mlflow/torch 로딩):
```bash
uv run python -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'monitoring'); import drift_worker; print('ok', drift_worker._gate_predictions.__name__)"
grep -n 'accuracy' monitoring/drift_worker.py
uv run pytest -q
```
Expected: `ok _gate_predictions`. grep은 `WorkerState.champion_accuracy` 필드(이전부터 미사용) 한 줄과 `GATE_SAMPLE_SIZE` 주석의 과거 기록만 남는다 — 둘 다 이번 변경이 만든 미사용이 아니므로 손대지 않는다. pytest는 169 passed (165 − 5 제거 + 8 신규 + 1 pass-through 테스트).

- [ ] **Step 7: 문서 문구**

`docs/STRUCTURE.md`의 gate.py 세 줄:

```
│       ├── gate.py                 ·  G1 원본 eval에서 놓친 불량 개수(champion+1 이내)
│       │                           ·  G2 라벨 도착 창의 오탐·놓침 건수를 따로 비교 — 둘 다
│       │                           ·  후퇴 없고 한쪽 개선. 정상 라벨 없는 창은 거부. 섀도우도 동일
```

`README.md` §2-7의 문장 "원본 eval 성능 유지(G1)와 라벨 도착 구간 정확도 개선(G2)을 둘 다 확인한 뒤에도"를 "원본 eval 성능 유지(G1)와 라벨 도착 창에서 오탐·놓침이 둘 다 후퇴하지 않고 한쪽은 개선(G2, 정상 라벨이 없는 창은 판정 불가로 거부)을 둘 다 확인한 뒤에도"로 바꾼다.

- [ ] **Step 8: 커밋**

```bash
git add monitoring/drift_worker.py docs/STRUCTURE.md README.md
git commit -m "feat: gate and shadow verdicts use two-sided false-alarm/miss comparison"
```

---

### Task 4: 라이브 재현 3종으로 검증하고 결과 기록 (수동)

**Files:**
- Modify: `docs/specs/2026-09-02-cnc-two-sided-gate-design.md` ("실행 결과에 따른 정정" 절 추가)
- Modify: `../tasks/todo.md` (리뷰 절)
- 코드 변경 없음.

**Interfaces:** 없음.

- [ ] **Step 1: 서버 여유와 사전 상태 확인**

```bash
who; uptime
curl -s http://127.0.0.1:8000/health || echo "no server (정상)"
md5sum data/processed/scaler.json data/model/feature_baseline.json data/model/model.pt | cut -c1-8
```
Expected: 해시 `9ab55583` / `6d7d3978` / `8841fd72`(champion v1 정본).

- [ ] **Step 2: 시나리오마다 DB 초기화 → 세 프로세스 실행 → 정리**

시나리오 전환 때마다 반드시 세 DB를 치운다(보관: `data/monitoring/_<시나리오>_<날짜>/`).

```bash
mkdir -p data/monitoring/_<시나리오>_$(date +%Y%m%d)
mv data/monitoring/labels.db data/monitoring/requests.db data/monitoring/shadow.db data/monitoring/_<시나리오>_$(date +%Y%m%d)/ 2>/dev/null
rm -rf data/timeline/<이전 시나리오>

# 터미널 1
nice -n 19 uv run uvicorn src.serving.app:app --app-dir . --port 8000
# 터미널 2 (거부 시 RAG 조치 생성에 .env 필요)
nice -n 19 uv run --env-file .env python monitoring/drift_worker.py <시나리오> --base-url http://127.0.0.1:8000 --poll-interval 5
# 터미널 3
nice -n 19 uv run python monitoring/simulate_timeline.py <시나리오> --serve-url http://127.0.0.1:8000 --days <N> --pace-seconds <P>
```

| 시나리오 | `--days` | `--pace-seconds` | 기대 (워커 로그) |
|---|---|---|---|
| fixture_loosening | 40 | 2 | Day 19·24·29 거부(오탐 회귀), **Day 34 거부 "정상 라벨 없음"**, 추정 원인 4/4 vibration_backlash |
| tool_wear | 40 | 2 | 5회 거부(Day 20·25 오탐 회귀, 30 놓침 회귀, 35·40 정상 라벨 없음), 추정 원인 5/5 tool_wear |
| temperature | 70 | 15 | 거부 몇 회(오탐 회귀) 뒤 게이트 통과 → 섀도우 → **승격**. 08-25와 같은 흐름 |

워커 로그에 `Day 40`(temperature는 `Day 70`)이 찍히면 worker → server 순서로 종료한다. `pkill -f`에 스크립트 인자를 넣으면 명령을 담은 셸 자신이 먼저 죽는다 — PID로 종료하거나 `pgrep -f '[d]rift_worker'`를 쓴다.

- [ ] **Step 3: MLflow 태그 확인**

```bash
uv run python - <<'EOF'
import sys; sys.path.insert(0, "src")
from lstm_ae.tracking import configure_tracking, EXPERIMENT_NAME
from mlflow.tracking import MlflowClient
configure_tracking(); c = MlflowClient(); exp = c.get_experiment_by_name(EXPERIMENT_NAME)
for scen in ["fixture_loosening", "tool_wear", "temperature"]:
    runs = c.search_runs([exp.experiment_id], filter_string=f"tags.scenario = '{scen}'", order_by=["attributes.start_time DESC"], max_results=8)
    for r in runs:
        t = r.data.tags
        print(scen, "day", t.get("trigger_day"), t.get("gate_decision"), "| n_good", t.get("gate_g2_n_good"), "n_bad", t.get("gate_g2_n_bad"),
              "fa_delta", t.get("gate_g2_fa_delta"), "miss_delta", t.get("gate_g2_miss_delta"), "| cause", t.get("estimated_cause"), "|", (t.get("gate_reject_reason") or "")[:40])
EOF
```
Expected: 거부 run마다 `gate_g2_*` 네 태그와 `estimated_cause`가 있고, `gate_g2_accuracy_delta`는 새 run에 없다. temperature의 섀도우 run에는 `shadow_n_good`, `shadow_fa_delta` 등이 있다.

- [ ] **Step 4: champion v1 복원 (temperature가 승격시킨 뒤)**

```bash
uv run python scripts/promote_model.py 1
uv run python - <<'EOF'
import sys; sys.path.insert(0, "src")
from pathlib import Path
from retraining.promotion import restore_backup
backups = sorted(Path("data/model_backup").iterdir())
restore_backup(backups[-1], Path("data/model"), Path("data/processed/scaler.json"))
print("restored from", backups[-1].name)
EOF
md5sum data/processed/scaler.json data/model/feature_baseline.json data/model/model.pt | cut -c1-8
```
Expected: 해시가 Step 1과 같다. 다르면 `data/model_backup/` 중 해시가 맞는 폴더로 다시 복원한다.

- [ ] **Step 5: 결과 기록 + 커밋**

스펙 `docs/specs/2026-09-02-cnc-two-sided-gate-design.md` 맨 아래에 "## 실행 결과에 따른 정정 (날짜)" 절을 추가한다 — 시나리오별 트리거 일자·창 구성·오탐/놓침 건수·판정·추정 원인 표, 사후 적용 표(구현 전 데이터 확인 절)와 다른 점이 있으면 그 원인, temperature 승격 버전과 복원 결과. `../tasks/todo.md`에도 리뷰 절을 붙인다.

```bash
git add docs/specs/2026-09-02-cnc-two-sided-gate-design.md ../tasks/todo.md
git commit -m "docs: record live validation of the two-sided gate"
```

---

## Self-Review

- **Spec coverage**: 판정 규칙 표 3행 → Task 1. `evaluate_gate` 시그니처·`evaluate_shadow`/`accuracy_from_pairs` 제거 → Task 2. `_gate_predictions`, 로그, 태그 교체, 섀도우 판정, 거부 경로 유지 → Task 3. 문서 문구 → Task 3 Step 7. 라이브 재현 3종과 정정 절 → Task 4. Part C는 보류라 계획 없음.
- **Placeholder**: 없음. 모든 코드 스텝에 실제 코드가 있다.
- **Type consistency**: `evaluate_two_sided` 반환 키(`n_good`, `n_bad`, `champion_false_alarms`, `candidate_false_alarms`, `champion_misses`, `candidate_misses`, `decision`, `reject_reason`)를 Task 2의 `PROMOTED_G2`, Task 3의 `_describe_g2`/`_g2_tags`가 같은 이름으로 쓴다. `_gate_predictions`의 반환 순서(truths, champion_preds, retrained_preds, contributions)와 Task 3 Step 4의 언패킹 순서가 같다.
