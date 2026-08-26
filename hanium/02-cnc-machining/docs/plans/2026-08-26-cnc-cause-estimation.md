# 재학습 거부 원인 추정 + 현장 조치 제안 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 게이트가 재학습을 거부한 순간, 챔피언 모델의 feature_contributions로 원인(`tool_wear`/`vibration_backlash`)을 추정하고 그 카테고리로 RAG 코퍼스를 필터링해 현장 조치를 제안한다.

**Architecture:** 새 드리프트 시나리오 `fixture_loosening`을 `simulate_timeline.py`에 추가하고, 순수 함수 `estimate_cause()`(`src/monitoring/cause_estimation.py`)가 게이트 거부 시 이미 계산된 챔피언 feature_contributions로 원인을 추정한다. `src/rag/`에 원인 카테고리 전용 함수 2개(`generate_cause_guide`, `build_cause_guide`)를 추가해 기존 개별 `/predict` 안내(`build_guide`)와 분리하고, `drift_worker.py`가 이 함수들을 직접 import해 콘솔+MLflow 태그에 결과를 남긴다.

**Tech Stack:** Python, pandas/numpy, pytest, MLflow, OpenAI API(gpt-4o-mini), FAISS(이번 작업에서는 벡터 검색을 쓰지 않음).

**Spec:** `docs/specs/2026-08-26-cnc-cause-estimation-design.md`

## Global Constraints

- 승격(승인)된 경우는 다루지 않는다 — 이번 스코프는 게이트 "거부" 이벤트에만 대응한다.
- `tool_wear` / `vibration_backlash` 두 카테고리만 구분한다. 세 번째 원인은 범위 밖.
- 개별 `/predict` 응답에 쓰이는 `build_guide()`/`generate_guide()`는 변경하지 않는다.
- `fault_category`로 필터링한 뒤에는 벡터 재검색을 하지 않고 필터링된 청크를 그대로 LLM에 전달한다.
- drift_worker.py는 서버에 새 HTTP 엔드포인트를 요청하지 않고 `src/rag/`, `src/monitoring/`의 함수를 직접 import해서 쓴다(기존 `src/retraining/` 사용 패턴과 동일).

---

### Task 1: `fixture_loosening` 드리프트 시나리오 추가

**Files:**
- Modify: `monitoring/simulate_timeline.py`
- Modify: `monitoring/sweep_drift_constants.py`
- Test: `tests/monitoring/test_simulate_timeline.py` (신규)

**Interfaces:**
- Produces: `apply_fixture_loosening(df: pd.DataFrame, progress: float) -> pd.DataFrame`, `VIBRATION_RATE: float`, `VIBRATION_LABEL_FLIP_DAY: int`, `VIBRATION_COLUMNS: list[str]` — `PERTURBATIONS["fixture_loosening"]`와 `true_label()`에서 씀.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/monitoring/test_simulate_timeline.py`:
```python
import numpy as np
import pandas as pd
import pytest

from monitoring.simulate_timeline import apply_fixture_loosening


def _position_df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "X_ActualPosition": rng.normal(100, 5, size=n),
        "Y_ActualPosition": rng.normal(100, 5, size=n),
        "Z_ActualPosition": rng.normal(100, 5, size=n),
        "X_ActualVelocity": rng.normal(0, 1, size=n),
        "Y_ActualVelocity": rng.normal(0, 1, size=n),
        "Z_ActualVelocity": rng.normal(0, 1, size=n),
    })


def test_apply_fixture_loosening_no_change_at_zero_progress():
    df = _position_df()

    out = apply_fixture_loosening(df, progress=0.0)

    pd.testing.assert_frame_equal(out, df)


def test_apply_fixture_loosening_keeps_mean_but_increases_spread():
    df = _position_df()

    out = apply_fixture_loosening(df, progress=1.0)

    for col in df.columns:
        assert out[col].mean() == pytest.approx(df[col].mean(), abs=df[col].std() * 0.5)
        assert out[col].std() > df[col].std()
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `uv run pytest tests/monitoring/test_simulate_timeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_fixture_loosening'`

- [ ] **Step 3: 최소 구현**

`monitoring/simulate_timeline.py` 상단에 `import numpy as np` 추가(현재 `pandas`만 import되어 있음). `WEAR_LABEL_FLIP_DAY = 21` 줄 아래에 추가:

```python
VIBRATION_LABEL_FLIP_DAY = 21  # WEAR_LABEL_FLIP_DAY 와 동일 — Step 6에서 필요시 조정
VIBRATION_RATE = 0.2           # WEAR_RATE 채택값과 동일 자리 — Step 6에서 스윕 후 확정
```

`WEAR_COLUMNS = [...]` 아래에 추가:
```python
VIBRATION_COLUMNS = [
    "X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition",
    "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity",
]
```

`apply_tool_wear` 함수 뒤, `PERTURBATIONS = {...}` 줄 앞에 추가:
```python
def apply_fixture_loosening(df: pd.DataFrame, progress: float) -> pd.DataFrame:
    """고정구/척 풀림: 진행될수록 위치·속도 추종의 흔들림(분산)이 커진다.
    apply_tool_wear와 달리 평균은 그대로 두고 노이즈만 키운다."""
    out = df.copy()
    rng = np.random.default_rng(43)  # tool_wear 계열과 겹치지 않는 고정 시드
    for col in VIBRATION_COLUMNS:
        out[col] = out[col] + rng.normal(
            0, out[col].std() * VIBRATION_RATE * progress, size=len(out)
        )
    return out
```

`PERTURBATIONS` 딕셔너리를 다음으로 교체:
```python
PERTURBATIONS = {
    "temperature": apply_temperature,
    "tool_wear": apply_tool_wear,
    "fixture_loosening": apply_fixture_loosening,
}
```

`true_label()` 함수를 다음으로 교체:
```python
def true_label(scenario: str, day: int) -> str:
    """제품이 실제로 불량이냐. 온도는 제품 품질을 바꾸지 않는다."""
    if scenario == "temperature":
        return "good"
    if scenario == "fixture_loosening":
        return "bad" if day >= VIBRATION_LABEL_FLIP_DAY else "good"
    return "bad" if day >= WEAR_LABEL_FLIP_DAY else "good"
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `uv run pytest tests/monitoring/test_simulate_timeline.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add tests/monitoring/test_simulate_timeline.py monitoring/simulate_timeline.py
git commit -m "feat: add fixture_loosening drift scenario"
```

- [ ] **Step 6: 그리드 스윕으로 VIBRATION_RATE 캘리브레이션**

`monitoring/sweep_drift_constants.py`의 `main()` 끝(`tool_wear` 스윕 루프 뒤)에 추가:
```python
    print("=== fixture_loosening (VIBRATION_RATE = v) ===")
    for v in GRID:
        st.VIBRATION_RATE = v
        print(f"  v={v:<5} ratio={ratio_for(st.apply_fixture_loosening(base, 1.0), state):.2f}")
```

실행:
```bash
uv run python monitoring/sweep_drift_constants.py
```

출력된 `fixture_loosening` 표에서, `tool_wear` 채택값(0.2 → ratio 3.08)과 비슷한 수준(ratio 2.5~3.5, 실측 BAD 대역 1.00~3.79 안)이 되는 `v`를 찾는다. `simulate_timeline.py`의 `VIBRATION_RATE = 0.2` 를 그 값으로 갱신하고, `WEAR_RATE` 위 주석과 같은 방식으로 실측 표를 주석에 남긴다.

- [ ] **Step 7: 커밋**

```bash
git add monitoring/simulate_timeline.py monitoring/sweep_drift_constants.py
git commit -m "chore: calibrate VIBRATION_RATE against realistic drift band"
```

---

### Task 2: 원인 추정 함수 `estimate_cause()`

**Files:**
- Create: `src/monitoring/cause_estimation.py`
- Test: `tests/monitoring/test_cause_estimation.py`

**Interfaces:**
- Consumes: 없음(순수 함수, 외부 의존 없음)
- Produces: `estimate_cause(feature_contributions_batches: list[list[dict]]) -> str` — 반환값은 `"tool_wear"` 또는 `"vibration_backlash"`. 각 내부 dict는 `{"feature": str, "z_score": float, ...}` 형태(기존 `rank_feature_contributions()`의 반환 원소와 동일 스키마). Task 5에서 씀.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/monitoring/test_cause_estimation.py`:
```python
from monitoring.cause_estimation import estimate_cause


def test_estimate_cause_tool_wear_when_spindle_load_dominates():
    batches = [
        [
            {"feature": "S_OutputCurrent", "z_score": 10.0},
            {"feature": "X_ActualPosition", "z_score": 1.0},
        ]
        for _ in range(20)
    ]

    assert estimate_cause(batches) == "tool_wear"


def test_estimate_cause_vibration_backlash_when_position_variance_dominates():
    batches = [
        [
            {"feature": "S_OutputCurrent", "z_score": 1.0},
            {"feature": "X_ActualVelocity", "z_score": 10.0},
        ]
        for _ in range(20)
    ]

    assert estimate_cause(batches) == "vibration_backlash"


def test_estimate_cause_ignores_features_outside_both_groups():
    batches = [[{"feature": "M_CURRENT_FEEDRATE", "z_score": 999.0}]]

    assert estimate_cause(batches) == "tool_wear"  # 양쪽 다 0점 -> 동점 기본값


def test_estimate_cause_ties_default_to_tool_wear():
    batches = [[
        {"feature": "S_OutputCurrent", "z_score": 5.0},
        {"feature": "X_ActualPosition", "z_score": 5.0},
    ]]

    assert estimate_cause(batches) == "tool_wear"
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `uv run pytest tests/monitoring/test_cause_estimation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'monitoring.cause_estimation'`

- [ ] **Step 3: 최소 구현**

`src/monitoring/cause_estimation.py`:
```python
TOOL_WEAR_FEATURES = {
    "S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback",
    "X_OutputPower", "Y_OutputPower",
}
VIBRATION_BACKLASH_FEATURES = {
    "X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition",
    "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity",
}


def estimate_cause(feature_contributions_batches: list[list[dict]]) -> str:
    """챔피언 모델의 최근 N건 feature_contributions(배치별 피처-zscore 리스트)를
    받아, 두 피처 그룹 중 어느 쪽이 누적으로 더 크게 벗어났는지로 원인을
    추정한다. 반환값은 코퍼스의 fault_category 값과 동일하게 맞춘다
    ("tool_wear" / "vibration_backlash") — RAG 필터링에 그대로 쓰기 위함.

    simulate_timeline.py의 시나리오 이름은 "fixture_loosening"(물리적 원인
    이름)이지만, 여기서는 기존 Sandvik 코퍼스가 이미 쓰는 카테고리 값인
    "vibration_backlash"를 반환한다 — 시뮬레이션과 지식 코퍼스가 같은 현상을
    각자의 기존 명명 체계로 부르기 때문이다."""
    tool_wear_score = 0.0
    vibration_score = 0.0
    for contributions in feature_contributions_batches:
        for c in contributions:
            if c["feature"] in TOOL_WEAR_FEATURES:
                tool_wear_score += c["z_score"]
            elif c["feature"] in VIBRATION_BACKLASH_FEATURES:
                vibration_score += c["z_score"]
    return "tool_wear" if tool_wear_score >= vibration_score else "vibration_backlash"
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `uv run pytest tests/monitoring/test_cause_estimation.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/monitoring/cause_estimation.py tests/monitoring/test_cause_estimation.py
git commit -m "feat: add estimate_cause for drift-rejection root cause"
```

---

### Task 3: `generate_cause_guide()` — 원인 카테고리 전용 LLM 프롬프트

**Files:**
- Modify: `src/rag/generation.py`
- Test: `tests/rag/test_generation.py`

**Interfaces:**
- Consumes: 없음(테스트에서는 기존 `_FakeClient` 등 fixture 재사용)
- Produces: `generate_cause_guide(cause: str, retrieved_chunks: list[dict], client) -> dict` — 반환 스키마는 기존 `generate_guide()`와 동일(`cause_estimate`, `confidence_note`, `recommended_actions`, `safety_notes`, `sources`). Task 4에서 씀.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_generation.py` 파일 끝에 추가(기존 `_FakeClient`, `_PAYLOAD`를 그대로 재사용):
```python
from rag.generation import generate_cause_guide


def test_generate_cause_guide_parses_json_response():
    client = _FakeClient(json.dumps(_PAYLOAD))

    guide = generate_cause_guide("tool_wear", [], client)

    assert guide == _PAYLOAD


def test_generate_cause_guide_includes_confidence_principle_in_system_prompt():
    client = _FakeClient(json.dumps(_PAYLOAD))

    generate_cause_guide("tool_wear", [], client)

    messages = client.chat.completions.last_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "단정하지" in messages[0]["content"]


def test_generate_cause_guide_includes_cause_and_chunk_text_in_user_prompt():
    client = _FakeClient(json.dumps(_PAYLOAD))
    chunks = [{
        "title": "Sandvik", "url": "https://x", "content_type": "cause",
        "text": "Reduce cutting speed",
    }]

    generate_cause_guide("tool_wear", chunks, client)

    messages = client.chat.completions.last_kwargs["messages"]
    assert "tool_wear" in messages[1]["content"]
    assert "Reduce cutting speed" in messages[1]["content"]
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `uv run pytest tests/rag/test_generation.py -v -k cause_guide`
Expected: FAIL — `ImportError: cannot import name 'generate_cause_guide'`

- [ ] **Step 3: 최소 구현**

`src/rag/generation.py`의 `SYSTEM_PROMPT` 정의 뒤에 추가:
```python
CAUSE_SYSTEM_PROMPT = (
    "당신은 CNC 설비의 재학습 거부 사유를 설명하는 어시스턴트입니다.\n"
    "재학습이 거부됐다는 것은, 최근 며칠간의 변화가 정상 범위 재조정이 "
    "아니라 실제 설비 이상일 가능성이 높다는 뜻입니다. 아래 참고 문서를 "
    "바탕으로 추정 원인과 현장 조치를 제안하세요. 이 추정은 통계적 "
    "패턴 비교에 근거한 것이므로 단정하지 말고 확신도를 낮춘 표현을 "
    "쓰세요.\n"
    "아래 JSON 스키마로만 답하세요:\n"
    '{"cause_estimate": str, "confidence_note": str, '
    '"recommended_actions": [str], "safety_notes": [str], '
    '"sources": [{"title": str, "url": str}]}'
)


def _build_cause_user_prompt(cause: str, retrieved_chunks: list[dict]) -> str:
    lines = [f"통계적으로 추정된 원인 카테고리: {cause}", "\n참고 문서:"]
    for chunk in retrieved_chunks:
        lines.append(
            f"- [{chunk['title']}]({chunk['url']}) "
            f"{chunk['content_type']}: {chunk['text']}"
        )
    return "\n".join(lines)


def generate_cause_guide(cause: str, retrieved_chunks: list[dict], client) -> dict:
    model = os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CAUSE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_cause_user_prompt(cause, retrieved_chunks)},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `uv run pytest tests/rag/test_generation.py -v`
Expected: PASS (기존 테스트 포함 전체)

- [ ] **Step 5: 커밋**

```bash
git add src/rag/generation.py tests/rag/test_generation.py
git commit -m "feat: add generate_cause_guide for drift-rejection explanations"
```

---

### Task 4: `build_cause_guide()` — 카테고리 필터링 + 안전수칙 포함

**Files:**
- Modify: `src/rag/guide.py`
- Test: `tests/rag/test_guide.py`

**Interfaces:**
- Consumes: `generate_cause_guide(cause, retrieved_chunks, client) -> dict`(Task 3)
- Produces: `build_cause_guide(cause: str, rag_corpus: list[dict] | None, openai_client) -> dict | None`. Task 5에서 씀.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_guide.py` 파일 끝에 추가:
```python
from rag.guide import build_cause_guide

_CAUSE_CORPUS = [
    {"title": "Sandvik", "url": "https://x", "fault_category": "tool_wear",
     "content_type": "cause", "text": "공구 마모 조치"},
    {"title": "Sandvik", "url": "https://x", "fault_category": "vibration_backlash",
     "content_type": "cause", "text": "진동 조치"},
    {"title": "OSHA", "url": "https://y", "fault_category": "general",
     "content_type": "safety", "text": "안전 수칙"},
]


def test_build_cause_guide_returns_none_when_rag_unavailable():
    result = build_cause_guide("tool_wear", rag_corpus=None, openai_client=None)

    assert result is None


def test_build_cause_guide_filters_corpus_by_cause_and_includes_safety():
    captured = {}

    def fake_generate(cause, chunks, client):
        captured["cause"] = cause
        captured["chunks"] = chunks
        return {"cause_estimate": "x", "recommended_actions": []}

    import rag.guide as guide_module
    original = guide_module.generate_cause_guide
    guide_module.generate_cause_guide = fake_generate
    try:
        build_cause_guide("tool_wear", _CAUSE_CORPUS, openai_client=object())
    finally:
        guide_module.generate_cause_guide = original

    assert captured["cause"] == "tool_wear"
    texts = {c["text"] for c in captured["chunks"]}
    assert texts == {"공구 마모 조치", "안전 수칙"}


def test_build_cause_guide_returns_none_on_pipeline_exception():
    class _BrokenClient:
        pass

    result = build_cause_guide("tool_wear", _CAUSE_CORPUS, openai_client=_BrokenClient())

    assert result is None
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `uv run pytest tests/rag/test_guide.py -v -k cause_guide`
Expected: FAIL — `ImportError: cannot import name 'build_cause_guide'`

- [ ] **Step 3: 최소 구현**

`src/rag/guide.py`의 import 줄을 다음으로 교체:
```python
from .generation import generate_cause_guide, generate_guide
```

파일 끝에 추가:
```python
def build_cause_guide(cause: str, rag_corpus: list[dict] | None, openai_client) -> dict | None:
    if rag_corpus is None or openai_client is None:
        return None
    try:
        chunks = [c for c in rag_corpus if c["fault_category"] == cause]
        chunks += [c for c in rag_corpus if c["content_type"] == "safety"]
        return generate_cause_guide(cause, chunks, openai_client)
    except Exception as exc:
        print(f"원인 설명 생성 실패: {exc}")
        return None
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `uv run pytest tests/rag/test_guide.py -v`
Expected: PASS (기존 테스트 포함 전체)

- [ ] **Step 5: 커밋**

```bash
git add src/rag/guide.py tests/rag/test_guide.py
git commit -m "feat: add build_cause_guide with fault_category filtering"
```

---

### Task 5: `drift_worker.py` 통합

**Files:**
- Modify: `monitoring/drift_worker.py`

**Interfaces:**
- Consumes: `estimate_cause()`(Task 2), `build_cause_guide()`(Task 4), `load_rag_state()`(기존 `src/serving/app.py:52`)
- Produces: 없음(최종 소비자 — main 진입점)

이 파일은 실제 모델 파일/HTTP 클라이언트에 의존해 순수 유닛 테스트가 어렵고, 기존에도 라이브 재현(Task 6)으로만 검증돼 왔다. 이 태스크는 코드 변경 후 임포트/문법 확인만 하고, 실제 동작 검증은 Task 6에서 한다.

- [ ] **Step 1: `WorkerState`에 RAG 상태 필드 추가**

`monitoring/drift_worker.py`의 `WorkerState` 데이터클래스를 다음으로 교체:
```python
@dataclass
class WorkerState:
    flag_history: list[bool] = field(default_factory=list)
    cooldown_remaining: int = 0
    champion_missed: int = 1                # 현 champion 실측 — 불량 11개 중 1개 놓침
    champion_accuracy: float = 0.0          # 첫 게이트 평가 시 측정값으로 대체
    shadow: ShadowState | None = None
    rag_corpus: list[dict] | None = None
    openai_client: object | None = None
```

- [ ] **Step 2: import 추가**

파일 상단 import 블록(`from retraining.trigger import ...` 다음 줄)에 추가:
```python
from monitoring.cause_estimation import estimate_cause  # noqa: E402
from rag.guide import build_cause_guide  # noqa: E402
```

- [ ] **Step 3: `_predict_labels()`가 전체 result를 반환하도록 변경**

`_predict_labels()` 함수를 다음으로 교체:
```python
def _predict_labels(batch_paths, model, scaler_dict, threshold, baseline, window_size):
    """배치들을 주어진 모델로 판정한다. HTTP를 타지 않는다 — /predict 를 부르면
    요청 로그에 게이트 평가용 가짜 트래픽이 쌓여 드리프트 윈도우가 오염된다."""
    import pandas as pd

    from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
    from serving.inference import predict_experiment

    results = []
    for path in batch_paths:
        result = predict_experiment(
            pd.read_csv(path), model, FEATURE_COLUMNS, scaler_dict, window_size,
            threshold, "mean", baseline, SETUP_CONSTANT_COLUMNS,
        )
        results.append(result)
    return results
```

- [ ] **Step 4: `_gate_accuracies()`가 champion의 feature_contributions도 반환하도록 변경**

`_gate_accuracies()` 함수 반환형 주석과 본문을 다음으로 교체(시그니처 반환 타입만 바뀜, 인자는 그대로):
```python
def _gate_accuracies(result: dict, current_day: int, scenario: str) -> tuple[float, float, int, list[list[dict]]]:
    """G2 입력 — 라벨 도착 구간에서 champion과 재학습 모델의 정확도, 그리고
    champion의 배치별 feature_contributions(원인 추정용).

    두 모델을 같은 배치에 대고 직접 돌려 비교한다. champion 판정을 predict_log
    에서 꺼내오지 않는 이유는 배치 식별자가 로그에 없어 짝을 맞출 수 없기
    때문이고, /predict 를 다시 부르지 않는 이유는 위 _predict_labels 주석과 같다.
    """
    import json

    import torch

    from lstm_ae.model import LSTMAutoencoder
    from monitoring.labels import get_arrived_labels
    from preprocessing.columns import FEATURE_COLUMNS
    from retraining.gate import accuracy_from_pairs
    from retraining.runner import TRAINING_CONFIG
    from serving.app import load_model_state

    arrived = get_arrived_labels(current_day, LABELS_DB)[-GATE_SAMPLE_SIZE:]
    if not arrived:
        return 0.0, 0.0, 0, []

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

    return (
        accuracy_from_pairs(truths, champion_preds),
        accuracy_from_pairs(truths, retrained_preds),
        len(truths),
        champion_contributions,
    )
```

- [ ] **Step 5: `_decide_and_start_shadow()`에서 거부 시 원인 추정 + RAG 호출**

`_decide_and_start_shadow()` 함수의 게이트 판정 이후 부분(`champion_accuracy, retrained_accuracy, sample_size = _gate_accuracies(...)`부터 함수 끝까지)을 다음으로 교체:
```python
    champion_accuracy, retrained_accuracy, sample_size, champion_contributions = _gate_accuracies(
        result, current_day, scenario
    )
    verdict = evaluate_gate(
        retrained_missed=result["missed"],
        champion_missed=state.champion_missed,
        retrained_accuracy=retrained_accuracy,
        champion_accuracy=champion_accuracy,
    )

    extra_tags = {
        "gate_g1_missed": verdict["g1_missed"],
        "gate_g2_accuracy_delta": verdict["g2_accuracy_delta"],
        "gate_g2_sample_size": sample_size,
    }

    print(f"  게이트: G1 놓침={verdict['g1_missed']}건 (champion {state.champion_missed}건, "
          f"허용 {state.champion_missed + 1}건) / "
          f"G2 {retrained_accuracy:.2f} vs {champion_accuracy:.2f} "
          f"(표본 {sample_size}건)", flush=True)

    if verdict["decision"] == "rejected":
        cause = estimate_cause(champion_contributions)
        guide = build_cause_guide(cause, state.rag_corpus, state.openai_client)
        extra_tags["estimated_cause"] = cause
        extra_tags["recommended_action"] = (
            "; ".join(guide["recommended_actions"]) if guide else ""
        )
        _tag(mlflow_client, result["run_id"], scenario, current_day,
             decision="rejected", reason=verdict["reject_reason"], extra=extra_tags)
        action_desc = (
            guide["recommended_actions"] if guide
            else "(RAG 비활성 — OPENAI_API_KEY 또는 코퍼스 없음)"
        )
        print(f"  거부 — {verdict['reject_reason']}  (champion 유지, 사람 확인 필요)", flush=True)
        print(f"  추정 원인: {cause} / 권장 조치: {action_desc}", flush=True)
        return "rejected"

    _tag(mlflow_client, result["run_id"], scenario, current_day,
         decision=verdict["decision"], reason="", extra=extra_tags)
    _start_shadow(client, state, result, current_day)
    return "shadow_started"
```

- [ ] **Step 6: `main()`에 시나리오 추가 + RAG 상태 로딩**

`main()`의 `parser.add_argument("scenario", choices=["temperature", "tool_wear"])` 줄을 교체:
```python
    parser.add_argument("scenario", choices=["temperature", "tool_wear", "fixture_loosening"])
```

`main()`의 import 블록(`from monitoring.labels import get_latest_produced_day` 다음 줄)에 추가:
```python
    from serving.app import load_rag_state
```

`state = WorkerState()` 줄을 다음으로 교체:
```python
    state = WorkerState()
    state.rag_corpus, _, state.openai_client = load_rag_state()
```

- [ ] **Step 7: 임포트/문법 확인**

Run: `uv run python -c "import sys; sys.path.insert(0, 'monitoring'); sys.path.insert(0, 'src'); import drift_worker"`
Expected: 에러 없이 종료(모듈이 임포트만으로 성공해야 함 — 실제 서버 연결은 시도하지 않음)

Run: `uv run pytest -q`
Expected: 기존 전체 테스트(Task 1~4 포함) PASS, drift_worker.py 관련 새 실패 없음

- [ ] **Step 8: 커밋**

```bash
git add monitoring/drift_worker.py
git commit -m "feat: estimate cause and generate RAG guide on gate rejection"
```

---

### Task 6: 라이브 재현으로 통합 검증 (수동)

**Files:** 없음(코드 변경 없음, 실행/관찰만)

**Interfaces:** 없음

- [ ] **Step 1: labels.db / requests.db / shadow.db 초기화**

기존 40일 재현 때와 동일하게 세 DB를 비운다(정확한 절차는 README의 "완전 분리 실행" 절 참고).

- [ ] **Step 2: 3-프로세스 라이브 재현 실행**

```bash
# 프로세스 1: 서버
nice -n 19 uv run uvicorn src.serving.app:app --app-dir . --port 8000

# 프로세스 2: feeder
nice -n 19 uv run python monitoring/simulate_timeline.py fixture_loosening \
  --serve-url http://127.0.0.1:8000 --days 40 --pace-seconds 2

# 프로세스 3: 드리프트 워커
nice -n 19 uv run python monitoring/drift_worker.py fixture_loosening \
  --base-url http://127.0.0.1:8000
```

- [ ] **Step 3: 게이트 거부와 원인 추정 로그 확인**

드리프트 워커 콘솔에서 `Day XX  ... action=rejected` 다음 줄에 `추정 원인: vibration_backlash / 권장 조치: [...]`가 출력되는지 확인한다. `tool_wear` 재현 때처럼 거부가 아예 안 나오면 Task 1 Step 6의 `VIBRATION_RATE`/`VIBRATION_LABEL_FLIP_DAY`가 너무 약한 것이니 값을 올려 재실행한다.

- [ ] **Step 4: MLflow 태그 확인**

```bash
uv run python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_ids=['0'], filter_string=\"tags.scenario = 'fixture_loosening'\", order_by=['start_time DESC'], max_results=5)
for r in runs:
    print(r.info.run_id, r.data.tags.get('gate_decision'), r.data.tags.get('estimated_cause'), r.data.tags.get('recommended_action'))
"
```
`gate_decision=rejected`인 run에 `estimated_cause=vibration_backlash`와 비어있지 않은 `recommended_action`이 있는지 확인한다(실험 ID는 실제 MLflow 설정에 맞게 조정).

- [ ] **Step 5: 결과를 spec 문서에 기록**

`docs/specs/2026-08-26-cnc-cause-estimation-design.md`의 "알려진 한계" 섹션 아래에 "실행 결과에 따른 정정" 절을 추가해, 실제 캘리브레이션 값과 재현에서 관찰된 것(정정 사항이 있다면)을 기록한다.

```bash
git add docs/specs/2026-08-26-cnc-cause-estimation-design.md
git commit -m "docs: record fixture_loosening live reproduction results"
```
