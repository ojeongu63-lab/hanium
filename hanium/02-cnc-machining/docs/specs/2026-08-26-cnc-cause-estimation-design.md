# 재학습 거부 원인 추정 + 현장 조치 제안 설계

## 배경

드리프트 워커(`monitoring/drift_worker.py`)는 재구성오차가 연속 3일 기준을
넘으면 재학습을 트리거하고, 게이트(`src/retraining/gate.py`)가 재학습된
모델과 챔피언 모델을 실측 라벨로 비교해 승격/거부를 정한다. 이 판정은
순수하게 결과(정확도) 기반이다 — **게이트는 원인을 모른다.** 트리거도
마찬가지다(`drift_worker.py` 모듈 docstring: "센서만 틀어진 것"과 "설비가
실제로 망가진 것"을 구분하지 못한다").

그 결과 "거부됨"이라는 이벤트가 발생해도 지금은 사유(정확도 수치)만 남고,
현장에서 무엇을 봐야 하는지에 대한 안내가 없다.

한편 `/predict`에 연결된 RAG(`src/rag/`)는 불량 판정 시 개별 배치의 top
feature_contributions(순간 z-score 3개)로 쿼리를 만들어 코퍼스를 벡터
검색하는데, 코퍼스가 작고(26개 청크) 카테고리 필터링이 없어 서로 다른
원인이 거의 항상 같은 결과로 수렴하는 문제가 실측으로 확인됐다(2026-08-26
세션). 순간 스냅샷보다 "여러 날에 걸친 추세"가 원인별로 더 뚜렷하다는
것도 함께 확인했다 — `tool_wear`는 스핀들 부하 계열의 평균이 선형
증가하고, 새로 추가하는 `fixture_loosening`은 위치·속도 추종 계열의
분산만 커진다.

## 목표

게이트가 **재학습을 거부한 순간**, 이미 계산되어 있는 챔피언 모델의
feature_contributions(G2 평가에 쓰인 최근 20건)를 재사용해 원인을
`tool_wear` / `fixture_loosening` 둘 중 하나로 추정하고, 그 카테고리로
RAG 코퍼스를 필터링해 현장 조치를 생성한다. 결과는 콘솔 로그와 MLflow
run 태그에 남긴다.

## 비목표

- 승격(승인)된 경우의 사후 설명 — 이번 스코프는 "거부" 이벤트에만 대응한다.
- `temperature`나 그 외 원인의 구분 — 지금 거부로 이어지는 시나리오는
  `tool_wear`, `fixture_loosening` 둘뿐이라 이 둘만 구분한다. 나중에
  다른 거부 원인이 생기면 그때 확장한다(알려진 한계로 문서화).
- 개별 `/predict` 응답의 RAG 가이드(`build_guide`) 변경 — 그대로 둔다.
- fault_category를 이용한 벡터 재랭킹 — 카테고리로 필터링한 뒤 남는
  청크가 6~14개뿐이라 그대로 LLM에 넘긴다. 벡터 검색을 얹지 않는다.

## 아키텍처 개요

```
[챔피언 모델] --최근 20건 feature_contributions--> [estimate_cause()]
                                                         |
                                                    "tool_wear" 또는
                                                    "fixture_loosening"
                                                         |
                                                         v
[corpus.json] --fault_category로 필터링--> [build_cause_guide()] --> {cause_estimate, recommended_actions, ...}
                                                         |
                                    +--------------------+--------------------+
                                    v                                         v
                              콘솔 로그 출력                          MLflow run 태그
```

새 컴포넌트 하나(`cause_estimation.py`)와 기존 `src/rag/` 모듈에 대한
작은 확장, 그리고 `simulate_timeline.py` / `drift_worker.py`에 대한
수정으로 구성된다. 서버에 새 HTTP 엔드포인트는 추가하지 않는다 —
drift_worker가 지금도 `src/retraining/`의 순수 함수를 직접 호출하는
패턴을 그대로 따라, `src/rag/`의 함수도 직접 import해서 쓴다.

## 컴포넌트 상세

### 1. `monitoring/simulate_timeline.py` — 새 시나리오

```python
VIBRATION_NOISE_RATE = ...  # WEAR_RATE(=1.0)와 같은 자리 — 계획 단계에서 캘리브레이션
VIBRATION_LABEL_FLIP_DAY = ...  # WEAR_LABEL_FLIP_DAY와 같은 자리

def apply_fixture_loosening(df: pd.DataFrame, progress: float) -> pd.DataFrame:
    """고정구/척 풀림: 진행될수록 위치·속도 추종의 흔들림(분산)이 커진다.
    apply_tool_wear와 달리 평균은 그대로 두고 노이즈만 키운다."""
    out = df.copy()
    rng = np.random.default_rng(43)  # tool_wear와 다른 시드
    cols = ["X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition",
            "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity"]
    for col in cols:
        out[col] = out[col] + rng.normal(0, out[col].std() * VIBRATION_NOISE_RATE * progress, size=len(out))
    return out


PERTURBATIONS = {
    "temperature": apply_temperature,
    "tool_wear": apply_tool_wear,
    "fixture_loosening": apply_fixture_loosening,
}
```

`true_label()`에 `fixture_loosening` 분기를 추가한다 — `tool_wear`와
같은 방식으로 "진행이 `VIBRATION_LABEL_FLIP_DAY`를 넘으면 `bad`"로
전환한다. `VIBRATION_NOISE_RATE`와 `VIBRATION_LABEL_FLIP_DAY`의 정확한
숫자는 `sweep_drift_constants.py`와 동일한 방법론(실측 GOOD/BAD
재구성오차 대역을 벗어나지 않는 값)으로 계획 단계에서 캘리브레이션한다
— 이 문서는 두 상수가 있어야 한다는 설계 의도만 고정한다.

### 2. `src/monitoring/cause_estimation.py` — 신규

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
    순수 함수 — 모델이나 I/O에 의존하지 않는다."""
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

두 그룹 어디에도 속하지 않는 피처(예: `M_CURRENT_FEEDRATE`)는 계산에서
제외한다. 두 점수가 정확히 0(입력이 비정상적으로 비어있는 등)이면
`tool_wear`로 판정한다 — 극단적 엣지 케이스이며, 실제로는 게이트가
거부할 정도의 편차가 있었다는 전제 자체가 이걸 막는다.

### 3. `src/rag/generation.py` — 새 프롬프트 빌더

기존 `generate_guide()`는 개별 배치 판정 스키마(`predicted_label_text`,
`score`, `threshold`)에 묶여 있어 그대로 재사용할 수 없다. 별도 함수를
추가한다:

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


def generate_cause_guide(cause: str, retrieved_chunks: list[dict], client) -> dict:
    model = os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL)
    lines = [f"통계적으로 추정된 원인 카테고리: {cause}", "\n참고 문서:"]
    for chunk in retrieved_chunks:
        lines.append(
            f"- [{chunk['title']}]({chunk['url']}) "
            f"{chunk['content_type']}: {chunk['text']}"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CAUSE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

### 4. `src/rag/guide.py` — 새 진입점

```python
def build_cause_guide(cause: str, rag_corpus, openai_client) -> dict | None:
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

`fixture_loosening`은 새 코퍼스 문서 없이 기존 Sandvik
`vibration_backlash` 카테고리(6개 청크, "약한 고정구" 등)를 그대로
사용한다 — `fault_category` 값 자체를 `"vibration_backlash"`로 맞춰서
`estimate_cause`의 리턴값과 코퍼스 필터링 키를 통일한다(즉
`estimate_cause`는 `"tool_wear"` / `"vibration_backlash"`를 반환하도록
정의한다 — 함수명과 반환값의 혼동을 피하기 위해 아래 "명명" 절 참고).

### 5. `monitoring/drift_worker.py` — 통합

- `main()`의 `choices=["temperature", "tool_wear"]`에 `"fixture_loosening"`
  추가.
- `main()`에서 `from serving.app import load_rag_state`로 RAG 상태
  (`rag_corpus`, `openai_client` — `rag_index`는 이 흐름에서 불필요)를
  1회 로드해 `WorkerState`에 보관한다.
- `_predict_labels()`가 라벨 텍스트만 모으던 것을 전체 `result` 리스트를
  반환하도록 바꾼다. 호출부(`_gate_accuracies`)에서 필요한 값
  (`predicted_label_text`, `feature_contributions`)을 각각 뽑아 쓴다.
- `_gate_accuracies()`가 champion 쪽 `feature_contributions` 배치
  리스트도 함께 반환하도록 반환 튜플을 확장한다.
- `_decide_and_start_shadow()`에서 `verdict["decision"] == "rejected"`
  분기에 추가:

```python
if verdict["decision"] == "rejected":
    cause = estimate_cause(champion_contributions_batches)
    guide = build_cause_guide(cause, state.rag_corpus, state.openai_client)
    _tag(mlflow_client, result["run_id"], scenario, current_day,
         decision="rejected", reason=verdict["reject_reason"],
         extra={
             "gate_g1_missed": verdict["g1_missed"],
             "gate_g2_accuracy_delta": verdict["g2_accuracy_delta"],
             "gate_g2_sample_size": sample_size,
             "estimated_cause": cause,
             "recommended_action": "; ".join(guide["recommended_actions"]) if guide else "",
         })
    action_desc = guide["recommended_actions"] if guide else "(RAG 비활성 — OPENAI_API_KEY 또는 코퍼스 없음)"
    print(f"  거부 — {verdict['reject_reason']}", flush=True)
    print(f"  추정 원인: {cause} / 권장 조치: {action_desc}", flush=True)
    return "rejected"
```

## 명명 정합성

`simulate_timeline.py`의 시나리오 이름은 `"fixture_loosening"`(물리적
원인을 가리키는 이름, MLflow `scenario` 태그에 쓰임)이지만,
`estimate_cause()`와 코퍼스 필터링은 `"vibration_backlash"`(기존
Sandvik 코퍼스가 이미 쓰는 카테고리 값)를 쓴다. 같은 현상을 두 다른
문맥(시뮬레이션 vs 지식 코퍼스)에서 각자의 기존 명명 체계로 부르는
것이므로, 두 이름이 어디서 왜 갈리는지 `estimate_cause()` 옆에 주석으로
남긴다.

## 에러 처리

- `OPENAI_API_KEY` 없음 또는 코퍼스 미구축 → `build_cause_guide`가
  `None` 반환, 드리프트 워커는 원인 추정(`estimated_cause` 태그)까지는
  남기고 조치 제안 없이 진행(기존 `/predict`의 "키 없어도 서버는
  정상 기동" 원칙과 동일).
- `generate_cause_guide` 내부에서 OpenAI 호출 실패 → `build_cause_guide`
  가 예외를 잡아 `None` 반환(기존 `build_guide`와 동일 패턴).
- `estimate_cause`에 빈 리스트가 들어오는 경우 — G2 표본이 0건이면
  게이트 자체가 거부되지 않으므로(`_gate_accuracies`가 조기 반환) 이
  경로에서 발생하지 않는다.

## 테스트 전략

- `estimate_cause()`: 순수 함수, TDD로 유닛 테스트(스핀들 부하가 우세한
  입력 → `tool_wear`, 위치/속도 분산이 우세한 입력 → `vibration_backlash`,
  동점 → `tool_wear`).
- `apply_fixture_loosening()`: 평균은 유지되고 분산만 커지는지 확인하는
  유닛 테스트.
- `build_cause_guide()` / `generate_cause_guide()`: `rag_corpus=None` /
  `openai_client=None` 폴백 테스트, 카테고리 필터링이 올바른 청크만
  골라내는지 테스트(OpenAI 호출은 모킹).
- 통합 검증: 40일 이상 `fixture_loosening` 시나리오를 라이브로 돌려
  게이트가 실제로 거부하고, `estimated_cause` 태그가
  `vibration_backlash`로 남는지 수동 확인(기존 온도/공구마모 재현과
  같은 방식).

## 알려진 한계

- `tool_wear` 대 `vibration_backlash` 두 카테고리만 구분한다. 세 번째
  거부 원인이 추가되면 `estimate_cause`를 확장해야 한다.
- 두 피처 그룹의 경계에 걸치는 복합 원인(예: 마모와 풀림이 동시 진행)은
  점수가 큰 쪽으로만 판정되고 혼합 신호를 표현하지 못한다.
- `fixture_loosening` 캘리브레이션(노이즈 비율, 라벨 전환일)은 실측
  재구성오차 대역에 맞춰 계획 단계에서 조정이 필요할 수 있다.

## 실행 결과에 따른 정정 (2026-09-02)

계획서 Task 6대로 세 프로세스(서버 / feeder `--pace-seconds 2` / 워커
`--env-file .env`)로 40일을 라이브로 돌렸다. 워커·feeder·서버 로그와
`labels.db`/`requests.db` 원본은 `data/monitoring/_fixture_loosening_20260902/`,
`_tool_wear_20260902/`에 보관.

### fixture_loosening — 거부 3회 모두 `vibration_backlash`, 4번째는 게이트 통과

| 트리거 | G2 (재학습 vs champion, 표본 20) | 판정 | estimated_cause | recommended_action 첫 항목 |
|---|---|---|---|---|
| Day 19 (v41) | 0.80 vs 0.90 | 거부 | vibration_backlash | 고정구의 안정성을 확인하고 필요시 개선 |
| Day 24 (v42) | 0.45 vs 0.85 | 거부 | vibration_backlash | 고정구의 강도를 점검하고 필요에 따라 개선 |
| Day 29 (v43) | 0.60 vs 0.65 | 거부 | vibration_backlash | 고정구의 상태를 점검하고 필요시 강화 |
| Day 34 (v44) | **0.80 vs 0.60** | **통과 → 섀도우 시작** | (거부 아님) | — |

설계 목표는 충족됐다 — 거부가 날 때마다 원인이 `vibration_backlash`로
추정됐고(3/3, 캘리브레이션 `VIBRATION_RATE=3.65`·`VIBRATION_LABEL_FLIP_DAY=21`
변경 없음), 그 카테고리로 필터한 코퍼스에서 만든 RAG 조치는 매번 "고정구
점검"을 첫 항목으로 냈으며, `estimated_cause`·`recommended_action` 태그가
MLflow run에 남았다. champion은 v1 그대로(`/health`, scaler·baseline 해시
불변).

**계획과 달랐던 점 — 4번째 재학습이 게이트를 통과했다.** 원인 추정의
결함이 아니라 G2의 구조적 사각지대다. Day 34의 G2 창(라벨이 도착한 최근
20건 = 생산일 Day 24~27)은 **전부 불량 라벨**이다(전환일 21 + 지연 7).
후보 v44와 champion을 같은 배치에 대고 다시 돌려 확인했다:

- 후보는 champion과 배치 순위가 같고 score/threshold만 일률적으로 약 20%
  높다(임계값 0.62 vs 0.86). "같은 걸 보고 더 자주 불량이라 하는" 모델이다.
- 전부 불량인 Day 24~27 창에서는 그래서 이긴다 — 불량 판정 16/20 vs 12/20,
  정확도 0.80 vs 0.60.
- 같은 후보를 전부 정상인 Day 14~17 창(Day 24 게이트가 본 창)에 대면 진다 —
  정상을 불량이라 한 게 9/20 vs 3/20, 정확도 0.55 vs 0.85.

G2는 "최근 라벨 구간에서 더 정확한가"만 묻기 때문에, 창에 한 클래스만
있으면 과탐 경향을 개선으로 읽는다. `gate.py`가 G1에 대해 적어둔 경고("모든
것을 불량이라 판정하는 모델도 통과")가 G2에도 성립하는 조건이 이것이다.
시나리오 B(공구마모, 08-19)에서 5회 모두 거부됐던 것은 곱셈 램프를 정상으로
학습한 재학습 모델이 오히려 덜 잡았기 때문이고, 가산 노이즈인
fixture_loosening에서는 반대 방향으로 갈렸다 — 거부 여부가 드리프트의 물리적
형태에 따라 달라진다는 뜻이다.

섀도우는 이번엔 이걸 막지 못했다. 관찰 기준일이 Day 40(feeder 완주 시점)이라
이후 트래픽이 없어 `shadow_pending`으로 끝났고, `--days`를 늘려도 Day 41
이후 라벨 역시 전부 불량이라 같은 사각지대에 놓인다.

**후속 후보(미구현, 별도 스펙 필요)**: G2 창에 정상·불량이 모두 있어야
판정하도록 하거나, G2를 "불량 창 개선 AND 정상 창 무회귀"의 두 조건으로
나누는 것. 이번 스코프(원인 추정)와 무관한 게이트 설계 문제라 기록만 남긴다.

### tool_wear — 5회 모두 거부, 추정 원인 5/5 `tool_wear`

두 갈래 분류기의 반대쪽도 확인하려고 같은 절차로 `tool_wear`를 돌렸다
(계획서에는 없던 추가 검증).

| 트리거 | G2 (재학습 vs champion, 표본 20) | 판정 | estimated_cause | recommended_action 첫 항목 |
|---|---|---|---|---|
| Day 20 (v45) | 0.75 vs 0.90 | 거부 | tool_wear | 절삭 속도(vc)를 낮춰 절삭 온도를 낮춤 |
| Day 25 (v46) | 0.65 vs 0.80 | 거부 | tool_wear | 절삭 속도를 낮추고 이송량(fz) 증가 |
| Day 30 (v47) | 0.50 vs 0.55 | 거부 | tool_wear | 절삭 속도를 낮추고 이송량 증가 |
| Day 35 (v48) | 0.35 vs 1.00 | 거부 | tool_wear | 절삭 속도를 낮춤 |
| Day 40 (v49) | 0.40 vs 1.00 | 거부 | tool_wear | 절삭 속도 조정 |

08-19 시나리오 B와 같은 결론(5회 모두 거부)에 원인 추정이 얹혔다. 두
시나리오를 합쳐 거부 8건 중 8건에서 `estimate_cause`가 정답 카테고리를
냈고 반대쪽으로 넘어간 적이 없다. 전부 불량인 창(Day 35·40)에서는 재학습
모델이 20건 중 7~8건만 잡아 champion(20/20)에 크게 밀렸다 —
fixture_loosening의 Day 34와 정반대 방향이며, 위 "거부 여부가 드리프트의
물리적 형태에 따라 달라진다"는 관찰을 실측으로 뒷받침한다.

### 시뮬레이션 관찰 (버그 아님)

첫 트리거(Day 19) 이후 워커가 재학습으로 뒤처지는 동안 feeder가 40일을
완주해, Day 20부터는 `/drift-status` 값이 전부 2.61(Day 40 시점)로 찍힌다.
트리거가 정확히 쿨다운(5일)마다 걸린 이유이며, 게이트·원인 추정은 논리적
날짜의 라벨로 계산하므로 영향 없다(08-24 기록과 동일한 현상).
