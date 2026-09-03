# 플레이북 기반 조치 가이드 설계 (센서 서명 대조)

작성 2026-09-03. 선행 스펙: `2026-08-12-cnc-rag-action-guide-design.md`(`/predict`
가이드), `2026-09-02-cnc-diagnosis-agent-design.md`(보류 — 이 문서가 대체).

## 배경

### 확인된 문제 — 검색이 모델의 정보를 버린다

`/predict`의 가이드는 상위 3개 센서 이름을 넣은 한 문장을 임베딩해 코퍼스
26청크와 코사인 유사도로 3개를 뽑고, 그 문서를 읽은 LLM이 답을 쓴다. 코퍼스는
전부 절삭 조건 어휘(절삭 속도, 이송량, 인서트 등급)로 쓰여 있고 센서 어휘로
쓰인 청크는 KAMP의 "설정값 대비 실제값, 전류·전압·토크 신호 수집" 하나뿐이라,
무엇을 넣어도 그 청크가 1위가 되고 LLM은 진동 신호를 보고도 공구 마모로 답한다.

### 2026-09-03 실험 — 문서를 늘리는 것만으로는 부족하다

사용자 제안대로 "시나리오에 맞는 문서를 많이 써서 활용"하는 방식을 단계별로
실험했다. 입력은 합성 불량 3건(`synthetic/scenarios/*_predict_result.json`)과
실제 불량 experiment_07(`docs/examples/predict_response_experiment_07.json`).

| 단계 | 코퍼스 | 방식 | 맞게 고른 상황 | 비고 |
|---|---|---|---|---|
| 1 | 기존 26 + 시나리오 문서 4 | 임베딩 검색 | 4 / 4 | 문서가 4개뿐이라 맞은 것 |
| 2 | 기존 26 + 상황 문서 16 | 임베딩 검색 | 2 / 4 | 틀린 둘은 1·2위 차이 0.002 |
| 3 | 같음 | 규칙으로 그룹 선택 → 그룹 안에서 임베딩 | 그룹 4/4, 상황 3/4 | 공구 마모 대신 스핀들 전원 이상 |
| 4 | 같음 | **문서에 적힌 센서 코드와 상위 피처의 일치도** | 4 / 4 | 다른 그룹과 점수 차이 큼 |

임베딩이 실패하는 이유는 질의가 센서 이름 나열이고 상황 문서들이 서로 비슷한
센서를 언급하기 때문이다. 모델이 주는 정보는 정확히 "어떤 센서 코드가 얼마나
튀었나"이므로 그 코드로 대조하는 것이 맞다. 4단계 방식에 "시스템이 고른
상황"을 프롬프트로 알려주자 원인 문장도 4/4 깨끗해졌다(그 전에는 3/4에
"공구 마모"가 끼어 있었다).

### 타임라인 시나리오 600건 검증

4단계 방식을 `monitoring/simulate_timeline.py`의 세 시나리오(40일 × 5배치)에
champion v1로 오프라인 적용했다. 스크립트·로그·배치별 기록은
`data/monitoring/_signature_spike_20260903/`.

| 시나리오 | 구간 | 불량 판정 | 고른 상황 | 일치 중앙값 | 다른 그룹 최고 | 동률 |
|---|---|---|---|---|---|---|
| tool_wear | Day 21-40 (실제 불량) | 93/100 | 공구 마모 93 | 0.80 | 0.22 | 0 |
| fixture_loosening | Day 21-40 (실제 불량) | 77/100 | 고정구 풀림·채터 77 | 1.00 | 0.00 | 0 |
| temperature | Day 21-40 (실제 정상) | 79/100 | 공구 마모 43, 이송축 과부하 35, 절삭유 1 | 0.66 | 0.44 | 35 |
| 셋 공통 | Day 1-10 (변형 전) | 6/50 | 고정구 풀림·채터 6 | 0.58 | 0.00 | 6 |

고장 둘은 불량 시작 뒤 한 건도 다른 상황을 고르지 않았다. 온도는 고장이
아닌데 배치 하나만 보면 스핀들과 이송축이 함께 올라 고장 상황을 고른다. 다만
숫자가 고장과 다르다 — 고장은 한 그룹만 튀고 다른 그룹은 1위의 0.33 이하,
온도는 다른 그룹이 1위의 0.67(중앙값). 변형 전 오탐 6건은 같은 배치(이송
속도 20 실험)이고 상위 z가 4.4로 약하다. 그래서 판정을 세 단계로 나눈다.

| 시나리오·구간 | 확정 | 복합 징후 | 약한 신호 |
|---|---|---|---|
| tool_wear Day 21-40 | 공구 마모 93 | 0 | 0 |
| fixture_loosening Day 21-40 | 고정구 풀림 77 | 0 | 0 |
| temperature Day 21-40 | 이송축 과부하 10 | 62 | 7 |
| 변형 전 오탐 6건 | 0 | 0 | 6 |

## 사용자 결정 (2026-09-03)

- 문서를 시나리오대로 많이 써서 활용한다. 데모 서사는 "플레이북에 많은 상황이
  있고, 모델이 짚은 센서와 각 상황의 센서 서명을 대조해 상황을 고른 뒤 그
  문서로 조치 가이드를 쓴다".
- 진단 에이전트 API(요청 번호 저장, 도구 호출 루프)는 **보류**. Spring 연동은
  `/predict` 응답 확장으로 충분하다.
- 문서는 외부 자료가 아니라 팀이 쓴 플레이북임을 출처와 데모에서 밝힌다.

## 목표 / 비목표

**목표**

1. `/predict` 가이드가 16개 상황 중 맞는 것을 결정적으로 고르고, 그 문서로
   조치를 쓴다. 합성 3건 + experiment_07 4/4, 타임라인 고장 2종 100%.
2. 고장이 아닌 변화(온도 드리프트, 학습 범위 밖 프로그램)는 "복합 징후" 또는
   "약한 신호"로 나가고 고장이라고 단정하지 않는다.
3. 응답에 `fault` 필드를 더해 상황·판정·근거 센서를 기계가 읽을 수 있게 한다.
   기존 9개 키와 `guide` 스키마는 그대로.
4. OpenAI 키가 없어도 `fault`는 나온다(`guide`만 `null`).

**비목표**

- 진단 에이전트, 요청 번호 저장, `/diagnose` 엔드포인트(보류).
- 재학습 거부 경로의 `estimate_cause`(두 카테고리) 변경. 검증이 끝난 코드다.
- 외부 자료 추가. 플레이북은 자체 작성 문서다.
- 실제 실험(eval 14개)의 원인 정답 정의.
- 기준값 두 개(`WEAK_Z`, `COMPOSITE_RATIO`)의 일반화 검증. 이번 시나리오 결과에서
  여유를 두고 고른 값이며, 알려진 한계에 적는다.

## 데이터 흐름

```
/predict  CSV → 판정 + 기여도 40개
          → match_playbook(기여도, 코퍼스)          # 결정적, 임베딩 없음
             = {verdict, situation, category, coverage, alternatives, other_group, ...}
          → select_chunks(판정, 코퍼스, 임베딩 검색)  # 확정일 때만 임베딩으로 Sandvik 2개 고름
          → LLM 가이드 (프롬프트에 판정·상황·후보 명시)
          → 응답 + fault + guide
```

## Part A — 플레이북 문서 (`rag/sources/scenario_playbook.md`, 신규)

구역 4개는 원인이 아니라 **센서가 보여주는 모습**으로 나눈다. 항목마다 다섯
줄이 고정이다: `관련 센서`, `증상`, `가능 원인`, `현장 확인`, `조치`.

```
## 1. 스핀들 부하 상승

### 공구 마모 — 스핀들 부하가 서서히 상승
관련 센서: S_OutputCurrent, S_OutputPower, S_CurrentFeedback
증상: ...
가능 원인: ...
현장 확인: ...
조치: ...
```

작성 규칙:

- `관련 센서`는 `FEATURE_COLUMNS`의 코드만 쓴다. 빌드 시 검증한다. 대조에 쓰이지
  않을 항목(온도 드리프트, 소재 변경)은 `관련 센서: 없음`.
- 센서로 구분이 안 되는 원인들은 같은 구역에 두고 `현장 확인` 줄에 구분법을
  적는다.
- **각 구역의 첫 항목이 대표 상황**이다. 서명 점수가 같으면 앞선 항목을 고른다.
  (실험에서 임베딩으로 동률을 가른 결과와 같다 — 이송축 과부하 vs 윤활 불량,
  이송축 과부하 vs 절삭 조건 과다.)

| 구역 (카테고리) | 항목 | 관련 센서 |
|---|---|---|
| 1. 스핀들 부하 상승 (tool_wear) | 공구 마모 (대표) | S_OutputCurrent, S_OutputPower, S_CurrentFeedback |
| | 공구 파손·치핑 | S_OutputCurrent, S_OutputPower |
| | 절삭유 부족·과열 | S_OutputPower, S_OutputVoltage |
| | 스핀들 베어링 손상 | S_OutputCurrent, S_ActualVelocity, S_ActualAcceleration |
| | 스핀들 전원·인버터 이상 | S_DCBusVoltage, S_OutputVoltage |
| 2. 이송축 부하 상승 (feed_overload) | 이송축 과부하 (대표) | X_OutputCurrent, Y_OutputCurrent, X_OutputPower, Y_OutputPower |
| | 볼스크류·가이드 윤활 불량 | X_OutputCurrent, Y_OutputCurrent |
| | 절삭 조건 과다 | S_OutputPower, X_OutputPower, Y_OutputPower |
| 3. 축 위치·속도 편차 (vibration_backlash) | 고정구 풀림·채터 (대표) | X/Y/Z_ActualVelocity, X/Y/Z_ActualPosition |
| | 볼스크류 백래시 | X/Y/Z_ActualPosition, X/Y/Z_ActualAcceleration |
| | 공구 돌출 과다·홀더 불량 | S_CurrentFeedback, X_ActualVelocity, Y_ActualVelocity |
| | 엔코더·피드백 이상 | X/Y/Z_ActualPosition |
| 4. 고장이 아닌 변화 (general) | 온도 드리프트 (대표) | 없음 |
| | 이송 속도 설정 변경 | M_CURRENT_FEEDRATE, X_SetVelocity, Y_SetVelocity, Z_SetVelocity |
| | 가공 시작·종료 과도 구간 | X/Y/Z_ActualAcceleration |
| | 공작물 소재 변경 | 없음 |

본문 초안은 `data/monitoring/_signature_spike_20260903/scenario_playbook.md`
(실험에 쓴 것). 내용은 일반 가공 지식으로 쓴 것이라 도메인 검토가 필요하며,
검토 전이라도 데모 흐름 검증에는 쓸 수 있다.

### 코퍼스 빌드 (`rag/build_corpus.py`)

- `parse_playbook(text)`을 `src/rag/playbook.py`에 두고 빌드 스크립트가 가져다
  쓴다(테스트에서 import 가능하게). 청크 필드: 기존 6개 + `signature: list[str]`
  + `source: "playbook"` + `name`(제목에서 " — " 앞부분). 구역 1~3은
  `content_type: "cause"`, 구역 4는 `"context"`.
- 메타: `title = "팀 시나리오 플레이북(자체 작성)"`, `url =
  "rag/sources/scenario_playbook.md"`.
- 빌드 시 검증: 항목 16개, 모든 `관련 센서` 코드가 `FEATURE_COLUMNS`에 있음,
  구역 매핑 누락 없음. 결과 42청크(26 + 16), 임베딩은 한 번에 만든다(기존과 같음).
- 기존 청크에는 `signature`가 없다. 대조 코드는 `c.get("source") == "playbook"`로
  플레이북 항목만 고른다.

## Part B — 서명 대조 (`src/rag/playbook.py`, 신규)

```python
TOP_N = 5              # 대조에 쓰는 상위 피처 수
WEAK_Z = 10.0          # 상위 1 피처의 z가 이 미만이면 "약한 신호"
COMPOSITE_RATIO = 0.5  # 다른 그룹 최고 점수가 1위의 이 비율 이상이면 "복합 징후"

VERDICT_KO = {"confirmed": "확정", "composite": "복합 징후", "weak": "약한 신호",
              "unknown": "판단 불가", "none": "이상 없음"}


def coverage(signature: list[str], contributions: list[dict], top_n: int = TOP_N) -> float:
    """상위 top_n 피처를 1/순위로 가중해, 서명이 설명하는 비율(0~1). 소수 둘째 자리로 반올림."""


def match_playbook(contributions: list[dict], corpus: list[dict]) -> dict | None:
    """플레이북 항목이 코퍼스에 없으면 None.
    1. 항목별 coverage. best = 최고 점수, 동률이면 코퍼스 순서(구역 대표 우선).
    2. best == 0                       → verdict "unknown", situation·category None.
    3. contributions[0].z < WEAK_Z     → "weak" (best 항목은 참고로 채움).
    4. other = best와 fault_category가 다른 항목 중 최고 coverage.
       other >= COMPOSITE_RATIO * best  → "composite".
    5. 아니면 "confirmed".
    반환: {"verdict", "verdict_ko", "situation", "category", "coverage",
           "matched_features": 상위 top_n 중 서명에 있는 코드(순위순),
           "alternatives": 같은 구역의 다른 항목 이름 최대 2개(coverage 내림차순, 동률은 순서),
           "other_group": {"situation", "category", "coverage"} | None,
           "top_z": contributions[0].z}"""


NO_FAULT = {"verdict": "none", "verdict_ko": "이상 없음", "situation": None, "category": None,
            "coverage": 0.0, "matched_features": [], "alternatives": [], "other_group": None,
            "top_z": None}
```

임베딩을 쓰지 않는다. 같은 입력이면 같은 결과가 나오고, 키 없이도 돈다.

기준값 근거(타임라인 600건): 고장 시나리오 불량 구간의 상위 z 최소는 21.4,
변형 전 오탐은 4.4~5.2 → `WEAK_Z = 10`. 고장의 다른 그룹 비율 최대는 0.33,
온도의 중앙값은 0.67 → `COMPOSITE_RATIO = 0.5`.

## Part C — 청크 선택과 가이드 (`src/rag/guide.py`, `generation.py`)

### `predict_experiment` (`src/serving/inference.py`)

```python
fault = NO_FAULT if predicted_label_text == "good" else (
    match_playbook(feature_contributions, rag_corpus) if rag_corpus else None)
guide = build_guide({..., "fault": fault}, rag_corpus, rag_index, openai_client)
return {..., "fault": fault, "guide": guide}
```

`fault`는 코퍼스가 없거나 플레이북 항목이 없으면 `None`. 기존 9개 키는 그대로.

### `build_guide`

```python
def build_guide(predict_result, rag_corpus, rag_index, openai_client) -> dict | None:
    good → GOOD_GUIDE (기존)
    코퍼스·인덱스·클라이언트 없음 → None (기존)
    fault = predict_result.get("fault")
    fault is None → 기존 경로(질의 임베딩 top-3)         # 플레이북 없는 코퍼스 호환
    chunks = select_chunks(fault, rag_corpus, rag_index, embed_fn, predict_result)
    return generate_guide(predict_result, chunks, openai_client, fault)
    예외 → None (기존)
```

### `select_chunks(fault, corpus, index, embed_fn, predict_result)`

| verdict | 청크 |
|---|---|
| confirmed | 선택 항목 + `alternatives` 항목(≤2) + 같은 카테고리의 비플레이북 cause 청크를 질의 임베딩 순위로 ≤2 + 안전 3 |
| composite | 선택 항목 + `other_group` 항목 + 구역 4 항목 전부 + 안전 3 |
| weak | 선택 항목(참고) + 구역 4 항목 전부 + 안전 3 |
| unknown | 구역 4 항목 전부 + 안전 3 |

임베딩 호출은 confirmed일 때 한 번(기존과 같은 횟수). 나머지는 호출 없음.

### 프롬프트 (`generation.py`)

`SYSTEM_PROMPT`에서 "tool_condition을 학습한 적이 없다"는 문장을 뺀다(LLM이
공구 마모를 끼워 넣는 원인). 대신: 판정은 센서 신호의 통계적 이상만 본 것이므로
단정하지 말 것 / **시스템 판정과 상황을 원인의 중심에 둘 것** / 참고 문서에 없는
원인을 덧붙이지 말 것 / 같은 구역의 다른 후보는 "함께 확인할 것"으로만 / 확정이
아니면 조치보다 확인 절차를 앞세울 것. JSON 스키마는 그대로.

`_build_user_prompt(predict_result, chunks, fault)`에 판정 줄을 더한다.

```
시스템 판정: 확정 — 공구 마모 (센서 서명 일치 0.80, 일치 센서: S_OutputCurrent, S_CurrentFeedback, S_OutputPower)
같은 구역의 다른 후보(현장 확인으로 구분): 스핀들 베어링 손상, 절삭유 부족·과열
```
```
시스템 판정: 복합 징후 — 공구 마모(0.66)와 이송축 과부하(0.44)가 함께 나타남. 여러 센서가 같이 이동하는 드리프트일 수 있음. 라벨·추이 확인을 권할 것.
```
```
시스템 판정: 약한 신호 — 상위 센서 z 4.4 (기준 10 미만). 보류·재확인을 권할 것. 참고 상황: 고정구 풀림·채터
```
```
시스템 판정: 판단 불가 — 서명이 일치하는 상황 없음. 상위 센서: ... 현장 확인을 권할 것.
```

`generate_guide(predict_result, chunks, client, fault=None)` — `fault`가 None이면
기존 프롬프트(거부 경로 `generate_cause_guide`는 변경 없음).

### 재학습 거부 경로

`build_cause_guide`는 카테고리로 코퍼스를 거르므로 플레이북 항목이 자연히
포함된다(tool_wear 5개, vibration_backlash 4개 추가). 코드 변경 없음.

## `/predict` 응답 예 (불량, 확정)

```json
{
  "predicted_label": 1, "predicted_label_text": "bad", "score": 2.63, "threshold": 0.8566,
  "method": "mean", "feature_contributions": [...], "model_version": "1", "mlflow_run_id": "...",
  "fault": {
    "verdict": "confirmed", "verdict_ko": "확정",
    "situation": "공구 마모", "category": "tool_wear", "coverage": 0.80,
    "matched_features": ["S_OutputCurrent", "S_CurrentFeedback", "S_OutputPower"],
    "alternatives": ["스핀들 베어링 손상", "절삭유 부족·과열"],
    "other_group": {"situation": "공구 돌출 과다·홀더 불량", "category": "vibration_backlash", "coverage": 0.22},
    "top_z": 428.9
  },
  "guide": {"cause_estimate": "...", "confidence_note": "...", "recommended_actions": ["..."],
            "safety_notes": ["..."], "sources": [{"title": "팀 시나리오 플레이북(자체 작성)", "url": "rag/sources/scenario_playbook.md"}]}
}
```

정상이면 `fault`는 `NO_FAULT`, `guide`는 기존 `GOOD_GUIDE`.

## 에러 처리

- 플레이북 파싱 실패(관련 센서 줄 없음, 모르는 코드) → 빌드 스크립트가 assert로
  멈춘다. 서빙 시점에는 일어나지 않는다.
- `match_playbook`은 예외를 내지 않는다(순수 계산). 기여도가 비어 있으면 unknown.
- 임베딩·LLM 실패는 기존처럼 `guide = None`. `fault`는 남는다.

## 테스트 전략 (TDD)

- `tests/rag/test_playbook.py`: `coverage` 계산(1/순위 가중, 상위 5개만, 반올림).
  `parse_playbook`으로 실제 파일 파싱 — 16항목, 구역 4개, 관련 센서 코드가 모두
  `FEATURE_COLUMNS`에 있음, 구역 대표 4개 이름. `match_playbook` — 합성 3건 +
  experiment_07 기록으로 상황 4/4, 동률은 순서, alternatives 2개, 약한 신호(z<10),
  복합 징후(other ≥ 0.5·best), unknown(일치 0), 플레이북 없는 코퍼스 → None.
- `tests/rag/test_guide.py`: `select_chunks` verdict별 구성(임베딩은 confirmed에서만
  호출됨을 가짜 embed_fn 호출 횟수로 확인), `fault` 없는 결과는 기존 경로.
- `tests/rag/test_generation.py`: 프롬프트에 판정 줄, `fault=None`이면 기존 문구,
  시스템 프롬프트에 tool_condition 문장 없음.
- `tests/serving/test_inference.py`, `test_app.py`: 응답에 `fault`, good → `none`,
  코퍼스 없음 → `null`, 기존 9개 키 불변.

## 검증 방법

1. `uv run --env-file .env python rag/build_corpus.py` → 42청크, 분포 출력.
2. 오프라인 채점 `rag/eval_playbook.py`(신규, LLM 호출 없음): 합성 3건 +
   experiment_07의 상황·판정, 타임라인 3종 × 40일의 구간별 판정 표. 목표: 위
   배경 절의 표와 같은 결과. 결과를 `data/rag/eval_playbook.json`에 저장하고
   요약을 이 문서의 정정 절에 적는다. `nice -n 19`, 5분 내외.
3. 라이브: 서버를 `--env-file .env`로 띄워 합성 불량 3건 + experiment_07 +
   experiment_12를 `/predict`로 돌리고 `docs/examples/predict_response_*.json`
   5개를 갱신·추가한다. 기대: 확정 4건의 상황이 주입한 원인과 일치, 원인 문장에
   다른 상황이 섞이지 않음, 정상 1건은 `none`.
4. 키 없이 같은 요청 → `fault` 있음, `guide` null.
5. 회귀: 기존 테스트 169개 + 신규 전부 통과.

## 알려진 한계

- 플레이북은 시나리오를 알고 쓴 자체 문서이고 검증도 합성 시나리오 중심이다.
  현장 정확도의 근거가 아니라 데모 흐름의 검증이다.
- `WEAK_Z`, `COMPOSITE_RATIO`는 이번 시나리오에서 고른 값이다. 온도 배치 79건 중
  10건은 여전히 "확정 이송축 과부하"로 나간다.
- 상황 선택은 상위 5개 피처만 본다. 서명에 없는 센서가 튀는 새 고장은 unknown이
  된다(의도한 동작).
- 이송축 과부하·윤활 불량·절삭 조건 과다처럼 센서로 구분 안 되는 짝은 대표
  항목이 뽑히고 나머지는 후보로만 나간다.
- 가이드 문장은 호출마다 다르다. `fault`는 결정적이다.

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `rag/sources/scenario_playbook.md` (신규) | 상황 16개 |
| `src/rag/playbook.py` (신규) | `parse_playbook`, `coverage`, `match_playbook`, 상수, `NO_FAULT` |
| `rag/build_corpus.py` | 플레이북 파싱·검증·메타 추가 |
| `src/rag/guide.py` | `select_chunks`, `build_guide`가 `fault` 경로 사용 |
| `src/rag/generation.py` | 시스템 프롬프트 개정, 판정 줄, `generate_guide(..., fault=None)` |
| `src/serving/inference.py` | `fault` 계산·반환 |
| `rag/eval_playbook.py` (신규) | 오프라인 채점 |
| `tests/rag/test_playbook.py` (신규), 기존 테스트 4개 파일 | 위 변경 반영 |
| `docs/examples/predict_response_*.json` | 5개 갱신·추가 |
| `README.md`, `docs/STRUCTURE.md` | §2-1 `fault` 필드, §2-4 플레이북, 폴더 표 |

변경하지 않는 것: `src/retraining/*`, `monitoring/*`, `src/monitoring/*`,
`src/lstm_ae/*`, `src/preprocessing/*`, `src/rag/{query,retrieval,features}.py`,
`generate_cause_guide`.

## 실행 결과에 따른 정정 (2026-09-03, 구현 후)

`rag/eval_playbook.py` 결과(`data/rag/eval_playbook.json`, champion v1, 임베딩·LLM
없음). 기록 4건: 합성 tool_wear → 확정 공구 마모(0.80), 합성 feed_overload →
확정 이송축 과부하(0.66), 합성 vibration_backlash → 확정 고정구 풀림·채터(1.00),
실제 experiment_07 → 확정 이송축 과부하(0.66).

| 시나리오 | 구간 | 불량 판정 | 확정 | 복합 징후 | 약한 신호 | 확정 상황 |
|---|---|---|---|---|---|---|
| tool_wear | Day 11-20 | 11/50 | 6 | 3 | 2 | 공구 마모 6 |
| tool_wear | Day 21-40 (실제 불량) | 93/100 | 93 | 0 | 0 | 공구 마모 93 |
| fixture_loosening | Day 11-20 | 9/50 | 6 | 0 | 3 | 고정구 풀림·채터 6 |
| fixture_loosening | Day 21-40 (실제 불량) | 77/100 | 77 | 0 | 0 | 고정구 풀림·채터 77 |
| temperature | Day 11-20 | 6/50 | 0 | 0 | 6 | - |
| temperature | Day 21-40 (실제 정상) | 79/100 | 10 | 62 | 7 | 이송축 과부하 10 |
| 셋 공통 | Day 1-10 (변형 전) | 6/50 | 0 | 0 | 6 | - |

배경 절의 실험 표와 다른 점: tool_wear Day 11-20의 확정 1건이 실험에서는 스핀들
베어링 손상이었는데 공구 마모로 바뀌었다. 서명을 `관련 센서` 줄에 명시하면서
베어링 항목의 서명이 S_ActualVelocity·S_ActualAcceleration 중심으로 좁아진 결과다.
그 외 숫자는 모두 같다.

온도 시나리오에서 "확정 이송축 과부하"로 나간 10건은 **전부 원본 실험 2(이송
속도 20)에서 만든 배치**다. 이 실험은 변형 전 구간에서도 매번 오탐(z 4.4, 약한
신호)이 나는, 이전에 기록한 모델의 알려진 사각지대다. 온도 변형이 더해지면 X·Y축
출력 전류만 상위에 남고 스핀들 신호가 5위 밖으로 밀려 다른 그룹 점수가
0.15~0.20으로 떨어지므로 배치 하나로는 이송축 과부하와 구분되지 않는다. 나머지
원본 실험 7개에서 온 배치는 모두 복합 징후였다. 이 10건을 잡으려고
`COMPOSITE_RATIO`를 0.2 아래로 내리면 실제 고장(다른 그룹 최대 0.33)까지 복합
징후로 바뀌므로 기준값은 손대지 않는다.

### 라이브 `/predict` (키 있음, `docs/examples/predict_response_*.json`)

| 입력 | 판정 | verdict | situation (일치) | cause_estimate 요지 | 출처 |
|---|---|---|---|---|---|
| synthetic tool_wear | bad | confirmed | 공구 마모 (0.80) | 공구 마모가 원인일 가능성 | 플레이북, OSHA |
| synthetic feed_overload | bad | confirmed | 이송축 과부하 (0.66) | 이송축 과부하로 X·Y축 출력 전류 상승 가능성. 윤활 불량·절삭 조건 과다는 함께 확인 | 플레이북, Sandvik, OSHA |
| synthetic vibration_backlash | bad | confirmed | 고정구 풀림·채터 (1.00) | 고정구 풀림이나 채터로 추정 | 플레이북, Sandvik, OSHA |
| experiment_07 | bad | confirmed | 이송축 과부하 (0.66) | 이송축 과부하. 절삭 조건 과다·윤활 불량 가능성 병기 | 플레이북 |
| experiment_12 | good | none | - | 이상 없음 | - |

원인 문장에 다른 구역의 상황이 섞인 경우는 없다. 서버 로그에 `RAG 가이드 생성
실패` 0건. 키 없이 띄운 서버에서 같은 tool_wear 요청 → `fault`는 확정 공구
마모(0.80) 그대로, `guide`는 `null`.

관찰: tool_wear 가이드의 `confidence_note`가 서명 일치 0.80을 "맞을 확률이 높다"로
읽었다. 일치도는 확률이 아니므로 프롬프트에 그 구분을 한 줄 더하는 것이 다음
개선 후보다(이번 범위 밖).
