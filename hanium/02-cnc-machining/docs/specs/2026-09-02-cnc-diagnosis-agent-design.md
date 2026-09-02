# 진단 에이전트 + 카테고리 기반 RAG 설계

작성 2026-09-02. 선행 스펙: `2026-08-12-cnc-rag-action-guide-design.md`(`/predict`
가이드), `2026-08-26-cnc-cause-estimation-design.md`(거부 경로 원인 추정),
`2026-09-02-cnc-two-sided-gate-design.md`.

## 배경

### 실측으로 확인된 문제 — 검색이 모델의 정보를 버린다

`/predict`의 가이드는 "상위 3개 센서 이름을 넣은 한 문장"을 임베딩해 코퍼스
26청크와 코사인 유사도로 3개를 뽑고, 그 문서를 읽은 LLM이 답을 쓴다. 2026-09-02
추적 결과, 무엇을 넣어도 KAMP 가이드북의 "설정값 대비 실제값, 전류·전압·토크
신호 수집" 청크(공구마모 태그)가 1위였다. 이 청크는 센서 이름을 나열한 문장과
항상 가장 비슷하고 본문에 "공구수명"이 있어서, LLM이 진동 신호를 보고도 공구
마모로 답한다.

| 입력 | 상위 피처 (모델은 맞게 짚음) | 검색 1위 | 가이드의 원인 |
|---|---|---|---|
| 진동 합성 시나리오 | Z·Y·X축 실제 속도 | KAMP 신호 수집 청크 (tool_wear) | "공구 마모 또는 파손" |
| experiment_07 (실제 불량) | Y·X축 출력 파워, Z축 속도 | 같은 청크 | "공구 마모" |

즉 1·2단계(판정, 기여도)는 정확한데 4단계(검색)가 그 정보를 버린다. 재학습
거부 경로는 기여도로 카테고리를 먼저 정하고 그 카테고리 문서만 LLM에 넘기므로
같은 문제가 없었다(09-02 라이브: 10/10 정답).

### 왜 "에이전트"인가 — 판단은 규칙, 작성은 에이전트

팀원 Spring 앱의 "작업 조치 보고서"가 판정 하나가 아니라 추이·과거 사례·근거
문서를 종합한 진단을 필요로 한다. LLM에게 원인 판단까지 맡기면 위 수렴 문제가
더 그럴듯한 문장으로 포장될 뿐이므로, **원인 카테고리는 규칙으로 정하고 LLM은
읽기 전용 도구로 이력·문서·과거 사례를 모아 보고서를 쓰는 역할**로 제한한다.
카테고리와 수치는 결정적, 문장만 생성적이다.

## 사용자 결정 (2026-09-02 브레인스토밍)

- 소비자: **팀원 Spring 앱이 API로 호출.** 출력 JSON 스키마는 팀과 합의 대상.
- 입력: **`/predict`가 돌려주는 `request_id`.** 판정·기여도를 저장해 두고 ID로
  찾는다. 이력 도구가 가능해진다.
- 응답: **동기, 최대 60초.** 도구 호출 5회 상한, 전체 45초 예산.
- 접근: **A안 — 판단은 규칙, 작성은 에이전트.** `/predict` 가이드도 카테고리
  기반으로 바꾼다. 합성 시나리오로 채점한다.

## 목표 / 비목표

**목표**

1. `/predict` 가이드가 공구마모·이송과부하·진동을 구분해 답한다(합성 3종 채점).
2. `POST /diagnose/{request_id}`가 판정·추이·드리프트·근거 문서·과거 사례를
   종합한 진단 보고서 JSON을 60초 안에 돌려준다. OpenAI 키가 없어도 폴백
   보고서를 낸다.
3. 원인 카테고리·수치는 결정적이고, 같은 `request_id`는 같은 보고서를 재사용한다.

**비목표**

- 에이전트가 무언가를 실행(파라미터 변경, 재학습 트리거)하는 것. 도구는 전부
  읽기 전용이다.
- 재학습 거부 경로의 `estimate_cause`(두 카테고리) 변경. 검증이 끝난 코드다.
  세 카테고리 규칙과의 통합은 나중 과제.
- 온도 드리프트 등 새 카테고리. 카테고리는 3개 + unknown이다.
- 코퍼스 확장. 이송과부하 문서가 2청크뿐인 것은 알려진 한계로 둔다.
- 비동기 작업 큐. 동기 한 번으로 끝낸다.
- 실제 실험(eval 14개)의 원인 정답 정의. 공구 상태(worn/unworn)는 원인이
  아니라 분포 기술에만 쓴다.

## 데이터 흐름

```
/predict  CSV → 판정 + 기여도 40개 → classify_fault → 카테고리
          → 카테고리 문서로 가이드 생성 → predict_log 저장(id 발급)
          → 응답 + request_id + fault_category

/diagnose/{id}
          predict_log 조회 → good이면 즉시 "이상 없음" 보고서
          → 에이전트: get_judgment(미리 호출) → LLM이 도구 선택
             (get_recent_trend / get_drift_status / search_docs / get_past_cases)
             ≤ 5회 → 최종 JSON → 결정 필드는 서버 값으로 덮어씀
          → 실패·키 없음 → 폴백 보고서
          → diagnosis_log 저장 → 응답
```

## Part A — 판정 저장과 식별 (`src/monitoring/logging.py`, `src/serving/app.py`)

### `predict_log` 확장

| 컬럼 | 타입 | 내용 |
|---|---|---|
| `batch_id` | TEXT | 업로드 파일명(확장자 제외) |
| `predicted_label` | INTEGER | 0/1 |
| `threshold` | REAL | 판정에 쓴 임계값 |
| `method` | TEXT | mean/max/p95 |
| `model_version` | TEXT | champion 버전 |
| `feature_contributions_json` | TEXT | 기여도 40개 배열 JSON |

`_init_db`는 새 DB에는 전체 스키마로 만들고, 기존 DB에는 `PRAGMA
table_info`로 없는 컬럼만 `ALTER TABLE ... ADD COLUMN`한다. 기존 행은 새
컬럼이 NULL이며, 진단 도구는 NULL을 "정보 없음"으로 다룬다.

```python
def log_request(
    feature_means, score, predicted_label_text, db_path,
    *, batch_id=None, predicted_label=None, threshold=None,
    method=None, model_version=None, feature_contributions=None,
) -> int:
    """새 행의 id(= request_id)를 돌려준다. 기존 4개 위치 인자는 그대로."""

def get_request(request_id: int, db_path) -> dict | None
def get_requests_before(request_id: int, n: int, db_path) -> list[dict]
    """id < request_id 인 최근 n건, 최신순."""
```

### `diagnosis_log` (신규 테이블, 같은 DB)

`request_id INTEGER PRIMARY KEY, timestamp TEXT, generated_by TEXT, report_json TEXT`.
`save_diagnosis(request_id, report, db_path)`, `get_diagnosis(request_id, db_path)`.

### `/predict` 변경

- `predict_experiment`가 `fault_category`를 계산해 결과에 포함하고(Part B)
  `build_guide`에 넘긴다.
- `log_request`에 새 필드를 채워 넣고 반환된 id를 응답에 `request_id`로 붙인다.
- 응답에 `fault_category`(문자열) 추가. 기존 9개 키는 그대로.

## Part B — 고장 카테고리 규칙 (`src/diagnosis/classify.py`, 신규)

```python
FAULT_GROUPS = {
    "tool_wear": ["S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback"],
    "feed_overload": ["X_OutputCurrent", "Y_OutputCurrent", "X_OutputPower", "Y_OutputPower"],
    "vibration_backlash": [
        "X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition",
        "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity",
    ],
}
CATEGORY_KO = {
    "tool_wear": "공구 마모", "feed_overload": "이송축 과부하",
    "vibration_backlash": "진동·백래시", "unknown": "판단 불가", "none": "이상 없음",
}
MIN_GROUP_SCORE = 3.0


def group_scores(feature_contributions: list[dict]) -> dict[str, float]:
    """그룹별로 z_score 상위 2개의 평균. 그룹 크기(3/4/6) 차이를 없앤다."""


def classify_fault(predicted_label_text: str, feature_contributions: list[dict]) -> dict:
    """{"category", "category_ko", "group_scores"}.
    good → none. 최고 점수 < MIN_GROUP_SCORE → unknown. 아니면 최고 점수 그룹.
    동점이면 FAULT_GROUPS 정의 순서(tool_wear → feed_overload → vibration_backlash)."""
```

그룹은 합성 시나리오 3종이 건드리는 컬럼과 `simulate_timeline.py`의
`WEAR_COLUMNS`/`VIBRATION_COLUMNS`에서 가져왔다. 공구마모 그룹에서 X/Y
출력 파워를 뺀 이유는 이송과부하와 겹치기 때문이며, 스핀들 세 센서만으로
공구마모 합성 사례(S_OutputCurrent z=12418)와 타임라인 시나리오가 잡힌다.
기록된 합성 6건에 적용하면 tool_wear / feed_overload / vibration_backlash /
none / none / none이 나온다.

`MIN_GROUP_SCORE = 3.0`은 "정상 대비 3σ"라는 뜻이며, 온도 드리프트처럼 세
그룹 어디에도 뚜렷이 속하지 않는 약한 신호를 unknown으로 보내기 위한 것이다.
단 온도 드리프트가 위치·전류를 함께 움직이면 vibration이나 feed로 잡힐 수
있다 — 알려진 한계.

## Part C — `/predict` 가이드를 카테고리 기반으로 (`src/rag/guide.py`, `generation.py`)

```python
def build_guide(predict_result, rag_corpus, rag_index, openai_client) -> dict | None:
    category = predict_result["fault_category"]
    if category == "none": return dict(GOOD_GUIDE)
    if rag_corpus is None or rag_index is None or openai_client is None: return None
    if category == "unknown":
        chunks = 기존 방식(전체 벡터 검색 top_k=3)
    else:
        chunks = select_category_chunks(category, predict_result, rag_corpus, rag_index, embed_fn)
    return generate_guide(predict_result, chunks, openai_client)
```

`select_category_chunks`: `fault_category == category`인 청크를 모으고, 5개를
넘으면 기존 질의문의 임베딩과 유사도로 5개를 고른다(전체 인덱스를 `top_k=
len(corpus)`로 검색한 뒤 카테고리로 거른다). 여기에 `content_type == "safety"`
청크 3개를 더한다. 임베딩 호출은 카테고리 청크가 5개를 넘을 때만 일어난다.

`_build_user_prompt`에 한 줄 추가: `"시스템 추정 카테고리: {category_ko}"`.
시스템 프롬프트에 "추정 카테고리와 다른 원인을 주장하지 말 것"을 더한다.
`generate_guide`의 JSON 스키마는 그대로다(팀원 보고서 호환).

## Part D — 진단 에이전트

### 도구 (`src/diagnosis/tools.py`, 신규)

전부 읽기 전용 순수 함수. `DiagnosisContext`(db_path, corpus, index, embed_fn,
threshold, mlflow_client)를 받는다. `TOOL_SPECS`가 OpenAI 함수 호출 스키마,
`dispatch(name, args, ctx)`가 실행한다.

| 도구 | 입력 | 반환 |
|---|---|---|
| `get_judgment` | request_id | timestamp, batch_id, label, label_text, score, threshold, ratio, model_version, fault{category, category_ko, group_scores}, top_features[5]{feature, name_ko, z_score} |
| `get_recent_trend` | request_id, n=10 | window, n_available, bad_count, category_counts, ratios(오래된 순), ratio_trend("rising"/"falling"/"flat": 앞 절반 평균 대비 뒤 절반 평균 ±10%) |
| `get_drift_status` | 없음 | `compute_drift_status(최근 10건)` 결과 요약: sufficient_data, flagged_features 상위 5, ratio_to_threshold, output_flagged |
| `search_docs` | category, top_k=5 | Part C의 `select_category_chunks`와 같은 청크 + 안전 청크: heading, text, title, url, content_type |
| `get_past_cases` | category, n=5 | MLflow `cnc-lstm-ae` run 중 `gate_decision=rejected` 이고 `estimated_cause == category`인 것 최신순: date, scenario, trigger_day, reason, action. feed_overload는 항상 빈 목록(거부 경로가 그 카테고리를 안 냄) |

### 루프 (`src/diagnosis/agent.py`, 신규)

```python
def run_diagnosis(request_id, ctx, client, model, max_tool_calls=5, budget_seconds=45) -> dict
```

1. `judgment = get_judgment(request_id)`. `label_text == "good"`이면 에이전트
   없이 `report_for_good(judgment)` 반환.
2. 메시지: 시스템 프롬프트 + `judgment`를 "이미 호출된 도구 결과"로 넣는다.
   시스템 프롬프트 규칙: 카테고리는 시스템 판정을 따른다 / 근거는 도구 결과에
   있는 것만 쓴다 / 추이·드리프트·문서·과거 사례 중 필요한 것을 골라 부른다 /
   마지막에 정해진 JSON으로 답한다 / 확신도 표현은 낮춘다(기존 프롬프트 관례).
3. `chat.completions.create(tools=TOOL_SPECS, tool_choice="auto", timeout=20)`.
   tool_calls가 있으면 `dispatch`해 결과를 붙이고 반복. 호출 횟수 5회 또는 45초를
   넘으면 도구를 더 주지 않고 최종 답을 요구한다.
4. 최종 호출은 `tool_choice="none"`, `response_format={"type": "json_object"}`.
5. `assemble_report`: LLM JSON에서 narrative, recommended_actions, safety_notes,
   sources만 받고, judgment/fault/trend/past_cases는 서버가 도구 결과로 채운다.
   LLM이 카테고리나 수치를 바꿔 써도 무시된다.
6. 어느 단계든 예외·타임아웃·파싱 실패 → `build_fallback_report`.

### 폴백 (`src/diagnosis/report.py`, 신규)

`build_fallback_report(judgment, trend, drift, docs, cases)`: 같은 도구를
파이썬이 직접 불러 템플릿으로 채운다. summary는 `"{category_ko} 징후 — 상위
센서 {names}"`, evidence는 수치 문장, recommended_actions는 카테고리 문서의
"해결책:" 뒤 문장을 2개씩 최대 5개(출처 포함), safety_notes는 안전 청크의 첫
문장. `generated_by = "fallback"`. OpenAI 키가 없을 때 `/diagnose`는 이것을
낸다.

### 보고서 스키마 (`src/diagnosis/report.py`의 `REPORT_KEYS`, 팀과 합의 대상)

```json
{
  "request_id": 123,
  "judgment": {"label": 1, "label_text": "bad", "score": 3.2464, "threshold": 0.8566,
               "ratio": 3.79, "model_version": "1", "timestamp": "...", "batch_id": "experiment_07"},
  "fault": {"category": "feed_overload", "category_ko": "이송축 과부하",
            "group_scores": {"tool_wear": 4.1, "feed_overload": 28.4, "vibration_backlash": 15.2},
            "top_features": [{"feature": "Y_OutputPower", "name_ko": "Y축 출력 파워", "z_score": 35.6}]},
  "trend": {"window": 10, "n_available": 10, "bad_count": 6,
            "category_counts": {"feed_overload": 5, "none": 4, "unknown": 1},
            "ratio_trend": "rising", "drift_flagged": true, "drift_ratio": 2.61},
  "narrative": {"summary": "...", "evidence": ["..."], "confidence_note": "..."},
  "recommended_actions": [{"action": "...", "source": {"title": "...", "url": "..."}}],
  "safety_notes": ["..."],
  "past_cases": [{"date": "2026-09-02", "scenario": "tool_wear", "reason": "...", "action": "..."}],
  "sources": [{"title": "...", "url": "..."}],
  "generated_by": "agent",
  "tool_calls": ["get_judgment", "get_recent_trend", "search_docs"],
  "model": "gpt-4o-mini",
  "elapsed_seconds": 12.4
}
```

정상 판정 보고서는 `fault.category = "none"`, narrative.summary "이상 없음",
actions·safety·past_cases·sources 빈 배열, `generated_by = "rule"`.

### 엔드포인트 (`src/serving/app.py`)

`POST /diagnose/{request_id}?refresh=false`

- `get_request`가 None → 404 `"request_id {id} 없음"`.
- champion 미로드 → 기존 503.
- `refresh=false`이고 `diagnosis_log`에 있으면 그대로 반환(Spring 재시도 안전).
- 아니면 `run_diagnosis` → `save_diagnosis` → 반환. 동기 처리이며 FastAPI가
  스레드풀에서 돌리므로 `/predict`를 막지 않는다.
- OpenAI 클라이언트가 없으면 `run_diagnosis`가 즉시 폴백 경로를 탄다.
- 모델은 `OPENAI_CHAT_MODEL`(기본 gpt-4o-mini), 기존 관례와 같다.

## 에러 처리

- 도구 실행 예외는 도구 결과에 `{"error": "..."}`로 넣어 LLM이 다른 도구로
  넘어가게 하고, 예외 자체는 루프를 끊지 않는다.
- LLM 호출 예외·타임아웃·JSON 파싱 실패·필수 키 누락 → 폴백.
- MLflow 조회 실패(`get_past_cases`) → 빈 목록 + error 필드. 진단은 계속된다.
- `predict_log`의 구버전 행(기여도 NULL) → `get_judgment`가 `fault.category =
  "unknown"`, top_features 빈 배열로 돌려주고 보고서는 폴백으로 만든다.

## 테스트 전략 (TDD)

- `tests/diagnosis/test_classify.py`: 기록된 합성 6건의 기여도로 tool_wear /
  feed_overload / vibration_backlash / none×3. 약한 신호(모든 z < 3) → unknown.
  그룹 점수는 상위 2개 평균. good → none.
- `tests/monitoring/test_logging.py`: 새 컬럼 저장·조회, `log_request`가 id 반환,
  구스키마 DB에 컬럼이 추가되는 마이그레이션, `get_requests_before` 순서,
  `diagnosis_log` 저장·재조회.
- `tests/diagnosis/test_tools.py`: 임시 DB와 가짜 코퍼스로 도구 5개. 추이의
  rising/falling/flat 경계, 과거 사례는 MLflow 클라이언트를 가짜로.
- `tests/diagnosis/test_agent.py`: `tests/rag/test_generation.py`의 가짜 클라이언트
  관례를 확장해 tool_calls 시퀀스를 재생. 호출 순서 기록, 5회 상한, 예산 초과,
  JSON 파싱 실패 → 폴백, LLM이 카테고리를 바꿔 써도 서버 값 유지, good →
  에이전트 미호출.
- `tests/serving/test_app.py`: `/predict`에 `request_id`·`fault_category`,
  `/diagnose` 404·캐시·refresh·폴백(클라이언트 None).
- `tests/rag/test_guide.py`: 카테고리 청크 선택, unknown 폴백, 프롬프트에
  카테고리 줄.

## 검증 방법

1. 오프라인 채점 `diagnosis/eval_synthetic.py` → `diagnosis/summary.json`:
   합성 6건 카테고리 정답률(목표 6/6)과 실제 eval 14건의 카테고리 분포 및
   공구 상태(worn/unworn) 교차표(기술 통계, 목표값 없음).
2. 라이브: 서버를 `--env-file .env`로 띄워 합성 불량 3건과 experiment_07을
   `/predict` → `/diagnose`로 돌린다. 보고서 4개를 `docs/examples/diagnose_*.json`
   에 저장하고, 각각의 `tool_calls`·`elapsed_seconds`·`fault.category`를 이
   문서의 정정 절에 표로 기록한다. 기대: 합성 3건의 카테고리가 주입한 원인과
   일치, 소요 60초 미만.
3. 키 없이 같은 요청 → `generated_by = "fallback"` 보고서가 나오는지.
4. 회귀: 기존 테스트 169개 + 신규 전부 통과, `/predict` 기존 9개 키 불변.

## 알려진 한계 (미리 적어 두는 것)

- 카테고리는 3개 + unknown뿐이다. 온도 드리프트는 vibration이나 feed로 잡힐 수
  있다.
- 이송과부하 문서는 2청크라 그 카테고리의 조치가 얇고, 과거 사례도 없다.
- 실제 실험에는 원인 정답이 없어 규칙의 정확도는 합성 시나리오로만 검증된다.
- 에이전트의 문장은 호출마다 다르다. 카테고리·수치·과거 사례는 결정적이다.
- 보고서 스키마는 초안이며 Spring 쪽과 합의 후 바뀔 수 있다.
- 진단 1건당 OpenAI 호출 2~6회(gpt-4o-mini 기준 수 센트).

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `src/monitoring/logging.py` | 컬럼 6개 추가 + 마이그레이션, `log_request` id 반환, `get_request`, `get_requests_before`, `diagnosis_log` |
| `src/diagnosis/classify.py` (신규) | `FAULT_GROUPS`, `group_scores`, `classify_fault` |
| `src/diagnosis/tools.py` (신규) | 도구 5개, `TOOL_SPECS`, `dispatch`, `DiagnosisContext` |
| `src/diagnosis/agent.py` (신규) | `run_diagnosis` 루프 |
| `src/diagnosis/report.py` (신규) | 스키마, `assemble_report`, `build_fallback_report`, `report_for_good` |
| `src/serving/inference.py` | `fault_category` 계산·반환 |
| `src/serving/app.py` | `/predict` 저장 확장·`request_id`·`fault_category`, `POST /diagnose/{request_id}` |
| `src/rag/guide.py`, `generation.py` | 카테고리 청크 선택, 프롬프트 한 줄 |
| `diagnosis/eval_synthetic.py` (신규) | 오프라인 채점 |
| `tests/diagnosis/*`, 기존 테스트 3개 파일 | 위 변경 반영 |
| `docs/STRUCTURE.md`, `README.md` | 폴더 표에 `src/diagnosis/`·`diagnosis/`, `/diagnose` 실행법 |

변경하지 않는 것: `src/retraining/*`, `monitoring/drift_worker.py`,
`monitoring/simulate_timeline.py`, `src/monitoring/cause_estimation.py`,
`src/lstm_ae/*`, `src/preprocessing/*`, `rag/build_corpus.py`.
