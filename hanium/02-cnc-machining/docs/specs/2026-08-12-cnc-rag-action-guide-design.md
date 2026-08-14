# CNC RAG 기반 현장 조치 가이드 설계

- 날짜: 2026-08-12
- 상태: 설계 완료, 구현 전

## 배경

프로젝트의 최종 목표("모델 판정 + 피처별 재구성오차 → RAG 기반 원인 설명 →
현장 조치 가이드")를 완성하는 마지막 서브프로젝트다. 지금까지:
- `/predict`가 판정 + `feature_contributions`(피처별 z-score 기여도)를 반환함
- 합성 이상/정상 시나리오 6개로 이 파이프라인이 실제로 의도대로 동작함을 검증함
- CNC 가이드북(`data/guide/`)엔 "원인/조치" 매핑이 없음을 확인함(키워드 검색,
  공정 설명·실습 위주) — RAG가 검색할 코퍼스를 직접 구축해야 함

사용자와 논의를 거쳐 다음을 확정했다:
- 코퍼스는 **공개된 외부 문서**(국내 KOSHA 안전기술지침 + 해외 CNC 트러블슈팅
  자료)를 웹에서 조사해 구축한다 — 직접 저작한 텍스트가 아니라 실제 출처가
  있는 문서 기반.
- 임베딩·생성 둘 다 **OpenAI API**를 쓴다(로컬 임베딩 모델 대신 API로 통일).
- 검색은 **FAISS**(서버 없는 라이브러리 모드, `IndexFlatIP`)를 쓴다 — 코퍼스가
  작아 ANN 인덱스 자체의 이점은 없지만, 잘 알려진 라이브러리를 쓴다는 것 자체가
  "벡터DB 활용"의 정당한 근거가 되고 서버 운영 부담은 없다.
- Knowledge Graph(Neo4j 등)는 도입하지 않는다 — 결함유형이 3종류뿐인 도메인에
  그래프DB는 명백한 오버킬이라 판단(사용자 확인 완료). 대신 청크마다 구조화
  태그(`fault_category`, `content_type`)를 붙여 그래프의 이점(구조 기반 필터링)을
  가볍게 흡수한다.
- 이번 세션 초반에 확립한 원칙 — `tool_condition`이 모델 입력에 없어 공구마모를
  직접 학습한 적이 없으므로, RAG는 "공구마모입니다"라고 단정하지 말고 확신도를
  낮춰("~일 가능성이 있습니다") 서술해야 한다 — 를 생성 프롬프트에 명시적으로
  반영한다.

## 목표 / 비목표

**목표**
- 실제 공개 문서(KOSHA + 해외 트러블슈팅)로 코퍼스를 구축하고 FAISS로 검색
  가능하게 만든다.
- `/predict`가 "불량" 판정을 낼 때, 검색된 문서 + 판정 정보를 OpenAI에 넣어
  현장 조치 가이드를 생성하고 응답에 포함시킨다.
- 확신도를 낮춘 서술 원칙이 실제 생성된 문장에 반영되는지 확인한다.
- RAG 실패(API 에러 등)가 핵심 판정 기능(`predicted_label` 등)을 막지 않게 한다.

**비목표**
- Knowledge Graph 구축 (위 배경에서 이미 기각)
- 코퍼스 자동 업데이트/새 문서 크롤링 파이프라인 (이번엔 1회성 구축)
- 판정 자체의 정확도 개선 (이건 별개 트랙, OOD 게이트 관련 논의에서 다룰 사안)

## Part A — 코퍼스 구축 (오프라인, `02-cnc-machining/rag/build_corpus.py`)

**문서 소스** (브레인스토밍 단계에서 실제 접근성 검증 완료):
- ~~KOSHA `M-4-2016`~~ — **접근 불가로 제외**: KOSHA 사이트가 자동화 접근을
  차단함(WebFetch 시도 시 403 Forbidden, 우회 링크도 전부 죽은 링크). 나중에
  필요하면 사용자가 브라우저로 직접 PDF를 받아 `sources/`에 추가하는 수동
  경로로 처리(이번 구현 범위에는 포함 안 함).
- **OSHA(미국 산업안전보건청) lockout/tagout 공식 해석 문서** (영어) —
  안전조치용으로 KOSHA 대체. `https://www.osha.gov/laws-regs/standardinterpretations/2005-08-24`
  — WebFetch로 실제 접근 확인 완료.
- **Sandvik Coromant 밀링 트러블슈팅 가이드** (영어) — 원인+조치용, 진동/공구마모/
  칩막힘 3개 카테고리 전부 다룸(약 4,500단어). `https://www.sandvik.coromant.com/en-us/knowledge/milling/troubleshooting-milling`
  — WebFetch로 실제 접근 확인 완료.

**원문 확보 방식**: 브레인스토밍 세션에서 위 두 문서의 본문을 WebFetch로 이미
가져와 `02-cnc-machining/rag/sources/`에 로컬 텍스트 파일로 저장해둔다(아래 Task
목록 참고). `build_corpus.py`는 **라이브 웹 fetch를 하지 않고 이 로컬 파일만
읽는다** — 매번 재실행할 때 외부 사이트의 봇 차단·페이지 변경에 의존하지 않게
하기 위함(기존 `data/guide/`의 로컬 PDF 관례와 동일한 패턴).

**처리 절차**: `02-cnc-machining/rag/sources/*.md` 두 파일은 이미 `##`/`###`
마크다운 헤딩으로 원인/해결책이 소주제별로 정리돼 있으므로(브레인스토밍 단계에서
원문을 이 구조로 저장해둠), 복잡한 문단 분리 로직 없이 **`###`(Sandvik) 또는
`##`(OSHA) 헤딩 단위를 그대로 청크 경계로 삼는다**:
1. 마크다운을 헤딩 기준으로 분할 → 청크 = 헤딩 제목 + 그 아래 본문.
2. 각 청크에 태그 부여:
   - `fault_category`: 소스 파일 상단 주석(`fault_category candidates: ...`,
     Sandvik은 섹션 번호로 구분: 1=vibration_backlash, 2=tool_wear,
     3=feed_overload / OSHA는 `general`)에 따라 수동 지정
   - `content_type`: `cause`(원인 설명) | `action`(조치) | `safety`(안전수칙,
     OSHA 문서는 전부 `safety`)
4. OpenAI `text-embedding-3-small`로 청크별 임베딩 계산(1회, 배치 호출).
5. 저장:
   - `02-cnc-machining/data/rag/corpus.json` — 청크 원문 + 태그 + 출처(제목/URL)
   - `02-cnc-machining/data/rag/corpus.index` — FAISS `IndexFlatIP` 직렬화

**주의**: `02-cnc-machining/data/`는 이 프로젝트 전체에서 `.gitignore` 대상이다(README에
설명된 `cnc-data.tar.gz` 수동 전달 관례 — `data/model/model.pt`,
`data/processed/train.csv` 등 기존 산출물과 동일 취급). `sources/*.md`,
`corpus.json`, `corpus.index`도 git에는 안 올라가고 이 관례를 그대로 따른다 —
다른 PC에서 쓰려면 `cnc-data.tar.gz`에 포함시켜 옮기거나 `build_corpus.py`를
그 PC에서 다시 실행해야 한다(소스 md 파일 자체는 이번에 로컬에 저장해두므로
재실행 시 네트워크 필요 없음, OpenAI 임베딩 API만 필요).

## Part B — 런타임 검색+생성 (`src/rag/` 신규, `/predict`에 통합)

이번 서브프로젝트는 (loocv/synthetic과 달리) **서빙 파이프라인의 정식 일부**라
`src/`에 새 패키지로 추가하고, 기존 관례대로 pytest 단위테스트를 갖춘다.

### `src/rag/features.py`
피처 코드 → 한국어 설명 매핑(신규 사전, 최소한 시나리오 관련 피처는 전부 포함):
```python
FEATURE_DESCRIPTIONS = {
    "S_OutputCurrent": "스핀들 출력 전류",
    "S_OutputPower": "스핀들 출력 파워",
    "S_CurrentFeedback": "스핀들 전류 피드백",
    "X_OutputCurrent": "X축 출력 전류",
    "X_OutputPower": "X축 출력 파워",
    "Y_OutputCurrent": "Y축 출력 전류",
    "Y_OutputPower": "Y축 출력 파워",
    "X_ActualPosition": "X축 실제 위치",
    "Y_ActualPosition": "Y축 실제 위치",
    "Z_ActualPosition": "Z축 실제 위치",
    "X_ActualVelocity": "X축 실제 속도",
    "Y_ActualVelocity": "Y축 실제 속도",
    "Z_ActualVelocity": "Z축 실제 속도",
    # 나머지 피처는 fallback으로 원본 컬럼명을 그대로 노출
}
```
`describe_feature(code: str) -> str`: 매핑에 있으면 설명, 없으면 원본 컬럼명 반환.

### `src/rag/query.py`
`build_query(feature_contributions: list[dict], top_n: int = 3) -> str`:
상위 `top_n`개 `feature_contributions`(이미 z-score 내림차순 정렬되어 있음,
`serving/inference.py`의 `rank_feature_contributions` 참고)를 문장으로 변환.
예: `"다음 센서 값이 정상 대비 크게 벗어났습니다: 스핀들 출력 전류(z=36.1),
스핀들 전류 피드백(z=35.9), 스핀들 출력 파워(z=21.1)"`

### `src/rag/retrieval.py`
`search(query: str, corpus, index, top_k: int = 3) -> list[dict]`:
쿼리를 OpenAI 임베딩으로 변환 → FAISS `index.search()`로 top-k 청크 검색 →
`corpus.json`에서 해당 청크의 원문+태그+출처를 붙여 반환.

### `src/rag/generation.py`
`generate_guide(predict_result: dict, retrieved_chunks: list[dict]) -> dict`:
OpenAI Chat Completion API 호출. 모델명은 환경변수 `OPENAI_CHAT_MODEL`로 받고
기본값 `gpt-4o-mini`(비용 효율적이고 JSON 출력 강제가 안정적인 모델) — 구현
시점에 이 모델이 여전히 사용 가능한지 확인하고, 아니면 그 시점의 동급 모델로
교체한다. 시스템 프롬프트에 반드시 포함:
- "이 모델은 `tool_condition`(공구 마모 여부)을 입력으로 학습한 적이 없다 —
  간접적인 센서 신호(전류/파워 등)로만 추정하는 것이므로, 원인을 단정하지 말고
  '~일 가능성이 있습니다', '~로 추정됩니다' 같은 확신도를 낮춘 표현을 써라."
- 검색된 청크 원문(출처 포함)과 판정 정보(어떤 피처가 얼마나 벗어났는지)를
  컨텍스트로 제공.
- 출력 형식(JSON)을 강제해 아래 스키마로 파싱.

반환 스키마:
```json
{
  "cause_estimate": "공구 마모 가능성이 있습니다 (추정)",
  "confidence_note": "이 모델은 공구 마모 여부를 직접 학습하지 않았으며, 전류·파워 신호로만 간접 추정합니다.",
  "recommended_actions": ["가공을 일시 중단하고 공구 상태를 육안 점검하세요"],
  "safety_notes": ["공구 교체 시 반드시 전원을 차단하세요"],
  "sources": [{"title": "Sandvik Coromant Milling Troubleshooting", "url": "https://..."}]
}
```

### `/predict` 통합 (`src/serving/inference.py`, `src/serving/app.py` 수정)
- `predicted_label_text == "bad"`: 위 파이프라인(`build_query` → `search` →
  `generate_guide`) 전체 실행, 결과를 `guide` 필드에 포함.
- `predicted_label_text == "good"`: LLM 호출 생략, 고정 응답:
  ```json
  "guide": {"cause_estimate": "이상 없음", "confidence_note": null,
             "recommended_actions": [], "safety_notes": [], "sources": []}
  ```
- **에러 처리**: OpenAI API 호출 실패(네트워크/키 누락/응답 파싱 실패) 시
  `guide: null` + 서버 로그(`print` 또는 기존 관례에 맞는 로깅) 남기고, 판정
  자체(`predicted_label`, `score`, `feature_contributions` 등)는 정상 반환 —
  RAG 실패가 핵심 기능을 막지 않는다.
- 기동 시(`load_model_state()` 확장 또는 별도 lifespan 단계): `corpus.json` +
  `corpus.index`를 로드해 메모리에 유지(매 요청마다 다시 안 읽음). 없으면(코퍼스
  미구축) 서버는 뜨되 `guide`는 항상 null.

## 신규 의존성 / 환경변수

| 항목 | 내용 |
|---|---|
| 의존성 추가 | `openai`(Python SDK), `faiss-cpu` — `02-cnc-machining/pyproject.toml` |
| 환경변수 | `OPENAI_API_KEY` — 없으면 서버 기동은 되지만 `guide`는 항상 null |
| 코퍼스 재구축 | `build_corpus.py`는 네트워크(문서 fetch) + OpenAI API(임베딩) 필요, 수동 실행 |

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `02-cnc-machining/rag/build_corpus.py` | 신규 — 코퍼스 구축 스크립트(1회성이지만 재실행 가능, `src/`에 안 넣음) |
| `02-cnc-machining/rag/sources/*.md` | 신규 — 브레인스토밍 단계에서 확보한 원문(Sandvik/OSHA). `data/` 밖에 둬서 **git 추적됨**(재fetch 없이 재현 가능하게) |
| `02-cnc-machining/data/rag/corpus.json`, `corpus.index` | 신규 — 구축 산출물, `.gitignore` 대상(`data/model`, `data/processed`와 동일 취급) |
| `02-cnc-machining/src/rag/features.py` | 신규 — 피처 설명 사전 |
| `02-cnc-machining/src/rag/query.py` | 신규 — 질의 생성 |
| `02-cnc-machining/src/rag/retrieval.py` | 신규 — FAISS 검색 |
| `02-cnc-machining/src/rag/generation.py` | 신규 — OpenAI 호출 + 프롬프트 |
| `02-cnc-machining/src/serving/inference.py` | 수정 — `predict_experiment()`가 `guide` 필드 포함하도록 확장 |
| `02-cnc-machining/src/serving/app.py` | 수정 — 기동 시 코퍼스/인덱스 로드, `OPENAI_API_KEY` 확인 |
| `02-cnc-machining/pyproject.toml` | 수정 — `openai`, `faiss-cpu` 의존성 추가 |

## 테스트 범위

**loocv/synthetic과 달리 이번엔 정식 pytest 단위테스트를 작성한다** — `src/`에
들어가는 서빙 파이프라인 정식 코드라 기존 70개 테스트와 같은 관례를 따른다.

- `features.py`: `describe_feature()` 매핑 존재/fallback 케이스 — 순수 함수, API 불필요
- `query.py`: `build_query()`가 상위 N개를 올바른 문장으로 조립하는지 — 순수 함수
- `retrieval.py`: 가짜 소규모 FAISS 인덱스(테스트 안에서 생성) + 스텁 임베딩으로
  top-k가 올바르게 나오는지 — 실제 OpenAI 호출 없이 임베딩 함수를 스텁으로 교체
- `generation.py`: OpenAI 클라이언트를 스텁으로 교체해 프롬프트에 확신도 원칙
  문구가 포함되는지, 응답 파싱이 스키마대로 되는지 — 실제 API 호출 없음
- `/predict` 통합: 기존 서빙 테스트 관례(`TestClient` + 의존성 주입 스텁) 그대로
  따라, `guide` 필드가 good/bad 분기와 에러 상황에서 올바른지 확인

## 검증 방법

1. `build_corpus.py` 실행 → `corpus.json`/`corpus.index` 생성 확인, 청크 개수와
   태그 분포 로그 출력.
2. pytest 전체 통과(기존 70개 + 신규 테스트).
3. 실제 champion 모델 + 합성 이상 시나리오(`synthetic/scenarios/tool_wear.csv`
   등)로 `/predict`를 실제로 호출해, `guide.cause_estimate`가 해당 시나리오와
   맞는 카테고리를 가리키는지, 확신도를 낮춘 문구("가능성", "추정")가 실제로
   포함되는지 육안 확인.
4. `OPENAI_API_KEY`를 일부러 비워 서버를 띄워 `guide: null` + 판정은 정상인지
   확인(에러 처리 검증).
5. 결과를 사용자에게 보고.
