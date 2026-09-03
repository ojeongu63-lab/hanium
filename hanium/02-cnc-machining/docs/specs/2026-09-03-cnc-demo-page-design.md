# 데모 화면 설계 — 배치 진단 흐름 + 시나리오 타임라인 + 루프 이벤트

작성 2026-09-03. 선행: `2026-09-03-cnc-playbook-guide-design.md`(fault·versions),
`2026-09-02-cnc-two-sided-gate-design.md`(루프 이벤트의 출처).

## 배경과 사용자 결정

미팅에서 "모델 결과 → 센서 근거 → 플레이북 대조 → 조치 가이드"와 "시나리오별로
시스템이 다르게 반응한다(고장은 확정·거부, 온도는 복합 징후·재보정)"를 한 화면에서
보여 준다. 작업은 회사 PC(WSL)에서 하고 push하며, 미팅은 개인 PC에서 pull 받아
진행한다.

- 하이브리드: 기록 모드가 기본, 같은 주소의 서버가 응답하면 실시간 버튼이 켜진다.
- 범위: 배치 진단 흐름 + 시나리오 타임라인 + 재학습 루프 이벤트.
- 개인 PC에서 실시간까지 준비한다.

## 목표 / 비목표

**목표**

1. pull 후 `demo/index.html`을 더블클릭하면 서버·파이썬·키·인터넷 없이 열리고, 실제
   결과(응답 예시 5건, 타임라인 3종, 루프 이벤트, 플레이북 16항목)가 전부 들어 있다.
2. 서버(`/demo`)로 열면 예시 5건과 업로드 파일을 실제 `/predict`로 다시 계산해 같은
   카드에 채운다.
3. 개인 PC 준비 절차가 README에 한 절로 있고, 회사 PC의 `data/`를 옮기는 것만으로
   실시간이 된다(재학습 불필요 — `tracking.py`가 MLflow 절대경로를 자동 수정).

**비목표**

- 타임라인 탭의 실시간 재계산(600건 추론은 미팅 중 할 일이 아니다).
- 외부 차트 라이브러리, 프레임워크, 빌드 도구. 순수 HTML/CSS/JS + 인라인 SVG.
- 판정·가이드·루프 코드 변경. 서버에는 라우트 2개만 더한다.
- 인증, 다중 사용자, 모바일 레이아웃.

## 파일

| 파일 | 역할 |
|---|---|
| `demo/template.html` | 화면 마크업·스타일·스크립트. `__DEMO_DATA__` 자리에 JSON이 들어간다 |
| `demo/build_demo.py` | 재료를 모아 `demo/index.html`을 만든다. 로그 파서 포함 |
| `demo/index.html` | 생성물. **커밋한다** — pull만 받은 PC에서 바로 열리게 |
| `src/serving/app.py` | `GET /demo`, `GET /demo/inputs/{key}` |
| `rag/eval_playbook.py` | 행에 `ratio`, `top`(상위 3 센서와 z) 추가 |
| `tests/demo/test_build_demo.py`, `tests/serving/test_app.py` | 아래 테스트 |
| `README.md` §2-9 | 미팅 데모 여는 법과 개인 PC 준비 |

## 내장 데이터 (`build_demo.py`가 만드는 JSON)

```
{
  "generated_at": "2026-09-03T15:40+09:00",
  "versions": {"playbook": "1c280cae", "corpus": "...", "chat_model": "gpt-4o-mini"},
  "playbook": [ {name, heading, category, category_ko, signature[], text} ×16, 문서 순서 ],
  "examples": [ {key, label, response} ×5 ],          # response = /predict 응답 JSON 그대로
  "scenarios": {
    "temperature": {
      "label": "온도 드리프트(재보정이 정답)", "fault_from_day": null, "days_total": 70,
      "days": [ {day, truth, ratio_mean,
                 batches: [ {index, ratio, pred, verdict, verdict_ko, situation, coverage, top: [[feature, z] ×3]} ×5 ]} ... ],
      "events": [ {day, kind, text, gate, reason, cause, actions} ... ]
    },
    "tool_wear": {..., "fault_from_day": 21, "days_total": 40}, "fixture_loosening": {...}
  }
}
```

재료와 출처:

- `examples`: `docs/examples/predict_response_*.json` 5개. key = `tool_wear`,
  `feed_overload`, `vibration_backlash`, `experiment_07`, `experiment_12`.
- `playbook`: `data/rag/corpus.json`에서 `source == "playbook"`인 청크. `category_ko`는
  구역 이름(스핀들 부하 상승 / 이송축 부하 상승 / 축 위치·속도 편차 / 고장이 아닌 변화).
- `scenarios[*].days`: `data/rag/eval_playbook.json`의 `timeline[*].rows`. 배치별 `ratio`와
  `top`이 필요해 `rag/eval_playbook.py`의 행에 두 필드를 더하고 다시 돌린다. temperature는
  `--days 70`으로 돌려 승격 구간까지 담는다(champion v1 기준 계산이라는 주석을 화면에 단다).
- `scenarios[*].events`: 워커 로그 3개(`data/monitoring/_tool_wear_20260902_v2/worker_tool_wear.log`,
  `_fixture_loosening_20260902_v2/worker_fixture_loosening.log`, `_temperature_20260902/worker_temperature.log`).
  파서가 읽는 줄:
  - `  [Day N] 트리거 발동 — 재학습 시작` → kind `trigger`
  - `  게이트: G1 ... / G2 ...` → 직후 판정 줄에 `gate` 문자열로 붙임
  - `  거부 — <사유>  (champion 유지, 사람 확인 필요)` → kind `rejected`, reason
  - `  통과 ...`, `섀도우 시작`류 → kind `shadow_started`; `승격`류 → kind `promoted`
    (정확한 문구는 구현 시 temperature 로그 Day 37·64 부근을 보고 정규식을 맞춘다)
  - `  추정 원인: <cause> / 권장 조치: [...]` → cause, actions(리스트 문자열은 그대로)
  - `Day NN  score/threshold=X  flagged=...  action=...` → 이벤트 없는 날은 무시
  ANSI 색상 코드는 걷어낸다. 로그 경로는 `build_demo.py` 상수이며 파일이 없으면 그
  시나리오의 events는 빈 배열이고 경고를 출력한다.

## 화면

공통: 상단 제목, 탭 2개, 우상단 상태 배지("기록 모드" / "실시간 연결됨 v1"). 페이지
로드 시 같은 origin의 `/health`를 3초 타임아웃으로 호출해 성공하면 실시간 UI를 켠다.
`file://`로 열렸으면 호출하지 않고 기록 모드로 고정하며 "서버로 열면 실시간 가능
(`/demo`)" 힌트를 보인다.

**탭 1 — 배치 진단 흐름.** 왼쪽 목록에서 예시 5개 중 하나를 고르면 오른쪽 카드가
채워진다. 실시간이 켜져 있으면 목록 아래에 "실시간으로 다시 계산" 버튼과 CSV 업로드
칸이 생기고, 결과가 오면 같은 카드를 갱신하며 소요 시간과 `versions`를 표시한다.
카드 순서와 내용:

1. 판정 — 정상/불량 배지, 점수·임계값·배율 막대(임계선 표시), method.
2. 센서 기여도 — z 상위 10개 가로 막대, 상위 5개(대조에 쓰이는 범위) 강조.
3. 플레이북 대조 — 16항목 표(구역별 그룹, 관련 센서, 일치도 막대). 일치도는 응답의
   `fault`와 같은 식(상위 5개 1/순위 가중)을 JS로 계산해 항목별로 보이고, 선택
   상황·후보·다른 구역 1위는 응답의 `fault` 값으로 표시한다(서버 값이 기준).
4. 판정 — verdict_ko 배지(확정/복합 징후/약한 신호/판단 불가/이상 없음)와 근거 문장
   (일치 센서, 다른 구역 점수, 상위 z와 기준 10, 절반 규칙).
5. 참고 문서 — 판정별 선택 규칙에 따라 어떤 문서가 LLM에 갔는지 이름 목록(플레이북
   항목·Sandvik·OSHA). 응답에는 청크 목록이 없으므로 규칙을 JS로 재현해 보인다.
6. 가이드 — cause_estimate, confidence_note, recommended_actions, safety_notes, sources.
   guide가 null이면 "키 없음 — fault만 계산됨" 안내.
7. versions — playbook·corpus·chat_model.

**탭 2 — 시나리오 타임라인.** 시나리오 3개 버튼. 위에 SVG 그래프: x축 Day, y축 배율,
배치 5개는 점, 일 평균은 선, 임계선 1.0, `fault_from_day`부터 배경을 연하게 칠해
"실제 불량 구간"이라 표시, 이벤트는 세로 마커(트리거 ▲, 거부 ✕, 섀도우 ◆, 승격 ★).
아래에 일별 판정 띠: 불량 판정 배치 수를 확정/복합/약한 신호/판단 불가 색으로
쌓는다. 날짜를 클릭하면 그날 배치 5개 표(index, 배율, 판정, verdict_ko, situation,
coverage, 상위 센서 3개)와 그날 이벤트 상세(게이트 문자열, 거부 사유, 추정 원인,
권장 조치)가 나온다. 시나리오 설명 한 줄과 "판정은 champion v1 기준 오프라인 계산,
이벤트는 09-02 라이브 워커 로그"라는 출처 주석을 단다.

## 서버

- `GET /demo` → `demo/index.html`을 `FileResponse`로. 파일이 없으면 404와 "demo/build_demo.py를
  먼저 실행" 메시지.
- `GET /demo/inputs/{key}` → key별 파일: `tool_wear`, `feed_overload`, `vibration_backlash` →
  `synthetic/scenarios/<key>.csv`; `experiment_07`, `experiment_12` → 데이터셋 폴더의 CSV.
  모르는 key → 404, 파일 없음(데이터셋 미배치) → 404에 경로 안내.
- 페이지의 실시간 호출은 같은 origin이라 CORS 설정이 필요 없다.

## 실시간 흐름과 오류 처리

예시 버튼: `GET /demo/inputs/{key}` → Blob → `POST /predict`(multipart) → 카드 갱신.
업로드: 파일 입력 → `POST /predict`. 타임아웃 90초. 실패하면 카드는 그대로 두고 상단에
오류 문자열(HTTP 상태·detail)을 보이고 버튼을 다시 활성화한다. `/health` 실패 시
실시간 UI를 숨긴다(기록 모드).

## 개인 PC 준비 (README §2-9에 적을 내용)

**기록 모드**: `git pull` 후 `02-cnc-machining/demo/index.html`을 브라우저로 연다. 끝.

**실시간 모드**:
1. 회사 PC에서 `tar -czf cnc-data.tar.gz --exclude=data/monitoring data` (60MB 안팎:
   dataset 15MB, processed 23MB, model 2MB, mlflow 21MB, rag 1MB). 개인 PC의
   `02-cnc-machining/` 바로 아래에 풀어 `data/` 구조가 README §1-2와 같게 한다.
   `tracking.py`가 첫 실행 때 mlflow.db의 절대경로를 이 PC 기준으로 고친다.
2. `uv sync`.
3. `.env`에 `OPENAI_API_KEY`. 없어도 판정·fault·기록 모드는 되고 guide만 null.
4. `uv run --env-file .env uvicorn serving.app:app --port 8899` → 브라우저에서
   `http://127.0.0.1:8899/demo`. WSL이면 README §3.
5. 미팅 전 점검: 상태 배지가 "실시간 연결됨"인지, 예시 하나를 다시 계산해 가이드가
   오는지, 소요 시간.

## 테스트

- `tests/demo/test_build_demo.py`: 로그 파서 — 위 문구를 담은 짧은 가짜 로그로
  trigger/rejected/promoted 이벤트와 gate·cause·actions 추출, ANSI 제거. 데이터 조립 —
  임시 디렉터리에 가짜 예시 2개·eval json·corpus를 두고 JSON 구조(키, 시나리오 일수,
  배치 5개) 확인. 템플릿 주입 — 결과 HTML에 `<script id="demo-data"`와 JSON이 있고
  `__DEMO_DATA__`가 남아 있지 않음.
- `tests/serving/test_app.py`: `/demo`가 200과 `text/html`, 본문에 `demo-data`;
  `/demo/inputs/tool_wear`가 200과 CSV 첫 줄; `/demo/inputs/nope`가 404.
- `rag/eval_playbook.py` 행 필드 추가는 기존 검증 범위(스크립트)라 실행으로 확인.

## 검증 방법

1. `uv run python rag/eval_playbook.py --scenarios temperature --days 70`와 나머지 둘 40일 →
   `data/rag/eval_playbook.json` 갱신(`nice -n 19`, 10분 안팎).
2. `uv run python demo/build_demo.py` → `demo/index.html` 생성. 크기와 내장 JSON 파싱 확인.
3. 서버 없이: 파일을 브라우저로 열어 두 탭이 그려지는지(WSL이면 Windows 탐색기에서).
4. 서버로: `/demo` 열어 상태 배지 "실시간 연결됨", 예시 재계산 5건 성공, 업로드 1건.
5. 키 없이 서버: 실시간 재계산 시 guide null 안내가 보이는지.
6. 회귀: 전체 테스트 통과.

## 알려진 한계

- 타임라인의 판정은 champion v1로 오프라인 계산한 값이라, temperature Day 64 승격
  뒤 실제 운영 판정(새 champion)과 다르다. 화면에 주석으로 밝힌다.
- 이벤트는 09-02 라이브 실행 로그이므로 시뮬레이터나 게이트를 바꾸면 다시 돌려야 한다.
- 브라우저는 최신 Chrome/Edge 기준. 인라인 SVG와 fetch만 쓴다.

## 실행 결과에 따른 정정 (2026-09-03, 구현 후)

- 생성물 `demo/index.html` 256KB(내장 JSON: 예시 5, 플레이북 16, 시나리오 temperature
  70일·tool_wear 40일·fixture_loosening 40일). 이벤트는 로그에서 temperature 12건(트리거 5,
  거부 3, 섀도우 시작 2, 섀도우 종료 1, 승격 1 — Day 22·27·32 거부, Day 37 섀도우, Day 64
  승격, Day 69 두 번째 섀도우), tool_wear 10건, fixture_loosening 10건(각 트리거 5 + 거부 5).
- 채점 재실행: temperature `--days 70`(350배치), 고장 2종 40일. `rag/eval_playbook.py`가
  시나리오별로 병합 저장하도록 바뀌어 두 번 나눠 돌렸다.
- 서버: `/demo` 200 text/html, `/demo/inputs/{key}` 5개 200(120~490KB), 모르는 키 404.
  키 없이 띄운 서버에서 재계산 → `fault` 확정 공구 마모, `guide` null, `versions.chat_model`
  null. 키 있는 서버에서 재계산 → 가이드 포함 8.6초(대부분 LLM 호출).
- 테스트 200 → 205(조립 4 + 라우트 1).
- 계획과 달랐던 점: 없음. 브라우저 확인은 회사 PC의 Windows 브라우저로 `demo/index.html`
  (기록 모드)과 `http://127.0.0.1:8899/demo`(실시간)를 열어 진행.
- 실행 중 배운 것: 서버를 띄우는 줄과 `pkill -f '<서버 패턴>'`을 한 명령 문자열에 같이
  넣으면 패턴이 그 셸 자신의 명령줄과도 일치해 셸이 먼저 죽는다. 서버 종료는 PID 파일이나
  별도 명령으로.
