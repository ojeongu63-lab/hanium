# 시연 자료 (2026-10 제출용)

제출 시점의 코드를 바꾸지 않고 만든 시연·보고용 자료. 루프 실행은 정본 `data/`가 아니라 복사본 데이터 루트
(`~/project/hanium-demo-data/`, `CNC_DATA_ROOT`)에서 돌리므로 승격이 정본 모델·MLflow를 바꾸지 않는다
(2026-10-21 리허설 뒤 정본 `model.pt`·`scaler.json`·`mlflow.db` md5 불변 확인).

| 파일 | 내용 |
|---|---|
| `run_live_loop.sh` | 재학습 루프 라이브 시연 도우미 — `reset / serving / worker / feeder / status / mlflow-ui / rehearsal` |
| `rehearsal/` | 2차 리허설 로그(`serving.log`, `worker.log`, `feeder.log`, `summary.txt`) |
| `capture_demo_page.py` | `demo/index.html`(기록 재생 페이지) 탭 1·2·3 캡처 스크립트 |
| `demo_1_batch*.png` `demo_2_timeline*.png` `demo_3_sim*.png` | 데모 페이지 캡처(1440×900, `_full`은 전체 페이지) |

## 1. 재학습 루프 라이브 시연

README §2-7의 세 터미널(서빙·감시 워커·feeder)을 그대로 쓰되, 루프 상수(`CNC_*`, `src/config.py`)를 화면용으로
줄여 **트리거 → 재학습 → 게이트 → 섀도우 → 승격**을 8분 안에 본다. 판정 규칙은 그대로고 상수만 다르다.

| 상수 | 설계값 | 시연값 | 뜻 |
|---|---|---|---|
| `CNC_DRIFT_WINDOW_SIZE` | 10 | 5 | 드리프트 판정에 쓰는 최근 요청 수(= 하루치) |
| `CNC_CONSECUTIVE_K` | 3 | 3 | 연속 flagged 횟수(트리거) |
| `CNC_COOLDOWN_DAYS` | 5 | 2 | 재트리거 쿨다운 |
| `CNC_GATE_SAMPLE_SIZE` | 20 | 5 | 게이트·섀도우 표본(라벨 건수) |
| `CNC_DRIFT_START_DAY` / `CNC_TOTAL_DAYS` | 10 / 40 | 3 / 4 | 변형 시작일 / 변형이 최대(진행도 1.0)에 닿는 날 |
| `CNC_DRIFT_MAX_PROGRESS` | 없음 | 1.0 | 변형 상한(설계 Day 40 수준에서 고정) |
| `CNC_LABEL_DELAY_DAYS` | 7 | 1 | QC 라벨 도착 지연 |
| `CNC_TRAIN_EPOCHS` | 50 | 50 | 재학습 epoch(그대로 — 실제 재학습) |
| feeder `--days` / `--pace-seconds` | 40 / 15 | 9 / 50 | 가상 일수 / 하루치 전송 뒤 쉬는 시간 |

### 녹화 절차 (터미널 3개 + 브라우저)

```bash
cd ~/project/hanium
./docs/시연자료_202610/run_live_loop.sh reset       # 복사본 데이터 루트 새로 만들기(이전 시연 흔적 제거)
./docs/시연자료_202610/run_live_loop.sh serving     # 터미널 1 — /health 가 model_version 1 을 돌려주면 준비
./docs/시연자료_202610/run_live_loop.sh worker      # 터미널 2 — "감시 시작" 이 찍힌 뒤
./docs/시연자료_202610/run_live_loop.sh feeder      # 터미널 3 — 여기서부터 녹화 시작
./docs/시연자료_202610/run_live_loop.sh mlflow-ui   # (선택) 터미널 4 — 레지스트리 화면, 승격 뒤 champion alias 이동
./docs/시연자료_202610/run_live_loop.sh status      # 아무 때나 — /health 모델 버전 + champion alias
```

끝낼 때는 각 터미널에서 Ctrl-C. 다시 돌리기 전에는 반드시 `reset`(labels.db·requests.db·shadow.db 를 비우지 않으면
워커가 이전 실행의 날짜를 이어받는다 — README §2-7).

### 2차 리허설 기록 (2026-10-21, 이 PC·WSL2, `nice -n 19`)

| 시각(feeder 시작 기준) | 워커 로그 | 뜻 |
|---|---|---|
| Day 1~3 | ratio 0.74 / 0.61 / 0.75, flagged=False | 변형 전 기준 구간 |
| Day 4~6 | 1.88 / 2.12 / 2.17, flagged=True | 변형(온도 드리프트) 시작, 연속 3회 |
| Day 6 (+4분 25초) | `트리거 발동 — 재학습 시작` → 23초 → `게이트: G1 놓침=1건(champion 1건, 허용 2건) / G2 정상 5건 — 오탐 후보 1 vs champion 5` → `섀도우 시작 — version 65 (관찰 기준일 Day 6)` | 재학습·게이트 통과·섀도우 투입 |
| Day 7 | `shadow_pending` | 섀도우가 Day 7 생산분을 병행 추론 |
| Day 8 (+6분 40초) | `섀도우 종료 — 정상 5건 — 오탐 후보 0 vs champion 5 → promoted`, `승격 완료 — version 65` | 승격, `/health` model_version 1 → 65 |
| Day 9 | 0.76, flagged=False | 새 champion 기준으로 정상 |
| 전체 | 7분 47초 (feeder 9일 466초) | |

- 버전 번호가 65인 것은 복사본 `mlflow.db`에 과거 리허설의 재학습 63회(v2~v64)가 남아 있기 때문이다. "v1 → v2"로
  보이게 하려면 `reset` 뒤 복사본 레지스트리에서 v2~v64를 지우면 된다(정본과 무관).
- **1차 리허설(pace 12초)은 승격에 이르지 못했다**: 재학습(31초)보다 feeder가 빨라 섀도우가 시작되는 순간 하루가
  반쯤 전송된 상태였고(관찰 기준일 Day 8, Day 9 배치 5개 중 섀도우 기록 3개), 워커의 섀도우 판정은 "기준일 이후
  첫 5건에 모두 섀도우 기록이 있어야" 넘어가므로 계속 `shadow_pending`이었다. 코드가 아니라 시연 페이스 문제라
  pace 를 50초로 올렸다. 녹화 중 PC 가 느리면 `CNC_DEMO_PACE=60`으로 더 여유를 둔다.
- 워커 `worker.log`에는 학습 진행 출력의 NUL 바이트가 섞여 `grep`이 바이너리로 본다 — `grep -a`.

## 2. 데모 페이지 캡처

`demo/index.html`은 서버 없이 열리는 기록 재생 페이지(README §2-9). 탭 1(배치 진단 흐름: 판정 → 센서 기여도 →
플레이북 대조 → 조치 가이드), 탭 2(시나리오 3종 70일 타임라인 + 루프 이벤트), 탭 3(하루씩 재생하는 시뮬레이션,
Day 22 트리거·거부 카드)을 찍었다.

```bash
# Playwright(Chromium) 가 있는 Python 으로
python3 docs/시연자료_202610/capture_demo_page.py docs/시연자료_202610
```

**캡처 중 발견해 고친 결함(2026-10-26)**: `demo/template.html`의 `#tab-batch { display:grid }`·`#tab-sim { display:grid }`가
탭 섹션의 `hidden` 속성(브라우저 기본 `display:none`)을 덮어써, 탭을 바꿔도 "배치 진단 흐름"과 "시뮬레이션"
섹션이 사라지지 않고 세 탭이 세로로 겹쳐 보였다. `template.html`과 생성물 `index.html`의 `<style>`에
`[hidden] { display:none !important; }` 한 줄을 넣어 고쳤고, 위 캡처는 고친 페이지를 그대로 찍은 것이다.
