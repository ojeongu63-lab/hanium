# 재학습 루프 통합 테스트 + compose 설계

작성 2026-09-15. 선행: `2026-08-19-cnc-drift-triggered-retraining-design.md`(루프 원형),
`2026-08-25-cnc-shadow-deployment-design.md`(섀도우), `2026-09-02-cnc-two-sided-gate-design.md`
(현재 게이트 규칙). 브랜치 `main`.

## 배경

루프(트리거 → 재학습 → 게이트 → 섀도우 → 승격/거부)는 세 프로세스(서버·워커·feeder)를
사람이 터미널 세 개로 띄워 40일을 돌리고 결과를 문서에 옮기는 방식으로만 검증돼 왔다.
09-02까지 열 번쯤 돌렸고 그때마다 사람이 로그를 읽었다. CI(`.github/workflows/cnc-tests.yml`)는
단위 테스트만 돌린다. 지금까지 찾은 루프 버그(폴링 사이 날짜 스킵, 섀도우 시작일, feeder
페이스, 승격 직후 재트리거)는 전부 **프로세스 경계**에서 나왔는데, 그 경계를 자동으로 타는
테스트가 없다.

CI에는 `data/`가 없다. champion 모델(MLflow), 재학습용 `eval.csv`·`scaler.json`, feeder가
배치를 만들 원본 실험 8개가 전부 거기 있고, 경로는 `ROOT / "data"`로 코드 6곳에 박혀 있다.
루프 상수(연속 횟수, 쿨다운, 게이트 표본, 라벨 지연 등)도 상수라 40일 미만으로 줄일 수 없다.

### 사용자 결정 (2026-09-15)

- 목표는 "실제 MLOps 관점". 판정의 옳고 그름이 아니라 **배관이 자동으로 검증되는 것**.
- CI 테스트는 배관만 보장한다. 판정 재현(temperature 승격, tool_wear 거부)은 기록된 라이브
  실행이 담당하며 이번 범위가 아니다.
- pytest가 서버·워커를 실제 서브프로세스로 띄운다. docker는 CI 테스트에 쓰지 않는다.
  compose는 같은 세 명령을 담은 로컬·배포용 파일이고 CI에서는 빌드만 확인한다.

## 목표 / 비목표

**목표**

1. `uv run pytest -m integration` 한 번으로, 실데이터 없이, 트리거 → 재학습 → 게이트 →
   MLflow 태그 → 섀도우 → `/reload-model` → 승격(파일 교체·백업) 경로와 거부(원인 추정 태그·
   champion 불변) 경로가 끝까지 닿는지 확인한다.
2. 같은 테스트가 GitHub Actions에서 매 push·PR마다 돈다.
3. 데이터 루트와 루프 상수를 환경변수로 바꿀 수 있다. 환경변수가 없으면 지금과 동일하게 동작한다.
4. `docker compose --profile demo up` 한 줄로 서버·워커·feeder가 뜬다.

**비목표**

- 판정 정확도, 모델 품질, 드리프트 임계값, 게이트 규칙의 변경.
- CI에서 docker compose 스택을 실제로 띄워 검증하는 것(빌드까지만).
- 관측성(metrics, 대시보드), 상태 저장소 교체(SQLite → Postgres), 인증. 별도 작업.
- `demo/`, `rag/`, `synthetic/`, `loocv/`, `augmentation/`의 경로 외부화. 루프에 안 들어간다.

## 구조

| 파일 | 역할 |
|---|---|
| `src/config.py` (신규) | 데이터 루트·루프 상수·학습 epochs·변형 진폭을 환경변수에서 읽는다. 기본값 = 현재 값 |
| `src/lstm_ae/tracking.py` | `MLFLOW_DIR = config.DATA_ROOT / "mlflow"` |
| `src/serving/app.py` | `DB_PATH`·`SHADOW_DB`·`DATASET_DIR`·RAG 경로·companion 폴백·`DRIFT_WINDOW_SIZE`를 config에서. 모듈 속성 이름 유지(기존 테스트가 monkeypatch) |
| `src/retraining/runner.py` | `root` 인자를 `data_root`로. `TRAINING_CONFIG["epochs"]`를 config에서 |
| `scripts/run_preprocessing.py`, `scripts/run_lstm_training.py` | 데이터 경로·epochs를 config에서 |
| `monitoring/simulate_timeline.py` | 경로·상수·진폭을 config에서. `progress_for`에 상한 적용. `--start-day` 옵션 |
| `monitoring/drift_worker.py` | 경로·상수를 config에서. `champion_missed`를 champion run의 `mean_fn` 지표에서 읽음 |
| `tests/integration/fixture_dataset.py` (신규) | 합성 데이터셋 생성기 |
| `tests/integration/conftest.py` (신규) | 임시 데이터 루트 부트스트랩, 서버·워커 서브프로세스 픽스처, 하루씩 feed 헬퍼 |
| `tests/integration/test_loop.py` (신규) | 승격 경로·거부 경로 테스트 2개 |
| `tests/test_config.py` (신규) | 기본값·환경변수 파싱 |
| `pyproject.toml` | `httpx2`를 본 의존성으로, `integration` 마커 등록, 기본 실행에서 제외 |
| `docker-compose.yml` (신규) | serving · worker · feeder(프로필 `demo`) |
| `.github/workflows/cnc-tests.yml` | `loop-integration` job, `docker-build` job(main push만) |
| `README.md`, `docs/STRUCTURE.md` | compose 실행법, 환경변수 표, 새 폴더 |

## 1. 설정 외부화 (`src/config.py`)

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # 02-cnc-machining/
DATA_ROOT     = Path(env "CNC_DATA_ROOT", PROJECT_ROOT / "data")
DATASET_DIR   = DATA_ROOT / "dataset" / "CNC 비식별화 원본데이터_1209"   # 인덱스 train.csv
EXPERIMENT_DIR = DATASET_DIR / "CNC Virtual Data set _v2"               # experiment_XX.csv
```

| 환경변수 | 기본값 | 쓰는 곳 |
|---|---|---|
| `CNC_DRIFT_WINDOW_SIZE` | 10 | `serving/app.py` |
| `CNC_CONSECUTIVE_K` | 3 | `drift_worker.py` |
| `CNC_COOLDOWN_DAYS` | 5 | `drift_worker.py` |
| `CNC_GATE_SAMPLE_SIZE` | 20 | `drift_worker.py` |
| `CNC_TOTAL_DAYS` | 40 | `simulate_timeline.py` |
| `CNC_BATCHES_PER_DAY` | 5 | `simulate_timeline.py`, `serving/app.py`(`/demo/timeline` 범위 검사) |
| `CNC_DRIFT_START_DAY` | 10 | `simulate_timeline.py` |
| `CNC_LABEL_DELAY_DAYS` | 7 | `simulate_timeline.py` |
| `CNC_LABEL_FLIP_DAY` | 21 | `simulate_timeline.py` (`WEAR_LABEL_FLIP_DAY`·`VIBRATION_LABEL_FLIP_DAY` 둘 다 이 값. 지금도 같은 값) |
| `CNC_DRIFT_MAX_PROGRESS` | 없음(상한 없음) | `simulate_timeline.progress_for` |
| `CNC_POS_DRIFT`, `CNC_CUR_DRIFT`, `CNC_WEAR_RATE`, `CNC_VIBRATION_RATE` | 0.02, 0.02, 0.2, 3.65 | `simulate_timeline.py` |
| `CNC_TRAIN_EPOCHS` | 50 | `run_lstm_training.py`, `retraining/runner.py` |

정수·실수 파싱 실패는 시작 시 `ValueError`로 죽는다(조용히 기본값으로 떨어지지 않는다).
파일 6개는 모듈 상단에서 `from config import ...`로 받아 **같은 이름의 모듈 속성**에 넣는다.
`tests/serving/test_app.py`가 `app_module.DB_PATH`처럼 속성을 monkeypatch하므로 이름을 바꾸지 않는다.

`progress_for`의 상한: `CNC_DRIFT_MAX_PROGRESS`가 있으면 `min(progress, 상한)`. 없으면 지금처럼
`TOTAL_DAYS`를 넘어도 계속 커진다. 08-25 섀도우 스펙이 "알려진 한계"로 남긴 70일 초과 변형은 이
값을 1.0으로 주면 사라진다(이번 작업에서 기존 실행 명령은 바꾸지 않는다).

`champion_missed`: 워커 `main()`이 시작할 때 champion alias → run → `run.data.metrics["mean_fn"]`을
읽어 `WorkerState.champion_missed`로 쓴다(`build_run_metrics`가 모든 run에 기록한다). 없으면
`KeyError`로 죽는다. 하드코딩 1은 사라진다. 승격 뒤 갱신하는 기존 로직은 그대로.

## 2. 합성 데이터셋 (`tests/integration/fixture_dataset.py`)

`write_dataset(dataset_dir, seed=0)`가 KAMP 원본과 같은 구조를 만든다.

- `<dataset_dir>/train.csv`: 인덱스. 컬럼 `No, material, feedrate, clamp_pressure, tool_condition,
  machining_finalized, passed_visual_inspection`. `No`는 `split.py`의 22개 ID
  (train 8, eval_good 3, eval_bad 11). eval_bad는 `passed_visual_inspection=no`, 나머지 `yes`.
- `<dataset_dir>/CNC Virtual Data set _v2/experiment_XX.csv`: 48컬럼(`FEATURE_COLUMNS` 41 +
  `DEAD_SENSOR_COLUMNS` 4 + `Machining_Process`, `M_sequence_number`, `M_CURRENT_PROGRAM_NUMBER`),
  150행. 피처는 실험별 시드의 사인파 + 잡음(값 범위는 실제와 무관해도 된다 — StandardScaler가 맞춘다).
  eval_bad 실험은 스핀들 전류·파워 컬럼을 3배로 올려 champion이 무언가를 잡을 수 있게 한다.
  `S_SystemInertia`, `M_CURRENT_FEEDRATE`는 실험당 상수.
- **train 실험 하나(17)는 잡음을 3배로 준다.** 임계값은 train 실험 8개 점수의 p95라 사실상
  최댓값 근처인데, 8개가 서로 비슷하면 정상 배치의 점수/임계값 비율이 1.0 근처가 되어 변형
  전 구간에서도 출력 드리프트(비율 > 0.8)가 켜진다. 잡음 큰 실험 하나가 임계값을 끌어올리면
  나머지 7개의 비율은 0.5 안팎이고, 그 실험의 복사본이 하루 5배치 중 하나로 섞여도 창 평균은
  0.8 아래에 머문다. 계단 변형 뒤에는 어차피 입력 드리프트(z > 2)로 확정 판정된다.
- 결정적: 같은 seed면 같은 파일.

이 구조 위에서 `scripts/run_preprocessing.py` → `scripts/run_lstm_training.py` →
`scripts/promote_model.py 1`이 **코드 변경 없이** 돌아 champion v1이 생긴다. feeder의
`generate_batch`도 같은 파일에서 train 실험을 읽는다.

## 3. 통합 테스트 (`tests/integration/test_loop.py`)

### 부트스트랩 (session 픽스처)

1. `tmp_path_factory`로 데이터 루트를 만들고 `write_dataset`.
2. 환경 `CNC_DATA_ROOT=<루트>`, `CNC_TRAIN_EPOCHS=2`와 아래 루프 상수를 붙여 스크립트 3개를
   서브프로세스로 실행(`uv run` 아님 — 현재 인터프리터 `sys.executable`). champion v1 생성.
3. 빈 포트를 잡아 `uvicorn serving.app:app`을 서브프로세스로 띄우고 `/health` 200까지 대기(60초).

루프 상수(두 테스트 공통): 배치/일 5, 드리프트 창 5, 연속 3, 쿨다운 2, 게이트 표본 5, 라벨 지연 1,
드리프트 시작일 2, 총 일수 3, 진행도 상한 1.0(→ Day 3부터 변형이 계단으로 고정), 라벨 뒤집는 날 3.
`CNC_TOTAL_DAYS`는 램프 기울기에만 쓰이고, 실제로 며칠을 보낼지는 feeder의 `--days`로 준다
(승격 경로 12일, 거부 경로 8일).

### 진행 방식 — 하루씩 동기화

워커는 `drift_worker.py <scenario> --base-url ... --poll-interval 0.2`로 서브프로세스, stdout을
파일로. feeder는 테스트가 맡는다: `simulate_timeline.py <scenario> --serve-url ... --start-day N
--days N`을 하루 단위로 서브프로세스 실행하고, 워커 로그에 `Day NN` 줄이 찍힐 때까지(최대 120초)
기다린 뒤 다음 날을 보낸다. 재학습에 몇 초가 걸리든 워커가 그 날을 끝낸 뒤에만 다음 날이 오므로
결과가 타이밍에 의존하지 않는다. 섀도우가 관찰할 "이후 생산분"도 반드시 섀도우 시작 뒤에 들어간다.

`--start-day`(기본 1)는 feeder에 새로 넣는 옵션이다. `range(start_day, days + 1)`.

### 승격 경로 — `temperature`, 12일

라벨은 전부 정상. `CNC_POS_DRIFT=8`(위치 3축이 8σ 이동 → 입력 드리프트 z>2 확정),
`CNC_CUR_DRIFT=1.0`. 계단 변형이라 Day 3 이후 배치는 서로 같은 분포다.

기대 흐름: Day 3·4·5 flagged → Day 5 트리거 → 정상 라벨 배치(생산일 ≤ 4, 계단 이후 포함)로 재학습 →
G1(원본 eval 재스케일) 통과, G2 창(생산일 4, 정상 5건)에서 champion 오탐 5 대 후보 오탐 적음 →
섀도우 시작(기준일 5) → Day 6 생산분이 섀도우로 기록 → Day 7에 라벨 5건 도착 → 섀도우 통과 →
`swap_with_rollback` → alias → `/reload-model` → `/health` 검증 → 승격. 첫 시도가 "개선 없음"으로
거부되더라도 쿨다운 뒤 재시도하므로 12일 안에 승격된다.

확인하는 것:
- 워커 로그의 `Day` 줄이 1..12 빠짐없이 순서대로.
- Day 5 이전에 트리거 없음(`action != none`이 처음 나오는 날 ≥ 5 — 연속 3회의 최소 시점.
  변형 전 구간에서 헛트리거가 나면 여기서 잡힌다).
- 12일 안에 `action=shadow_started` 다음 `action=promoted`가 나온다.
- MLflow(`sqlite:///<루트>/mlflow/mlflow.db`)에 `scenario=temperature` 태그 run이 있고
  `gate_decision`·`gate_g2_n_good`·`shadow_promoted`… 태그가 있다.
- `/health`의 `model_version`이 1이 아니고, MLflow champion alias가 가리키는 버전과 같다
  (12일 안에 승격이 두 번 일어나도 성립하는 표현).
- `<루트>/model/model.pt`·`processed/scaler.json`의 md5가 부트스트랩 때와 다르고, 승격 run의
  `retrain_dir` 안 파일과 같다. `<루트>/model_backup/`에 디렉터리가 하나 생겼다.
- `<루트>/monitoring/shadow.db`에 생산일 6 배치 5건이 기록됐다.

### 거부 경로 — `tool_wear`, 8일

`CNC_LABEL_FLIP_DAY=3`이라 계단과 같은 날부터 불량 라벨. `CNC_WEAR_RATE=20`.

기대 흐름: Day 5 트리거 → 재학습(정상 라벨 = 생산일 1·2, 10배치) → G2 창(생산일 4, 불량 5건)에
정상 라벨 0건 → "G2 판정 불가: 창에 정상 라벨 없음" 거부 → `estimate_cause` → 태그. 쿨다운 2는
tick마다 하나씩 줄어 Day 7에 0이 되므로 Day 7 재트리거 → 같은 거부.

확인하는 것:
- 거부 run이 1개 이상, `gate_decision=rejected`, `gate_reject_reason`에 "정상 라벨 없음",
  `estimated_cause`가 `tool_wear` 또는 `vibration_backlash`, `recommended_action` 태그 존재(값은
  RAG 없음이라 빈 문자열).
- `/health` 버전이 1 그대로, `model.pt`·`scaler.json` md5 불변, `model_backup/` 없음.
- `shadow_started`가 한 번도 없다.

두 테스트는 각각 자기 데이터 루트를 쓴다(서로 오염 없음). 워커·서버는 픽스처 종료 시 SIGTERM,
3초 뒤 SIGKILL. 워커 로그는 실패 시 pytest 출력에 붙인다.

### 마커와 기본 실행

`pyproject.toml`: `markers = ["integration: 서버·워커 서브프로세스를 띄우는 루프 테스트"]`,
`addopts`에 `-m "not integration"` 추가. `uv run pytest`는 지금처럼 단위 테스트만(20초),
`uv run pytest -m integration`이 통합 테스트만 돈다(명령줄 `-m`이 addopts의 `-m`을 덮어쓴다 —
구현 시 확인하고 아니면 `--override-ini`로 바꾼다).

## 4. CI (`.github/workflows/cnc-tests.yml`)

| job | 조건 | 내용 |
|---|---|---|
| `test` (기존) | push main, PR, 수동 | `uv run pytest -q` |
| `loop-integration` (신규) | 같음 | `uv run pytest -m integration -q`, `timeout-minutes: 20` |
| `docker-build` (신규) | push main, 수동 | `docker compose build` — Dockerfile이 지금까지 한 번도 자동 검증된 적이 없다 |

예상 시간: 통합 job 3~5분(uv sync 1~2분 + 테스트 2개 2~3분).

## 5. `docker-compose.yml`

```yaml
services:
  serving:
    build: .
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]
    env_file: [{path: .env, required: false}]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 5s
      retries: 24
  worker:
    build: .
    command: uv run --no-sync python monitoring/drift_worker.py ${SCENARIO:-temperature} --base-url http://serving:8000
    volumes: ["./data:/app/data"]
    env_file: [{path: .env, required: false}]
    depends_on:
      serving: {condition: service_healthy}
  feeder:
    build: .
    profiles: ["demo"]
    command: uv run --no-sync python monitoring/simulate_timeline.py ${SCENARIO:-temperature} --serve-url http://serving:8000 --days ${DAYS:-40} --pace-seconds ${PACE:-15}
    volumes: ["./data:/app/data"]
    depends_on:
      serving: {condition: service_healthy}
```

- `/health`는 champion이 로드돼야 200이므로 healthy = 모델 준비됨.
- `data/`는 호스트 볼륨이라 재학습·백업·MLflow 기록이 호스트에 남는다. 컨테이너 안 경로
  `/app/data`는 `PROJECT_ROOT / "data"`와 같아 환경변수가 필요 없다.
- `httpx2`가 dev 그룹에만 있어 `--no-dev` 이미지로는 워커·feeder가 import에 실패한다.
  본 의존성으로 옮긴다(런타임 의존성이 맞다).
- 시나리오 전환 전 `labels.db`·`requests.db`·`shadow.db`·`data/timeline/<이전>`을 비워야 하는
  기존 제약은 그대로다. README에 그대로 적는다.

## 테스트 (TDD)

- `tests/test_config.py`: 환경변수 없을 때 값 8개가 현재 상수와 같다 / `CNC_GATE_SAMPLE_SIZE=5`가
  int 5 / `CNC_DATA_ROOT`가 `DATA_ROOT`와 `DATASET_DIR`에 반영 / 파싱 불가 값은 `ValueError`.
  (config는 import 시 읽으므로 테스트는 `importlib.reload`로 확인한다.)
- `tests/integration/test_fixture_dataset.py`: 인덱스 ID 22개가 `split.py`와 일치 / 라벨이 분할과
  모순 없음 / 실험 파일 48컬럼·150행·NaN 없음 / 같은 seed → 같은 바이트.
- `tests/monitoring/test_simulate_timeline.py`: `progress_for`가 상한 있으면 잘리고 없으면 계속
  커진다 / `--start-day 5 --days 6`이 5·6만 보낸다(`feed_day`를 monkeypatch해 호출 기록).
- `tests/monitoring/test_drift_worker.py`(신규, 순수 함수만): `champion_missed_from_metrics`가
  `mean_fn`을 int로 / 없으면 `KeyError`.
- `tests/integration/test_loop.py`: 위 두 시나리오. `integration` 마커.
- 회귀: 기존 212개가 그대로 통과(기본값 불변).

## 검증 방법 (완료 기준)

1. `uv run pytest -q` 통과, 통합 테스트는 수집만 되고 실행 안 됨(`deselected` 표시).
2. `nice -n 19 uv run pytest -m integration -q` 이 서버에서 통과, 소요 시간 기록.
3. GitHub Actions 세 job 녹색(main push 후 확인, 워크플로 링크를 정정 절에).
4. 이 서버에서 `docker compose --profile demo up`으로 temperature를 5일치(`DAYS=5 PACE=2`)
   돌려 워커 로그에 `Day 01`~`Day 05`가 찍히는지. 끝나면 `labels.db`·`requests.db`·
   `data/timeline/temperature` 원상복구.
5. 실데이터 경로 불변 확인: 환경변수 없이 세 터미널 방식으로 temperature 3일 스모크(트리거 없는
   구간) → 기존 08-24 스모크와 같은 `flagged=False`. DB·timeline 원상복구.
6. 결과를 이 문서 "실행 결과에 따른 정정" 절에 적는다.

## 알려진 한계

- 합성 champion은 아무 의미 없는 모델이다. 테스트가 보장하는 것은 배관이지 판정이 아니다.
- 서브프로세스마다 torch·mlflow import에 5~7초가 든다. 두 테스트에 서버 2, 워커 2, feeder 20회
  (feeder는 torch를 import하지 않아 1초 안팎).
- Windows는 대상이 아니다(포트·시그널·경로).
- 승격 경로가 첫 트리거에서 "개선 없음"으로 거부될 수 있어 12일을 준다. 12일 안에도 승격이 안
  되면 테스트가 실패하는데, 그건 배관이 아니라 계단 변형 크기의 문제이므로 진폭을 키운다.
- `CNC_DRIFT_MAX_PROGRESS`는 테스트가 필요로 해서 넣는 값이다. 기존 실행 명령엔 붙이지 않는다.

## 손대지 않는 것

`src/retraining/gate.py`·`trigger.py`·`promotion.py`의 판정 로직, `src/monitoring/drift.py`의
임계값, 상수의 기본값, `demo/`, `rag/`, `synthetic/`, `loocv/`, `augmentation/`,
`monitoring/simulate_drift.py`, `monitoring/sweep_drift_constants.py`.
