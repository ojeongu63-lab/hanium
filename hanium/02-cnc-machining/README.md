# 02 · CNC 가공 이상탐지 MLOps (본 결과물)

CNC 가공 데이터로 LSTM-Autoencoder 기반 비지도 이상탐지 모델을 학습하고,
MLflow(sqlite 백엔드)로 실험을 추적·관리하며, FastAPI로 champion 모델을
서빙하는 프로젝트.

**성능: precision 0.91 / recall 0.91** (eval 14개 실험 — TP 10 / FP 1 / FN 1 / TN 2,
mean 임계값 기준). 정상 실험 11개 LOOCV에서도 9개를 정상으로 맞춰, 고정 분할
결과가 우연이 아님을 확인했다.

이 트랙은 [`../01-cn7-injection-molding/`](../01-cn7-injection-molding/)에서 세운
LSTM-AE 방법론을 **다른 제조공정에 재적용해 유효성을 검증**하고, 거기에
MLOps·RAG를 얹은 것이다. 전체 맥락은 [저장소 README](../README.md) 참고.

## 1. 다른 PC에서 처음 설정하기

### 1-1. 코드 받기

```bash
git clone git@github.com:ojeongu63-lab/hanium.git
cd hanium/02-cnc-machining
```

처음 clone하는 PC라면 GitHub SSH 키가 그 PC에 없을 수 있음 — 그 경우:

```bash
ssh-keygen -t ed25519 -C "ojeongu63@gmail.com"
cat ~/.ssh/id_ed25519.pub   # 출력된 걸 https://github.com/settings/keys 에 등록
```

### 1-2. 데이터 배치 (★ 경로 중요)

`data/` 폴더는 git에 안 올라가 있음(`.gitignore` 대상, 원본/전처리 데이터·학습된
모델·MLflow 기록이 들어있어서). 별도로 옮긴 `cnc-data.tar.gz`를 **`02-cnc-machining/`
바로 아래**(즉 `hanium/02-cnc-machining/data/`)에 풀어야 함:

```bash
# cnc-data.tar.gz를 이 PC로 옮긴 뒤, hanium/02-cnc-machining/ 안에서:
tar -xzf ~/cnc-data.tar.gz -C .
```

풀고 나면 아래 구조가 나와야 정상:

```
02-cnc-machining/data/
├── dataset/    # 원본 CNC CSV (실험별 raw 데이터, 폴더명에 한글/공백 포함)
├── processed/  # train.csv, eval.csv, scaler.json, manifest.json (전처리 결과)
├── model/      # 학습 산출물 (model.pt, evaluation_report.json 등)
└── mlflow/     # MLflow sqlite DB + 모델 아티팩트 (mlflow.db, artifacts/)
```

경로가 이거랑 다르면(`data/`가 `02-cnc-machining/` 밖에 있거나 한 단계 더 들어가 있으면)
아래 실행 명령이 전부 실패함 — `src/lstm_ae/tracking.py`의 `ROOT`가
`02-cnc-machining/` 기준으로 `data/mlflow/`를 하드코딩해서 찾기 때문.

### 1-3. 의존성 설치

```bash
uv sync
```

### 1-4. RAG용 OpenAI API 키 설정 (선택)

`/predict`가 불량 판정을 낼 때 현장 조치 가이드(`guide` 필드)를 생성하려면
OpenAI API 키가 필요함. `02-cnc-machining/` 바로 아래에 `.env` 파일을 만들고:

```
OPENAI_API_KEY=sk-...
```

`.env`는 `.gitignore` 대상이라 git에 안 올라감 — 다른 PC에서는 매번 새로
만들어야 함. **키가 없어도 서버는 정상 기동되고 판정(`predicted_label` 등)도
그대로 나옴** — `guide`만 항상 `null`로 나옴(RAG 기능만 비활성화).

## 2. 실행

### 2-1. 추론 서버 (FastAPI)

```bash
cd 02-cnc-machining
nice -n 19 uv run uvicorn serving.app:app --port 8899
```

- Swagger UI: `http://127.0.0.1:8899/docs`
- `/health`: 현재 로드된 champion 모델 버전 확인
- `/predict`: 실험 CSV 업로드 → 양품/불량 판정 + 피처별 재구성오차 기여도(`feature_contributions`,
  train(정상) 기준 z-score 큰 순 정렬) — 어떤 변수가 이 샷에서 평소 대비 가장 크게 벗어났는지 확인 가능
  + `guide`: 불량 판정일 때 RAG(코퍼스 검색 + OpenAI 생성)로 만든 현장 조치
  가이드(원인 추정/확신도 설명/권장 조치/안전수칙/출처). 정상 판정이면 고정
  메시지, 코퍼스 미구축이거나 `OPENAI_API_KEY` 없으면 `null`(아래 2-4 참고)

테스트용 실험 CSV 경로 (경로에 공백 있으니 항상 따옴표):

```
data/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2/experiment_XX.csv
```

정답 확인용 (eval 세트, 학습에 안 쓰인 데이터):
- 정상: `experiment_12`, `experiment_18`, `experiment_22`
- 불량: `experiment_04`, `experiment_05`, `experiment_06`, `experiment_07`,
  `experiment_08`, `experiment_09`, `experiment_10`, `experiment_16`,
  `experiment_20`, `experiment_21`, `experiment_23`

### 2-2. MLflow UI (실험 추적 대시보드)

```bash
cd 02-cnc-machining
nice -n 19 uv run mlflow ui --backend-store-uri sqlite:///$(pwd)/data/mlflow/mlflow.db --port 5099
```

- 접속: `http://127.0.0.1:5099`
- `cnc-lstm-ae` 실험 → run별 params/metrics, "Models" 탭에서 champion alias 확인

**⚠ Python 3.14 환경에서 알려진 문제:** mlflow 3.14.0의 UI 서버가 `assistant`
기능 모듈에서 `ImportError: cannot import name 'Traversable' from
'importlib.abc'` 에러로 계속 죽었다 재시작하기를 반복함 (mlflow가 아직
Python 3.14를 제대로 지원 안 해서 생기는 상류 버그, 우리 학습/서빙 코드와는
무관). `mlflow ui`가 안 뜨고 워커가 계속 재시작되면 아래 한 줄을 고쳐서 해결:

```bash
FILE=".venv/lib/python3.14/site-packages/mlflow/assistant/skill_installer.py"
sed -i 's/from importlib.abc import Traversable/from importlib.resources.abc import Traversable/' "$FILE"
```

이건 `.venv` 안 서드파티 파일을 직접 고치는 로컬 우회라서, `uv sync
--reinstall`이나 새로 `uv sync`를 하면(패키지가 다시 깔리면) 매번 다시
적용해야 함.

### 2-3. 새로 학습 / champion 승격 (참고용, 매번 할 필요 없음)

```bash
cd 02-cnc-machining
nice -n 19 uv run python scripts/run_lstm_training.py   # MLflow에 새 run 기록 + 모델 등록
uv run python scripts/promote_model.py <등록된 버전 번호>  # 그 버전을 champion으로 승격
```

### 2-4. RAG 코퍼스 구축 (최초 1회, OpenAI API 키 필요)

`/predict`의 `guide` 필드가 검색할 지식 코퍼스를 만드는 스크립트. 실제 공개
문서(Sandvik Coromant 밀링 트러블슈팅, OSHA 기계 안전수칙 — 원문은
`rag/sources/*.md`에 이미 로컬로 저장돼 있어 웹 접근 불필요)를 청크로 쪼개
OpenAI 임베딩으로 변환하고 FAISS 인덱스로 저장함:

```bash
cd 02-cnc-machining
uv run --env-file .env python rag/build_corpus.py
```

`data/rag/corpus.json`, `data/rag/corpus.index`가 생성됨(`data/` 하위라 git엔
안 올라감 — 서버 기동 전에 다른 PC에서도 한 번 돌려야 함). 코퍼스 소스
문서(`rag/sources/*.md`)를 바꾸지 않는 한 다시 돌릴 필요 없음.

### 2-5. LOOCV 검증 (참고용, 진단 스크립트)

정상 실험 11개에 대해 leave-one-out 교차검증을 돌려, 고정 8/3 분할 결과가
대표값인지 확인하는 일회성 진단 스크립트. champion 모델이나 서빙에는 영향
없음(완전히 분리된 산출물):

```bash
cd 02-cnc-machining
who && top -bn1 | head -6   # 서버 여유 확인 (11번의 전체 학습이 순차 실행됨, 수 분 소요)
nice -n 19 uv run python loocv/run_loocv.py
```

결과는 `loocv/summary.json`(집계), `loocv/summary.csv`(폴드별 상세)에 저장됨.
`loocv/folds/`(폴드별 중간 산출물)는 용량이 커서 git에 안 올라감.

### 2-6. 합성 데모 시나리오 생성 (참고용)

실제 정상 실험 위에 도메인 지식으로 이상 패턴(공구마모/이송축부하/진동)을
주입해, 발표·데모용 `/predict` 입력 CSV 6개(이상 3 + 정상 변형 3)를 만드는
스크립트. champion 모델로 실제 검증하며 생성함(진폭을 자동으로 조절):

```bash
cd 02-cnc-machining
nice -n 19 uv run python synthetic/generate_synthetic.py
```

결과는 `synthetic/scenarios/*.csv`(입력)와 `*_predict_result.json`(검증
기록)에 저장됨, git에 포함됨. 데모 때 `/predict`에 그대로 업로드해서 쓰면 됨:

```bash
curl -s -X POST "http://127.0.0.1:8899/predict" -F "file=@synthetic/scenarios/tool_wear.csv"
```

## 3. WSL에서 Windows 브라우저로 접속이 안 될 때

- 기본은 `http://localhost:<port>` 또는 `http://127.0.0.1:<port>`로 바로 열림
  (WSL2가 자동으로 포워딩해줌).
- 안 열리면 VS Code의 PORTS 패널이 해당 포트를 자기가 먼저 잡고 있을 수 있음
  (Remote-WSL 확장의 자동 포워딩이 꼬이는 경우) — PORTS 탭에서 그 포트
  forwarding을 끄거나, 다른 포트 번호로 서버를 띄워서 우회.
- Windows 쪽에서 `netstat -ano | findstr :<port>`로 그 포트를 누가 물고
  있는지 먼저 확인하면 원인 파악이 빠름.

## 4. 프로젝트 구조

> 파이프라인 흐름과 파일별 역할까지 주석으로 정리한 문서가 따로 있다 —
> **[`docs/STRUCTURE.md`](docs/STRUCTURE.md)**. 처음 보는 사람에게 공유하기 좋다.

폴더가 두 종류로 나뉜다. **`src/`는 import해서 쓰는 라이브러리 코드**이고,
**최상위의 나머지 폴더들은 각각 하나의 일회성 분석**으로, 실행 스크립트와 그
결과물을 한 곳에 담고 있다.

`src/rag/`와 `rag/`처럼 이름이 겹치는 쌍이 있는데, **`src/` 안쪽이 라이브러리,
바깥쪽이 그걸 돌리는 스크립트**로 구분하면 된다.

### 라이브러리 (`src/`)

| 경로 | 내용 |
|---|---|
| `src/preprocessing/` | 원본 CSV → train/eval 분할, 스케일링 |
| `src/lstm_ae/` | LSTM-Autoencoder 모델·학습·채점, `tracking.py`(MLflow 설정) |
| `src/serving/` | 추론 로직(`inference.py`) + FastAPI 앱(`app.py`) |
| `src/rag/` | RAG 검색(`retrieval.py`) + 생성(`generation.py`) + 오케스트레이션(`guide.py`) |
| `src/monitoring/` | 드리프트 계산(`drift.py`), 요청 로깅, MLflow 지표 기록 |

### 실행 스크립트 + 결과물

| 경로 | 하는 일 | 산출물 |
|---|---|---|
| `scripts/` | 전처리·학습·champion 승격 (주 파이프라인) | `data/` 하위 |
| `rag/` | RAG 코퍼스 구축 (§2-4) | `data/rag/` (원문은 `rag/sources/*.md`) |
| `loocv/` | 정상 실험 LOOCV 검증 (§2-5) | `loocv/summary.{json,csv}` |
| `synthetic/` | 데모용 합성 이상 시나리오 생성 (§2-6) | `synthetic/scenarios/*.csv` |
| `augmentation/` | 희소 샘플 증강 실험 | `augmentation/combined_dataset/` (git 제외) |
| `monitoring/` | 드리프트 상황 시뮬레이션 | 콘솔 출력 |

`loocv/`, `synthetic/`, `augmentation/`, `monitoring/`은 **검증·실험용이라 서빙
경로에 영향을 주지 않는다.** 서버를 띄우고 `/predict`를 쓰는 데는 `src/`,
`scripts/`, `data/`만 있으면 된다.

### 그 외

| 경로 | 내용 |
|---|---|
| `docs/specs/`, `docs/plans/` | 설계 스펙 / 구현 계획 문서 |
| `data/` | 원본·전처리·모델·MLflow 기록 (git 제외, §1-2 참고) |

> 각 스크립트는 자기 파일 위치로 프로젝트 루트를 역산한다
> (`Path(__file__).parent.parent`). **폴더 깊이가 코드에 박혀 있으므로**
> 스크립트를 다른 깊이로 옮기면 경로가 깨진다.
