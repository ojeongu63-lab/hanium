# CNC 이상탐지 MLOps 데모

CNC 가공 데이터로 LSTM-Autoencoder 기반 비지도 이상탐지 모델을 학습하고,
MLflow(sqlite 백엔드)로 실험을 추적·관리하며, FastAPI로 champion 모델을
서빙하는 프로젝트.

## 1. 다른 PC에서 처음 설정하기

### 1-1. 코드 받기

```bash
git clone git@github.com:ojeongu63-lab/hanium.git
cd hanium/version_2
```

처음 clone하는 PC라면 GitHub SSH 키가 그 PC에 없을 수 있음 — 그 경우:

```bash
ssh-keygen -t ed25519 -C "ojeongu63@gmail.com"
cat ~/.ssh/id_ed25519.pub   # 출력된 걸 https://github.com/settings/keys 에 등록
```

### 1-2. 데이터 배치 (★ 경로 중요)

`data/` 폴더는 git에 안 올라가 있음(`.gitignore` 대상, 원본/전처리 데이터·학습된
모델·MLflow 기록이 들어있어서). 별도로 옮긴 `cnc-data.tar.gz`를 **`version_2/`
바로 아래**(즉 `hanium/version_2/data/`)에 풀어야 함:

```bash
# cnc-data.tar.gz를 이 PC로 옮긴 뒤, hanium/version_2/ 안에서:
tar -xzf ~/cnc-data.tar.gz -C .
```

풀고 나면 아래 구조가 나와야 정상:

```
version_2/data/
├── dataset/    # 원본 CNC CSV (실험별 raw 데이터, 폴더명에 한글/공백 포함)
├── processed/  # train.csv, eval.csv, scaler.json, manifest.json (전처리 결과)
├── model/      # 학습 산출물 (model.pt, evaluation_report.json 등)
└── mlflow/     # MLflow sqlite DB + 모델 아티팩트 (mlflow.db, artifacts/)
```

경로가 이거랑 다르면(`data/`가 `version_2/` 밖에 있거나 한 단계 더 들어가 있으면)
아래 실행 명령이 전부 실패함 — `src/lstm_ae/tracking.py`의 `ROOT`가
`version_2/` 기준으로 `data/mlflow/`를 하드코딩해서 찾기 때문.

### 1-3. 의존성 설치

```bash
uv sync
```

## 2. 실행

### 2-1. 추론 서버 (FastAPI)

```bash
cd version_2
nice -n 19 uv run uvicorn serving.app:app --port 8899
```

- Swagger UI: `http://127.0.0.1:8899/docs`
- `/health`: 현재 로드된 champion 모델 버전 확인
- `/predict`: 실험 CSV 업로드 → 양품/불량 판정 + 피처별 재구성오차 기여도(`feature_contributions`,
  train(정상) 기준 z-score 큰 순 정렬) — 어떤 변수가 이 샷에서 평소 대비 가장 크게 벗어났는지 확인 가능

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
cd version_2
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
cd version_2
nice -n 19 uv run python scripts/run_lstm_training.py   # MLflow에 새 run 기록 + 모델 등록
uv run python scripts/promote_model.py <등록된 버전 번호>  # 그 버전을 champion으로 승격
```

## 3. WSL에서 Windows 브라우저로 접속이 안 될 때

- 기본은 `http://localhost:<port>` 또는 `http://127.0.0.1:<port>`로 바로 열림
  (WSL2가 자동으로 포워딩해줌).
- 안 열리면 VS Code의 PORTS 패널이 해당 포트를 자기가 먼저 잡고 있을 수 있음
  (Remote-WSL 확장의 자동 포워딩이 꼬이는 경우) — PORTS 탭에서 그 포트
  forwarding을 끄거나, 다른 포트 번호로 서버를 띄워서 우회.
- Windows 쪽에서 `netstat -ano | findstr :<port>`로 그 포트를 누가 물고
  있는지 먼저 확인하면 원인 파악이 빠름.

## 4. 프로젝트 구조 참고

- `src/preprocessing/`: 원본 CSV → train/eval 분할, 스케일링
- `src/lstm_ae/`: LSTM-Autoencoder 모델, 학습, 채점, `tracking.py`(MLflow 설정)
- `src/serving/`: 추론 로직(`inference.py`) + FastAPI 앱(`app.py`)
- `docs/specs/`, `docs/plans/`: 설계 스펙 / 구현 계획 문서
