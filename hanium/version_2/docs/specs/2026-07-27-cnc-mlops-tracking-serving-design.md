# CNC LSTM-AE 실험 추적 및 추론 서빙(MLOps) 설계

- 날짜: 2026-07-27
- 상태: 설계 완료, 구현 전

## 배경

전처리(`2026-07-24-cnc-preprocessing-pipeline-design.md`)와 LSTM-AE 모델링(`2026-07-24-cnc-lstm-autoencoder-design.md`)은 완료되어, 리키지를 제거한 최종 결과(precision/recall 0.8~0.9대, 방법별 상이)를 확보했다.

한이음 캡스톤 프로젝트는 모델링 결과 자체뿐 아니라 엔지니어링 성숙도(실험 관리, 모델 버전 관리, 서빙)도 보여줘야 한다. 최종 발표 목표는 "데이터 입력 → 모델 판정 → RAG 기반 원인 설명 → 조치 방향 제시"를 담은 데모 영상이며, RAG 설명/조치 제안은 이후 별도 스펙에서 다룬다. 이 스펙은 그 데모의 백본이 될 **학습 → 추적/등록 → 서빙** 파이프라인만 다룬다.

사용자와 논의를 거쳐, MLflow Model Registry까지 연결하되(더 "MLOps다운" 그림), 상시 `mlflow server` 프로세스 없이 **sqlite 파일 백엔드**로 구현하기로 결정했다 — Registry 기능(모델 등록/별칭 조회)은 DB 백엔드가 있으면 되고, 그 DB가 로컬 sqlite 파일이면 별도 서버 프로세스 없이 학습 스크립트와 서빙 앱이 같은 파일에 직접 접근할 수 있다. 데모 중 별도 프로세스를 띄우고 유지해야 하는 리스크를 없앤다.

## 목표 / 비목표

**목표**
- 학습을 실행할 때마다 설정(config)·지표(precision/recall 등)·모델 아티팩트가 MLflow에 자동으로 기록된다.
- 기록된 모델 버전 중 하나를 사람이 검토 후 "champion"으로 명시적으로 승격한다.
- FastAPI 서버가 champion 모델을 로드해, 원본 실험 CSV 하나를 입력받아 양품/불량 판정과 근거 점수를 JSON으로 반환한다.
- 서빙 응답에 어느 MLflow run/모델 버전에서 나온 판정인지 포함해, "지금 서빙 중인 모델"과 "그 모델을 만든 실험 기록"을 연결한다.

**비목표**
- RAG 기반 원인 설명·조치 제안 로직 (다음 스펙)
- 센서(피처)별 재구성오차 분해 — RAG 연결 시 별도 설계
- CI, 실시간 드리프트 모니터링
- 지도학습 분류기 비교 트랙
- 리키지 수정 이전 결과의 소급 기록 — 아티팩트가 이미 사라졌고, 스펙 문서의 "사후 보정" 절로 이미 기록돼 있어 생략
- 모델 자동 승격(성능 기준 자동 champion 교체) — 아래 "모델 등록과 승격" 절 참고

## 전체 아키텍처

```
전처리 (기존, 변경 없음)
  → data/processed/{train.csv, eval.csv, scaler.json, manifest.json}

학습 (scripts/run_lstm_training.py, 수정)
  → mlflow.start_run()
       params:  training_config 전체 + split 정보(manifest.json에서) + (MLflow가 자동 기록하는 git commit)
       metrics: {mean,max,p95} × {threshold,precision,recall,tp,fp,fn,tn}
       model:   mlflow.pytorch.log_model(..., registered_model_name="cnc-lstm-ae")
  → data/model/ 아래 기존 산출물도 그대로 저장 (하위 호환, 로컬 확인용)

승격 (scripts/promote_model.py, 신규, 수동 실행)
  → 지정한 버전에 alias "champion" 부여

서빙 (src/serving/, 신규, FastAPI)
  → 기동 시 mlflow.pytorch.load_model("models:/cnc-lstm-ae@champion") + 해당 run의 metrics(threshold) 조회
  → POST /predict: 원본 실험 CSV 업로드 → 전처리(scaler.json) → 윈도잉 → 추론 → 집계 → 판정 반환
```

## 1. 실험 추적 + 모델 레지스트리 (MLflow)

### 백엔드 구성

- Tracking/Registry URI: `sqlite:///{ROOT}/data/mlflow/mlflow.db` (`ROOT` = `version_2/`)
- 실험(experiment) 이름: `cnc-lstm-ae`, 최초 생성 시 `artifact_location`을 `{ROOT}/data/mlflow/artifacts`(로컬 경로)로 명시 지정 — 지정하지 않으면 MLflow가 기본값으로 서버 프록시가 필요한 경로를 쓸 수 있어, 반드시 로컬 파일 경로로 고정한다.
- `data/mlflow/`는 `data/`(이미 `.gitignore`에 포함) 하위이므로 git에 올라가지 않는다. `model.pt` 등 기존 산출물과 동일한 성격(재실행하면 재생성되는 산출물)이라 일관적이다.

### 학습 스크립트 변경 (`scripts/run_lstm_training.py`)

`run_lstm_pipeline()`(순수 함수, `lstm_ae/pipeline.py`)은 건드리지 않는다. MLflow 로깅은 이 함수를 호출하는 진입점(`run_lstm_training.py`)에서 감싸는 방식으로 추가한다 — 학습 로직과 추적 로직을 분리 유지.

```python
with mlflow.start_run():
    mlflow.log_params({**training_config, "train_experiment_ids": ..., "eval_good_experiment_ids": ..., "eval_bad_experiment_ids": ...})
    summary = run_lstm_pipeline(...)
    for method in ["mean", "max", "p95"]:
        r = summary["results"][method]
        mlflow.log_metrics({
            f"{method}_threshold": summary["thresholds"][method],
            f"{method}_precision": r["precision"],
            f"{method}_recall": r["recall"],
            f"{method}_tp": r["tp"], f"{method}_fp": r["fp"],
            f"{method}_fn": r["fn"], f"{method}_tn": r["tn"],
        })
    mlflow.pytorch.log_model(
        summary["model"], artifact_path="model", registered_model_name="cnc-lstm-ae",
        serialization_format="pickle",
    )
```

`serialization_format="pickle"`은 필수다 — mlflow 최신 버전의 기본값(`pt2`, `torch.export` 기반 그래프 트레이싱)은 `nn.LSTM`을 포함한 모델에서 `Constraints violated (dynamic_dim)` 오류로 실패함을 스파이크로 확인했다(cuDNN 기반 LSTM 구현이 `torch.export`로 완전히 트레이싱되지 않는 것으로 보임). `pickle` 포맷은 트레이싱 없이 객체를 그대로 직렬화하므로 문제없이 동작한다.

- split 정보(`train_experiment_ids` 등)는 `data/processed/manifest.json`에서 읽어 params로 남긴다 — "이 모델이 어떤 실험 구성으로 학습됐는지"가 리키지 수정 전/후를 구분하는 핵심 정보이기 때문.
- git commit hash는 MLflow가 git 저장소 안에서 실행되면 자동으로 태그(`mlflow.source.git.commit`)로 남기므로 별도 코드 불필요.
- `mlflow.pytorch.log_model(..., registered_model_name=...)`은 아티팩트 저장과 Registry 버전 등록을 한 번에 수행한다.
- 현재 `run_lstm_pipeline()`은 학습된 `model` 객체를 반환하지 않고 `data/model/model.pt`에 저장만 하므로, **반환값(`summary`)에 `"model"` 키로 학습된 모델 객체를 추가**해야 한다(모델을 다시 로드하지 않고 바로 `mlflow.pytorch.log_model`에 넘기기 위해) — 이 스펙의 유일한 `pipeline.py` 변경점.

### 모델 등록과 승격 (alias)

학습을 돌릴 때마다 **자동으로** Registry에 새 버전이 등록되지만, 그중 어떤 버전을 서빙이 쓸지는 **자동으로 정하지 않는다**. 이유: 재학습이 항상 이전보다 나은 결과를 낸다는 보장이 없고(하이퍼파라미터 실험 중 성능이 떨어지는 시도도 정상적으로 발생), 자동 승격은 실수로 나쁜 모델이 서빙에 올라가는 사고로 이어질 수 있다. 대신:

- `scripts/promote_model.py <version>` (신규, 수동 실행): 지정한 버전 번호에 `MlflowClient().set_registered_model_alias("cnc-lstm-ae", "champion", version)` 호출.
- 개발자가 `mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db`로 여러 run의 metrics를 비교한 뒤, 원하는 버전을 수동으로 승격한다.
- 최초 1회는 지금 확보된 리키지 수정 후 결과를 재현하는 학습 1회를 이 파이프라인으로 실행하고, 그 버전을 champion으로 승격하는 것으로 시작한다.

## 2. 추론 서빙 (FastAPI)

### 패키지 구조

```
version_2/src/serving/
  inference.py   # 순수 로직: 모델 로드, 전처리, 윈도잉, 추론, 집계, 판정
  app.py         # FastAPI 라우팅만 (inference.py 호출)
```

`inference.py`는 FastAPI에 의존하지 않는 순수 함수들로 구성해, 서빙 프레임워크 없이도 단위 테스트할 수 있게 한다(기존 `lstm_ae`/`preprocessing` 패키지 스타일과 동일).

### 모델 로딩 (기동 시 1회)

FastAPI `lifespan`에서:
1. `mlflow.pytorch.load_model("models:/cnc-lstm-ae@champion")`로 모델 로드.
2. 같은 버전의 `run_id`로 `MlflowClient().get_run(run_id).data.metrics`를 조회해 `{method}_threshold` 3종을 가져온다 — threshold를 로컬 파일이 아니라 MLflow에서 가져오게 해, "서빙 중인 모델과 그 threshold가 항상 같은 run에서 나온다"는 일관성을 보장한다.
3. champion alias가 아직 없으면(최초 배포 전) 또는 로드 중 에러가 나면, 기동을 실패시키지 않고 에러를 로그로 남긴 뒤 모델 미로딩 상태로 서버를 그대로 기동한다. 이 상태에서는 `GET /health`, `POST /predict`가 503을 반환하며, `scripts/promote_model.py`를 실행해 champion을 지정하면 다음 기동부터 정상 로드된다.

`scaler.json`(전처리 산출물)은 지금처럼 `data/processed/scaler.json`을 직접 읽는다 — 전처리는 이 스펙의 추적 대상이 아니고(전처리 자체가 자주 바뀌지 않음), 매 학습마다 이 파일이 달라지지 않으므로 MLflow에 별도로 얹지 않는다.

### `POST /predict` 계약

- **요청**: multipart 파일 업로드, 원본 실험 CSV 1개(`experiment_XX.csv`와 동일한 원시 컬럼 포맷 — 48개 원시 센서/메타데이터 컬럼, `FEATURE_COLUMNS` 41개 포함). 쿼리 파라미터 `method`(`mean`|`max`|`p95`, 기본값 `mean`).
- **처리**:
  1. 필수 컬럼(`FEATURE_COLUMNS`) 존재 확인 — 누락 시 400, 누락된 컬럼 목록 반환.
  2. `scaler.json`으로 스케일링(수동 계산: `(x - mean) / std`, `preprocessing.scaling`의 sklearn 객체를 재구성하지 않고 dict 그대로 사용해 단순화).
  3. `experiment_id` 컬럼에 상수값(0)을 부여해 `lstm_ae.sequencing.make_eval_windows` 재사용(윈도잉 로직 중복 방지).
  4. 업로드된 실험 길이가 `window_size`(champion run의 param에서 조회, MLflow params는 문자열로 저장되므로 int 변환) 미만이면 400, "최소 N행 필요" 메시지 반환.
  5. 모델 추론 → 윈도우 오차 → `lstm_ae.scoring.aggregate_window_errors_by_experiment`로 집계 → 요청된 `method`의 threshold와 비교.
- **응답 예시**:
  ```json
  {
    "predicted_label": 1,
    "predicted_label_text": "bad",
    "score": 3.21,
    "threshold": 0.857,
    "method": "mean",
    "model_version": "3",
    "mlflow_run_id": "a1b2c3..."
  }
  ```

### `GET /health`

champion 모델 로드 여부와 `model_version`/`mlflow_run_id`를 반환 — 데모 중 "지금 어떤 모델이 떠 있는지" 바로 보여줄 수 있는 용도.

### 데모용 데이터

발표 데모 영상에 넣을 실험 CSV는 **eval 세트(양품 3개, 불량 11개)에서 재사용**한다 — 양품 1개, 불량 1개를 골라 `POST /predict`에 넣는다. eval은 학습에 전혀 쓰이지 않으므로 "모델이 본 적 없는 데이터"라는 조건을 충족하고, 데모에 쓴다고 precision/recall 등 이미 계산된 지표가 바뀌지도 않는다.

제외된 중복 실험(19/24/25, `EXCLUDED_DUPLICATE_EXPERIMENT_IDS`)은 데모용으로도 쓰지 않는다 — 실험14/24는 사실상 동일 데이터라 24를 데모에 쓰면 "학습 데이터를 그대로 다시 넣어서 맞히는" 리키지 데모가 되고, 19/25는 라벨이 서로 모순돼 정답을 신뢰할 수 없다.

eval을 더 쪼개 "데모 전용"으로 따로 떼어두는 대안도 검토했으나, 안 그래도 얇은 eval(14개 실험)이 더 줄어들어 통계적 신뢰도가 낮아지는 단점이 재사용의 이점보다 크다고 판단해 채택하지 않았다.

## 3. 재현성 연결고리

서빙 응답의 `mlflow_run_id`로 MLflow UI에서 해당 run의 params(split 구성, 하이퍼파라미터)와 git commit 태그를 바로 확인할 수 있다. 향후 지도학습 분류기를 추가하면 같은 `cnc-lstm-ae` 실험(experiment) 아래 별도 run으로 자연스럽게 쌓여, 두 방법론을 같은 MLflow UI에서 비교할 수 있다(이번 스펙 범위 밖이지만 설계가 막지 않음).

## 4. 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `version_2/src/lstm_ae/pipeline.py` | `run_lstm_pipeline()`이 학습된 `model` 객체를 반환값에 포함하도록 수정 (1줄) |
| `version_2/scripts/run_lstm_training.py` | MLflow run으로 감싸 params/metrics/model 로깅 추가 |
| `version_2/scripts/promote_model.py` | 신규 — champion alias 수동 지정 CLI |
| `version_2/src/serving/inference.py` | 신규 — 모델 로드, 전처리~판정 순수 로직 |
| `version_2/src/serving/app.py` | 신규 — FastAPI 라우팅(`/predict`, `/health`) |
| `version_2/pyproject.toml` | `mlflow`, `fastapi`, `uvicorn` 의존성 추가 |

## 5. 테스트 범위

- `inference.py`의 순수 함수(컬럼 검증, 스케일링, 판정 로직)는 실제 MLflow 레지스트리 없이 가짜 모델/scaler dict로 단위 테스트.
- FastAPI 라우팅은 `TestClient` + 의존성 주입으로 로드된 모델을 스텁으로 교체해 테스트(실제 sqlite MLflow 스토어를 테스트에서 띄우지 않음).
- MLflow 로깅 코드 자체(`run_lstm_training.py`의 `mlflow.log_*` 호출)는 별도 단위 테스트를 만들지 않는다 — 실제 MLflow 스토어에 의존하는 통합 성격이라, 수동 실행 후 `mlflow ui`로 눈으로 확인하는 것으로 충분하다고 판단.

## 6. 검증 방법 (수동 시나리오)

1. `scripts/run_lstm_training.py` 실행 → `mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db`로 run이 기록됐는지, params/metrics/모델 아티팩트가 다 보이는지 확인.
2. `scripts/promote_model.py <version>` 실행 → `MlflowClient().get_model_version_by_alias(...)`로 champion이 지정한 버전을 가리키는지 확인.
3. FastAPI 서버 기동 → `GET /health`로 champion 버전 확인.
4. `data/dataset/`의 실제 실험 CSV(양품 1개, 불량 1개) 각각을 `POST /predict`에 넣어 라벨과 일치하는 판정이 나오는지 확인.
5. 컬럼이 누락된 CSV, `window_size`보다 짧은 CSV로 각각 400 에러가 의도한 메시지와 함께 오는지 확인.
