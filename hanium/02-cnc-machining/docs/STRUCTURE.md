# 개발 파이프라인 디렉터리 구조

CNC 가공 이상탐지 파이프라인(`02-cnc-machining/`)의 폴더·파일 구성과 각각의
역할을 정리한 문서입니다. 실행 방법은 [README](../README.md)를 참고하세요.

---

## 1. 현재 상태

**구현 완료** — 전처리 · 학습/평가 · MLflow 실험추적 · champion 승격 · FastAPI
서빙 · RAG 조치 가이드 · 드리프트 모니터링까지 파이프라인 전 구간이 동작합니다.
단위 테스트 100개 통과.

**성능** — 고정 분할(eval: 정상 3 + 불량 11개 실험) 기준 **precision 0.91 /
recall 0.91**. 이 결과가 분할 운에 기댄 것인지 확인하기 위해 정상 실험 11개로
LOOCV를 돌렸고, 9개를 정상으로 맞춰 대표성을 확인했습니다.

**알려진 한계** — LOOCV에서 오탐이 난 2개가 모두 `feedrate=20` 조건이었습니다.
이 조건의 정상 샘플이 데이터셋 전체에 exp2(unworn)/exp22(worn) **단 2개뿐**이라,
모델이 "이 구간의 정상"을 학습할 근거 자체가 부족합니다. OOD 게이트 · 소규모
증강 · LOOCV 폴드 앙상블 세 가지 접근을 각각 구현해 검증했으나 모두 실패했고,
앙상블에서는 11개 모델이 만장일치로 틀렸습니다. 원인이 모델이나 하이퍼파라미터가
아니라 **해당 조건의 표본 부족**임이 확인돼, 무리하게 맞추는 대신 알려진 한계로
문서화하는 쪽으로 정리했습니다. 상세 검증 과정은
`docs/specs/2026-08-12-cnc-loocv-validation-design.md`와
`docs/specs/2026-08-12-cnc-rare-sample-augmentation-design.md`에 있습니다.

## 2. 저장소 전체

프로젝트는 두 트랙으로 나뉘며, **아래 `02-` 트랙이 현재 진행 중인 파이프라인**입니다.

```
hanium/
├── 01-cn7-injection-molding/   1차 트랙 — 사출성형(CN7). 방법론 검증용, 종료
└── 02-cnc-machining/           본 파이프라인 — CNC 가공 + MLOps + RAG  ★
```

1차 트랙에서 LSTM-Autoencoder 이상탐지 방법론을 세웠으나, 평가셋의 진짜 불량이
18건뿐이라 성능 지표를 신뢰하기 어려웠습니다. 그래서 **같은 방법론을 불량 표본이
확보된 CNC 데이터에 재적용해 유효성을 검증**했고(precision 0.91 / recall 0.91),
거기에 실제 운용에 필요한 MLOps와 RAG를 얹은 것이 `02-` 트랙입니다.

## 3. 파이프라인 흐름

데이터가 흘러가는 순서와, 각 단계를 담당하는 폴더의 대응 관계입니다.

```
 ① 원본 CNC 실험 CSV (25개 실험)
      │              data/dataset/
      ▼
 ② 전처리 ─────────  src/preprocessing/   ← scripts/run_preprocessing.py 로 실행
      │              피처 41개 선택 · 라벨 부여 · 정규화 · train/eval 분할
      ▼              산출물: data/processed/
 ③ 학습 · 평가 ────  src/lstm_ae/         ← scripts/run_lstm_training.py 로 실행
      │              정상 실험만 학습 → 재구성오차 → 임계값 산정
      ▼              산출물: data/model/ + MLflow 기록
 ④ 실험 추적 ─────  src/lstm_ae/tracking.py
      │              MLflow에 params/metrics/모델 등록
      ▼              저장소: data/mlflow/ (sqlite)
 ⑤ champion 승격 ─  scripts/promote_model.py <버전번호>
      │              특정 버전에 champion alias 부여 = 서빙 대상 확정
      ▼
 ⑥ 추론 서빙 ─────  src/serving/          ← uvicorn serving.app:app
      │              CSV 업로드 → 판정 + 피처별 기여도
      ├─────────────  src/rag/            불량 판정 시 현장 조치 가이드 생성
      └─────────────  src/monitoring/     요청 로깅 + 입력/출력 드리프트 감지
```

**핵심 설계**: 임계값을 라벨에서 역산하지 않습니다. train(정상) 데이터의 오차
분포만으로 정하고 eval에는 적용만 합니다. 라벨에 맞춰 임계값을 튜닝하면 지표는
좋아지지만 현장에서 재현되지 않기 때문입니다.

## 4. 디렉터리 구조

폴더는 **두 종류**입니다. `src/`는 import해서 쓰는 **라이브러리**이고, 최상위의
나머지 폴더들은 각각 하나의 **일회성 분석**으로 실행 스크립트와 결과물을 한 곳에
담고 있습니다. `src/rag/`와 `rag/`처럼 이름이 겹치는 쌍은 **안쪽이 라이브러리,
바깥쪽이 그걸 돌리는 스크립트**로 구분됩니다.

```
02-cnc-machining/
│
├── src/                            ■ 라이브러리 (실행 진입점 아님)
│   │
│   ├── preprocessing/              ② 전처리
│   │   ├── pipeline.py             ·  전처리 전체 오케스트레이션
│   │   ├── columns.py              ·  피처 41개 정의. 죽은 센서 4개 / 설정 상수 1개
│   │   │                           ·  / 메타데이터 3개 컬럼은 학습에서 제외
│   │   ├── cleaning.py             ·  Machining_Process 값의 "end" → "End" 표기 통일
│   │   ├── labels.py               ·  라벨 부여. 가공완료 & 육안검사 통과 = 양품(0),
│   │   │                           ·  둘 중 하나라도 아니면 불량(1)
│   │   ├── split.py                ·  train/eval 실험 ID 고정 분할
│   │   │                           ·  train 8개 / eval 정상 3개 + 불량 11개
│   │   │                           ·  (중복 실험 19·24·25는 제외)
│   │   ├── scaling.py              ·  StandardScaler 학습·적용 + scaler.json 저장
│   │   └── manifest.py             ·  전처리 이력 기록 (재현성 확보)
│   │
│   ├── lstm_ae/                    ③ 모델 학습·평가
│   │   ├── model.py                ·  LSTM-Autoencoder 정의
│   │   ├── sequencing.py           ·  시계열 → 고정 길이 윈도우로 절단
│   │   ├── training.py             ·  학습 루프
│   │   ├── pipeline.py             ·  학습~평가 전체 흐름. 윈도우별/피처별/
│   │   │                           ·  시점별 재구성오차 계산
│   │   ├── scoring.py              ·  실험 단위 오차 집계, 임계값 산정,
│   │   │                           ·  precision/recall 평가
│   │   ├── tracking.py             ·  MLflow 설정. 실험명 cnc-lstm-ae,
│   │   │                           ·  champion alias 관리, 경로 자동 복구
│   │   └── plotting.py             ·  혼동행렬·점수분포 등 시각화 (MLflow 첨부)
│   │
│   ├── serving/                    ⑥ 추론 서버
│   │   ├── app.py                  ·  FastAPI 앱. 엔드포인트 3개:
│   │   │                           ·  GET /health · POST /predict · GET /drift-status
│   │   └── inference.py            ·  판정 로직 + 피처별 기여도 순위화
│   │
│   ├── rag/                        ⑥ 현장 조치 가이드 생성
│   │   ├── query.py                ·  기여도 상위 피처 → 검색 질의문 구성
│   │   ├── features.py             ·  센서 코드 → 사람이 읽는 설명 매핑
│   │   ├── retrieval.py            ·  임베딩 + FAISS 유사도 검색
│   │   ├── generation.py           ·  검색 결과 기반 조치 가이드 생성 (OpenAI)
│   │   └── guide.py                ·  위 단계 오케스트레이션. 정상 판정 시 고정 응답
│   │
│   └── monitoring/                 ⑥ 운영 모니터링
│       ├── logging.py              ·  요청/판정 이력 sqlite 적재
│       ├── drift.py                ·  드리프트 판정. 입력 z>2.0, 불량비율>0.8
│       └── mlflow_logging.py       ·  드리프트 지표를 champion run에 기록
│
├── scripts/                        ■ 주 파이프라인 실행 진입점
│   ├── run_preprocessing.py        ·  ② 전처리 실행
│   ├── run_lstm_training.py        ·  ③ 학습 + ④ MLflow 기록
│   └── promote_model.py            ·  ⑤ 지정 버전을 champion으로 승격
│
├── rag/                            ■ RAG 지식 코퍼스 구축 (최초 1회)
│   ├── build_corpus.py             ·  원문 → 청크 → 임베딩 → FAISS 인덱스
│   └── sources/                    ·  공개 기술문서 원문 (로컬 저장, 웹 접근 불필요)
│       ├── sandvik_milling_troubleshooting.md
│       └── osha_machine_guarding_lockout.md
│
├── loocv/                          ■ 교차검증 — 고정 분할이 대표값인지 확인
│   ├── run_loocv.py                ·  정상 실험 11개 leave-one-out
│   ├── run_loocv_augmented.py      ·  증강 데이터 적용판
│   └── summary.{json,csv}          ·  결과: 정상 11개 중 9개 정확 분류
│
├── synthetic/                      ■ 데모용 합성 이상 시나리오
│   ├── generate_synthetic.py       ·  정상 실험에 도메인 지식 기반 이상 주입
│   └── scenarios/                  ·  공구마모 / 이송축부하 / 진동 3종 ×
│                                   ·  (이상 + 정상변형) = 입력 CSV 6개 + 검증 기록
│
├── augmentation/                   ■ 희소 샘플 증강 실험
│   └── generate_augmented.py       ·  결과물은 용량 문제로 git 제외
│
├── monitoring/                     ■ 드리프트 상황 시뮬레이션
│   └── simulate_drift.py           ·  운영 중 분포 변화 재현 (콘솔 출력)
│
├── tests/                          ■ 단위 테스트 100개 — src/ 구조를 그대로 반영
│   ├── preprocessing/  lstm_ae/  serving/  rag/  monitoring/
│   └── test_mlops_dependencies.py  ·  MLOps 의존성 설치 확인
│
├── docs/
│   ├── STRUCTURE.md                ·  이 문서
│   ├── specs/                      ·  기능별 설계 스펙 (10건)
│   └── plans/                      ·  스펙에 대응하는 구현 계획 (9건)
│
├── data/                           ■ git 미포함 — 별도 전달 필요
│   ├── dataset/                    ·  원본 CNC 실험 CSV
│   ├── processed/                  ·  ② 산출물: train/eval.csv, scaler.json, manifest.json
│   ├── model/                      ·  ③ 산출물: model.pt, evaluation_report.json 등
│   ├── mlflow/                     ·  ④ MLflow sqlite DB + 모델 아티팩트
│   ├── rag/                        ·  FAISS 인덱스 + 코퍼스
│   ├── monitoring/                 ·  요청 이력 DB
│   └── guide/                      ·  참고 가이드북 PDF
│
├── README.md                       ·  설치 · 실행 · 트러블슈팅
├── pyproject.toml                  ·  의존성 (uv 관리, PyTorch는 CPU 빌드)
└── uv.lock
```

## 5. 읽는 순서 추천

파이프라인을 처음 보신다면 아래 순서가 이해하기 쉽습니다.

| 순서 | 파일 | 왜 |
|---|---|---|
| 1 | `src/preprocessing/split.py` | 어떤 실험을 학습/평가에 썼는지가 모든 지표의 전제 |
| 2 | `src/preprocessing/columns.py` | 41개 피처와 제외 컬럼의 근거 |
| 3 | `src/lstm_ae/pipeline.py` | 학습~평가 전체 흐름 (가장 핵심) |
| 4 | `src/lstm_ae/scoring.py` | 임계값을 어떻게 정하는지 |
| 5 | `src/serving/app.py` | 실제 서비스 형태 |

## 6. 참고 사항

- **`data/`는 git에 포함되지 않습니다.** 원본 데이터·학습된 모델·MLflow 기록이
  들어 있어 용량이 크기 때문이며, 별도 전달이 필요합니다(README 1-2절).
- **각 스크립트는 자기 파일 위치로 프로젝트 루트를 역산합니다**
  (`Path(__file__).parent.parent`). 폴더 깊이가 코드에 반영돼 있어, 스크립트를
  다른 깊이로 옮기면 경로가 깨집니다.
- `loocv/`, `synthetic/`, `augmentation/`, `monitoring/`은 **검증·실험용이라 서빙
  경로에 영향을 주지 않습니다.** 서버 운용에는 `src/`, `scripts/`, `data/`만
  필요합니다.
