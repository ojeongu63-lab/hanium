# CNC 드리프트 모니터링 설계

- 날짜: 2026-08-12
- 상태: 설계 완료, 구현 전

## 배경

멘토 제안: "제조공정은 시간이 지나며 드리프트가 생기니, MLOps 관점에서 이걸
우리 모델에도 반영해보자."

**참고 전례**: 같은 리포지토리의 CN7 사출성형 트랙에서 실제로 드리프트를 겪은
적이 있다 — unlabeled(train)와 eval의 시간범위가 거의 안 겹쳐 정상 샷
재구성오차가 특정 시점 이후 수백~수만 배로 치솟았고, "롤링 z-score 보정"으로
해결했다. CNC(02-cnc-machining)에는 이런 보정 로직이 전혀 없다(`lstm_ae/pipeline.py`
확인 완료, detrend 관련 코드 없음).

**CNC의 제약**: 데이터가 25개 실험짜리 고정된 공개 데이터셋이라 실시간
스트리밍 데이터가 없다 — "진짜 드리프트"가 자연발생적으로 존재하지 않는다.
따라서 이번 작업은 (1) 실제 `/predict` 요청을 누적해 드리프트를 감지하는
**진짜 동작하는 기능**을 만들고, (2) 검증은 `synthetic/generate_synthetic.py`
방식(실제 정상 실험에 도메인지식 기반 변형 주입)을 재사용해 **점진적으로
커지는 합성 배치 시퀀스**로 한다.

## 목표 / 비목표

**목표**
- `/predict` 호출마다 입력 피처 요약값 + 판정 점수를 로컬에 누적 기록한다.
- 최근 N번의 요청을 기존 train 기준 아티팩트(`scaler.json`,
  `feature_baseline.json`)와 비교해 입력/출력 드리프트를 감지한다.
- 드리프트 상태를 조회할 수 있는 새 엔드포인트를 제공한다.
- 점진적 합성 시퀀스로 이 메커니즘이 실제로 반응하는지 검증한다.

**비목표**
- 자동 재학습/재승격 (이 프로젝트는 champion 승격을 이미 수동으로만 하기로
  정했음 — `promote_model.py` 수동 실행 — 드리프트 감지도 같은 원칙: 사람이
  판단하고 실행)
- 실제 라이브 스트리밍 데이터 수집(존재하지 않음, 비목표)
- 알림(이메일/슬랙 등) 발송 — 조회 가능한 엔드포인트까지만

## Part A — 요청 로그 누적 + 드리프트 계산 (`src/monitoring/` 신규)

### 저장 방식
Python 표준 라이브러리 `sqlite3`(새 의존성 없음)로 `data/monitoring/requests.db`에
누적. `data/` 하위라 기존 관례대로 `.gitignore` 대상(런타임에 계속 쌓이는
운영 데이터라 애초에 git에 넣을 이유가 없음).

테이블 `predict_log`:
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `timestamp` | TEXT | ISO 8601 |
| `feature_means_json` | TEXT | 41개 피처의 **스케일링된(scaled) 값**을 실험 내 평균낸 것, JSON 딕셔너리 |
| `score` | REAL | 그 요청의 `score`(원본, 스케일링 안 함) |
| `predicted_label_text` | TEXT | "good"/"bad" |

**왜 스케일링된 값을 저장하는가**: `scaler.json`은 train에 fit돼 있어서, train과
같은 분포라면 스케일링 후 평균이 0 근처에 와야 한다. 즉 **저장 시점에 이미
"몇 시그마 벗어났는지"에 가까운 단위**라, 드리프트 체크 때 별도 정규화 계산이
필요 없다 — `predict_experiment()`가 이미 계산해둔 `scaled` 데이터프레임의
평균을 그대로 재사용.

### `src/monitoring/logging.py`
- `log_request(feature_means: dict[str, float], score: float, predicted_label_text: str, db_path: Path) -> None`
- `get_recent_requests(n: int, db_path: Path) -> list[dict]` — 최신 순으로 최대 n개

### `src/monitoring/drift.py` (순수 함수, 테스트 쉬움)
- `compute_drift_status(recent_requests: list[dict], threshold: float, window_size: int) -> dict`
- 로직:
  1. `recent_requests`가 `window_size`(기본 10)개 미만이면 `sufficient_data: false`로 조기 반환
  2. **입력 드리프트**: 최근 N개의 `feature_means_json`을 피처별로 평균 → 그 값의 절댓값이 **2.0**(스케일링 단위 기준 2 표준편차)을 넘는 피처를 `flagged_features`로 표시
  3. **출력 드리프트**: 최근 N개의 `score` 평균을 `threshold`와 비교한 비율(`avg_score / threshold`)이 **0.8**을 넘으면 `flagged: true` (실제로 넘기 전에 추세를 미리 보기 위한 여유선 — 정확한 "판정"이 아니라 "주의" 신호)
  - 두 임계값(2.0, 0.8)은 휴리스틱이며, 실제 운영 데이터가 쌓이면 조정 대상

### `/predict` 통합 (`serving/app.py`만 수정, `inference.py`/`predict_experiment()`는 안 건드림)
`predict_experiment()`의 반환 스키마는 그대로 유지한다(기존 클라이언트 호환성
유지 — `scaled` 데이터프레임은 그 함수 내부 지역변수라 밖에서 접근 불가하고,
반환값에 새 필드를 추가하지 않기로 했으므로 재사용하지 않는다). 대신
`app.py`의 `/predict` 라우트에서 **`scale_features()`를 한 번 더 호출**해
(이미 `serving.inference`에 있는 함수 재사용, 계산량 미미) 피처별 평균을
독립적으로 구하고, `predict_experiment()` 호출 결과(`score`,
`predicted_label_text`)와 합쳐 `log_request(...)`를 호출한다:

```python
scaled = scale_features(df, FEATURE_COLUMNS, state.scaler_dict)
feature_means = scaled[FEATURE_COLUMNS].mean().to_dict()
log_request(feature_means, result["score"], result["predicted_label_text"], DB_PATH)
```

### 신규 엔드포인트 `GET /drift-status`
```json
{
  "n_requests_logged": 47,
  "window_size": 10,
  "sufficient_data": true,
  "input_drift": {
    "flagged_features": [{"feature": "S_OutputCurrent", "avg_scaled_mean": 2.3}],
    "all_feature_avg_scaled_means": {"...": "41개 전부, 참고용"}
  },
  "output_drift": {
    "avg_score_recent": 0.72,
    "threshold": 0.857,
    "ratio_to_threshold": 0.84,
    "flagged": true
  },
  "checked_at": "2026-08-12T13:00:00"
}
```

## 검증 — 합성 점진적 드리프트 시뮬레이션

**스크립트**: `02-cnc-machining/monitoring/simulate_drift.py`(일회성 검증 스크립트,
`src/`에 안 넣음 — `loocv`/`synthetic`과 같은 관례).

- `fastapi.testclient.TestClient(app)`로 실제 앱(실제 champion 모델 로드,
  실제 lifespan 실행)에 대고 호출 — 별도 서버 프로세스 없이 검증.
- 실제 정상 실험(`experiment_01.csv`) 기준으로, `synthetic/generate_synthetic.py`의
  jittering 방식을 재사용하되 **자동보정 없이 고정된 진폭 단계**를 순서대로 적용:
  진폭 `[0.0, 0.02, 0.04, 0.06, 0.08, 0.10]` 6단계, 각 단계마다 `window_size`(10)번
  씩 `/predict` 호출(같은 진폭 CSV를 시드만 바꿔 10번 — 매번 완전히 동일한
  요청이면 "누적"의 의미가 없으므로).
- 마지막에 `GET /drift-status` 호출 → `input_drift.flagged_features`에 뭔가
  잡히는지, `avg_scaled_mean`이 진폭에 따라 실제로 커지는 추세인지 확인.
- **검증 전 가정하지 않는다**: 진폭 0.10에서도 안 잡히면 임계값(2.0/0.8)을
  조정해야 한다는 뜻으로, 실패도 있는 그대로 보고한다.

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `02-cnc-machining/src/monitoring/logging.py` | 신규 |
| `02-cnc-machining/src/monitoring/drift.py` | 신규 |
| `02-cnc-machining/src/serving/app.py` | 수정 — `/predict`에 로그 기록 추가, `GET /drift-status` 신규 |
| `02-cnc-machining/monitoring/simulate_drift.py` | 신규 — 검증 스크립트 |
| `02-cnc-machining/data/monitoring/` | 신규 — `.gitignore` 대상(`data/` 관례) |

## 테스트 범위

`src/`에 들어가는 정식 서빙 로직이라 pytest 단위테스트 작성(RAG와 동일 관례):
- `drift.py`의 `compute_drift_status()`: 순수 함수, 데이터 부족/정상/드리프트
  3가지 케이스를 가짜 `recent_requests` 리스트로 테스트 — DB/모델 불필요.
- `logging.py`: 임시 SQLite 파일(`tmp_path` fixture)로 `log_request`→`get_recent_requests`
  왕복 테스트.
- `/drift-status` 라우트: 기존 `test_app.py` 관례(`TestClient`+의존성 오버라이드)로
  통합 테스트.

## 검증 방법

1. pytest 전체 통과(기존 84개 + 신규).
2. `simulate_drift.py` 실행 → 진폭 단계별 `avg_scaled_mean` 로그 출력, 마지막
   `/drift-status` 응답 전체 출력.
3. 진폭이 커질수록 드리프트 지표가 실제로 커지는 추세인지, 어느 단계부터
   `flagged`가 `true`로 바뀌는지 사용자에게 있는 그대로 보고(가정하지 않음).
4. 결과를 사용자에게 보고.
