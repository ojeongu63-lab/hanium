# CNC 합성 데이터 현실성 보정 설계

- 날짜: 2026-08-12
- 상태: 설계 완료, 구현 전
- 대체 대상: `2026-08-12-cnc-synthetic-anomaly-scenarios-design.md`(이상 시나리오),
  `2026-08-12-cnc-drift-monitoring-design.md`의 "검증 — 합성 점진적 드리프트
  시뮬레이션" 절

## 배경

기존 합성 데이터 2종(이상 시나리오, 드리프트 시뮬레이션)이 모델 판정만 맞추도록
설계돼, **산출된 값 자체가 물리적으로 말이 안 되는 수준**임이 확인됐다.

실측(현 champion 모델로 실제 25개 실험 전부 `predict_experiment()` 실행):

| 대상 | top1 재구성오차 z-score | score/threshold 비율 |
|---|---|---|
| GOOD 실험 10개(exp22 제외) | 0.70 ~ 4.44 | 0.43 ~ 1.30 |
| BAD - 공정중단형 6개 | 12.79 ~ 35.59 | 1.72 ~ 3.79 |
| BAD - 품질불합격형 5개 | 4.18 ~ 13.64 | 1.00 ~ 3.15 |
| **합성 `feed_overload`** | **29790.2** | — |
| **합성 `tool_wear`** | **12418.7** | — |
| 합성 `vibration_backlash` | 42.3 | — |
| **드리프트 시뮬 마지막 단계** | — | **70.4** |

즉 실제로 가능한 최댓값(z≈36, 비율≈3.8)의 **수백~수천 배**를 만들어내고 있었다.
원인은 두 스크립트 모두 "목표 라벨이 나올 때까지 진폭을 배가/증가"시키는 방식이라,
**현실성 상한이 설계에 존재하지 않았기 때문**이다.

이 데이터의 용도는 (1) 추후 구축할 RAG 조치 가이드와 엮어 보여줄 이상/정상
시나리오 데모, (2) MLOps 드리프트 모니터링 시연이다. 둘 다 **실제 값처럼 봐도
이상하지 않아야** 데모로서 의미가 있다.

### 새로 확인된 실제 라벨 정보

`data/dataset/CNC 비식별화 원본데이터_1209/train.csv`에 25개 실험의 원본 라벨이
있고, 기존에 `BAD_EXPERIMENT_IDS`로 뭉뚱그려 온 11개가 사실 **서로 다른 두 종류의
실제 실패**임을 확인했다. 이 구분은 이번 설계의 기준이 된다.

| 그룹 | 조건 | 실험 | 실측 패턴 |
|---|---|---|---|
| 공정중단형 | `machining_finalized == "no"` | 4, 5, 7, 16, 20, 23 | `Y_OutputPower`/`X_OutputPower` 주도 |
| 품질불합격형 | `passed_visual_inspection == "no"` | 6, 8, 9, 10, 21 | `CurrentFeedback`/`Velocity` 계열 주도 |

## 목표 / 비목표

**목표**
- 실측 BAD 실험의 z-score / score 범위를 산출해 재사용 가능한 기준값으로 저장한다.
- 이상 시나리오 3개의 진폭을 그 실측 대역 안으로 재보정한다.
- 드리프트 시뮬레이션을 "일부 피처만, 실측 범위 안에서 점진 증가"로 재설계한다.
- 재보정 결과가 실제로 대역 안에 들어왔는지 **스크립트가 자체 검증**한다.

**비목표**
- 모델 재학습·성능 개선 (이 데이터는 학습에 쓰지 않는다)
- RAG 연결 (다음 서브프로젝트로 보류 — 사용자 확정)
- 테스트/운영 트래픽 분리 (`predict_log`에 `source` 컬럼 추가 등)
  — 별개 문제로 확인됐고 이번 스코프 밖 (사용자 확정)
- 시나리오가 건드리는 **대상 피처 목록 변경** — 기존 컬럼셋 유지 (사용자 확정).
  `tool_wear`의 컬럼셋이 실측 품질불합격형 패턴과 완전히 일치하지는 않지만,
  변경 범위를 진폭 보정으로 한정한다.

## Part 0 — 실측 이상 기준 산출 (신규)

**파일**: `synthetic/analyze_real_anomalies.py` → `synthetic/real_anomaly_reference.json`

25개 실험 전부를 현 champion 모델로 `predict_experiment()`에 통과시켜, 실제
라벨(`machining_finalized`, `passed_visual_inspection`)로 3그룹(good /
process_interrupted / quality_failed)을 나눈 뒤 그룹별 통계를 뽑는다.

산출 JSON 구조:

```json
{
  "generated_at": "2026-08-12T...",
  "model_version": "<champion 모델 버전>",
  "mlflow_run_id": "...",
  "threshold_mean": 0.8566,
  "groups": {
    "good": {
      "experiment_ids": [1, 2, 3, 11, 12, 13, 14, 15, 17, 18, 22],
      "top1_z_score": {"min": 0.70, "median": 1.23, "max": 36.13},
      "score_ratio_to_threshold": {"min": 0.43, "median": 0.53, "max": 2.31}
    },
    "process_interrupted": {
      "experiment_ids": [4, 5, 7, 16, 20, 23],
      "top1_z_score": {"min": 12.79, "median": 27.88, "max": 35.59},
      "score_ratio_to_threshold": {"min": 1.72, "median": 2.27, "max": 3.79}
    },
    "quality_failed": {
      "experiment_ids": [6, 8, 9, 10, 21],
      "top1_z_score": {"min": 4.18, "median": 10.03, "max": 13.64},
      "score_ratio_to_threshold": {"min": 1.00, "median": 1.35, "max": 3.15}
    }
  },
  "per_experiment": {
    "4": {"group": "process_interrupted", "predicted": "bad", "score": 1.470,
          "top3": [["Y_OutputPower", 14.59], ["S_CurrentFeedback", 12.85], ["S_DCBusVoltage", 11.36]]}
  }
}
```

위 수치는 이미 실측으로 확인된 값이며, 스크립트는 이를 재현·저장하는 역할이다
(모델이 재학습되면 다시 돌려 갱신).

`good` 그룹의 max(z=36.13, 비율 2.31)는 exp22, 그 다음(비율 1.30)은 exp2로,
둘 다 **이미 알려진 오탐**이다(feedrate=20 데이터 커버리지 한계, 3가지 접근으로
해결 실패 확인됨). 대역 산정 시 이 두 건은 정상 상한의 근거로 쓰지 않는다.

`synthetic/` 하위(= `data/` 밖)라 git 추적 대상이다 — 설계 근거를 남기는 작은
JSON이므로 코드와 함께 버전 관리한다.

## Part A — 이상 시나리오 진폭 재보정 (`synthetic/generate_synthetic.py`)

### 시나리오 ↔ 실측 그룹 매핑 및 목표 대역

| 시나리오 | 대응 실측 그룹 | 목표 top1 z-score (중앙값 / 허용범위) | 근거 |
|---|---|---|---|
| `feed_overload` | process_interrupted | 27.9 / 12.8 ~ 35.6 | 이송축 부하 급증 = 공정 중단으로 이어지는 파워 급증 패턴과 일치 |
| `tool_wear` | quality_failed | 10.0 / 4.2 ~ 13.6 | 공구마모 = 가공은 완료되나 표면 품질 불합격 패턴과 일치 |
| `vibration_backlash` | quality_failed (보수적 적용) | 10.0 / 4.2 ~ 13.6 | **실제 대응 라벨 없음.** 근거 없이 큰 값을 만들지 않기 위해 두 그룹 중 낮은 쪽 대역을 상한으로 채택 |

### `calibrate()` 교체

기존(제거): 초기 진폭에서 시작해 목표 라벨이 나올 때까지 2배씩 증폭, 최대 5회.
→ 상한이 없어 라벨만 맞으면 z=29790도 통과.

신규:

1. **고정 진폭 후보 목록**을 전부 시도한다 (배가 루프 없음).
   후보: `[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]`
   — 하한은 기존 "정상 변형"이 쓰던 0.05보다 아래까지 내려 정상 쪽 경계를
   충분히 탐색하고, 상한 1.0은 기존 이상 시나리오 초기 진폭과 동일하게 둔다.
2. 각 후보마다 CSV 생성 → `predict_experiment()` → `(amplitude, label, top1_z)` 기록.
3. **이상 시나리오**: `label == "bad"`인 후보 중 `|top1_z - 목표중앙값|`이 최소인
   것을 선택.
4. **정상 변형**: `label == "good"`인 후보 중 **진폭이 가장 큰** 것을 선택
   — "이 정도까지는 정상"이라는 경계를 데모에서 보여주기 위함.
5. 조건을 만족하는 후보가 하나도 없으면 **에러로 중단**한다(조용히 넘어가지 않음).
6. 선택된 이상 시나리오의 `top1_z`가 **허용범위 밖이면 에러로 중단**한다.
   이 assert가 이번 설계의 핵심 안전장치 — 진폭 상한이 없어서 생긴 버그를
   런타임에 직접 막는다.

`select_best_amplitude()`는 부작용 없는 순수 함수로 분리해, 위 선택 규칙이
`predict_experiment()` 호출과 뒤섞이지 않게 한다.

### perturbation 함수

`tool_wear()`, `feed_overload()`, `vibration_backlash()`의 **대상 피처와 변형
방식은 그대로 유지**한다(사용자 확정). 바뀌는 것은 어떤 진폭을 고르느냐뿐이다.

## Part B — 드리프트 시뮬레이션 재설계 (`monitoring/simulate_drift.py`)

기존 문제 2가지:
1. `FEATURE_COLUMNS` **41개 전부**를 동시에 같은 방향으로 shift — 실제 드리프트는
   보통 일부 센서만 서서히 틀어진다.
2. 진폭 상한 없음 (`AMPLITUDES` 마지막 8.0 → score/threshold 비율 70.4).

### 신규 설계

- **이동 대상 피처를 2개로 축소**: `Y_OutputPower`, `X_OutputPower`
  — 실측 공정중단형에서 일관되게 주도 피처로 등장하는 조합이므로,
  "실제로 생길 법한 드리프트"에 가장 가깝다.
- **진폭 스케줄을 실측 비율 대역 안으로 재보정**: 목표 score/threshold 비율
  6단계 `[0.5, 1.0, 1.5, 2.2, 3.0, 3.8]`
  — 0.5는 정상 수준(GOOD 실측 0.43~1.30 대역), 3.8은 실측 BAD 최댓값(3.79)에
  해당. 즉 "정상에서 출발해 최악의 실제 불량 수준까지" 점진적으로 악화되는
  그림이 된다.
- 각 목표 비율을 만드는 실제 진폭 값은 **구현 시 경험적 스윕으로 결정**한다:
  일회성 스윕 스크립트로 진폭 그리드(예: `0.05`부터 `2.0`까지 10~15개 값)마다
  `shift()` 적용 → `predict_experiment()` 1회 → 달성 비율을 기록하고, 목표
  6개 각각에 대해 **가장 가까운 비율을 낸 진폭**을 고른다. 확정된 6개 값을
  `simulate_drift.py`에 상수로 박고 유도 근거(스윕 결과 표)를 주석으로 남긴다.
  스크립트는 매 단계 **실제 달성 비율을 출력**하므로, 모델 재학습 등으로
  상수가 낡으면 즉시 눈에 보인다.
- 단계별 `/predict` 호출 횟수(`DRIFT_WINDOW_SIZE`=10)와 `TestClient` 사용,
  `shift()`의 "항상 같은 방향(+) 고정" 원칙은 기존 그대로 유지한다
  (방향을 섞으면 같은 윈도우 안에서 상쇄돼 드리프트가 안 잡힘 — 이미 확인된 사실).

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `synthetic/analyze_real_anomalies.py` | 신규 |
| `synthetic/real_anomaly_reference.json` | 신규 (생성 산출물, git 추적) |
| `synthetic/generate_synthetic.py` | 수정 — `calibrate()` → `select_best_amplitude()` + 대역 assert |
| `synthetic/scenarios/*.csv`, `*.json` | 재생성 (덮어씀) |
| `monitoring/simulate_drift.py` | 수정 — 대상 피처 2개로 축소, 진폭 스케줄 재보정 |
| `src/` 전체 | **변경 없음** — 서빙/모델 로직은 건드리지 않는다 |

## 테스트 범위

정식 pytest 단위테스트는 추가하지 않는다. 이유:
- `.pth` 설정상 `src/`만 import 가능해 `synthetic/`·`monitoring/` 스크립트는
  pytest에서 직접 import되지 않는다. 테스트하려면 이 데모 스크립트를 위해
  `src/`에 새 패키지를 만들어야 하는데, 단일 용도 코드에 과한 구조다.
- 더 중요하게는, **스크립트 자체의 런타임 assert가 실제 champion 모델을 상대로
  검증**하므로 가짜 데이터를 쓰는 단위테스트보다 강한 검증이다. 이번에 고치려는
  버그(비현실적 진폭)를 직접 겨냥하는 것도 이 assert다.
- 기존 `loocv/`, `synthetic/`, `monitoring/` 스크립트 관례와도 일치한다.

기존 pytest 100개는 그대로 통과해야 한다(`src/` 무변경이므로 회귀 없어야 정상).

## 검증 방법

1. `analyze_real_anomalies.py` 실행 → `real_anomaly_reference.json`의 그룹별
   범위가 위 표와 일치하는지 확인.
2. `generate_synthetic.py` 실행 → 6개 시나리오 재생성. 이상 3개의 top1 z-score가
   각 목표 허용범위 안에 들어오는지 확인(범위 밖이면 스크립트가 에러로 중단).
   정상 변형 3개는 `"good"` 판정 + 선택된 진폭이 로그에 출력되는지 확인.
3. `simulate_drift.py` 실행 → 6단계 각각의 **실제 달성 score/threshold 비율**이
   목표 `[0.5, 1.0, 1.5, 2.2, 3.0, 3.8]`에 근접하는지, 단조 증가하는지 확인.
   `input_drift.flagged_features`가 어느 단계부터 잡히는지도 함께 기록.
4. `pytest` 전체 통과 확인(100개).
5. 결과를 사용자에게 있는 그대로 보고 — 목표 대역에 못 맞추는 시나리오가 있으면
   숨기지 않고 보고하고 대역/후보 목록 재조정 여부를 논의한다.

## 남은 리스크

- 진폭 후보 목록(8개)이 이산적이라, 목표 중앙값에 정확히 맞는 진폭이 없을 수
  있다. 허용범위 안에만 들어오면 성공으로 본다(중앙값은 목표일 뿐 필수 아님).
- `vibration_backlash`는 대응하는 실제 라벨이 없어 대역 선택이 판단에 기반한다.
  이 가정은 위에 명시했으며, 나중에 근거가 생기면 재조정 대상이다.
- 드리프트 진폭 상수는 현 champion 모델에 종속적이다. 모델이 바뀌면 달성 비율이
  달라지지만, 스크립트가 매번 실제 비율을 출력하므로 조용히 틀리지는 않는다.
