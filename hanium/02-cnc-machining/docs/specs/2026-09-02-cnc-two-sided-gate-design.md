# 두 방향 승격 게이트 + 재학습 임계값 홀드아웃 보정 설계

작성 2026-09-02. 선행 스펙: `2026-08-19-cnc-drift-triggered-retraining-design.md`
(게이트 G1·G2), `2026-08-25-cnc-shadow-deployment-design.md`(섀도우),
`2026-08-26-cnc-cause-estimation-design.md`(거부 원인 추정 — 이 문서의
"실행 결과에 따른 정정" 절이 본 스펙의 출발점).

**상태 (2026-09-02)**: Part A·B(두 방향 게이트)는 구현 범위. Part C(홀드아웃
임계값 보정)는 구현 전 데이터 확인에서 이 데이터셋에 맞지 않는 것으로
드러나 **보류** — 맨 아래 "구현 전 데이터 확인" 절 참고.

## 배경

### 실측으로 드러난 사각지대

2026-09-02 `fixture_loosening` 40일 라이브 재현에서 4번째 재학습(Day 34,
v44)이 게이트를 통과해 섀도우로 들어갔다. 원인 추정의 결함이 아니라
**G2가 "더 좋다"를 재는 방식**의 문제였다.

G2는 라벨이 도착한 최근 20건에서 champion과 후보의 정확도를 비교한다.
Day 34의 창(생산일 24~27)은 불량 시작(Day 21)과 라벨 지연(7일) 때문에
**정답이 전부 불량**이었다. 정답이 한 클래스뿐이면 "더 자주 불량이라
하는" 모델이 무조건 이긴다.

후보 v44를 champion과 같은 배치에 다시 돌려 확인한 결과:

| 창 | champion | 후보 v44 |
|---|---|---|
| 전부 불량인 Day 24~27 | 12/20 정답 | 16/20 정답 → 통과 |
| 전부 정상인 Day 14~17 | 17/20 정답 (오탐 3) | 11/20 정답 (오탐 9) |

후보는 champion과 배치 순위가 같고 score/threshold만 일률적으로 약 20%
높았다(임계값 0.62 vs 0.86). 무엇을 더 잘 구분하는 게 아니라 기준선만
낮아진 모델이며, 승격됐다면 정상 제품의 오경보가 3배가 됐을 것이다.
섀도우도 같은 정확도 비교라 같은 사각지대에 있다(Day 41 이후 라벨도
전부 불량).

### 후보가 예민해진 이유 — 임계값이 학습 오차로 정해진다

| | champion v1 | 후보 v44 |
|---|---|---|
| 학습 행 수 | 14,654 | 155,691 |
| 학습 배치 | 실험 8개 | 배치 85개 (같은 8개 실험의 변형 복사본) |
| 학습 오차 중앙값 | 0.50 | 0.32 |
| 임계값 (학습 오차 p95) | 0.86 | 0.62 |

재학습 데이터는 같은 8개 패턴의 복사본이라 양이 10배여도 다양성은
같다. 모델은 8개 패턴을 더 빡빡하게 외우고, 학습 오차가 내려간 만큼
임계값도 내려가지만, 처음 보는 배치의 오차는 그만큼 내려가지 않는다.
분모만 작아져 모든 배치가 20% 더 이상해 보인다. 임계값을 학습에 쓴
데이터로 정하는 한 데이터가 늘수록 이 방향으로 밀린다.

### 왜 둘 다 고쳐야 하는가

- 임계값 보정만 고치면: 이번 후보가 튄 방향 하나를 막을 뿐이다. 실제
  운영에서 재학습 모델이 어느 방향으로 튈지는 통제할 수 없다.
- 게이트만 고치면: 예민한 후보를 걸러내긴 하지만 재학습이 계속 예민한
  후보만 만들어 내면 승격이 영영 안 된다.
- 게이트가 더 중요하다. 어느 방향으로 튀어도 걸러내는 쪽이 안전의
  본체이고, 임계값 보정은 후보의 품질을 올리는 보조다.

(2026-09-02 데이터 확인 후 결정: 보조인 임계값 보정은 이 데이터셋에서
반대쪽으로 지나쳐 보류. 게이트만으로 안전은 확보된다 — 맨 아래 절.)

## 목표 / 비목표

**목표**

1. G2(그리고 섀도우 판정)가 오탐과 놓침을 **따로** 세어, 한쪽을 다른
   쪽과 맞바꾸는 후보와 한쪽을 아예 잴 수 없는 창을 통과시키지 않는다.
2. ~~재학습 후보의 임계값과 feature_baseline을 학습에 쓰지 않은 정상
   배치(홀드아웃)의 오차로 정한다.~~ → **보류** (데이터 확인 결과, 맨 아래 절).
3. `temperature` 시나리오의 기존 동작(정상 라벨뿐인 창에서 오탐 개선으로
   승격)은 유지한다.

**비목표**

- G1(원본 eval 놓친 개수) 변경. 그대로 둔다.
- 창 크기 `GATE_SAMPLE_SIZE = 20` 변경. 창을 넓히면 측정 대상이 바뀌어
  champion에 유리해진다는 8/19 결론을 유지한다.
- 정상·불량을 각각 N건씩 모으는 계층화 창. 같은 이유로 제외하며,
  `temperature`에는 불량 라벨이 아예 없어 성립하지 않는다.
- champion(최초 학습) 경로의 임계값 산출 방식 변경. `scripts/run_lstm_training.py`
  는 손대지 않는다 — 헤드라인 성능(0.91/0.91)의 근거가 바뀌면 안 된다.
- 트리거 시점에 라벨 창이 전부 불량이면 재학습 자체를 건너뛰는 최적화.
  거부 run이 MLflow에 남는 지금의 기록 방식을 유지한다.
- 라벨 지연·불량 시작일 등 시뮬레이션 파라미터 변경.

## 판정 규칙 (사용자 확정)

| 창의 상태 | 판정 |
|---|---|
| 정상 라벨 0건 | **거부.** 사유 "정상 라벨 없음(오탐 회귀 확인 불가)". 거부 경로이므로 원인 추정 + RAG 조치까지 실행. 라벨이 며칠째 전부 불량이라는 것 자체가 고장 신호다. |
| 정상 라벨만 있음 | 오탐 건수만 비교. 후보 오탐 < champion 오탐이면 통과. 놓침 쪽은 G1이 맡는다(현행 `temperature` 경로와 동일). |
| 정상·불량 모두 있음 | 후보 오탐 ≤ champion 오탐 **그리고** 후보 놓침 ≤ champion 놓침, **그 위에** 둘 중 하나는 엄격히 더 적어야 통과. |

건수(정수)로 비교한다. 정상 12건이면 오탐 1건이 8.3%p라 소수 정확도는
그 눈금을 가리는 가짜 정밀도다(G1과 같은 이유).

## Part A — 게이트 (`src/retraining/gate.py`)

### `evaluate_two_sided` (신규, 순수 함수)

```python
def evaluate_two_sided(
    truths: list[str], champion_preds: list[str], candidate_preds: list[str]
) -> dict:
    """라벨 창을 정상/불량으로 나눠 오탐(정상→bad)과 놓침(불량→good)을
    두 모델 각각 센 뒤, 판정 규칙 표대로 promoted/rejected를 낸다.
    세 리스트 길이가 다르면 ValueError. 빈 창은 '정상 라벨 0건'으로 거부."""
```

반환:

```python
{
    "n_good": int, "n_bad": int,
    "champion_false_alarms": int, "candidate_false_alarms": int,
    "champion_misses": int, "candidate_misses": int,
    "decision": "promoted" | "rejected",
    "reject_reason": str,   # promoted면 ""
}
```

판정 순서:

1. `n_good == 0` → rejected, `"G2 판정 불가: 창에 정상 라벨 없음(오탐 회귀 확인 불가)"`.
2. `fa_ok = candidate_false_alarms <= champion_false_alarms`
3. `miss_ok = n_bad == 0 or candidate_misses <= champion_misses`
4. `improved = candidate_false_alarms < champion_false_alarms
   or (n_bad > 0 and candidate_misses < champion_misses)`
5. `promoted = fa_ok and miss_ok and improved`. 사유는 어긋난 조건마다
   한 줄씩 `"; "`로 잇는다 — `"G2 오탐 회귀: 후보 9건 > champion 3건 (정상 20건 중)"`,
   `"G2 놓침 회귀: 후보 5건 > champion 4건 (불량 8건 중)"`,
   `"G2 개선 없음: 오탐·놓침 모두 champion과 동일"`.

### `evaluate_gate` (시그니처 변경)

```python
def evaluate_gate(
    retrained_missed: int,
    champion_missed: int,
    g2: dict,                      # evaluate_two_sided() 결과
    extra_misses_allowed: int = EXTRA_MISSES_ALLOWED,
) -> dict:
```

G1 판정은 그대로(`retrained_missed <= champion_missed + extra_misses_allowed`).
G2는 `g2["decision"] == "promoted"`. 반환은 `decision`, `g1_pass`, `g2_pass`,
`g1_missed`, `reject_reason`을 유지하고, `g2_accuracy_delta` 대신 `g2`
딕셔너리를 그대로 포함한다(`"g2": g2`). `reject_reason`은 G1 사유와
`g2["reject_reason"]`을 `"; "`로 잇는다.

### `evaluate_shadow`, `accuracy_from_pairs` (제거)

섀도우 종료 판정은 `evaluate_two_sided`를 직접 쓴다. G1은 트리거 시점에
이미 확인했고 원본 eval은 시간이 지나도 안 바뀐다는 8/25의 논리는
그대로다 — 섀도우는 G2에 해당하는 비교만 반복하며, 그 비교가 이제 두
방향이 된 것뿐이다. `accuracy_from_pairs`는 호출처가 없어지므로 함께
제거한다.

## Part B — 워커 (`monitoring/drift_worker.py`)

### `_gate_accuracies` → `_gate_predictions`

```python
def _gate_predictions(result, current_day, scenario) -> tuple[
    list[str], list[str], list[str], list[list[dict]]
]:
    """(truths, champion_preds, candidate_preds, champion_contributions)"""
```

정확도 계산을 워커에서 걷어내고 판정은 전부 Part A가 한다. 도착한 라벨이
없으면 네 리스트 모두 빈 값 — `evaluate_two_sided`가 "정상 라벨 없음"으로
거부한다(현행: 0.0 vs 0.0으로 "개선 없음" 거부. 결과는 같고 사유가
정직해진다).

### `_decide_and_start_shadow`

```python
truths, champ, cand, champion_contributions = _gate_predictions(result, current_day, scenario)
g2 = evaluate_two_sided(truths, champ, cand)
verdict = evaluate_gate(result["missed"], state.champion_missed, g2)
```

콘솔 로그:

```
게이트: G1 놓침=0건 (champion 1건, 허용 2건) /
        G2 정상 12건 — 오탐 후보 3 vs champion 5 · 불량 8건 — 놓침 후보 2 vs champion 2
```

정상 또는 불량이 0건이면 해당 부분을 `"정상 0건"`처럼 건수만 찍는다.
거부 경로(사유 불문)는 지금처럼 `estimate_cause` → `build_cause_guide` →
태그 기록까지 실행한다.

### `_check_shadow`

라벨·예측을 모은 뒤 `evaluate_two_sided(truths, champion_preds, candidate_preds)`
로 판정한다. 로그와 태그 이름만 바뀐다.

### MLflow 태그

| 제거 | 추가 |
|---|---|
| `gate_g2_accuracy_delta` | `gate_g2_n_good`, `gate_g2_n_bad`, `gate_g2_fa_delta`(후보−champion), `gate_g2_miss_delta` |
| `shadow_accuracy_delta`, `shadow_candidate_accuracy`, `shadow_champion_accuracy` | `shadow_n_good`, `shadow_n_bad`, `shadow_fa_delta`, `shadow_miss_delta` |

`gate_g2_sample_size`, `gate_g1_missed`, `gate_decision`, `gate_reject_reason`,
`estimated_cause`, `recommended_action`, `scenario`, `trigger_day`는 그대로.

## Part C — 재학습 임계값 홀드아웃 보정 (보류)

**이번 구현 범위에서 제외한다.** 아래 설계는 참고용으로 남긴다. 보류 근거는
"구현 전 데이터 확인" 절 — 요약하면, 보정 집합의 p95가 feedrate=20 실험(2)
유래 배치 위에 떨어져 후보가 champion보다 둔해진다.

### `src/retraining/runner.py`

```python
CALIBRATION_EVERY = 5           # 생산일별로 5개 중 1개를 보정용으로
MIN_CALIBRATION_BATCHES = 5     # 이보다 적으면 기존 방식(학습 오차)으로 폴백


def split_calibration(good_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """생산일별로 도착 순서 기준 CALIBRATION_EVERY번째마다(첫 배치 포함)
    보정용으로 뗀다. 반환 (train_records, calibration_records)."""


def collect_normal_batches(
    arrived_labels, timeline_dir, current_day, lookback_days=30
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(train_raw, calibration_raw). 보정 배치가 MIN_CALIBRATION_BATCHES
    미만이면 전부 학습에 넣고 calibration_raw는 None."""
```

"생산일별로 첫 배치"를 택하는 이유: 최근 며칠을 통째로 떼면
`temperature` 재학습에 가장 필요한 "가장 새로운 정상"이 학습에서 빠진다.
일자별로 하나씩 떼면 보정 집합이 전 기간을 고르게 덮고, 시뮬레이션에서는
하루 5배치가 실험 8개를 순환하므로(5와 8이 서로소) 보정 배치도 8개 원본
패턴을 모두 거친다.

`run_retraining` 변경:

- scaler는 **train_raw로만** fit한다. calibration_raw는 같은 scaler로
  변환해 `calibration.csv`(FEATURE_COLUMNS + experiment_id)로 저장한다.
- `run_lstm_pipeline(..., calibration_csv_path=str(retrain_dir / "calibration.csv"))`.
  폴백이면 `None`.
- params에 `retrain_calibration_batch_count`(정수, 폴백이면 0)와
  `calibration_fallback`(bool)을 추가한다. `NOT_INSTALLED`에
  `"calibration.csv"`를 더해 정본 자리로 복사되지 않게 한다.

### `src/lstm_ae/pipeline.py`

`run_lstm_pipeline(..., calibration_csv_path: str | None = None)`.

```python
if calibration_csv_path is not None:
    calib_df = pd.read_csv(calibration_csv_path)
    calib_windows, calib_ids = make_eval_windows(calib_df, feature_columns, window_size)
    calib_scores = aggregate_window_errors_by_experiment(
        compute_window_errors(model, calib_windows), calib_ids
    )
    thresholds = compute_thresholds(calib_scores, percentile=threshold_percentile)
    baseline_scores = aggregate_feature_errors_by_experiment(
        compute_feature_errors(model, calib_windows), calib_ids, feature_columns
    )
else:
    thresholds = compute_thresholds(train_experiment_scores, percentile=threshold_percentile)   # 기존
    baseline_scores = train_feature_error_scores                                               # 기존
feature_baseline = {"mean": baseline_scores[feature_columns].mean().to_dict(),
                    "std": baseline_scores[feature_columns].std().to_dict()}
```

- 보정 배치는 `make_eval_windows`(stride 1)로 자른다. 임계값이 적용되는
  자리(`serving.inference.predict_experiment`)가 stride 1이므로 같은
  잣대로 정해야 한다. 학습 배치는 지금처럼 `make_train_windows`.
- feature_baseline도 보정 배치에서 계산한다. z-score 기여도의 "정상 오차
  프로필"이 학습 적합도가 아니라 처음 보는 정상에 대한 오차여야 하는
  이유가 임계값과 같다.
- 인자가 `None`이면 기존과 바이트 단위로 같은 산출물이 나와야 한다
  (champion 학습 경로 보호).
- `training_config.json`에 `"calibration_batches": n`(없으면 0)을 기록하고,
  진단용으로 `calibration_window_errors.csv`를 남긴다.

### 왜 champion은 그대로 두는가

champion은 실험 8개로 학습됐다. 여기서 보정용을 떼면 학습이 6개가 되고
헤드라인 성능의 근거가 바뀐다. 두 모델의 임계값 산출 방식이 달라지지만,
게이트는 임계값이 아니라 라벨 결과(오탐·놓침 건수)로 비교하므로 비교의
공정성은 유지된다. 이 비대칭은 알려진 한계로 적는다.

## 서빙 계약

변경 없음. `{mean,max,p95}_threshold` 메트릭과 `window_size` param, companion
아티팩트(`scaler.json`, `feature_baseline.json`)는 그대로 만들어진다.
`promotion.verify_serving_contract`도 그대로다.

## 에러 처리

- `evaluate_two_sided`에 길이가 다른 리스트 → `ValueError`(현행
  `accuracy_from_pairs`와 동일). 빈 리스트는 예외가 아니라 "정상 라벨
  없음" 거부.
- 보정 배치 부족(< 5) → 예외 없이 폴백 + params 기록. 재학습이 루프를
  죽이면 안 된다.
- 보정 CSV 경로가 주어졌는데 파일이 없으면 → 예외(설정 오류이므로 조용히
  넘기지 않는다).

## 테스트 전략 (TDD)

`tests/retraining/test_gate.py`

- `evaluate_two_sided` 여섯 경우: 전부 정상 + 오탐 감소 → promoted /
  전부 정상 + 오탐 동일 → rejected "개선 없음" / 전부 불량 → rejected
  "정상 라벨 없음" / 혼합 + 오탐↑·놓침↓ 맞바꾸기 → rejected "오탐 회귀" /
  혼합 + 둘 다 이하 + 하나 개선 → promoted / 길이 불일치 → ValueError.
- 빈 창 → rejected "정상 라벨 없음".
- `evaluate_gate`: G1 경계 테스트 3개는 `g2`를 promoted 딕셔너리로
  고정해 유지. G1·G2 동시 위반 시 사유 두 줄 연결.
- `accuracy_from_pairs`·`evaluate_shadow` 테스트 5개 삭제.

`tests/retraining/test_runner.py` (Part C — 보류, 구현 시 제외)

- `split_calibration`: 하루 5배치면 그날 첫 배치만 보정, 하루 10배치면
  2개, 보정·학습이 겹치지 않고 합집합이 원본.
- `collect_normal_batches`: 보정 배치 5개 미만이면 `(train, None)`.
- scaler가 train_raw로만 fit되는지(calibration 평균이 0이 아님).
- 기존 4개 테스트는 반환이 튜플이 된 것만 반영.

`tests/lstm_ae/test_pipeline.py` (Part C — 보류, 구현 시 제외)

- `calibration_csv_path` 있음: `thresholds`가 보정 배치 점수의 p95와
  일치, `feature_baseline`이 보정 배치 기준, `training_config.json`에
  `calibration_batches` 기록.
- 없음: 기존 두 테스트 그대로 통과(산출물 불변).

`tests/monitoring/test_simulate_timeline.py` 등 워커 테스트: 태그 이름과
`_gate_predictions` 반환 형태만 반영.

## 검증 방법 — 라이브 재현 3종

DB 초기화(`labels.db`, `requests.db`, `shadow.db`) 후 세 프로세스로 실행.
원본 로그·DB는 `data/monitoring/_<시나리오>_<날짜>/`에 보관.

| 시나리오 | 실행 | 기대 |
|---|---|---|
| fixture_loosening | 40일, `--pace-seconds 2` | Day 19·24·29 거부(현행과 동일 방향), **Day 34 거부** — 사유 "정상 라벨 없음", `estimated_cause=vibration_backlash` |
| tool_wear | 40일, `--pace-seconds 2` | 5회 거부, `estimated_cause=tool_wear` 5/5 |
| temperature | 70일, `--pace-seconds 15` | 거부 몇 회 뒤 **게이트 통과 → 섀도우 → 승격**(8/25와 같은 흐름). 정상 라벨뿐인 창에서 오탐 감소로 통과해야 한다 |

게이트만 바뀌므로 세 시나리오의 판정은 "구현 전 데이터 확인" 절의 사후
적용 표와 같아야 한다(같지 않다면 구현 오류다). 결과는 이 문서에 "실행
결과에 따른 정정" 절로 기록한다.

## 알려진 한계 (미리 적어 두는 것)

- 고장이 라벨 지연보다 오래 이어지면 그동안 어떤 후보도 승격되지 않는다.
  의도된 동작이다 — 그 기간에 승격할 근거가 없다.
- 창 20건의 구성이 원본 실험 8개의 순환 순서에 좌우된다(09-02 배치별
  표에서 champion이 놓친 8건은 전부 실험 11·13·14·17 유래). 건수 눈금이
  작은 표본에서 우연에 흔들릴 수 있으나, 표본을 늘리면 측정 대상이
  바뀐다는 8/19 결론대로 창은 손대지 않는다.
- 재학습 후보의 임계값은 여전히 학습 오차 기준이라 데이터가 늘수록 과민
  쪽으로 밀린다(Part C 보류의 대가). 과민한 후보는 G2 오탐 회귀로 거부되지만,
  `temperature` 유형에서 승격되는 후보도 "champion보다 오탐이 적다"일 뿐
  절대 오탐률은 높을 수 있다(스파이크: Day 37 후보 오탐 11/20 vs champion 18/20).
- 정상 라벨만 있는 창에서 놓침 쪽을 G1에 맡기는데, G1은 실측에서 한 번도
  작동한 적이 없다(8/19). 이 경로의 안전성은 지금과 같고 더 나아지지는
  않는다.

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `src/retraining/gate.py` | `evaluate_two_sided` 신규. `evaluate_gate` 시그니처 변경(`g2` 딕셔너리). `evaluate_shadow`·`accuracy_from_pairs` 제거 |
| `monitoring/drift_worker.py` | `_gate_accuracies` → `_gate_predictions`. 게이트·섀도우 판정을 `evaluate_two_sided`로. 로그·태그 변경 |
| ~~`src/retraining/runner.py`~~ | (보류) `split_calibration`, (train, calibration) 반환, `calibration.csv` |
| ~~`src/lstm_ae/pipeline.py`~~ | (보류) `calibration_csv_path` 인자 |
| `tests/retraining/test_gate.py`, 워커 테스트 | 위 변경 반영 |
| `docs/STRUCTURE.md`, `README.md` | gate.py 설명(오탐·놓침 두 방향), 태그 이름 갱신 |

변경하지 않는 것: `scripts/run_lstm_training.py`, `src/serving/*`,
`src/retraining/promotion.py`, `src/retraining/trigger.py`,
`src/retraining/runner.py`, `src/lstm_ae/pipeline.py`,
`monitoring/simulate_timeline.py`, `GATE_SAMPLE_SIZE`.

## 구현 전 데이터 확인 (2026-09-02)

사용자 요청("데이터가 있으니 우리 데이터에 맞는지 먼저 보자")으로, 코드를
바꾸기 전에 버릴 스크립트로 두 가지를 확인했다.

### 확인 1 — 오늘 후보 9개(v41~v49)에 두 방향 규칙 사후 적용

보관해 둔 라벨·배치(`data/monitoring/_fixture_loosening_20260902/`,
`_tool_wear_20260902/`, `data/retrain/20260902_*`)로 각 트리거의 G2 창을
다시 만들어 champion v1과 후보를 판정시켰다. 재현한 정확도가 워커 로그와
전부 일치해 후보 매핑은 정확하다.

| 시나리오 | 트리거 | 창 구성 | 후보 vs champion | 기존 규칙 | 새 규칙 |
|---|---|---|---|---|---|
| fixture | Day 19 | 정상 20 | 오탐 4 vs 2 | 거부 | 거부, 오탐 회귀 |
| fixture | Day 24 | 정상 20 | 오탐 11 vs 3 | 거부 | 거부, 오탐 회귀 |
| fixture | Day 29 | 정상 10 · 불량 10 | 오탐 5 vs 3, 놓침 3 vs 4 | 거부 | 거부, 오탐 회귀 |
| fixture | Day 34 | 불량 20 | 놓침 4 vs 8 | **통과** | **거부, 정상 라벨 없음** |
| tool_wear | Day 20 | 정상 20 | 오탐 5 vs 2 | 거부 | 거부, 오탐 회귀 |
| tool_wear | Day 25 | 정상 20 | 오탐 7 vs 4 | 거부 | 거부, 오탐 회귀 |
| tool_wear | Day 30 | 정상 5 · 불량 15 | 오탐 1 vs 2, 놓침 9 vs 7 | 거부 | 거부, 놓침 회귀 |
| tool_wear | Day 35 | 불량 20 | 놓침 13 vs 0 | 거부 | 거부, 정상 라벨 없음 |
| tool_wear | Day 40 | 불량 20 | 놓침 12 vs 0 | 거부 | 거부, 정상 라벨 없음 |

- 기존 규칙과 갈리는 것은 문제였던 fixture Day 34 하나뿐이다.
- 맞바꾸기가 실제로 양방향으로 존재한다(fixture Day 29: 오탐↑ 놓침↓,
  tool_wear Day 30: 오탐↓ 놓침↑). 클래스별 회귀 금지가 둘 다 잡는다.
- `temperature`는 창이 정상뿐이라 새 규칙이 정확도 규칙과 수학적으로
  같다. Day 37 후보를 재생성한 배치로 판정하면 오탐 11 vs 18 → 통과
  (08-19 실측 0.45 vs 0.10과 일치).

### 확인 2 — 홀드아웃 임계값 보정 시제품 (실제 재학습 3회)

Part C 설계대로 정상 배치를 생산일별 1/5로 나눠 세 지점에서 재학습하고,
보정 배치(stride 1)의 p95로 임계값을 다시 정해 같은 창을 판정시켰다.

| 지점 | 임계값 학습오차 → 보정 | 정상 창 오탐: 후보 보정 전 → 후 (champion) | 불량 창 놓침: 보정 전 → 후 (champion) | 원본 eval G1 놓침: 보정 전 → 후 (champion 1) |
|---|---|---|---|---|
| fixture Day 24 | 0.576 → 1.052 | 10 → 3 (3) | — | 0 → 2 |
| fixture Day 34 | 0.630 → 1.065 | 6 → 3 (3) | 6 → 14 (8) | 0 → 2 |
| temperature Day 37 | 0.562 → 1.077 | 11 → 3 (18) · 섀도우 창 10 → 2 (20) | — | 0 → 2 |

- **의도한 효과는 난다.** 과민이 사라져 정상 창 오탐이 세 지점 모두
  champion 수준이 되고, `temperature`의 승격 근거가 훨씬 확실해진다.
- **그러나 반대쪽으로 지나친다.** 보정 임계값 1.05~1.08은 세 경우 모두
  **feedrate=20 실험(2) 유래 배치**가 만든 값이다(보정 배치 점수 상위가
  전부 실험 2, 1.05~1.09; 그다음은 0.7~0.8). 실험 2는 정상이어도 늘 높게
  나오는 알려진 한계 데이터인데 전체의 12.5%라, 보정 집합 크기와 무관하게
  p95가 이 배치 위에 떨어진다. 결과로 후보가 champion보다 둔해진다 —
  원본 eval 놓침 0 → 2(G1 허용치 champion+1에 딱 걸림, 놓친 것은 실험
  6·21), fixture 불량 창 놓침 6 → 14(champion 8).
- champion의 임계값 0.857도 실험 2(0.97)가 끌어올린 값이다. 8개 값의
  p95 보간이 우연히 중간에 떨어졌을 뿐, 같은 구조다.

### 결정

- **Part A·B 구현.** 두 방향 게이트만으로 과민한 후보는 오탐 회귀로,
  전부 불량인 창은 정상 라벨 없음으로 걸러진다. `temperature` 승격은
  지금처럼 된다.
- **Part C 보류.** 이 데이터셋에서는 보정이 후보를 둔하게 만들고, p95를
  피해 가는 다른 통계량을 고르는 것은 결과가 나올 때까지 손잡이를 돌리는
  일이라 하지 않는다. feedrate=20 조건의 정상 표본이 더 확보되면 다시 본다.
