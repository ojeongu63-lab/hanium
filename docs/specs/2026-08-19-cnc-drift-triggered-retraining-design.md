# CNC 드리프트 트리거 기반 자동 재학습 설계

- 날짜: 2026-08-19
- 상태: 설계 완료, 구현 전
- 선행 결정 변경: `2026-08-12-cnc-drift-monitoring-design.md`가 "자동 재학습/재승격"을
  명시적 **비목표**로 두었으나(champion 승격은 사람이 판단), 이번 설계에서 이를 뒤집는다.
  근거는 아래 배경 절에 남긴다.

## 배경

멘토 제안: "실제 MLOps/AIOps 환경처럼, 시간이 지나며 모델에 드리프트가 발생했다고
어떤 트리거에 의해 판단되면 자동으로 재학습되는 것까지 보여달라."

기존 트랙에서 **드리프트 감지**는 이미 완성돼 있다 — `/predict` 요청 누적
(`src/monitoring/logging.py`), 입력/출력 드리프트 계산(`src/monitoring/drift.py`),
`GET /drift-status`, MLflow 기록(`src/monitoring/mlflow_logging.py`). 없는 것은
**감지 이후**다: 트리거 판정 → 재학습 실행 → 승격 게이트 → 자동 재배포.

### 해결하려는 문제

주 문제는 이것이다: **제품은 정상인데 공정 파라미터가 변해서 모델이 불량으로
오판하는 상황.** 계절·온도 변화로 입력 분포가 이동하면 모델이 학습한 "정상의
정의"가 낡아 오탐이 늘어난다. 이때 필요한 것은 설비 정비가 아니라 **정상 기준의
갱신 = 재학습**이다.

### 왜 자동 재학습만으로는 안 되는가

같은 드리프트 신호가 정반대 원인에서도 나온다.

| 상황 | 입력 분포 | 제품 실제 품질 | 모델 판단 | 올바른 처방 |
|---|---|---|---|---|
| 계절·온도 변화 | 이동 | 정상 | 오탐 | **재학습** |
| 설비 노후화 (초기) | 이동 | 아직 QC 합격 | 오탐 | 재학습해도 규칙 위반은 아니나 **위험** |
| 설비 노후화 (진행) | 이동 | 불량 | 정탐 | **정비 — 재학습 금지** |

드리프트 지표만으로는 이 셋을 구분할 수 없다. 구분에 필요한 것은 원인이 아니라
**제품이 실제로 불량이냐**인데, 그 정보(QC 라벨)는 항상 늦게 도착한다.

세 번째 행에서 재학습을 하면 모델에게 "불량이 정상"이라고 가르치는 셈이다. 더
까다로운 것은 두 번째 행이다 — QC를 통과한 데이터라 라벨상 정상이고 재학습에
포함시켜도 규칙 위반이 아니지만, 그것을 정상 기준에 계속 넣으면 **기준이 열화를
따라 조금씩 내려간다.**

따라서 이 설계의 구조는 "트리거 → 재학습"이 아니라 **"트리거 → 재학습 → 게이트 →
승격 또는 거부"** 다. 게이트가 없는 자동 재학습은 위험하다는 것이 선행 설계에서
자동 승격을 비목표로 두었던 이유이고, 이번에 그 결정을 뒤집을 수 있는 근거도
**게이트를 함께 만들기 때문**이다.

## 목표 / 비목표

**목표**
- 드리프트가 지속되면 사람 개입 없이 재학습이 발동되고, 게이트를 통과한 경우에만
  champion이 교체되며, 돌고 있는 서빙이 새 모델을 집어드는 **닫힌 루프**를 만든다.
- 온도·계절 시나리오에서 루프가 **승격**으로 끝나고, 공구마모 시나리오에서
  **거부 + 사람 호출**로 끝나는 것을 실제로 확인한다.
- 승격/거부 판단의 근거가 MLflow에 전부 기록으로 남는다(거부된 run도 보존).
- 자동화하면 터지는 기존 결함 2건(아래 Part F)을 함께 고친다.

**비목표**
- 모델 구조·하이퍼파라미터 개선 — 재학습은 기존 `run_lstm_pipeline()`을 그대로
  호출하며, 바뀌는 것은 **입력 데이터와 scaler**뿐이다.
- 알림 발송(이메일/슬랙) — 거부 시 워커 로그와 MLflow 태그까지만.
- 센서 고장(급변)·신제품 도입 시나리오 — 전자는 드리프트가 아니라 이상치 탐지
  문제이고, 후자는 계획된 변경이라 자동 트리거의 대상이 아니다.
- 실제 라이브 스트리밍 데이터 — 존재하지 않는다(공개 데이터셋 25개 실험 고정).

## 데이터 전략 — 가상 운영 타임라인

### 왜 합성인가, 그리고 어떤 합성인가

보유 데이터에는 **시간축이 없다.** 실험 CSV 25개는 각각 "가공 1회분"이고 서로
시간 순서가 없다. "시간이 지나며 나빠진다"를 보이려면 순서가 있어야 한다.

세 가지 길을 검토했다.

| 안 | 내용 | 판단 |
|---|---|---|
| A. 기존 실험에 시간 종속 변환 | train 8개 실험에 날짜 비례 변형을 주입해 스트림 생성 | **채택** |
| B. 생성 모델(TimeGAN 등) | 25개 실험으로 생성 모델 학습 | 기각 — 학습 표본이 없다. 검증 불가능한 층이 하나 더 생긴다 |
| C. CNC 절삭 물리 시뮬레이션 | 절삭력·주축부하 물리 모델 구축 | 기각 — 별도 프로젝트 규모이고, 증명 대상(루프)에 기여하지 않는다 |

A는 데이터를 지어내는 것이 아니라 **있는 실험에 시간축을 부여하는 것**에 가깝다.
변형 폭은 `synthetic/real_anomaly_reference.json`의 실측 대역 안에 가둔다 —
과거에 진폭 상한이 없어 z=29790 같은 비현실적 값을 만든 전례가 있으므로
(`2026-08-12-cnc-realistic-synthetic-data-design.md`), 이 제약이 중요하다.

### 타임라인 구조

- 가상 운영 기간 **40일**, 하루 **5건** 가공 → 총 **200 배치**
- 각 배치는 train 실험 8개를 순환하며 baseline으로 쓴다 — exp01 하나만 쓰면
  재학습 데이터가 feedrate 1개 조건으로 붕괴하고, 이미 알려진 `feedrate=20`
  커버리지 한계가 악화된다
- 배치 1개 = 원본과 같은 48컬럼 CSV 1개 = `/predict` 호출 1회
- 드리프트 윈도우가 10이므로 이틀치가 한 윈도우

변형 진행도: `progress = max(0, (day - 10) / 30)` — Day 1~10은 변형 없는
baseline 구간(감지기가 정상 상태를 학습할 시간), Day 11부터 선형 증가해 Day 40에
최대가 된다.

### 시나리오 A — 온도·계절 변화 (주 시나리오)

물리적 이야기: 계절이 바뀌며 공장 온도가 오른다. 주축·볼스크류가 열팽창하고,
서보 모터 권선 저항이 올라가며, 절삭유 점도가 변한다. **제품 품질은 변하지 않는다.**

데이터셋에 온도 컬럼은 없다(41개 피처가 전부 위치/속도/가속도/전류/전압/파워/
피드레이트). 온도를 직접 관측할 수 없고 그 **영향만** 센서에 나타나는 상황이라,
드리프트로만 감지된다 — 현실 그대로다.

| 온도가 하는 일 | 변형 대상 | 방식 |
|---|---|---|
| 열변위 | `X/Y/Z_ActualPosition` | 해당 피처 표준편차 × `POS_DRIFT` × progress 만큼 offset |
| 서보 권선 저항 증가 | `X/Y_OutputCurrent`, `X/Y_OutputPower` | 배율 `1 + CUR_DRIFT × progress` |

`SetPosition` 계열은 건드리지 않는다 — 지령값은 그대로인데 실제값만 벌어지는
것이 열변위의 본질이다.

- **QC 라벨: 전 구간 정상.** 제품은 멀쩡하다.
- 목표 도달점: Day 40에서 champion의 `score/threshold` ≈ **1.5~2.0**
  (실측 GOOD 대역 0.43~1.30을 막 넘어서는 수준)
- 기대 결말: Day 30 전후 오탐 시작 → 트리거 발동 → 이동된 정상 데이터와 새 scaler로
  재학습 → G1·G2 통과 → **승격** → 오탐 소멸. **루프 성공 경로.**

### 시나리오 B — 공구마모 (게이트 검증용)

물리적 이야기: 공구 마모가 누적되어 절삭 품질이 떨어진다. 어느 시점부터 **실제로
불량품이 나온다.**

`synthetic/generate_synthetic.py`의 `tool_wear()` perturbation을 그대로 재사용한다
(대상 피처 `S_OutputCurrent, S_OutputPower, S_CurrentFeedback, X_OutputPower,
Y_OutputPower`, 램프 배율). 진폭만 `WEAR_RATE × progress`로 날짜에 연동한다.

- **QC 라벨: Day 1~20 정상, Day 21~40 불량.** 마모가 임계를 넘어 표면 품질이
  검사를 통과하지 못하기 시작하는 시점을 Day 21로 둔다.
- 목표 도달점: Day 40에서 `score/threshold` ≈ **3.0** (실측 BAD 대역 1.00~3.79 안)

기대 결말과 그 메커니즘:

```
트리거가 Day 28쯤 발동
        ↓
라벨 지연 7일 → 그 시점에 쓸 수 있는 라벨은 Day 21까지
        ↓
그 중 "정상" 확정분은 Day 1~20
        ↓
Day 11~20은 이미 마모가 진행 중이지만 QC는 통과한 구간
        ↓
재학습기가 이를 정상 기준에 포함 (라벨상 틀린 판단이 아님)
        ↓
"정상"의 범위가 마모 방향으로 넓어짐
        ↓
진짜 불량품을 정상이라 판정하기 시작
        ↓
승격 거부 + 사람 호출
```

> **실행 결과에 따른 정정 (2026-08-19).** 위 마지막 단계를 설계 시점에는
> "원본 eval의 quality_failed 그룹을 놓쳐 G1(recall 회귀)이 거부한다"로
> 예상했으나, **실제로는 G1이 아니라 G2가 막았다.**
>
> 재학습 모델의 원본 eval recall은 매번 **1.000**이었다(tp=11, fn=0). 재학습이
> scaler를 새로 fit하기 때문에 원본 eval셋 전체가 낯선 좌표계로 옮겨져
> 거의 모두 불량으로 판정되고, 그 결과 recall이 자동으로 1.0이 된다.
> "마모를 정상으로 받아들이게 된 효과"가 좌표계 이동 효과에 묻힌 것이다.
>
> 예측한 둔감화 자체는 실재했고 **G2에서 드러났다.** Day 39 게이트 표본
> (Day 29~32, 전부 실제 불량 20건)에서 champion은 20건 전부를 불량으로 잡아
> 정확도 1.00, 재학습 모델은 8건만 잡아 **0.40** — 진짜 불량의 60%를 정상이라
> 판정했다.
>
> 이는 아래 Part D에 적어둔 "G1만으로는 부족하다"는 경고가 그대로 실현된
> 사례다. G1을 단독 방어선으로 뒀다면 이 시나리오는 승격됐을 것이다.

즉 **라벨이 완벽하게 정직해도 자동 재학습은 위험하다**는 것을 보이는 시나리오다.
실측 근거: `real_anomaly_reference.json`의 quality_failed 그룹 `score_ratio_to_threshold`
최솟값이 **1.00** — 턱걸이로 잡히고 있어 기준이 조금만 느슨해져도 놓친다.

두 시나리오는 물리적으로 구분되며, 피처 기여도에서도 다르게 나타나야 한다:

| | A (온도) | B (공구마모) |
|---|---|---|
| 영향 축 | X·Y·Z 균일 | 주축(S) 중심 |
| 가공 구간 의존성 | 무관, 일정한 offset | 절삭 구간에서 부하 증가 |
| 위치 편차 | 동반(열변위) | 없음 |

### 변형 상수 결정 방법

`POS_DRIFT`, `CUR_DRIFT`, `WEAR_RATE`는 미리 못 박지 않는다. 구현 시 일회성
스윕으로 결정한다 — 후보 그리드마다 Day 40 배치를 만들어 `predict_experiment()`를
1회 돌리고, 위 목표 비율에 가장 가까운 값을 채택한다. 확정값은 상수로 박고 스윕
결과 표를 주석으로 남긴다. 스크립트는 매 단계 **실제 달성 비율을 출력**하므로,
모델이 바뀌어 상수가 낡으면 조용히 틀리지 않고 눈에 보인다
(`simulate_drift.py`에서 확립된 관례).

## Part A — 스트림 생성기 (`monitoring/simulate_timeline.py`, 신규)

일회성 실행 스크립트 관례를 따라 `src/`에 넣지 않는다(`loocv/`, `synthetic/`,
`monitoring/`과 동일).

- 시나리오(`temperature` | `tool_wear`)와 기간을 인자로 받는다
- Day 1~40, 하루 5배치를 순서대로 생성해 `data/timeline/<scenario>/day{NN}_{i}.csv`에
  저장하고, `fastapi.testclient.TestClient(app)`로 `/predict`에 업로드한다
  (별도 서버 프로세스 없이 실제 앱·실제 champion을 상대로 검증 — 기존 관례)
- 각 배치의 **진실 라벨**을 `arrived_at = day + LABEL_DELAY_DAYS`로 `qc_labels`
  테이블에 기록한다
- 매 가상 1일(5배치)마다 감시 워커의 1틱을 호출한다

배치 CSV를 파일로 남기는 이유는 재학습 러너가 같은 원본을 다시 읽어야 하기
때문이다. `data/` 하위라 `.gitignore` 대상이다.

## Part B — 트리거 (`src/retraining/trigger.py`, 신규)

부작용 없는 순수 함수로 만들어 테스트를 쉽게 한다.

```python
def should_retrain(
    flag_history: list[bool],
    consecutive_k: int = 3,
    cooldown_remaining: int = 0,
) -> bool:
    if cooldown_remaining > 0:
        return False
    if len(flag_history) < consecutive_k:
        return False
    return all(flag_history[-consecutive_k:])
```

`flag_history`의 각 원소는 한 번의 `/drift-status` 조회 결과로,
`output_drift.flagged`이거나 `input_drift.flagged_features`가 비어 있지 않으면
`True`다.

| 항목 | 값 | 근거 |
|---|---|---|
| 폴링 주기 | 가상 1일(5배치)마다 | |
| 발동 조건 | 연속 **3회** flagged | 단발 노이즈로 재학습이 돌면 안 된다 |
| 쿨다운 | 재학습 후 **5일** | 재학습 직후에는 로그에 옛 데이터가 남아 즉시 재발동한다 |

## Part C — 재학습 러너 (`src/retraining/runner.py`, 신규)

### 재학습 데이터 구성

1. 최근 **30일** 중 **라벨이 도착했고(`arrived_at <= 오늘`) 정상인** 배치의 원본
   CSV를 수집해 concat → `train_raw`. LSTM-AE는 정상만 학습하므로 불량 라벨은 제외.
2. `fit_scaler(train_raw, FEATURE_COLUMNS)` → **새 scaler**. 센서값이 이동한
   환경에서는 scaler 재fit이 필수다.
3. `transform_features(train_raw, ..., 새 scaler)` → `train.csv`

### eval셋도 새 scaler로 재생성해야 한다

`data/processed/eval.csv`는 **옛 scaler로 이미 스케일된 상태**로 저장돼 있고
(`preprocessing/pipeline.py:81-83`), `run_lstm_pipeline()`은 스케일링을 하지 않고
스케일된 입력을 전제한다(`lstm_ae/pipeline.py:82-83`). 새 scaler를 쓰면서 옛
eval.csv를 그대로 넣으면 **train과 eval의 좌표계가 어긋난다.**

따라서 러너는 `data/dataset/`의 원본 실험에서 eval 14개를 raw로 다시 읽어
새 scaler로 transform해 `eval.csv`를 만든다. 기존 `preprocessing`의
`_load_experiment` + `transform_features`를 재사용하며 새 로직은 없다.

**eval의 실험 구성과 라벨은 원본 그대로 고정한다** — 게이트의 기준셋이므로
드리프트와 무관하게 불변이어야 한다.

### 학습 실행

```python
run_lstm_pipeline(
    train_csv_path=<retrain_dir>/train.csv,
    eval_csv_path=<retrain_dir>/eval.csv,
    output_dir=<retrain_dir>,        # data/model/ 이 아니다 — Part F 참조
    feature_columns=FEATURE_COLUMNS,
    exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
    **TRAINING_CONFIG,               # 기존 상수 그대로
)
```

임계값은 기존 원칙대로 **train 분포에서만** 산정하므로 eval을 함께 넣어도
오염되지 않는다. 파이프라인이 반환하는 `results`에 precision/recall이 이미 있어
G1 계산에 그대로 쓴다. `feature_baseline.json`도 파이프라인이 `output_dir`에
생성하므로(`lstm_ae/pipeline.py:158-162`) 별도 작업이 필요 없다 — 다만 새 train
데이터 기준으로 계산된 값이라 **반드시 모델과 함께 이동해야 한다**(Part F).

MLflow에는 **새 run**으로 기록하고 모델을 등록하되, **승격은 하지 않는다**
(게이트 통과 후에만).

### 재학습 run이 지켜야 할 계약 (지키지 않으면 서빙이 죽는다)

`load_model_state()`는 champion run에서 다음을 **직접 읽는다**:

```python
thresholds = {m: run.data.metrics[f"{m}_threshold"] for m in ["mean", "max", "p95"]}   # app.py:68-70
window_size = int(run.data.params["window_size"])                                       # app.py:71
```

따라서 재학습 run이 이 metric 3개와 param 1개를 남기지 않으면, **승격 자체는
성공하는데 그 다음 `/reload-model`이 `KeyError`로 죽는다.** 자동 루프에서는
사람이 중간에 확인하지 않으므로 이 계약을 코드로 강제한다:

- 러너는 `mlflow.log_metrics(build_run_metrics(thresholds, results))`를 반드시
  호출한다 — 기존 함수가 `{method}_threshold`를 포함해 정확히 필요한 것을 만든다.
- 러너는 `mlflow.log_params(TRAINING_CONFIG)`를 호출한다 — `window_size`가 여기
  들어 있다.
- 승격 **직전에** 러너가 자기 run에서 위 4개를 다시 읽어 존재를 확인한다. 하나라도
  없으면 승격하지 않고 거부로 처리한다.

**`build_run_params()`는 재학습에 그대로 쓸 수 없다.** 이 함수는 전처리
manifest의 `experiment_split`(train/eval 실험 ID 목록)을 요구하는데
(`tracking.py:96-103`), 재학습의 학습 데이터는 실험 ID가 아니라 **날짜 배치**라
그 개념이 성립하지 않는다. 러너는 자체 params를 구성한다:

```python
mlflow.log_params({
    **TRAINING_CONFIG,                    # window_size 포함 — 위 계약 충족
    "source": "auto_retrain",
    "retrain_batch_days": "11-21",
    "retrain_batch_count": 55,
})
```

`src/lstm_ae/tracking.py`는 **수정하지 않는다** — 기존 함수는 최초 학습 경로용으로
그대로 두고, 재학습은 필요한 것만 골라 쓴다.

## Part D — 승격 게이트 (`src/retraining/gate.py`, 신규)

두 조건의 **AND**.

| | 조건 | 평가 데이터 | 역할 |
|---|---|---|---|
| **G1** | `recall >= champion_recall - 0.10` | 원본 실측 eval 14개 (고정) | 안전 — 열화 학습 차단 |
| **G2** | `accuracy > champion_accuracy` | 라벨이 도착한 최근 배치 | 근거 — 승격할 이유가 실제로 있는가 |

**G1의 0.10은 임의값이 아니다.** eval 불량이 11개라 1건 = 0.0909이므로,
"불량 1건 더 놓치는 것까지 허용, 2건이면 거부"라는 뜻이다.
10/11(0.909) → 9/11(0.818) 통과, 8/11(0.727) 거부.

**G1이 recall만 보고 precision을 보지 않는 이유**: 센서 좌표계가 이동한 환경에서
새 모델을 원본(옛 좌표계) eval셋에 적용하면 전반적으로 이상해 보여 precision이
떨어진다. 그 하락은 모델 결함이 아니라 좌표계 차이의 산물이라 판단 근거로 쓸 수
없다. 반면 recall은 "진짜 불량 패턴을 여전히 알아보는가"라는 안전 질문이라 좌표계가
바뀌어도 의미가 유지된다.

**G1만으로는 부족하다.** 모든 것을 불량이라 판정하는 망가진 모델도 recall은 1.0이라
G1을 통과한다. G1은 "승격하면 안 되는 경우"를 걸러낼 뿐이고, "승격해도 좋다"의
근거는 G2가 댄다.

G2는 라벨이 도착한 배치에 대해 champion과 재학습 모델을 각각 돌려 정확도를
비교한다. 시나리오 A에서는 champion이 오탐 중이라 새 모델이 이기고, 시나리오 B에서는
새 모델이 불량을 놓쳐 진다.

### 판정 후 동작

**거부**: `<retrain_dir>`을 그대로 보존(증거) → MLflow run에 거부 사유 태그 →
워커 로그에 사람 호출 메시지. **정본 파일과 champion alias는 전혀 건드리지 않는다.**

**통과**: 아래 순서를 지킨다. 이 순서가 중요한 이유는, 모델(MLflow alias)과
동반 파일(디스크)이 서로 다른 저장소에 있어 **중간에 실패하면 짝이 어긋나기**
때문이다.

```
1. 백업     현재 정본을 data/model_backup/<ts>/ 로 복사
              (data/model/ 전체 + data/processed/scaler.json)
2. 계약 확인 재학습 run에 {mean,max,p95}_threshold metric과
              window_size param이 실제로 있는지 재조회
              → 하나라도 없으면 여기서 중단하고 거부 처리
3. 파일 교체 <retrain_dir>/* → data/model/
              <retrain_dir>/scaler.json → data/processed/scaler.json
4. alias 교체 promote_to_champion(version)
5. 리로드    POST /reload-model
6. 검증      GET /health 의 model_version이 새 버전인지 확인
```

**3~5 중 어느 단계라도 실패하면 롤백한다**: 백업을 정본 자리로 되돌리고, alias를
이전 버전으로 되돌린 뒤, 다시 `/reload-model`을 호출한다. 롤백까지 실패하면
워커는 요란하게 실패하고 멈춘다 — 이 상태는 사람이 봐야 한다.

2번(계약 확인)을 파일 교체보다 **앞에** 두는 것이 핵심이다. 계약 위반은 파일을
건드리기 전에 잡아야 롤백할 것 자체가 생기지 않는다.

## Part E — 감시 워커 (`monitoring/drift_worker.py`, 신규)

Part B~D를 순서대로 엮는 얇은 루프. 판정 로직은 전부 `src/retraining/`에 있고
워커는 호출만 한다.

```
1틱:
  GET /drift-status  →  flag_history 갱신
  should_retrain()?  →  아니면 종료
  runner.retrain()   →  새 모델 + MLflow run
  gate.evaluate()    →  통과/거부
  통과 → 정본 이동 → promote_to_champion() → POST /reload-model
  거부 → 태그 기록 → 로그
  쿨다운 설정
```

`src/serving/app.py`를 폴링하는 별도 프로세스 구조라, 서빙과 학습이 한 프로세스에
섞이지 않는다(학습이 추론 응답을 지연시키는 안티패턴 회피).

## Part F — 자동화하면 터지는 기존 결함 2건 수정

수동 승격에서는 사람이 순서를 지켜 드러나지 않던 결함이다. 자동 루프에서는 즉시
문제가 된다.

### 결함 ① — 서빙이 champion을 시작 시 한 번만 로드한다

`src/serving/app.py:90` lifespan에서 `_state = load_model_state()`를 한 번 호출하고
끝이다. 승격해도 돌고 있는 서버는 옛 모델을 계속 쓴다 — 루프의 마지막 고리가
끊겨 있다.

**수정**: `POST /reload-model` 엔드포인트를 추가해 `_state`를 다시 로드한다.
실패 시 기존 `_state`를 유지하고 에러를 반환한다(교체 실패가 서빙 중단으로
번지면 안 된다).

### 결함 ② — scaler/baseline이 모델과 함께 버전 관리되지 않는다

`app.py:72-73`이 `data/processed/scaler.json`과 `data/model/feature_baseline.json`을
**고정 경로**에서 읽는다. MLflow 아티팩트가 아니다. 그런데 재학습은 이 파일들을
같은 자리에 덮어쓴다.

자동 루프에서 위험한 이유:
- 재학습은 게이트 판정 **전에** 실행된다 → 그 시점에 champion의 동반 파일이
  이미 덮어써진다
- 게이트가 거부해도 파일은 날아간 뒤다 → MLflow의 champion 모델은 멀쩡한데
  짝이 맞는 scaler가 사라진 상태가 되고, **에러 없이 조용히 틀린 스케일로 추론**한다
- 센서가 이동한 환경에서는 재학습이 새 scaler를 만드는 것이 맞으므로, 승격 시
  모델과 scaler가 **반드시 함께** 바뀌어야 한다

**수정 3가지**:
1. 재학습 산출물을 `data/retrain/<timestamp>/`에 격리한다. `data/model/`과
   `data/processed/`는 건드리지 않는다.
2. 게이트 **통과가 확정된 뒤에만** 정본 자리로 이동한다.
3. `scaler.json`과 `feature_baseline.json`을 MLflow run 아티팩트로도 업로드하고,
   `load_model_state()`가 **아티팩트에서 먼저 읽고 없으면 기존 고정 경로로 폴백**하게
   한다. 폴백이 필요한 이유는 현재 champion run에 이 아티팩트가 없기 때문이다 —
   하위 호환이 유지돼야 기존 100개 테스트와 현재 서빙이 깨지지 않는다.

## MLflow 기록

재학습은 champion run에 이어붙이지 않고 **새 run**으로 만든다(드리프트 metric은
기존대로 champion run에 붙는다 — 그 설계는 유지).

태그:

| 태그 | 값 |
|---|---|
| `scenario` | `temperature` / `tool_wear` |
| `trigger_day` | 트리거가 발동한 가상 날짜 |
| `retrain_data_range` | 재학습에 쓴 배치의 날짜 범위 |
| `gate_decision` | `promoted` / `rejected` |
| `gate_g1_recall` | 원본 eval recall |
| `gate_g2_accuracy_delta` | champion 대비 정확도 차 |
| `gate_reject_reason` | 거부 시 위반한 조건 |

**거부된 run도 지우지 않는다.** "왜 거부됐는지"가 남는 것이 이번 작업의 핵심
증거다.

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `src/retraining/__init__.py` | 신규 |
| `src/retraining/trigger.py` | 신규 — 순수 함수 |
| `src/retraining/runner.py` | 신규 — 재학습 데이터 구성 + 학습 실행 |
| `src/retraining/gate.py` | 신규 — G1/G2 평가 + 판정 |
| `src/monitoring/labels.py` | 신규 — `qc_labels` 테이블 적재/조회 |
| `src/serving/app.py` | 수정 — `POST /reload-model` 추가, `load_model_state()`에 아티팩트 우선 로드 + 폴백 |
| `monitoring/simulate_timeline.py` | 신규 — 스트림 생성기 |
| `monitoring/drift_worker.py` | 신규 — 감시 워커 |
| `data/timeline/`, `data/retrain/`, `data/model_backup/` | 신규 — `.gitignore` 대상(`data/` 관례) |
| `src/lstm_ae/`, `src/preprocessing/` | **변경 없음** — 함수를 호출만 한다 |
| `src/monitoring/drift.py`, `logging.py` | **변경 없음** — 그대로 재사용 |

기존 `predict_log` 스키마는 건드리지 않는다. QC 라벨은 별도 테이블로 둔다 —
실제 운영에서도 검사 결과는 다른 시스템에서 오므로 분리가 자연스럽다.

## 테스트 범위

`src/`에 들어가는 정식 코드는 pytest 단위테스트를 작성한다(기존 관례).

- `trigger.py`: 순수 함수. 데이터 부족 / 연속 미달 / 연속 충족 / 쿨다운 중 4케이스.
  DB·모델 불필요.
- `gate.py`: G1 통과·위반, G2 통과·위반, AND 조합을 가짜 지표 딕셔너리로 검증.
  경계값(recall 9/11 통과, 8/11 거부)을 명시적으로 테스트한다.
- `labels.py`: 임시 SQLite(`tmp_path`)로 적재→지연 조회 왕복. `arrived_at`이
  미래인 라벨이 조회되지 않는지 확인.
- `runner.py`: 데이터 구성 부분(라벨 필터링, scaler 재fit, eval 재생성)만
  작은 가짜 데이터로 검증한다. 실제 학습 호출은 단위테스트 대상이 아니다.
- **MLflow 계약 확인**: 가짜 run 객체로 (1) metric/param이 다 있으면 통과
  (2) `mean_threshold`가 없으면 거부 (3) `window_size`가 없으면 거부를 검증한다.
  이 테스트가 "승격은 됐는데 서빙이 KeyError로 죽는" 사고를 직접 겨냥한다.
- **승격 롤백**: `tmp_path`에 가짜 정본/백업 디렉터리를 만들고, 파일 교체 후
  alias 교체가 실패하는 상황을 주입해 정본이 원래대로 복원되는지 검증한다.
- `app.py`: `POST /reload-model` 라우트를 기존 `test_app.py` 관례
  (`TestClient` + 의존성 오버라이드)로 통합 테스트. 로드 실패 시 기존 상태가
  유지되는지도 확인한다.

`monitoring/`의 두 스크립트는 기존 관례대로 pytest 대상이 아니며, 런타임 assert와
실행 로그가 검증 역할을 한다.

기존 **100개 테스트는 그대로 통과해야 한다.** `app.py` 수정이 있으므로 회귀
확인이 특히 중요하다.

## 검증 방법

1. `pytest` 전체 통과(기존 100개 + 신규).
2. 변형 상수 스윕 실행 → Day 40 도달 비율이 목표 대역(A: 1.5~2.0, B: 3.0)에
   들어오는지 확인. 못 맞추면 대역/그리드를 재조정하고 그 사실을 보고한다.
3. **시나리오 A 전체 실행** → 트리거가 발동한 날, 게이트 G1/G2 값, 승격 여부,
   승격 후 오탐이 실제로 줄었는지를 로그로 확인.
4. **시나리오 B 전체 실행** → 트리거 발동 후 게이트가 **거부**하는지, 거부 사유가
   G1인지 확인.
5. MLflow UI에서 두 시나리오의 run과 태그, 거부된 run이 남아 있는지 확인.
6. 결함 ② 수정 검증: 게이트 거부 후 `data/processed/scaler.json`과
   `data/model/`이 **손상되지 않았는지**, champion 추론이 그대로 동작하는지 확인.
7. 결과를 있는 그대로 보고한다. 기대와 다르게 나오면(예: 시나리오 B가 게이트를
   통과해 버리면) 숨기지 않고 보고하고 게이트 기준 재조정을 논의한다.

## 실측으로 확인된 한계 — 표본 부족과 평가 창 민감도 (2026-08-19 추가)

### G1의 눈금은 실험 1개다

eval셋이 14개(정상 3 + 불량 11)뿐이라 지표가 뚝뚝 끊긴다. 실험 하나가
뒤집히면 recall이 0.0909 움직이고, precision은 정상이 3개뿐이라 오탐 1건에
0.33이 날아간다. 애초 G1 기준을 `recall >= champion - 0.10`으로 잡았는데
그 0.10이 사실상 "실험 1개"였다 — 소수점 넷째 자리까지 찍히는 가짜 정밀도다.

**고칠 수 없다.** eval을 늘리려면 실험이 더 있어야 하는데 공개 데이터셋 25개가
전부이고 3개는 중복이라 이미 뺐다. **합성 데이터로 eval셋을 채우면 안 된다** —
게이트 기준셋은 "진짜 불량이 이렇게 생겼다"는 앵커라, 여기에 생성한 데이터를
넣으면 게이트가 자기가 만든 답을 채점하게 된다.

대신 기준을 **놓친 개수로 표현**하도록 바꿨다(`missed <= champion_missed + 1`).
판정은 동일하고 가짜 정밀도만 사라진다. 표본 부족을 해결하는 게 아니라
숨기지 않는 조치다.

### 평가 창을 넓히면 구모델이 유리해진다

G2가 라벨 도착분 160건 중 20건만 쓰고 있어 60건으로 넓혀 봤다.
**시나리오 A가 승격에서 거부로 뒤집혔다.**

| Day 37 게이트 | 표본 20 | 표본 60 |
|---|---|---|
| 재학습 모델 | 0.95 | 0.48 |
| champion | 0.60 | 0.50 |
| 판정 | **승격** | **거부** |

Day 37 시점 라벨은 Day 30까지 도착한다. 표본 20은 Day 26~30(드리프트가 심해
champion이 오탐하던 구간)을 보지만, 표본 60은 Day 19~30이라 앞쪽에 champion이
멀쩡했던 구간이 섞인다.

재학습 모델은 최근 환경에 맞춰 학습되므로 최근 구간에서 강하다. 창을 과거로
넓히면 그 강점이 희석되는 반면 champion은 자기가 잘하던 옛 구간 점수를
벌어들인다. **드리프트 상황에서 평가 창 확대는 중립적이지 않고 구모델에
유리하다.** G2의 질문은 "지금 현장에서 더 나은가"인데 12일치 평균은 그
"지금"이 아니다.

`GATE_SAMPLE_SIZE = 20`으로 되돌렸다. 20이 통계적으로 넉넉해서가 아니라
넓히면 측정 대상 자체가 바뀌기 때문이다. **평가 창 크기가 승격 판정을 뒤집을
만큼 민감하다는 것이 이 설계의 알려진 한계다.**

30~40으로 절충해 승격이 살아나는지 볼 수도 있었으나 하지 않았다. 원하는
결과가 나올 때까지 손잡이를 돌리는 행위이고, 이 프로젝트는 `feedrate=20`
오탐에서 이미 같은 선을 그었다.

## 남은 리스크

- **시나리오 B가 게이트를 통과해 버릴 수 있다.** Day 11~20 구간의 마모가 약해서
  재학습 모델의 recall이 별로 안 떨어지는 경우다. 그러면 `WEAR_RATE`나 라벨 전환
  시점(Day 21)을 조정해야 하는데, 이는 "데모가 성립하도록 값을 맞추는" 행위이므로
  조정 사실과 근거를 문서에 남긴다.
- **G2가 두 시나리오를 충분히 가르지 못할 수 있다.** 라벨 도착 구간이 좁으면
  정확도 차이가 노이즈에 묻힌다. 이 경우 비교 구간을 넓히는 것으로 대응한다.
- 변형 상수는 현 champion 모델에 종속적이다. 모델이 바뀌면 달성 비율이 달라지지만,
  스크립트가 매번 실제 비율을 출력하므로 조용히 틀리지는 않는다.
- 온도 시나리오의 열변위 offset은 **실측 근거가 없는 판단**이다. 데이터셋에 온도
  기록이 없어 검증할 방법이 없으며, 변형 폭을 실측 대역 안에 가두는 것으로만
  현실성을 담보한다. 이 가정은 발표에서 명시되어야 한다.
- `feedrate=20` 커버리지 한계는 이번 작업으로 해결되지 않는다. 재학습 데이터를
  train 8개 실험 전부에서 만드는 것은 이 문제를 **악화시키지 않기 위한** 조치이지
  개선책이 아니다.
