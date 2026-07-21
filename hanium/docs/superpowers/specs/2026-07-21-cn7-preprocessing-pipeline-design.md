# CN7 사출성형 데이터 전처리 파이프라인 설계

- 날짜: 2026-07-21
- 상태: 설계 승인됨, 구현 대기

## 배경

KAMP 「사출성형기 AI 데이터셋」(자동차 앞유리 사이드 몰딩 사출 공정) 중 CN7 제품만
사용해 실시간 이상탐지(추후 LSTM 기반 오토인코더 + 변수별 복원오차 → RAG 불량유형
매칭) 모델을 만드는 것이 최종 목표다. 이 문서는 그중 **전처리 파이프라인**(정제 →
자가정제 → 스케일링 → 분할) 범위만 다룬다. LSTM 시퀀스 구성, 모델 아키텍처, RAG
연계는 별도 스펙으로 다룬다.

가이드북(`04. Guidebook_Molding.pdf`, `04. Guidebook_Molding_데이터부분.pdf`)의
분석 실습은 알고리즘 성능 비교 데모가 목적이라 본 프로젝트(현장 설명가능성 중심의
실시간 이상탐지)와 목적이 다르다. 가이드북은 데이터 계보 파악(어느 파일이 어떻게
만들어졌는지) 용도로만 참조했고, 파이프라인 자체는 우리 목적에 맞게 새로 설계한다.

## 데이터 계보 (조사 결과 요약)

- `labeled_data.csv`, `unlabeled_data.csv`(원본 다운로드본, 이제 삭제됨) — 여러
  제품/여러 사출기 혼재. `Reason`, `TimeStamp` 보유. 미가공 수치.
- KAMP가 제공한 `moldset_labeled_cn7.csv` 등 2차 가공 파일들은 이미
  StandardScaler로 스케일링되어 있고(재계산 시 std가 1.000413로 통일되는 것으로
  확인) `Reason`이 빠져 있어 우리 목적(RAG 매칭)에 맞지 않아 사용하지 않기로 함.
- 최종적으로 원본 2개 파일만 `PART_NAME` LIKE `CN7` & `EQUIP_NAME` ==
  `650톤-우진2호기` 조건으로 직접 필터링해 `data/dataset/cn7_labeled.csv`(6,736행),
  `data/dataset/cn7_unlabeled.csv`(35,239행) 두 파일만 남겼다. RG3 및 나머지
  원본 파일은 삭제함(공개 데이터라 KAMP 포털에서 재다운로드 가능).

## 목표 / 비목표

**목표**
- `cn7_labeled.csv`, `cn7_unlabeled.csv`를 입력으로 받아 모델 학습에 바로 투입
  가능한 정제·정규화된 train/eval 데이터셋을 만든다.
- 나중에 복원오차를 실단위(초/mm/℃/MPa)로 역변환해 현장에 설명 가능하도록
  스케일러 파라미터를 별도 보존한다.
- 데이터 리키지(정보 누수)를 피한다(스케일러는 train에만 fit, eval 라벨은 절대
  학습에 사용하지 않음).

**비목표 (다음 스펙에서 다룸)**
- LSTM 입력을 위한 샷(사이클) 시퀀스/윈도우 구성
- 오토인코더 등 최종 모델 아키텍처 설계
- RAG 불량유형 매칭, 비라벨 스트림 재생 시뮬레이션

## 입력

| 파일 | 행수 | 비고 |
|---|---|---|
| `data/dataset/cn7_labeled.csv` | 6,736 | `PassOrFail`(Y/N), `Reason` 보유 |
| `data/dataset/cn7_unlabeled.csv` | 35,239 | 라벨 없음, `ERR_FACT_QTY`(일자 단위 집계, 신뢰도 낮아 미사용) 보유 |

## 출력 (`data/processed/`)

| 파일 | 내용 |
|---|---|
| `train.csv` | 자가정제 + 스케일링된 unlabeled 데이터 (모델 학습용, "대부분 정상" 전제) |
| `eval.csv` | 스케일링된 labeled 데이터 전체 (정상 6,717 + 불량 19), 학습에 전혀 사용 안 함 |
| `scaler.json` | 컬럼별 mean/std (역변환용) |
| `removed_outliers.csv` | 자가정제 단계에서 train에서 제거된 행 (감사 로그) |
| `manifest.json` | 필터 조건, 드롭 컬럼 목록, 처리 일시 등 재현성 기록 |

## 파이프라인 단계

### 1. 컬럼 정제

공정변수 36개 중 12개를 드롭한다 (실측 확인됨):

- `Mold_Temperature_1,2,5,6,7,8,9,10,11,12` (10개): train/eval 양쪽 다 상시 0 —
  이 라인에서 사용하지 않는 죽은 센서 채널.
- `Switch_Over_Position`, `Barrel_Temperature_7` (2개): unlabeled 전체에서는
  48.5%가 0/51.5%가 실값(~17.3mm, ~30℃)으로 나타나지만, eval 구간
  (2020-10-16~11-03)에서는 100% 0. 특정 시점부터 센서가 꺼진 것으로 추정되며,
  이대로 두면 train/eval 분포가 근본적으로 달라 재구성오차를 왜곡시키므로 드롭.

나머지 process 변수(Injection_Time, Filling_Time, Plasticizing_Time, Cycle_Time,
Clamp_Close_Time, Cushion_Position, Plasticizing_Position, Clamp_Open_Position,
Max_Injection_Speed, Max_Screw_RPM, Average_Screw_RPM, Max_Injection_Pressure,
Max_Switch_Over_Pressure, Max_Back_Pressure, Average_Back_Pressure,
Barrel_Temperature_1~6, Hopper_Temperature, Mold_Temperature_3,4)은 모델 피처로
사용한다.

`PassOrFail`, `Reason`, `TimeStamp`는 모델 피처에서는 제외하되 `eval.csv`에는
그대로 보존한다(사후분석/RAG 매칭용). `ERR_FACT_QTY`는 eval에 아예 없는 컬럼이고
일자 단위 집계라 신뢰도가 낮아 피처로 쓰지 않는다. `_id`, `PART_FACT_PLAN_DATE`,
`PART_FACT_SERIAL`, `PART_NO`, `PART_NAME`, `EQUIP_CD`, `EQUIP_NAME` 등 메타데이터
컬럼은 필터링으로 이미 상수가 되었으므로 드롭한다.

### 2. 라벨 처리 (labeled 데이터만)

- `PassOrFail`: `Y`→`0`(정상), `N`→`1`(불량)
- `Reason == '초기허용불량'`인 행은 라인 초기가동 시 허용된 불량으로, 물리적
  불량현상과 성격이 달라 **정상(0)으로 재분류**한다.
- 최종 라벨 분포: 정상 6,717 / 불량 19 (가스 13 + 미성형 6)

### 3. 자가정제 (Self-cleaning, train에만 적용)

unlabeled 데이터는 라벨이 없어 정상/불량이 섞여 있을 수 있다(`ERR_FACT_QTY`를
날짜별로 확인한 결과 일자 단위 집계값이라 행 단위 오염률은 정확히 알 수 없음이
확인됨). 완전한 무결성을 요구하면 준지도 이상탐지 자체가 성립하지 않으므로, 아래
경량 정제 절차로 명백한 이상치만 솎아낸다.

- **IsolationForest**(scikit-learn)를 정제 전용 도구로 사용한다. 최종 모델
  (LSTM-AE)과는 별개이며 다음 스펙에서 설계할 최종 모델을 여기서 만들지 않는다.
- `contamination`은 labeled 데이터의 실측 불량률(0.3~0.6%)을 참고해 여유 있게
  **1%**로 설정한다.
- 상위 1% 이상치로 판정된 행을 train에서 제거하고 `removed_outliers.csv`에 원본
  인덱스와 이상치 점수를 기록한다.
- 정제된 나머지로 스케일링 단계를 진행한다.

### 4. 스케일링

- `StandardScaler`를 자가정제 후 train에만 `fit`한다.
- train과 eval 모두 동일한 스케일러로 `transform`한다 (eval에는 절대 재fit하지
  않는다 — 리키지 방지).
- 컬럼별 mean/std를 `scaler.json`에 저장해 나중에 복원오차를 실단위로 역변환할
  수 있게 한다.

### 5. 분할

- **train** = 자가정제 + 스케일링된 unlabeled 데이터 전체
- **eval** = 스케일링된 labeled 데이터 전체 (정상 6,717 + 불량 19), 학습에 전혀
  사용하지 않음
- 별도 validation split은 두지 않는다. 이유: 진짜 불량이 19건뿐이고 그마저
  10/16 05:21~05:57(17건 연속 발생, 하나의 사건)과 11/03 04:44(2건) 단 두
  사건에 몰려 있어, 3등분하면 val/test 중 한쪽에 불량이 거의 남지 않아 통계적
  의미가 없어진다. 대신 임계값(threshold)은 train(정상)의 재구성오차 분포에서
  통계적 기준(예: 평균+3표준편차, 99.7% 규칙)으로 정하고, eval에는 적용만 해서
  precision/recall을 산출한다(라벨을 보고 임계값을 짜맞추는 과적합 방지).

## 알려진 한계 / 리스크

- 평가셋의 진짜 불량이 19건뿐이라 precision/recall 수치의 통계적 신뢰도가
  낮다. 정량 지표와 별개로, 재구성오차 시계열이 실제 10/16 05:21~05:57 구간과
  11/03 04:44 시점에 정확히 튀는지 정성적으로 확인하는 검증을 병행할 것을
  권장한다(다음 스펙/구현 단계 과제).
- unlabeled 데이터의 행 단위 오염률은 정확히 계산할 방법이 없다(`ERR_FACT_QTY`가
  일자 단위 집계라 분모를 알 수 없음). 자가정제의 `contamination=1%`는 추정치이며
  향후 모델 성능을 보고 재조정이 필요할 수 있다.
- `Switch_Over_Position`, `Barrel_Temperature_7`을 드롭하는 결정은 eval 구간에서
  센서가 꺼져 있었다는 관측에 근거한 것으로, 향후 다른 시기 데이터를 추가할 때
  이 센서들이 다시 켜져 있다면 재검토가 필요하다.
