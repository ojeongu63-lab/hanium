# CNC 희귀 정상 샘플 증강 설계

- 날짜: 2026-08-12
- 상태: 설계 완료, 구현 전

## 배경

LOOCV 검증(`2026-08-12-cnc-loocv-validation-design.md`)으로 `feedrate=20` 정상
샘플이 데이터셋 전체에 exp2(unworn)/exp22(worn) 딱 2개뿐이라, 어느 쪽을
held-out해도 나머지 하나만으로는 못 커버해서 오탐이 남을 확인했다
(`good_side_loocv.mean.misclassified_fp = 2`, `fp_experiment_ids: [2, 22]`).
실제 CNC 장비로 추가 실험은 불가능(공개 데이터셋 전용, 기존 확인됨).

사용자의 최종 목표는 "학습 모델의 신뢰도를 높이는 것"이다. 다만 이번 작업은
그중에서도 **LOOCV로 콕 집어낸 특정 약점(feedrate=20 커버리지) 하나를 좁혀서
고치는 것**으로 스코프를 명확히 한다 — `feedrate 12/15`처럼 정상 샘플이 아예
0개인 구간을 추정해서 채우는 것(더 근본적인 "전체 데이터 부족" 해결)은 실제로
관측한 적 없는 값을 추정하는 위험이 크다고 판단해 배제했다(사용자 확인 완료).
그 방향은 별도로 "LOOCV 폴드 모델 앙상블"(다음 서브프로젝트)로 다룬다.

## 목표 / 비목표

**목표**
- 실제로 존재하는 희귀 샘플(exp2, exp22)에 소규모 변주를 만들어 학습 데이터에
  추가한다.
- LOOCV로 증강 전/후를 비교해 실제로 효과가 있는지 검증한다 — 효과를 미리
  장담하지 않는다.
- "신뢰도" 관점에서 오탐 개수뿐 아니라 폴드 간 성능 편차(안정성)까지 같이
  본다.

**비목표**
- `feedrate 12/15`처럼 정상 샘플이 0개인 구간을 추정해서 채우는 것 (위험하다고
  판단해 배제)
- champion 모델 교체/재승격 (이번 작업도 LOOCV와 마찬가지로 진단·검증 목적)
- 앙상블 서빙 (다음 서브프로젝트에서 별도로 다룸 — 이번에 만드는 증강 폴드
  모델을 그 재료로 재사용할 수 있음)

## Part A — 증강 데이터 생성 (`version_2/augmentation/generate_augmented.py`)

**대상**: exp2(unworn), exp22(worn) 각각 3개씩, 총 6개 신규 변주.

**기법**: 데모 정상변형(`synthetic/generate_synthetic.py`의 `vibration_backlash`,
진폭 0.05에서 "good" 유지가 이미 검증됨)과 같은 메커니즘 재사용 — 41개
`FEATURE_COLUMNS` 전체에, 그 피처 자체의 표준편차 × 0.05를 표준편차로 하는
가우시안 노이즈를 더한다(평균은 유지, 흔들림만 추가). 변주마다 다른 시드로
재현 가능하게 생성(exp2: seed 101/102/103, exp22: seed 221/222/223).

**새 실험 ID**: exp2 파생 → `2001,2002,2003`, exp22 파생 → `2201,2202,2203`
(실제 실험 ID 1~25와 겹치지 않음).

**메타데이터**: 원본 `train.csv`(25행)에서 exp2/exp22 행을 복사해 `No`만 새
ID로 바꾼 6행을 추가한 `combined_index.csv`(31행) 생성 — feedrate/tool_condition/
라벨은 원본과 동일(둘 다 정상이므로 그대로 유지).

**산출물**:
```
version_2/augmentation/
  generate_augmented.py
  combined_dataset/      # 원본 25개 CSV 복사본 + 신규 6개 CSV (31개)
  combined_index.csv     # 원본 25행 + 신규 6행
```
`combined_dataset/`, `combined_index.csv`는 스크립트 재실행으로 다시 만들 수
있는 산출물이라 git 제외(`data/`와 같은 취급).

## Part B — 증강 반영 LOOCV (`version_2/loocv/run_loocv_augmented.py`)

기존 `loocv/run_loocv.py`는 건드리지 않는다(이미 완료·병합된 진단 스크립트).
새 스크립트로 별도 작성.

### 핵심 설계 — 정보 누수 방지

held-out 후보(정상 11개, real ID)는 기존과 동일. 각 폴드의 학습 데이터를:

```
fold_train_ids = (정상 11개 − held_out) + (held_out을 "제외한" 나머지 실험들의 증강 변주)
```

로 구성한다. 즉 **exp2가 held-out인 폴드에서는 exp2 자신의 증강 변주 3개도
같이 빠지고**(안 그러면 "exp2를 안 배운 상태에서 검증"이라는 LOOCV 취지가
깨짐 — 사실상 변형판을 몰래 학습하는 정보 누수), exp22의 증강 변주 3개는
그대로 남는다. 이게 "exp22(같은 feedrate=20)의 변주가 학습에 있으면 exp2를
더 잘 맞히는가"라는, 이번 작업이 검증하려는 가설 그 자체다.

- `held_out=2`: train = `[1,3,11,12,13,14,15,17,18,22]`(10) + `[2201,2202,2203]`(exp22 변주) = 13개
- `held_out=22`: train = `[1,2,3,11,12,13,14,15,17,18]`(10) + `[2001,2002,2003]`(exp2 변주) = 13개
- 그 외 9개 폴드: train = 10개(real) + `[2001,2002,2003,2201,2202,2203]`(exp2·exp22 둘 다 fold_train에 있으므로 둘의 변주 다 포함) = 16개

`eval_good=[held_out]`, `eval_bad=BAD_EXPERIMENT_IDS`(11개, 기존과 동일)는
안 바뀐다. `run_pipeline()`/`run_lstm_pipeline()`은 기존 LOOCV와 동일하게
그대로 재사용하되, `experiment_index_path`/`experiment_dir`만 Part A가 만든
`combined_index.csv`/`combined_dataset/`을 가리키게 한다.

### 산출물

```
version_2/loocv/
  run_loocv_augmented.py
  augmented_folds/            # 폴드별 산출물 (기존 folds/와 같은 구조), git 제외
  summary_augmented.json
  summary_augmented.csv
```

## 검증 — "신뢰도" 관점 비교

기존 `loocv/summary.json`(증강 전) vs `summary_augmented.json`(증강 후)을
비교한다:

1. **오탐 개수**: `good_side_loocv.mean.misclassified_fp`가 2에서 줄어드는지,
   `fp_experiment_ids`에서 2/22가 빠지는지.
2. **다른 폴드 부작용 없는지**: 나머지 9개 폴드가 여전히 정상 판별되는지
   (`correct_tn`이 늘어난 만큼이 아니라 그 이상으로 새로운 오탐이 안 생겼는지).
3. **안정성(편차) — 신뢰도 지표**: `bad_side_stability`에 표준편차(`std`)를
   새로 계산해 추가한다(기존 스키마엔 min/max/mean만 있었음). 편차가 줄어들면
   "어떤 정상 실험을 빼도 결과가 비슷하게 안정적"이라는 뜻으로, 이게 "신뢰도"를
   수치로 보여주는 지표다.
4. **전체 합산 성능**: 11개 폴드를 합산한 pooled precision/recall이 증강 전
   (recall 평균 0.760, `bad_side_stability.mean.mean`)보다 나아지는지.

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `version_2/augmentation/generate_augmented.py` | 신규 |
| `version_2/loocv/run_loocv_augmented.py` | 신규 |
| 기존 `src/`, `loocv/run_loocv.py` | 변경 없음 |

## 테스트 범위

정식 pytest 단위테스트는 만들지 않는다 — `loocv`/`synthetic`과 동일한 관례
(일회성 진단/검증 스크립트, 런타임 assertion으로 검증).

- `generate_augmented.py`: 생성된 6개 CSV의 행 수가 원본과 같은지, 메타데이터
  컬럼(Machining_Process 등)이 원본과 동일한지 assertion.
- `run_loocv_augmented.py`: 각 폴드의 `fold_train_ids` 개수가 예상대로인지
  (held_out이 2/22면 13개, 그 외는 16개), `combined_index.csv`에 신규 6개 ID가
  전부 있는지 assertion.

## 검증 방법

1. `generate_augmented.py` 실행 → 6개 CSV + `combined_index.csv` 생성 확인.
2. `run_loocv_augmented.py` 실행 → 11개 폴드 전부 assertion 통과 확인(수 분
   소요 예상, 폴드당 학습 데이터가 기존보다 약간 커짐).
3. `summary.json`과 `summary_augmented.json`을 나란히 비교해 위 4가지 지표
   보고.
4. 결과를 사용자에게 보고 — 효과가 없거나 부작용이 있으면 그것도 그대로
   보고(미리 좋아질 거라 가정하지 않음). 다음 서브프로젝트(앙상블) 진행 여부
   결정.
