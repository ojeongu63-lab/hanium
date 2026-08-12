# CNC 정상 실험 LOOCV 검증 설계

- 날짜: 2026-08-12
- 상태: 설계 완료, 구현 전

## 배경

OOD(Out-of-Distribution) 게이트를 설계하던 중, 정상(양품) 실험이 usable 22개(정상
11 + 불량 11) 중 11개뿐이고 그중 8개만 학습에 쓰고 있다는 점이 드러났다. 지금
서빙 중인 모델은 이 8/3 고정 분할 한 번의 결과(precision=0.909, recall=0.909,
mean 방식 기준, `data/model/evaluation_report.json`)를 기준으로 판단하고 있는데,
정상 실험이 11개뿐인 상황에서 "8/3으로 한 번 나눈 결과"가 우연히 좋게 나온 건지
아니면 안정적인 결과인지 확인할 방법이 없었다.

특히 eval 정상 3개 중 하나인 실험22(worn 공구 × feedrate=20)는 그 조합의 정상
샘플이 데이터셋 전체에 그것 하나뿐이라 오탐(FP)이 났는데, 이게 "이 특정 분할의
불운"인지 "이 조합 자체가 데이터에 근본적으로 없어서 어떤 분할을 해도 못 맞히는
것"인지 구분이 안 됐다. 실제 CNC 장비로 추가 실험은 불가능(공개 데이터셋 전용)
하다는 것도 이미 확인된 상태다.

이 스펙은 학습/평가 분할 전략을 **고정 8/3 분할 → 정상 11개에 대한
leave-one-out 교차검증(LOOCV)**으로 바꿔, (1) 성능 추정치의 신뢰도를 높이고
(2) 실험22류의 실패가 우연인지 데이터의 근본 한계인지를 통계적으로 확정하는 것을
목표로 한다. 서빙 중인 champion 모델이나 MLflow 등록 로직은 건드리지 않는
별도의 진단/검증 작업이다.

## 목표 / 비목표

**목표**
- 정상 실험 11개 각각을 정확히 한 번씩 "완전히 학습에 안 쓴 상태"로 검증한다.
- 폴드별로 그 정상 실험이 정상으로 맞았는지(tn) 오탐(fp)났는지 집계해, exp22류
  실패가 반복되는지 확인한다.
- 불량 11개에 대한 recall/precision이 학습 구성(어느 정상 실험이 빠졌는지)에
  따라 얼마나 흔들리는지 확인해, 지금의 단일 분할 결과(0.909/0.909)가 대표값인지
  판단할 근거를 만든다.
- 기존 `data/model/`, `data/processed/`, MLflow 레지스트리는 전혀 건드리지
  않는다 — 완전히 분리된 산출물 트리에 결과를 남긴다.

**비목표**
- champion 모델 교체나 재승격 (이 작업은 진단용이지 새 모델을 배포하려는 게
  아니다)
- OOD 게이트 자체의 재설계 (이 검증 결과를 보고 다음에 논의)
- feedrate 12/15 커버리지 문제 해결 (데이터에 그 조합의 정상 샘플이 아예 없어
  분할 전략을 바꿔도 못 채워짐 — 이미 확인됨)

## 전체 절차

기존 `preprocessing.pipeline.run_pipeline()`과 `lstm_ae.pipeline.run_lstm_pipeline()`은
둘 다 실험 ID 리스트/CSV 경로를 인자로 받는 순수 함수라 **수정 없이 그대로
재사용**한다. 새로 만드는 건 이 둘을 11번 반복 호출하는 오케스트레이션 스크립트뿐이다.

```
정상 실험 11개 = [1, 2, 3, 11, 12, 13, 14, 15, 17, 18, 22]
불량 실험 11개 = [4, 5, 6, 7, 8, 9, 10, 16, 20, 21, 23]  (모든 폴드에서 고정)

for held_out in 정상 11개:
    fold_train_ids = 정상 11개 - {held_out}   # 10개
    run_pipeline(
        train_experiment_ids=fold_train_ids,
        eval_good_experiment_ids=[held_out],
        eval_bad_experiment_ids=불량 11개,
        output_dir=loocv/folds/fold_<held_out>/processed,
    )
    summary = run_lstm_pipeline(
        train_csv_path=loocv/folds/fold_<held_out>/processed/train.csv,
        eval_csv_path=loocv/folds/fold_<held_out>/processed/eval.csv,
        output_dir=loocv/folds/fold_<held_out>/model,
        window_size=20, hidden_size=64, latent_dim=16, epochs=50,
        batch_size=64, learning_rate=1e-3, random_seed=42,
        threshold_percentile=95.0,   # 현재 champion 모델과 동일 설정
    )
    fold 결과 기록
```

- 스케일러는 폴드마다 그 폴드의 10개 정상 실험 기준으로 새로 fit된다(`run_pipeline`이
  항상 하는 동작 그대로) — 리키지 없음.
- MLflow에는 등록하지 않는다. 11개 다 후보 모델이 아니라 진단용 일회성 실행이고,
  champion 승격 로직과 무관하다.
- `experiment_index_path`/`experiment_dir`는 `scripts/run_preprocessing.py`와 동일한
  원본 경로를 그대로 쓴다: `data/dataset/CNC 비식별화 원본데이터_1209/train.csv`,
  `data/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2/`.

## 산출물

```
version_2/loocv/
  run_loocv.py                     # 오케스트레이션 스크립트 (src/에 추가 안 함, 일회성)
  folds/
    fold_01/processed/{train.csv, eval.csv, scaler.json, manifest.json}
    fold_01/model/{model.pt, evaluation_report.json, training_config.json, ...}
    fold_02/...
    ... (정상 11개 실험 ID별로 11개 폴드)
  summary.csv                      # 폴드별 1행
  summary.json                     # 집계 결과
  .gitignore                       # folds/ 는 재생성 가능한 대용량 산출물이라 git 제외
```

`fold_<id>/model/`은 `run_lstm_pipeline()`이 이미 만드는 표준 산출물(`data/model/`과
동일한 파일 구성)을 그대로 재사용한다 — 새 저장 포맷을 만들지 않는다.

### `summary.csv` (폴드별 1행)

| 컬럼 | 설명 |
|---|---|
| `held_out_experiment_id` | 이 폴드에서 검증용으로 뺀 정상 실험 |
| `mean_correctly_classified` | held_out 실험이 mean 방식으로 정상 판정됐는지(bool) |
| `mean_bad_recall` | 이 폴드 모델의 불량 11개에 대한 recall(mean 방식) |
| `mean_bad_precision` | 이 폴드 모델의 불량 11개+held_out 1개에 대한 precision(mean 방식) |
| (동일 패턴으로 `max_*`, `p95_*`) | |

### `summary.json` (집계)

두 성격이 다른 지표를 구분해서 담는다 — 섞으면 통계적 의미가 달라진다:

```json
{
  "good_side_loocv": {
    "note": "정상 실험 11개, 각각 정확히 1번씩 완전 홀드아웃 평가 (진짜 LOOCV)",
    "mean": {"n": 11, "correct_tn": <int>, "misclassified_fp": <int>, "fp_experiment_ids": [...]},
    "max": {...}, "p95": {...}
  },
  "bad_side_stability": {
    "note": "불량 실험 11개는 폴드마다 매번 재평가됨(학습 구성이 폴드마다 다른 모델로) - 독립 표본 아님, recall이 학습 구성에 얼마나 민감한지 보는 안정성 체크",
    "mean": {"recall_per_fold": [...11개 값...], "min": .., "max": .., "mean": ..},
    "max": {...}, "p95": {...}
  },
  "comparison_to_fixed_split": {
    "fixed_split_result": { "...data/model/evaluation_report.json 값 그대로 인용..." }
  }
}
```

`good_side_loocv`가 이번 작업의 핵심 질문("exp22 실패가 우연인가")에 대한 답이고,
`bad_side_stability`는 부가적으로 기존 0.909/0.909가 얼마나 대표성 있는 값인지
보여주는 참고 지표다.

## 예상 결과 (검증 전 가설, 사후에 실제 값으로 교체)

exp22가 held-out인 폴드에서도 여전히 오탐(fp)이 날 것으로 예상한다 — 나머지 10개
정상 실험에도 worn×feedrate=20 조합이 없기 때문이다. 이게 실제로 일어나면 "우연이
아니라 데이터의 근본 한계"라는 게 통계적으로 확정된다. 반대로 다른 폴드에서
새로운 실험이 오탐 나거나, exp22가 의외로 맞는다면 그 자체로 추가로 조사해야 할
새로운 정보다.

## 연산 부담 및 실행 방식

11번의 전체 학습(각 폴드 10개 정상 실험, 현재 8개보다 약간 더 큰 규모, CPU,
50 epoch)을 순차 실행한다. 실행 전 `who`/`top`으로 서버 상태를 확인하고
`nice -n 19`로 실행한다. 병렬화하지 않는다(공유 서버 예의 + 폴드 수가 적어
순차로도 수 분 내 완료될 것으로 예상).

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `version_2/loocv/run_loocv.py` | 신규 — 기존 `run_pipeline`/`run_lstm_pipeline`을 11번 호출하는 오케스트레이션 스크립트 |
| `version_2/loocv/.gitignore` | 신규 — `folds/` 제외 |
| 기존 `src/` 코드 | **변경 없음** — 두 파이프라인 함수 모두 이미 실험 ID 리스트/CSV 경로를 인자로 받아 수정 없이 재사용 가능 |

## 테스트 범위

이 스크립트는 일회성 진단 도구라 별도 단위 테스트는 만들지 않는다(기존 프로젝트에서
`baseline_isolation_forest.py` 같은 일회성 분석 스크립트에 적용한 것과 동일한 관례).
대신 스크립트 자체에 아래 sanity check를 넣어 실행 중 바로 검증한다:
- 매 폴드 `fold_train_ids`가 정확히 10개인지
- 매 폴드 eval 대상이 held_out 1개 + 불량 11개 = 12개인지
- `summary.csv`가 정확히 11행인지
- `good_side_loocv.mean.n == 11`이고 `correct_tn + misclassified_fp == 11`인지

## 검증 방법

1. `run_loocv.py` 실행 → 위 sanity check 전부 통과 확인.
2. `summary.json`의 `good_side_loocv`로 exp22류 실패가 반복되는지, 다른 실험에서도
   나오는지 확인.
3. `bad_side_stability`의 폴드별 recall 범위를 기존 고정분할 결과(0.909)와 비교.
4. 결과를 사용자에게 보고 — OOD 게이트 재논의 여부는 이 결과를 보고 결정.
