# 01 · CN7 사출성형 이상탐지 (1차 트랙, 종료)

> **이 트랙은 종료됐습니다.** 현재 진행 중인 결과물은
> [`../02-cnc-machining/`](../02-cnc-machining/)입니다. 이 폴더는 방법론을 처음
> 세우고 검증했던 기록으로 보존돼 있습니다.

사출성형기(650톤 우진2호기)에서 CN7 부품을 찍은 **샷 단위 공정 데이터**로,
정상 데이터만 학습한 LSTM-Autoencoder의 재구성 오차를 이용해 불량 샷을
탐지한다.

## 결론 먼저

| 지표 | 값 |
|---|---|
| precision | **0.044** (탐지 204건 중 실제 불량 9건) |
| recall | **0.50** (실제 불량 18건 중 9건 탐지) |
| TP / FP / FN / TN | 9 / 195 / 9 / 3,761 |

**성능이 실용 수준에 못 미쳤다.** 오탐이 195건으로, 이대로 현장에 걸면 알람을
아무도 안 보게 되는 수준이다.

다만 원인을 따져보니 **방법론이 아니라 데이터 쪽 문제**였다. 평가셋에 들어 있는
진짜 불량이 **18건뿐**이라, precision/recall이 몇 건의 판정 차이로 크게 출렁인다.
이 숫자로는 "LSTM-AE 방식이 제조 이상탐지에 안 맞는다"고 결론 낼 근거가 부족했다.

그래서 **같은 방법론을 불량 표본이 더 확보된 CNC 가공 데이터에 그대로 적용해**
방법론 자체의 유효성을 따로 검증하기로 했고, 그 결과가 `02-cnc-machining/`이다
(precision 0.91 / recall 0.91 — 방법론은 유효했다).

## 데이터

`data/`는 git에 포함되지 않는다. 아래 두 파일이 `data/dataset/`에 있어야 한다.

| 파일 | 내용 |
|---|---|
| `cn7_labeled.csv` | 6,736행 — `PassOrFail` 라벨이 있는 구간 (평가용) |
| `cn7_unlabeled.csv` | 35,239행 — 라벨 없는 구간 (학습용, 정상으로 가정) |

필터 조건: `PART_NAME LIKE 'CN7%' AND EQUIP_NAME == '650톤-우진2호기'`

**라벨 정의**: `PassOrFail == 'N'`이면 불량(1). 단 **초기 기동 불량은 정상(0)으로
처리**한다 — 설비를 켜고 안정화되기 전 나오는 불량은 공정 이상이 아니라
정상적인 워밍업 과정이라서.

## 파이프라인

```
cn7_labeled.csv / cn7_unlabeled.csv
  │
  ├─ 전처리  scripts/run_preprocessing.py
  │    완전 중복 제거 (labeled 6,736 → 3,974행)
  │    계량없는 샷 제거 (unlabeled 35,239 → 17,087행)
  │    IsolationForest 자가정제 (contamination=0.01) → 16,917행
  │    컬럼별 표준화 (24개 피처)
  │    → data/processed/{train,eval}.csv, scaler.json, manifest.json
  │
  └─ 학습·평가  scripts/run_lstm_training.py
       롤링 z-score 드리프트 보정 (window=200)
       12샷 시퀀스 구성 → LSTM-AE (hidden 64, latent 16, 50 epochs)
       train 오차 분포의 95 percentile로 임계값 결정
       → data/model/{model.pt, threshold.json, evaluation_report.json}
```

**임계값은 라벨을 보고 정하지 않는다.** train(정상) 오차 분포의 95 percentile로
정한 뒤 eval에 적용만 한다. 라벨에 맞춰 임계값을 역산하면 지표는 좋아지지만
현장에서는 재현되지 않기 때문이다.

## 실행

```bash
cd 01-cn7-injection-molding
uv sync
uv run python scripts/run_preprocessing.py    # 전처리
nice -n 19 uv run python scripts/run_lstm_training.py   # 학습 + 평가
uv run pytest                                  # 테스트
```

> 작업 서버(da20-suresoft)는 공용이므로 학습처럼 무거운 작업은 `nice -n 19`로
> 실행한다.

## 이 트랙에서 겪은 문제

**공정 드리프트.** 초기 학습 결과, 정상 샷인데도 재구성 오차가 날짜에 따라
수백~수만 배씩 달라졌다. train(unlabeled, ~10/20)과 eval(labeled, 10/16~11/3)의
시간 범위가 거의 겹치지 않아 생긴 문제다. 각 피처를 절대값이 아니라 **최근 200샷
기준 롤링 z-score**로 변환해 완화했다.

윈도우 크기는 실측으로 정했다 — `window=100`은 불량 사건(최대 17연속 샷)이 자기
기준선을 오염시켜 성능이 뚜렷이 나빠졌고, 200과 400은 차이가 거의 없어 200을 썼다.

**계량 없는 샷.** 원본에 계량 공정이 실행되지 않은 샷이 절반 가까이(18,152건)
섞여 있었다. 이걸 그대로 학습하면 모델이 "정상"의 기준을 잘못 잡는다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `src/preprocessing/` | 정제 → 자가정제 → 스케일링 → 분할 |
| `src/lstm_ae/` | 드리프트 보정(`detrend.py`), 시퀀스 구성, 모델, 학습, 채점 |
| `scripts/` | 파이프라인 실행 진입점 2개 |
| `tests/` | 단위 테스트 |
| `docs/specs/`, `docs/plans/` | 설계 스펙 / 구현 계획 |
| `docs/CN7_이상탐지_모델링_결과_리포트.docx` | 결과 리포트 |
| `data/` | 원본·전처리·모델 산출물 (git 제외) |
