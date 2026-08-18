# 작업: LSTM+AE 이상탐지에 OOD(신뢰도) 게이트 추가

> **v2 — 실제 코드/데이터로 검증 후 확정.** 변경 이력은 맨 아래 "v1 대비 변경" 참고.
> 아직 구현 전 — 이 문서는 설계/검증 단계 결과물이다.

## 배경
- LSTM+AE 모델이 재구성오차(champion v5 기준 mean threshold=**0.8463**) 하나로만 정상/불량을 판정 중
- 문제 사례: eval exp22 (feedrate=20, worn tool)가 FP로 오분류됨 — champion(v5)으로 재검증 완료: score=1.8598 > threshold=0.8463, 실제 라벨은 good → FP 확인
  - 원인: "worn tool × feedrate=20" 조합이 train 8개 실험 중 단 한 번도 없었음 — `train.csv` 메타데이터로 직접 확인: train의 유일한 feedrate=20 샘플(exp2)은 unworn, exp22는 worn×20이라 train에 없는 조합
  - train에 feedrate=20인 샘플은 exp2 하나뿐인데 unworn(n=1) — 확인됨
  - **(추가 발견)** exp2 자신도 champion 모델에서 재구성오차가 이례적으로 높음(score=1.11, train 8개 중 최댓값, 나머지는 0.37~0.73) — 자기 자신이 학습 데이터인데도 threshold(0.846)를 넘어 "bad"로 자가-오판정됨. exp22와는 독립적인 두 번째 증거.
  - 재구성오차만으로는 "진짜 이상"과 "모델이 처음 보는 조건이라 서투른 것"을 구분 못함

## 목표
재구성오차와 별개로 "이 입력이 train 분포에서 얼마나 낯선가"를 재는
novelty score를 추가해서, 재구성오차-novelty score 2축으로 4분면 판정 로직 구현:
- 재구성오차 낮음 + novelty 낮음 → 정상
- 재구성오차 높음 + novelty 낮음 → 진짜 이상 (RAG 조치 추천 대상)
- novelty 높음 (재구성오차 무관) → 판단불확실/OOD (RAG 호출 안 함, "수동 확인 필요"만 표시)

## 구현 방식 (확정)

1. **41개 피처의 train 기준 평균/표준편차 — 새로 계산할 것 없음.**
   `data/processed/scaler.json`(`StandardScaler`, `src/preprocessing/scaling.py::fit_scaler`로
   train에서 적합)에 이미 있다. **주의**: `data/model/feature_baseline.json`은 이것과 다르다 —
   그건 train의 *재구성오차*(reconstruction error) 분포이지 *원본 입력값* 분포가 아니다.
   feature_baseline을 novelty score에 쓰면 재구성오차와 상관돼버려서 두 축의 독립성이 깨진다.
   `feature_baseline.json`은 기존 용도(`feature_contributions` 원인 랭킹)로만 계속 쓴다.

2. `serving/inference.py::scale_features()`가 이미 `(x - train_mean) / train_std`를 계산하므로,
   **스케일된 입력값 자체가 곧 41개 피처 각각의 z-score**다. 새 입력의 스케일된 윈도우에서
   `abs(scaled_value)`를 41피처 × window_size 전체에 대해 평균 또는 최댓값으로 집계하면
   그게 novelty score.
   (공분산 기반 Mahalanobis distance는 샘플 수(8개)가 적어 불안정할 수 있어 1차는 단순
   z-score 평균/최댓값으로 시작)

3. novelty_threshold는 train 8개 실험 자체의 novelty score 분포(최댓값 또는 95th percentile)로
   잠정 설정 — `src/lstm_ae/scoring.py::compute_thresholds()`와 동일한 percentile 패턴을 그대로
   따른다(기존 컨벤션 일치).

4. exp2(train)와 exp22(eval FP)의 novelty score를 비교해서 가설 검증(exp22가 유의미하게 높아야
   함). exp2 자체도 champion에서 이례적 고오차를 보이므로, exp2의 novelty score가 다른 train
   7개보다 높게 나오는지도 함께 검증(검증 계획 4번).

## 사전 조사 결과 (확인 완료)
- 41개 피처 목록: `src/preprocessing/columns.py::FEATURE_COLUMNS`
- train/eval 분할: `src/preprocessing/split.py` — `TRAIN_EXPERIMENT_IDS=[1,2,3,11,13,14,15,17]`, `EVAL_GOOD_EXPERIMENT_IDS=[12,18,22]`, `EVAL_BAD_EXPERIMENT_IDS=[4,5,6,7,8,9,10,16,20,21,23]`
- train 기준 41개 피처 원본값 mean/std: `data/processed/scaler.json` (novelty score의 유일한 재료, 이미 존재)
- train 기준 41개 피처 재구성오차 mean/std: `data/model/feature_baseline.json` (novelty score에는 부적합 — 기존 원인 랭킹 전용으로 유지)
- 기존 z-score 랭킹 코드: `src/serving/inference.py::rank_feature_contributions()` — 원인 랭킹용, novelty score와는 다른 통계를 쓰므로 재사용하지 않고 별도 함수로 구현
- threshold 산출 로직: `src/lstm_ae/scoring.py::compute_thresholds()` (percentile 기반) — novelty_threshold 설계 시 그대로 본뜬다
- 모델 추론/재구성오차 계산: `src/serving/inference.py::predict_experiment()`, `src/lstm_ae/pipeline.py::compute_window_errors()`, `src/lstm_ae/scoring.py::aggregate_window_errors_by_experiment()`
- FastAPI `/predict`: `src/serving/app.py` — multipart 파일 업로드 + `method`(mean/max/p95) 쿼리파라미터

## 검증 계획
1. train 8개 + eval 14개 전체에 대해 novelty score 계산
2. exp22가 4분면 중 "OOD" 칸에 들어가는지 확인 — 재구성오차 조건은 이미 확인됨(champion 기준 FP 맞음)
3. exp21(threshold를 근소하게 초과하는 경계 케이스)은 novelty score가 낮게 나오는지 확인
   - v1은 이 케이스를 "FN"으로 전제했으나, 근거였던 `data/model/evaluation_report.json`/`experiment_scores.csv`는 stale한 이전 학습 run(threshold=0.879)의 결과였음. champion(v5, threshold=0.8463)으로 재검증한 결과 exp21 score=0.8646로 이미 TP로 정확히 판정되고 있음. 다만 격차가 2%로 여전히 경계에 가깝고, exp21(feedrate=3, clamp=4.0, unworn)은 train의 exp11과 조건이 거의 동일 — "낯선 조건이 아니라 판정 경계 문제"라는 원래 검증 목적은 유효.
4. exp2(train)의 novelty score가 다른 train 7개보다 유의미하게 높은지 확인
   - champion 재검증 결과 exp2는 train 안에서도 재구성오차가 이례적으로 높음(score=1.11 vs 나머지 0.37~0.73, threshold 0.846을 넘어 자가-오판정). novelty score로도 재현되면 "feedrate=20 조합 자체가 모델에게 낯설다"는 가설이 exp22와 독립적으로 두 번 뒷받침된다.

## v1 대비 변경 사항 요약
| 항목 | v1 | v2 |
|---|---|---|
| threshold 수치 | 0.857 (출처 불명확) | champion v5 실측값 0.8463으로 교체 |
| exp21 성격 | FN | threshold 근접 경계 케이스(champion 기준 현재 TP) — 근거 파일이 stale했음을 확인 후 정정 |
| 1단계 재료 | "평균/표준편차 계산 (재사용 가능하면)" — 모호 | `data/processed/scaler.json`으로 확정, 새 계산 불필요. `feature_baseline.json`은 다른 통계(재구성오차)라 명시적으로 배제 |
| feedrate=20의 위치 | 명시 안 됨 | FEATURE_COLUMNS엔 없고 train.csv 메타데이터에만 존재 — novelty score는 시계열 패턴을 통한 간접 탐지임을 명시 |
| 검증 대상 | exp2 vs exp22 | exp2가 train 내부에서도 이상치라는 점을 별도 검증 항목(4번)으로 추가 |
| "확인해야 할 것" | 미확인 TODO | 전부 실제 파일 경로로 확인 완료 |
