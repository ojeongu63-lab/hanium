# 드리프트 트리거 기반 자동 재학습 (2026-08-19)

## 목적

멘토 제안: "드리프트가 감지되면 자동으로 재학습되는 것까지 보여달라."

해결하려는 문제는 **제품은 정상인데 공정 파라미터가 변해서 모델이 불량으로
오판하는 상황**이다. 계절·온도 변화로 입력 분포가 이동하면 모델이 학습한
"정상의 정의"가 낡아 오탐이 늘고, 이때 필요한 건 설비 정비가 아니라 재학습이다.

단, 같은 드리프트 신호가 정반대 원인(설비가 실제로 망가져 불량품이 나옴)에서도
나온다. 지표만으로는 구분할 수 없으므로 구조는 "트리거 → 재학습"이 아니라
**"트리거 → 재학습 → 게이트 → 승격 또는 거부"** 가 된다.

- 스펙: `02-cnc-machining/docs/specs/2026-08-19-cnc-drift-triggered-retraining-design.md`
- 구현 계획(코드 포함): `02-cnc-machining/docs/plans/2026-08-19-cnc-drift-triggered-retraining.md`

## 결정 사항

- **데이터**: 생성 모델(TimeGAN)이나 물리 시뮬레이션이 아니라, train 실험 8개에
  날짜 비례 변형을 주입해 가상 운영 타임라인을 만든다. 25개 실험으로는 생성
  모델을 학습시킬 수 없고, 물리 모델은 별도 프로젝트 규모다.
- **시나리오 2종**: 온도·계절(승격 경로) / 공구마모(거부 경로). 두 경로가 다
  있어야 게이트가 장식이 아님을 보일 수 있다.
- **게이트 2조건 AND**: G1 원본 eval에서 놓친 불량이 champion보다 1건까지만
  많을 것, G2 라벨 도착 구간(최근 20건) 정확도가 champion보다 개선.
  G1을 소수 recall이 아니라 개수로 표현하는 이유는 eval 불량이 11개뿐이라
  recall 눈금이 실험 1개당 0.0909로 끊기기 때문이다(아래 리뷰 참조).
- **선행 결정 번복**: `2026-08-12-cnc-drift-monitoring-design.md`가 자동 재학습을
  비목표로 뒀으나 뒤집는다. 근거는 게이트를 함께 만들기 때문이다.
- **기존 결함 2건 동시 수정**: 서빙이 champion을 시작 시 한 번만 로드하는 문제,
  scaler/baseline이 모델과 함께 버전 관리되지 않는 문제. 수동 승격에서는 안
  드러나지만 자동 루프에서는 즉시 터진다.
- `src/lstm_ae/`, `src/preprocessing/`, `src/monitoring/drift.py`, `logging.py`는
  **수정하지 않는다.** 호출·재사용만 한다.

## 작업

- [x] 1. QC 라벨 저장소 `src/monitoring/labels.py`
      → 검증: 지연 도착 조회 왕복 테스트 3개, 전체 103개 통과
- [x] 2. 재학습 트리거 `src/retraining/trigger.py` (연속 3회 + 쿨다운)
      → 검증: 순수 함수 테스트 8개, 전체 111개 통과
- [x] 3. 승격 게이트 `src/retraining/gate.py` (G1/G2)
      → 검증: 경계값(1건 초과 통과·2건 초과 거부) 포함, 전체 117개 통과
- [x] 4. 서빙 계약 검증 + 백업/교체/롤백 `src/retraining/promotion.py`
      → 검증: 실패 주입 롤백 테스트 포함 7개, 전체 124개 통과
- [x] 5. `POST /reload-model` 추가 (결함 ①)
      → 검증: 실패 시 기존 상태 유지 테스트, 전체 126개 통과
- [x] 6. 동반 아티팩트 버전 결합 + 폴백 (결함 ②)
      → 검증: 실제 champion이 폴백 경로로 로드되는지, 전체 128개 통과
- [x] 7. 재학습 데이터 구성 `src/retraining/runner.py`
      → 검증: 라벨 필터·eval 재스케일링 테스트 6개, 전체 134개 통과
- [x] 8. 재학습 실행 + MLflow 서빙 계약 충족
      → 검증: `window_size` param 존재 테스트, 전체 135개 통과
- [x] 9. 타임라인 스트림 생성기 `monitoring/simulate_timeline.py`
      → 검증: perturbation이 `SetPosition`을 건드리지 않는지 직접 확인
- [x] 10. 변형 상수 스윕으로 확정 (자리값 0.0 교체)
      → 검증: Day 40 도달 비율이 목표 대역(A 1.5~2.0, B 3.0) 안인지
- [x] 11. 감시 워커 `monitoring/drift_worker.py`
      → 검증: import 성공, 전체 135개 통과
- [x] 12. 시나리오 A 실행 — 승격 경로
      → 검증: `action=promoted`, 승격 후 오탐 감소
- [x] 13. 시나리오 B 실행 — 거부 경로 + 결함 ② 수정 확인
      → 검증: `action=rejected` + 거부 후 정본 파일 해시 불변

## 주의

- 공유 서버다. 재학습·스윕·타임라인 실행은 전부 `nice -n 19`를 붙이고,
  시작 전 `who` / `top`으로 부하를 확인한다.
- Task 10 전까지 `simulate_timeline.py`의 `POS_DRIFT`/`CUR_DRIFT`/`WEAR_RATE`는
  자리값 `0.0`이다. Task 10 Step 4의 assert가 교체 누락을 잡는다.
- Task 12·13에서 기대와 다른 결과가 나오면 **값을 조정해 통과시키지 않는다.**
  관측값을 그대로 보고하고 논의한다. 조정하게 되면 그 사실과 근거를 스펙의
  "남은 리스크" 절에 남긴다.

## 리뷰 — 시나리오 A (온도·계절) — **승격으로 종료**

첫 드리프트 감지 Day 17, 트리거는 네 번 발동했고 **앞의 세 번은 거부, 네 번째에 승격**.

| 트리거 | G1 놓친 개수 (허용 2건) | G2 재학습 vs champion | 판정 |
|---|---|---|---|
| Day 22 | 0건 놓침 ✓ | 0.65 vs 0.90 | 거부 |
| Day 27 | 0건 놓침 ✓ | 0.65 vs 0.90 | 거부 |
| Day 32 | 0건 놓침 ✓ | 0.45 vs 0.60 | 거부 |
| **Day 37** | 0건 놓침 ✓ | **0.45 vs 0.10** | **승격** |

승격 후 오탐 감소 확인: ratio 1.76(Day 37) → 1.12(Day 39) → 1.09(Day 40).
Day 38의 2.38은 드리프트 윈도우(최근 10건)에 승격 전 요청이 섞여 있어서다.

### 계획과 달랐던 점 — 세 번 거부는 버그가 아니었다

계획 때는 "트리거 발동 → 바로 승격"을 예상했으나 실제로는 세 번 거부됐다.
원인은 **라벨 지연 7일**이다. Day 22 시점에 도착한 라벨은 Day 15까지인데, 그
구간은 드리프트가 약해 champion이 아직 잘 하고 있었다(정확도 0.90). 즉
**champion을 바꿔야 한다는 증거가 라벨에 아직 없었다.**

Day 32에 champion 정확도가 0.90 → 0.60으로 떨어지며 드리프트가 라벨 구간까지
도달했고, Day 37에는 0.10까지 무너져 재학습 모델(0.45)이 역전하면서 비로소
승격됐다.

**G2가 설계 의도대로 작동한 증거다.** 드리프트 지표는 Day 17부터 울렸지만
모델을 교체할 근거는 Day 37에야 생겼다 — 자동화는 신호가 울리자마자 조치하는
게 아니라 근거가 쌓일 때까지 기다릴 줄 알아야 한다는 것을 보여준다.

### 구현 중 고친 것

1. `TestClient`를 `with` 블록 없이 써서 `/predict`가 503. lifespan이 안 돌아
   champion이 로드되지 않았다. `simulate_drift.py`가 이미 쓰던 관례를 놓쳤다.
2. G2가 G1과 같은 지표가 될 뻔했다(재학습 정확도를 recall로 대충 채움).
   두 모델을 같은 배치에 직접 돌려 비교하도록 고쳤다. 이때 `/predict`를 다시
   부르면 요청 로그에 게이트 평가용 가짜 트래픽이 쌓여 드리프트 윈도우가
   오염되므로 HTTP 없이 모델을 직접 로드해 추론한다.
3. `predict_experiment` 시그니처가 계획과 달랐다(위치 인자 순서, `threshold` 단수형).

## 리뷰 — 시나리오 B (공구마모) — **다섯 번 모두 거부**

| 트리거 | G1 놓친 개수 (허용 2건) | G2 재학습 vs champion | 판정 |
|---|---|---|---|
| Day 19 | 0건 놓침 ✓ | 0.90 vs 0.90 | 거부 |
| Day 24 | 0건 놓침 ✓ | 0.60 vs 0.85 | 거부 |
| Day 29 | 0건 놓침 ✓ | 0.60 vs 0.60 | 거부 |
| Day 34 | 0건 놓침 ✓ | 0.55 vs 1.00 | 거부 |
| Day 39 | 0건 놓침 ✓ | **0.40 vs 1.00** | 거부 |

champion은 v1 그대로 유지. 재학습 모델은 갈수록 나빠졌다(0.90 → 0.40).

### 결함 ② 수정 검증 — 통과

거부 5회 후 정본 파일 해시가 실행 전과 완전히 동일:
`9ab55583`(scaler) / `8841fd72`(model.pt) / `6d7d3978`(feature_baseline).
재학습이 다섯 번 돌았는데도 champion의 동반 파일을 한 번도 건드리지 않았다.

### 계획과 달랐던 점 — G1이 아니라 G2가 막았다

설계 시 "열화 데이터를 정상으로 학습 → 원본 eval의 불량을 놓침 → G1(recall
회귀)이 거부"를 핵심 논리로 잡았으나 **G1은 한 번도 작동하지 않았다.**
재학습 모델의 원본 eval recall은 매번 1.000(tp=11, fn=0)이었다.

원인은 scaler 재학습이다. 재학습은 새 좌표계를 쓰는데 원본 eval셋을 그리로
옮기면 전부 낯선 영역에 떨어져 거의 모두 불량으로 판정되고, recall이 자동으로
1.0이 된다. "마모를 정상으로 받아들이게 된 효과"가 좌표계 이동 효과에 묻힌다.

**예측한 둔감화 자체는 실재했고 G2에서 드러났다.** Day 39 게이트 표본
(Day 29~32, 전부 실제 불량 20건)에서 champion은 20건 전부를 잡아 정확도 1.00,
재학습 모델은 8건만 잡아 0.40 — 진짜 불량의 60%를 정상이라 판정했다.

이는 스펙 Part D에 적어둔 "G1만으로는 부족하다"는 경고가 그대로 실현된
사례다. **G1을 단독 방어선으로 뒀다면 이 시나리오는 승격됐을 것이다.**
스펙의 해당 서술은 실측값으로 정정했다.

## 실행 중 추가로 발견한 것 — 결함 ②의 남은 구멍

시나리오 A로 v5가 승격된 뒤 `promote_model.py 1`로 alias만 되돌렸더니,
**디스크의 동반 파일은 v5 것이 그대로 남아** champion v1이 threshold는 자기
것(0.8566)을 쓰면서 scaler는 v5 것을 읽는 상태가 됐다. 에러는 나지 않았다.

Task 6의 수정(MLflow 아티팩트 우선 + 디스크 폴백)은 아티팩트를 가진 모델에만
작동한다. v1은 최초 학습본이라 companion 아티팩트가 없어 폴백을 탔고, 그
폴백이 가리키는 디스크에 v5 파일이 있었다.

조치 2가지:
1. `swap_with_rollback`이 만든 백업(`data/model_backup/20260819_120005_375820/`)
   에서 복원. 해시가 원본과 정확히 일치했다 — **백업/복원 메커니즘이 실제로
   작동함이 부수적으로 검증됐다.**
2. champion v1 run에 companion 아티팩트를 소급 등록. 이제 v1도 자기 scaler를
   MLflow에서 읽어 디스크 상태와 무관해졌다.

이 문제는 **수동 승격 경로에도 원래 있던 것**이며, 자동 루프를 만들지
않았으면 계속 드러나지 않았을 결함이다.

## 후속 — 게이트 표본·기준 개선 시도 (2026-08-19)

"eval 표본이 14개뿐이라 평가값이 크게 흔들리는 것 아니냐"는 지적에서 출발해
두 가지를 시도했다.

### ① G1을 놓친 개수 기준으로 — 채택

```
이전:  retrained_recall >= champion_recall - 0.10
이후:  retrained_missed <= champion_missed + 1
```

판정은 동일하고 가짜 정밀도만 사라진다. eval 불량이 11개뿐이라 recall 눈금이
실험 1개당 0.0909로 끊기는데, 소수점 넷째 자리 임계값은 그 사실을 가린다.
두 시나리오 재실행에서 판정이 바뀌지 않음을 확인했다.

**표본 부족 자체는 고칠 수 없다.** eval을 늘리려면 실험이 더 있어야 하는데
공개 데이터셋 25개가 전부고 3개는 중복이라 이미 뺐다. 합성으로 eval을 채우면
게이트가 자기가 만든 답을 채점하게 되므로 하면 안 된다.

### ② G2 표본 20 → 60 — 시도했다가 되돌림

라벨 도착분이 160건 쌓이는데 20건만 쓰고 있어 넓혀봤더니
**시나리오 A가 승격에서 거부로 뒤집혔다.**

| Day 37 게이트 | 표본 20 | 표본 60 |
|---|---|---|
| 재학습 모델 | 0.45 | 0.48 |
| champion | 0.10 | 0.50 |
| 판정 | **승격** | **거부** |

Day 37 시점 라벨은 Day 30까지 도착한다. 표본 20은 Day 26~30(champion이
오탐하던 구간)을 보지만 표본 60은 Day 19~30이라 champion이 멀쩡했던 앞
구간이 섞인다.

**드리프트 상황에서 평가 창 확대는 중립적이지 않고 구모델에 유리하다.**
재학습 모델은 최근 환경에 맞춰 학습되므로 최근 구간에서 강한데, 창을 과거로
넓히면 그 강점이 희석되는 반면 champion은 자기가 잘하던 옛 구간 점수를
벌어들인다. G2의 질문은 "지금 현장에서 더 나은가"인데 12일치 평균은 그
"지금"이 아니다.

20으로 되돌렸다. 30~40으로 절충해 승격이 살아나는지 볼 수도 있었으나 하지
않았다 — 원하는 결과가 나올 때까지 손잡이를 돌리는 행위이고, 이 프로젝트는
`feedrate=20` 오탐에서 이미 같은 선을 그었다.
**평가 창 크기가 승격 판정을 뒤집을 만큼 민감하다는 것이 알려진 한계다.**

### 보고 착오 정정

1차 실행 로그는 승격 시 `G2 delta=+0.3500`만 찍고 절대값을 출력하지 않았는데,
당시 사용자에게 "0.95 vs 0.60"이라고 보고했다. 이는 로그에서 읽은 값이 아니라
직전 줄에서 유추한 추측이었고 틀렸다. 실제 값은 `0.45 vs 0.10`(delta 동일)이다.
이번 변경으로 절대값을 출력하게 되면서 드러났다.

## 남은 작업

- [x] `drift_worker.py`에 독립 실행 진입점(`main`) — 완료 (2026-08-24)
- [x] `README.md`에 자동 재학습 루프 실행법 추가 — 완료 (2026-08-24, §2-7)
- [x] `main` 브랜치 머지 — 완료 (브랜치는 `main` 하나만 남음, 2026-09-02 확인)

## 리뷰 — 독립 실행 진입점 (2026-08-24)

기존엔 `simulate_timeline.py` 하나가 `TestClient`로 배치 주입과 감시(`tick()`)를
같은 프로세스에서 다 했다. 선택한 구조("별도 감시 워커 프로세스")를 실제로
만족하려면 배치를 실제 서버에 흘리는 쪽(feeder)과 감시하는 쪽(worker)이 각각
독립된 OS 프로세스여야 한다 — 완전 분리로 진행(사용자 선택).

**막힌 문제**: 워커가 실제 프로세스로 분리되면 "오늘이 며칠째인지"를 어디서
아는가? feeder의 내부 카운터에 접근할 수 없다. 해결: `labels.db`(이미 두
프로세스가 공유하는 SQLite)에 `get_latest_produced_day()`를 신설해, 워커가
매 폴링마다 "feeder가 지금까지 기록한 최신 produced_day"를 물어보게 했다.
두 프로세스가 서로 다른 날짜를 셀 위험이 없다.

**변경 파일**:
- `src/monitoring/labels.py` — `get_latest_produced_day(db_path) -> int` 추가
  (TDD, 테스트 2개: 최댓값 추적 / DB 없을 때 0)
- `monitoring/simulate_timeline.py` — `--serve-url` 옵션 추가. 지정하면
  `TestClient` 대신 진짜 HTTP(`httpx2.Client`)로 배치만 흘리고 `tick()`은
  호출하지 않는다(별도 프로세스가 감시하므로). 생략 시 기존 동작 그대로라
  기록된 시나리오 A/B 재현 커맨드는 안 바뀐다.
- `monitoring/drift_worker.py` — `main()` 추가. 실제 서버를 `httpx2.Client`로
  폴링하며, `get_latest_produced_day()`가 마지막으로 처리한 날보다 커지면
  `tick()`을 호출한다.

**실행법** (세 터미널):
```bash
uv run uvicorn src.serving.app:app --app-dir . --port 8000
uv run python monitoring/simulate_timeline.py temperature --serve-url http://127.0.0.1:8000
uv run python monitoring/drift_worker.py temperature --base-url http://127.0.0.1:8000 --poll-interval 5
```

**실행 중 발견한 버그 2건 (구현 중 수정)**:
1. `main()`을 정의만 하고 `if __name__ == "__main__": main()` 가드를 빠뜨려
   스크립트로 실행해도 아무 일도 안 일어났다. 스모크 테스트에서 발견.
2. `import drift_worker`가 mlflow/torch 로딩 때문에 약 7초 걸린다는 걸 모르고
   첫 스모크에서 `timeout 5`를 줘서 매번 죽었다 — "워커가 멈췄다"로 오인할
   뻔했다. 타임아웃을 넉넉히 주니 정상 작동.

**스모크 테스트로 확인한 설계상 제약 (버그 아님, 운영 규칙)**: `labels.db`는
`qc_labels` 테이블에 시나리오 구분 컬럼이 없다. 이전 시나리오 실행분이 DB에
남은 채로 새 시나리오를 얹으면 `get_latest_produced_day()`가 옛 시나리오의
최댓값(예: 40)을 그대로 돌려줘 워커가 첫 폴링부터 엉뚱한 날짜로 튄다. 실제로
스모크 중 재현했다. **시나리오를 바꿔 실행하기 전엔 반드시 `labels.db`를
비울 것** — 기존에도 있던 제약이고 이번에 새로 생긴 게 아니다.

**검증**: uvicorn(포트 8010, `nice -n 19`)을 띄우고 feeder 3일 → 워커가 실제
HTTP로 폴링해 Day 01~03을 순서대로 감지, 매번 `flagged=False action=none`
(Day 1~9는 `DRIFT_START_DAY=10` 이전이라 변형이 없어 트리거가 절대 안 걸리는
안전한 구간 — champion과 MLflow run 수는 스모크 전후 불변 확인). 스모크가
남긴 `labels.db`/`requests.db`/`data/timeline/temperature`는 백업에서 복원 및
삭제로 원상복구. 전체 테스트 141개 통과(신규 2개 포함).

부수 발견 → **2026-08-24에 수정함**: `tests/serving/test_app.py`의 예측
테스트 일부가 `DB_PATH`를 monkeypatch하지 않아 `uv run pytest`를 돌릴
때마다 실제 `data/monitoring/requests.db`에 요청 2건이 실기록됐다.
여러 번 발견해 직접 지워 복구하다가, 사용자 요청으로 근본 수정.
오염 원인은 `test_predict_returns_prediction_for_valid_csv`와
`test_predict_response_includes_guide_field` 두 개뿐이었다(둘 다
`/predict` 성공 경로 — `predict_experiment`가 성공해야 `log_request`에
도달하므로, 400 에러 테스트 3개는 애초에 그 줄에 안 닿아 오염과 무관함을
개별 실행으로 확인). 두 테스트에 `monkeypatch.setattr(app_module,
"DB_PATH", tmp_path / "requests.db")` 추가, 전체 스위트 두 번 연속
돌려도 `requests.db` count가 202로 안 바뀜을 확인. 커밋 `3f0c99c`.

## 리뷰 — 사용자가 직접 실행해 잡은 버그 (2026-08-24)

내가 한 스모크(짧은 구간, 조건 맞춰 준비)는 통과했지만, 사용자가 직접 세
터미널로 실행하자 준비 안 된 실제 조건에서 두 가지가 드러났다.

1. **진짜 버그 — 폴링 사이 날짜가 통째로 스킵됨.** `main()`이
   `latest_day > last_day`일 때 `last_day = latest_day`로 한 번에
   점프하는 방식이었다. feeder가 폴링 주기(2초)보다 빠르게 5일치를
   쏟아내자 실측 로그에 `Day 02` 다음 `Day 04`가 찍히고 `Day 03`이
   사라졌다. `flag_history`에서 날짜가 통째로 빠지면 트리거의 "연속
   3회" 판정 의미가 깨진다. `days_to_process(last_day, latest_day)`
   순수 함수(TDD, 테스트 3개)로 고쳐 폴링 사이 날짜를 전부 순회하게
   했다. 커밋 `39734a3`.
2. **내 안내 실수 — `requests.db`도 시나리오 전환 시 비웠어야 했다.**
   드리프트 판정은 `requests.db`의 최근 10개 요청 평균을 본다
   (`DRIFT_WINDOW_SIZE=10`). 지난 시나리오 B 40일 실행의 낡은 요청이
   남아있는 채로 새로 5개(Day 01)를 보내니 최근 10개 창에 낡은 것 5개가
   섞여 `score/threshold=1.57, flagged=True`가 튀었다. `labels.db`만
   비우라고 안내하고 `requests.db`는 빠뜨렸던 것 — 코드 결함이 아니라
   내 안내 실수다.

정리 후 재검증: `requests.db`/`labels.db`/`data/timeline/temperature`를
전부 백업(`*.bak-20260819-scenarioB`) 후 비우고 다시 실행 → Day 01~05
빠짐없이 순서대로, `flagged=False` 일관. Day 01·02는 요청이 10개 안 차
`sufficient_data=false`라 `ratio=0.00`(정상, 버그 아님), Day 03부터
0.65~0.68로 무변형 기준(0.72대)과 일치.

**교훈**: 내가 준비한 스모크가 통과해도 실제 사용자 실행에서 타이밍이나
누락된 전제조건 때문에 다른 결함이 드러날 수 있다. "내가 검증했다"와
"사용자가 직접 돌려봤다"는 다른 검증이다.

## 리뷰 — 완전 분리 구조로 40일 전체 재현 (2026-08-24)

버그 수정 후 세 프로세스(서버/워커/feeder, `--days 40`)로 시나리오 A를
다시 돌려 트리거→재학습→게이트→승격까지 실제로 확인했다.

**결과 — 8/19 시나리오 A 실측과 사실상 동일하게 재현됨:**

| 트리거 | G2 판정 | 결과 |
|---|---|---|
| Day 22 | 0.65 ≤ 0.90 | 거부 |
| Day 27 | 0.65 ≤ 0.90 | 거부 |
| Day 32 | 0.45 ≤ 0.60 | 거부 |
| Day 37 | (통과) | **승격 (version 27)** |

게이트가 실제 승격/거부를 가른 근거(G2 수치)까지 8/19 기록과 거의
일치한다 — 로직 자체는 견고하다는 뜻.

**새로 발견한 현상 — "Day N" 로그가 그 시점을 대표하지 않을 수 있다.**
Day 23~26·28~31·33~36이 각각 정확히 `1.97`로 고정되어 나왔다(원인 조사
전엔 이상해 보였다). 원인: `/drift-status`는 항상 `requests.db`의
"최근 10개 요청"을 본다. feeder는 워커의 재학습(50 에포크, 수십 초)을
기다리지 않고 이미 Day 40까지 다 쏴버리므로, 워커가 뒤늦게 "Day 23을
처리한다"고 tick 해도 그 순간의 "최근 10개"는 실제로 feeder가 이미
도달한 최신 시점의 요청이다 — Day 23 시점의 스냅샷이 아니다. Day 38의
`3.52`도 같은 이유(승격 직후 threshold만 바뀐 채 여전히 최신 시점 값).

**버그로 보지 않기로 함.** 실제 프로덕션엔 "가상의 day" 개념이 없다 —
그냥 실시간이고, 재학습 중에도 `/drift-status`가 최신 실시간 상태를
보여주는 게 오히려 맞는 동작이다. 문제는 이 시뮬레이션이 로그에 "Day
N"이라는 라벨을 붙여서 스냅샷인 것처럼 보이게 한다는 점뿐이다. **게이트
판정(승격/거부)은 이 현상과 무관하다** — `_gate_accuracies`가
`/drift-status`가 아니라 champion·재학습 모델을 직접 로드해 추론하기
때문에 오염되지 않는다. 실제 영향은 트리거 판정(`flagged`)에 국한되고,
이번엔 값이 임계값을 훨씬 넘긴 상태라 결론이 우연히 안 바뀌었다 —
경계값 근처였다면 트리거 타이밍이 왜곡될 수 있었다는 게 알려진 한계다.
사용자 확인 후 문서화만 하고 코드는 건드리지 않기로 함(코드 수정 대신
"완전 분리"를 택한 것 자체의 트레이드오프로 남겨둠).

**뒷정리**: champion을 `promote_model.py 1` + `restore_backup()`(파일
동반 복원, alias만 바꾸는 결함②를 반복하지 않도록)으로 v1으로 되돌림 —
threshold `0.8566220998764038`, scaler/baseline 해시(`edee2506`/
`85c9aacf`) 전부 원래와 일치 확인. `/reload-model`로 서버도 v1 반영.
`labels.db`/`requests.db`도 시나리오 B 정본(200건/202건)으로 복원.

**부수 발견**: "시나리오 B 정본"이라고 이름 붙였던 백업(`requests.db.
bak-20260819-scenarioB`)이 실제로는 8/19 순수 정본이 아니라 그 사이
확인 작업(Day 1~5) 잔여물 27건이 섞인 상태였다 — 되돌리기 작업 자체를
검증 없이 신뢰하면 오염이 조용히 누적된다. `id > 202`인 행을 지워
202건으로 다시 맞췄다(바이트 해시는 sqlite 내부 구조상 달라지지만
count·내용은 8/19 정본과 동일 확인).

## 섀도우 배포 (2026-08-25)

멘토 요구가 아니라 사용자가 "MLOps 포트폴리오로 발전시키고 싶다"는 목적에서
시작. 브레인스토밍 → 스펙(`02-cnc-machining/docs/specs/2026-08-25-cnc-shadow-deployment-design.md`)
→ 계획(`02-cnc-machining/docs/plans/2026-08-25-cnc-shadow-deployment.md`)
→ 인라인 구현 순서로 진행.

**만든 것**: 게이트(G1+G2, 트리거 시점) 통과 후 즉시 100% 승격하던 걸,
candidate를 실트래픽에 병행 투입(응답엔 영향 없음)해 라벨 20건 도착까지
관찰한 뒤 champion vs candidate 정확도를 다시 비교해 최종 승격/폐기를
정하는 섀도우 단계로 바꿨다. 신규: `src/monitoring/shadow_log.py`.
수정: `src/serving/app.py`(`/start-shadow`, `/stop-shadow`, `/predict`
병행 추론), `src/retraining/gate.py`(`evaluate_shadow`),
`monitoring/drift_worker.py`(섀도우 상태·감시·승격), `monitoring/
simulate_timeline.py`(`--pace-seconds` 추가).

**실측 검증**: 시나리오 A(온도)를 세 번 재현하며 스펙에 없던 버그 3개를
찾아 고쳤다(라벨 오프셋을 개수가 아니라 생산일 기준으로, feeder에 페이스
조절 추가, 섀도우 시작 기준일을 논리적 트리거일이 아니라 실제 서버 반영
시점으로 재조회). 최종적으로 `--days 70`, `--pace-seconds 15`로 섀도우가
실제로 끝까지(시작→실시간 관찰→라벨 매칭→승격) 작동하는 걸 확인 —
champion v1 → 승격 v39.

**추가로 발견해 고친 것**: 승격 직후 쿨다운 없이 즉시 재트리거되는 문제
(섀도우 기간이 쿨다운보다 훨씬 길어서) — 섀도우 종료 시에도 쿨다운을
재설정하도록 수정.

**알려진 한계로 남긴 것**: `--days`가 `TOTAL_DAYS`(40)를 넘으면 온도
변형이 설계 상한(Day 40 기준 1.0)의 거의 2배까지 커진다. v39 승격 시
champion이 20건 중 0건을 맞힌 게 이 때문 — 섀도우 로직의 결함이 아니라
시뮬레이션 파라미터 문제이며, 섀도우 검증 기간을 확보하려면(`--days`를
`TOTAL_DAYS`보다 크게 잡아야 함) 피할 수 없는 트레이드오프다. 코드는
안 건드리고 스펙에 정정 기록만 남겼다.

**뒷정리**: champion을 `promote_model.py 1` + `restore_backup()`으로 v1
복원(threshold `0.8566220998764038`, scaler/baseline 해시 `edee2506`/
`85c9aacf` 전부 원래와 일치 확인). `labels.db`/`requests.db`도 정본
백업에서 복원. 전체 테스트 153개(신규 12개 포함) 통과.

## 리뷰 — 재학습 거부 원인 추정 라이브 검증 (2026-09-02)

08-26에 코드·단위 테스트까지 커밋됐지만 계획서 Task 6(라이브 재현)은 안 돈
상태였다 — MLflow에 `fixture_loosening` run 0건, `estimated_cause` 태그 0건,
스펙에 실행 결과 절 없음. 세 프로세스(서버 / feeder `--pace-seconds 2` /
워커 `--env-file .env`)로 40일을 돌려 확인했다. 원본 로그·DB는
`02-cnc-machining/data/monitoring/_fixture_loosening_20260902/`,
`_tool_wear_20260902/`.

**fixture_loosening 결과**: 거부 3회(Day 19·24·29) 모두
`estimated_cause=vibration_backlash`, 권장 조치는 매번 "고정구 점검"이 첫
항목, MLflow 태그(`estimated_cause`, `recommended_action`) 정상. champion v1
유지(scaler/baseline/model.pt md5 `9ab55583`/`6d7d3978`/`8841fd72` 불변,
v1 companion 아티팩트와 바이트 단위로 동일). **해시 착오 정리**: 08-24
기록에서 "출처 불분명"이라 했던 md5 값과 "정본"이라 한 `edee2506`/
`85c9aacf`는 **같은 두 파일의 md5와 sha256**이다(직접 계산해 확인). 두 기록
모두 같은 정본을 가리키고 있었고 파일이 바뀐 적은 없다.

**계획과 달랐던 점**: 4번째(Day 34)가 게이트를 통과해 섀도우로 갔다(v44,
G2 0.80 vs 0.60). 후보를 champion과 같은 배치에 다시 돌려보니 임계값이
낮아(0.62 vs 0.86) 모든 배치를 20%쯤 높게 보는 "더 자주 불량이라 하는"
모델이었고, Day 34의 G2 창이 전부 불량 라벨(생산일 24~27)이라 그쪽이
이겼다. 같은 후보를 전부 정상인 창(Day 14~17)에 대면 0.55 vs 0.85로
진다. **G2는 창에 한 클래스만 있으면 과탐을 개선으로 읽는다** — 원인 추정
결함이 아니라 게이트 설계의 사각지대. 섀도우도 Day 41 이후가 전부 불량이라
같은 사각지대(이번 실행은 feeder 완주 후라 `shadow_pending`으로 끝남).
스펙 정정 절에 배치별 표와 함께 기록했고 코드는 안 건드렸다. 고치려면
"G2 창에 양쪽 클래스 필수" 또는 "불량 창 개선 AND 정상 창 무회귀"로
나누는 별도 스펙이 필요하다.

**tool_wear 결과(계획서에 없던 반대쪽 검증)**: 5회(Day 20·25·30·35·40)
모두 거부, `estimated_cause=tool_wear` 5/5, 권장 조치는 절삭 속도 저하·
이송량 증가·내마모성 등급·런아웃 점검 등 공구마모 문서 내용. 두 시나리오
합쳐 거부 8건 중 8건 정답, 반대쪽으로 넘어간 적 없음. 전부 불량인 창(Day
35·40)에서 재학습 모델이 20건 중 7~8건만 잡아(G2 0.35·0.40 vs 1.00)
fixture_loosening Day 34와 정반대 — 곱셈 램프는 정상으로 학습돼 덜 잡고,
가산 노이즈는 더 잡는다. champion v1 유지. 오늘 생성된 재학습 run은 v41~v49.

**후속 — 두 방향 게이트 스펙 (2026-09-02)**: 위 G2 사각지대를 닫는 설계를
브레인스토밍(사용자 결정: 정상 라벨 없는 창은 거부+원인 추정, 규칙은
클래스별 회귀 금지) 후 `02-cnc-machining/docs/specs/2026-09-02-cnc-two-sided-gate-design.md`
로 작성. 구현 전에 사용자 요청으로 우리 데이터에 맞는지 버릴 스크립트로
확인했다 — ① 오늘 후보 9개에 새 규칙을 사후 적용하니 fixture Day 34만
"정상 라벨 없음" 거부로 뒤집히고 나머지 8개는 기존과 같은 거부(맞바꾸기
양방향 실재 확인). ② 홀드아웃 임계값 보정을 세 지점에서 실제 재학습해
보니 과민(정상 창 오탐 10~11 → 3)은 사라지지만 보정 p95가 feedrate=20
실험 2 배치 위에 떨어져 후보가 둔해짐(원본 eval 놓침 0 → 2, fixture
불량 창 놓침 6 → 14). **결정: 게이트(Part A·B)만 구현, 보정(Part C)은
보류.** 상세 표는 스펙의 "구현 전 데이터 확인" 절.

## 두 방향 게이트 구현 (2026-09-02)

스펙 `02-cnc-machining/docs/specs/2026-09-02-cnc-two-sided-gate-design.md`
(Part A·B), 계획 `02-cnc-machining/docs/plans/2026-09-02-cnc-two-sided-gate.md`.

- [x] Task 1 `evaluate_two_sided` — 오탐·놓침을 따로 세는 순수 함수 (TDD 8개)
- [x] Task 2 `evaluate_gate`가 두 방향 결과를 받도록, `evaluate_shadow`·`accuracy_from_pairs` 제거
- [x] Task 3 워커 통합 — `_gate_predictions`, 로그·MLflow 태그 교체, STRUCTURE/README 문구
- [x] Task 4 라이브 재현 3종(fixture 40일, tool_wear 40일, temperature 70일) → 스펙 정정 절, champion v1 복원

### 리뷰 — 두 방향 게이트 라이브 재현 (2026-09-02)

| 시나리오 | 트리거 | 판정 | 원인 추정 | 기존 규칙과의 차이 |
|---|---|---|---|---|
| fixture_loosening 40일 | 5회 | 5회 모두 거부 | 5/5 vibration_backlash | Day 30(맞바꾸기)·Day 35(전부 불량 창)는 기존 규칙이면 통과였음 |
| tool_wear 40일 | 5회 | 5회 모두 거부 | 5/5 tool_wear | 판정 동일, 사유가 정직해짐 |
| temperature 70일 | 4회 + 승격 후 1회 | 거부 3회 → Day 37 통과 → 섀도우 → Day 64 승격(v63) | (범위 밖, tool_wear로 찍힘) | 08-25와 같은 흐름 |

거부 사유 네 종류(개선 없음 / 오탐 회귀 / 놓침 회귀 / 정상 라벨 없음)가 전부
실제로 나왔고, 거부 10건 모두 원인 추정·RAG 조치·새 태그 4종이 MLflow에
남았다. 승격 후 champion을 `promote_model.py 1` + `restore_backup()`으로 v1
복원(해시 원래대로). 상세 표는 스펙의 "실행 결과에 따른 정정" 절. 로그·DB는
`data/monitoring/_<시나리오>_20260902_v2/`(temperature는 `_temperature_20260902/`).

**사용자 결정(2026-09-02)**: 이 상태를 결과물로 고정한다. 추가 기능 작업은
하지 않는다.

**실행 중 배운 것**: `pkill -f 'drift_worker.py <시나리오>'`처럼 패턴에
스크립트 인자를 넣으면 그 명령을 담은 셸 자신도 매칭돼 먼저 죽는다 —
PID 파일(`kill $(cat worker.pid)`)이나 `pgrep -f '[d]rift_worker'` 형태로.

## 플레이북 기반 조치 가이드 (2026-09-03)

배경: 사용자가 RAG가 "구시대적"이라 느껴 브레인스토밍 → 09-02 에이전트 스펙
작성 → 09-03 사용자 제안("시나리오 문서를 많이 써서 활용")을 실험으로 검증.
문서 4개는 4/4 맞지만 16개로 늘리면 임베딩 검색 2/4. 문서의 `관련 센서` 코드와
상위 피처를 대조하는 방식이 4/4 + 타임라인 600건(고장 2종 100%, 온도는
복합 징후·약한 신호로 분리)에서 검증됨. 증거는
`02-cnc-machining/data/monitoring/_signature_spike_20260903/`.

**사용자 결정(2026-09-03)**: 플레이북 + 서명 대조 + 세 단계 판정으로 간다.
진단 에이전트 API(09-02 스펙 Part A·D)는 보류.

스펙 `02-cnc-machining/docs/specs/2026-09-03-cnc-playbook-guide-design.md`,
계획 `02-cnc-machining/docs/plans/2026-09-03-cnc-playbook-guide.md`.

- [x] Task 1 플레이북 문서 16항목 + `parse_playbook`
- [x] Task 2 `coverage`·`match_playbook`·세 단계 판정 (기록 4건 4/4)
- [x] Task 3 `build_corpus.py`에 플레이북 포함, 코퍼스 재빌드(42청크)
- [x] Task 4 프롬프트 — 시스템 판정 줄, 공구 마모 유도 문장 제거
- [x] Task 5 판정별 청크 선택 `select_chunks`, `build_guide` 새 경로
- [x] Task 6 `/predict` 응답 `fault` 필드, README·STRUCTURE
- [x] Task 7 `rag/eval_playbook.py` 오프라인 채점 → 스펙 정정 절
- [x] Task 8 라이브 검증 5건 → `docs/examples/`, 키 없이 폴백 확인, push

미결: 플레이북 16항목 내용의 도메인 검토(팀·멘토). 팀원 Spring 쪽에 `fault`
필드 추가를 알릴 것(기존 9개 키·`guide` 스키마는 불변).

### 리뷰 — 플레이북 가이드 구현 (2026-09-03)

커밋 713569b → 3c75d9f(8개, push 완료). 테스트 169 → 198개 통과. 코퍼스 42청크.

| 검증 | 결과 |
|---|---|
| 기록 4건(합성 3 + experiment_07) | 4/4 확정, 기대한 상황 |
| 타임라인 tool_wear Day 21-40 | 93/93 확정 공구 마모 |
| 타임라인 fixture_loosening Day 21-40 | 77/77 확정 고정구 풀림·채터 |
| 타임라인 temperature Day 21-40 (정상) | 복합 징후 62, 약한 신호 7, 확정 10 |
| 변형 전 오탐 6건 | 전부 약한 신호 |
| 라이브 /predict 5건 | 확정 4건 상황 일치, 원인 문장에 다른 구역 섞임 없음, 정상 1건 none |
| 키 없이 | `fault` 그대로, `guide` null |

상세는 스펙 "실행 결과에 따른 정정" 절. 배운 것: Python 3.14 `json.tool`은 파일로
리다이렉트해도 색상 코드를 넣는다(`NO_COLOR=1` 또는 json.dump로 저장). 전체
테스트가 cwd에 `mlflow.db`(790KB)를 만들어 놓는다 — 기존 현상, 지웠음. 다음 개선
후보: 가이드 `confidence_note`가 서명 일치도를 확률로 읽는 문구(프롬프트 한 줄).

**추가 (2026-09-03 오후)**: 팀원 요청으로 `/predict` top-level에 `versions`
객체(playbook 해시·코퍼스 빌드 시각·chat_model) 추가. `data/rag/corpus_meta.json`을
빌드 시 기록하고 서버가 읽는다. 테스트 200개, 예시 5개 재생성.


## 미팅 데모 화면 (2026-09-03 오후)

스펙 `02-cnc-machining/docs/specs/2026-09-03-cnc-demo-page-design.md`, 계획
`docs/plans/2026-09-03-cnc-demo-page.md`. 사용자 결정: 하이브리드(기록 모드 기본, 서버가
응답하면 실시간), 범위는 배치 진단 흐름 + 시나리오 타임라인 + 루프 이벤트, 개인 PC에서
pull 받아 미팅하되 실시간까지 준비.

- [x] Task 1 채점 기록에 ratio·top 추가, 시나리오별 병합, temperature 70일 재실행
- [x] Task 2 `src/demo/build.py` 로그 파서·조립·주입 + 테스트 4개
- [x] Task 3 `demo/template.html` + `demo/build_demo.py` → `demo/index.html`(256KB, 커밋)
- [x] Task 4 `/demo`, `/demo/inputs/{key}` + README §2-9 + STRUCTURE
- [x] Task 5 키 없이/있음 서버 검증, 실시간 재계산 8.6초, 스펙 정정, push

### 리뷰

테스트 205개 통과. 기록 모드는 pull 후 `demo/index.html` 더블클릭, 실시간은 회사 PC의
`data/`(감시 기록 제외 60MB) 옮기기 + `uv sync` + `.env` + 서버 기동. 남은 일: 사용자가
브라우저에서 두 탭을 눈으로 확인, 개인 PC에서 리허설.

**배운 것**: 서버 기동 줄과 `pkill -f '<패턴>'`을 같은 명령 문자열에 넣으면 패턴이 셸
자신과 일치해 셸이 죽는다(exit 144). 종료는 PID 파일이나 별도 명령으로 — `tasks/lessons.md`.

## 미팅 데모 — 시뮬레이션 탭 (2026-09-03 저녁)

사용자 요청: 정적 화면보다 MLOps 루프와 RAG 연결을 동적 시뮬레이션으로. 결정: 재생 모드
(09-02 실행 기록을 하루 단위로 재생, 재학습은 그 자리에서 안 돌림) + 좌우 분할 + "이 배치
지금 계산"으로 RAG 실시간 연결 시연. 스펙 `02-cnc-machining/docs/specs/2026-09-03-cnc-demo-simulation-design.md`.

- [x] Sim 1 채점 기록 top10 추가·재실행, `_batch` 확장, `guide_key`·`pick_representatives`, `guides` 조립
- [x] Sim 2 대표 가이드 13개 생성(`build_demo.py --representative` → `data/rag/demo_guides.json`)
- [x] Sim 3 `/demo/timeline/{scenario}/{day}/{index}` + 테스트 2개
- [x] Sim 4 탭 3(재생 엔진 `demo/sim_engine.js` + `node demo/test_sim.mjs`, 그래프, 이벤트 로그, 진단 패널, 지금 계산)
- [x] Sim 5 서버 검증(타임라인 배치 → /predict 5.1초), README·STRUCTURE·스펙 정정, push

테스트 211개 + node 1개 통과. `demo/index.html` 685KB. 남은 일: 사용자가 브라우저에서
시뮬레이션 탭 재생·이벤트 정지·칩 전환·지금 계산을 눈으로 확인, 개인 PC 리허설.

## 서빙 — `/predict` 이벤트 루프 블로킹 수정 (2026-09-15)

배경: 09-15 코드 리뷰에서 `/predict`가 `async def`인데 안에서 torch 추론과 OpenAI 호출을
동기로 실행하는 것을 발견. 실측(experiment_07을 60배로 이어 붙인 33,900행): predict 4.4초
동안 `/health`가 3.9초 대기. RAG를 켜면 LLM 호출 5~10초 동안 데모 페이지의 `/health` 폴링과
섀도우 추론까지 같이 멈춘다. 사용자 결정: "실제 MLOps 관점" 격차 중 1번으로 먼저 수정.
`main`에서 작업(`sim-realistic`은 실험 브랜치).

- [x] RED — `tests/serving/test_app.py::test_predict_does_not_block_other_requests`
      (추론이 막혀 있는 동안 `/health`가 답해야 함. 타이밍이 아니라 순서로 검증 —
      `/health`가 답한 뒤에야 추론을 풀어 준다) → 현재 코드로 실패 확인(5초 타임아웃까지 막힘)
- [x] GREEN — `src/serving/app.py`: `async def predict` → `def predict`,
      `await file.read()` → `file.file.read()`. FastAPI가 동기 엔드포인트를 스레드풀에서
      돌린다(`/health`·`/drift-status`·`/reload-model`은 원래부터 동기라 같은 방식)
      → 단독 통과, 전체 212개 통과
- [x] 실서버 재측정(포트 8917, champion v1) → predict 4.6초 실행 중 `/health` 0.004초

### 리뷰

| | 수정 전 | 수정 후 |
|---|---|---|
| predict (33,900행) | 4.4초 | 4.6초 |
| 그 동안 `/health` | 3.9초 | 0.004초 |

**실행 중 발견**: 새 테스트가 단독으로는 통과하고 전체 스위트에서는 `[True, True]`로
실패했다. 기존 `/start-shadow` 테스트가 엔드포인트를 실제로 호출해 전역 `_shadow_state`를
세팅한 채 되돌리지 않아서(monkeypatch가 아님), 뒤에 오는 테스트의 predict가 섀도우 추론까지
한 번 더 탄다. 새 테스트 안에서 `_shadow_state`를 None으로 격리해 해결. 기존 테스트의
누수 자체는 손대지 않았다 — 다른 테스트에도 영향을 줄 수 있으니 정리 후보.

스레드 안전성: 전역 `_state`는 요청 시작 시 `Depends`로 참조를 잡고 `/reload-model`은
참조를 통째로 바꾸므로 진행 중 요청은 옛 모델로 끝난다. torch 추론, SQLite(호출마다 새
연결), OpenAI 클라이언트는 스레드에서 호출해도 된다.

뒷정리: 측정용 predict 호출이 `requests.db`에 남긴 4행(09-15 타임스탬프)은 삭제해 8행으로
복원. 테스트 서버는 PID 파일로 종료.

## 재학습 루프 통합 테스트 + compose (2026-09-15)

스펙 `02-cnc-machining/docs/specs/2026-09-15-cnc-loop-integration-test-design.md`,
계획 `02-cnc-machining/docs/plans/2026-09-15-cnc-loop-integration-test.md`.
사용자 결정: 배관만 보장, pytest가 서버·워커를 서브프로세스로 기동, compose는 CI에서 빌드만.

- [x] Task 1 `src/config.py` — 환경변수 설정 모듈
- [x] Task 2 src 모듈·스크립트가 config를 읽도록
- [x] Task 3 feeder — config, 진행도 상한, `--start-day`
- [x] Task 4 워커 — config, champion 놓침 수를 MLflow에서
- [x] Task 5 pyproject — httpx2 본 의존성, integration 마커
- [x] Task 6 합성 데이터셋 생성기
- [x] Task 7 통합 테스트 하네스 + 스모크
- [x] Task 8 승격 경로 테스트
- [x] Task 9 거부 경로 테스트
- [x] Task 10 CI job 2개
- [x] Task 11 docker-compose + README/STRUCTURE
- [x] Task 12 실데이터 스모크, 스펙 정정 절, 리뷰

### 리뷰

- 통합 테스트 3개, 이 서버(`nice -n 19`)에서 152.3초. 단위 229개(3 deselected). 커밋 747b9b9~e8d1a8d 14개와
  이 기록 커밋, push 안 함. 승격 경로는 Day 5 트리거 → Day 7 v2 승격, 거부 경로는 Day 5·7 두 번 다 "정상 라벨
  없음"으로 거부. 실데이터 3일 스모크(환경변수 없음)는 Day 01~03 flagged=False(0.00 / 0.68 / 0.68), G1 기준
  1건, 정본 해시 불변, 실데이터 원상복구. 수치는 스펙 "실행 결과에 따른 정정 (2026-09-18)".
- 실행 중 발견한 것:
  - 독립 실행 워커가 09-03(6b5f12b)부터 시작 직후 죽고 있었다 — `load_rag_state()` 4-튜플을 3개로 언패킹(a3d8e25).
  - feeder가 배치마다 라벨을 적어 워커가 그날 배치 일부만 보고 그날을 처리했다 — 다 보낸 뒤 라벨(e325547).
  - 2 epoch 합성 champion은 사실상 미학습이라 잡음 ×3이면 Day 1부터 flagged(0.94) → Day 3 조기 트리거. ×10으로.
  - 계획의 테스트 단언 2개가 MLflow가 int로 주는 버전을 문자열 "1"과 비교했다(하나는 항상 통과, 하나는 항상 실패).
  - 스펙 §5의 "httpx2가 dev 전용이라 이미지에서 import 실패"는 틀렸다(openai 3.0이 이미 끌어옴). 직접 의존성 선언은 유지.
  - `/drift-status`가 champion run에 지표를 적으므로 실데이터 스모크는 `mlflow.db`까지 백업·복원했다.
  - 최종 리뷰로 넘긴 것: 섀도우 종료 시 `trigger_day` 태그 덮어쓰기, 단위 스위트가 실제 `shadow.db`에 매번
    2행 기록(`test_app.py` 격리 누수, 지금 46행).
- 최종 리뷰 수정 웨이브: 324f595·3f3dbe1·9667d67·839212d·66767e4 — shadow.db 누수 차단·정리, trigger_day 태그, 단언 강화, 하네스 보강
- 배운 것: 계획서 속 테스트 코드도 돌려 보기 전엔 가설이다. 반환 형태를 바꾸면 모든 호출부를 grep한다.
  둘 다 `tasks/lessons.md`.
- 남은 일: CI 결과 확인(push 후), 개인 PC 에서 `docker compose build` 와 `--profile demo` 리허설
- 후속 과제(최종 리뷰에서 보류, 동작 영향 없음):
  - `test_app.py`만 단독 실행하면 MLflow 기본 URI로 cwd에 `mlflow.db`가 생긴다. 전체 스위트에선
    `test_tracking.py`가 전역 URI를 tmp로 남겨 우연히 안 생긴다(순서 의존). autouse 픽스처에서 tmp tracking URI 설정.
  - 승격 run의 `trigger_day == "5"` 단언은 스펙이 허용한 재시도 승격을 막는다. `shadow_started`가 찍힌 날
    집합과 비교하는 형태로 완화(옛 버그도 계속 잡힌다).
  - CI 실패 시 워커·서버 로그를 아티팩트로 올리기(`--basetemp` + `upload-artifact`, 버전 확인 후).
  - `tests/test_config.py`의 서브프로세스 3개를 1개로 합치기(단위 스위트 +10~15초의 원인).
  - `conftest.py` `stop()`에서 kill 뒤 `wait(timeout=5)`가 드물게 `TimeoutExpired`를 내면 나머지 정리를 건너뛴다.

## 통합 테스트 변화 크기를 실제 운영 수준으로 (2026-09-18)

배경: 테스트 로그의 드리프트 비율(온도 58, 공구마모 9,634)이 실제 운영(9월 2일 실데이터 40일째 온도 1.80,
공구마모 3.08)과 너무 달라 발표 때 "이게 맞나"라는 질문을 부를 수 있다(사용자 지적). 근사 계산상 변화를
줄여도 승격·거부 흐름은 같다: 위치 2σ·전류 1.15배 → 비율 2.05, 스핀들 최대 1.3배 → 비율 2.80.

- [x] 1. `tests/integration/test_loop.py` 설정값 3개와 docstring 변경 → 확인: 옛 값(8, 1.0, 20) 없음
- [x] 2. 통합 테스트 3회 → 확인: 매번 3 passed, 세 번의 Day 줄 동일, 비율 약 2.05 / 2.80
- [x] 3. 스펙 정정 절·이 절에 결과 기록, 커밋

결과: 3회 모두 3 passed(162~181초), Day 줄 세 번 동일. 비율은 온도 1.86~2.07, 공구마모 2.51~2.71로 계산
(2.05 / 2.80)과 거의 같고 실데이터 기록(1.80 / 3.08)과 같은 범위. 흐름은 그대로: 승격 경로 Day 5 트리거 →
Day 7 v2 승격 → 이후 0.36~0.49, 거부 경로 Day 5·7 "정상 라벨 없음" 거부·원인 tool_wear·champion v1 유지.

## 새 데이터셋(두 번째 트랙) 취소 (2026-09-21)

사용자 결정: `test/` 폴더에 올려 두었던 AI-Hub 기계시설물·NASA Milling 샘플로 두 번째 트랙을 만드는 계획은
취소한다. 폴더도 삭제됨. 데이터셋은 기존 KAMP CNC 22실험 그대로 간다.

## main 병합 · CI 첫 실행 (2026-09-21)

- [x] 단위 테스트 재확인 → 229 passed·3 deselected, 34.5초
- [x] `loop-integration-test` → `main` fast-forward 병합(27커밋, `cbc7a7f..1f46527`)
- [x] `origin/main` push → 확인: CI 3개 job 전부 success
      [run 35549348692](https://github.com/ojeongu63-lab/hanium/actions/runs/35549348692)
      (`test` 55초, `loop-integration` 132초, `docker-build` 27초 — 셋 다 첫 실행)
- [x] 결과를 스펙 "실행 결과에 따른 정정" 절에 기록
- 병합된 로컬 브랜치 `loop-integration-test` 삭제(커밋은 main 에 있음, 원격에는 올린 적 없음)
- 남은 일: docker 있는 PC 에서 `docker compose --profile demo up` 기동 리허설, 위 후속 과제 5건

## 팀 연동 준비 (2026-09-21)

회의에서 스프링 중앙 오케스트레이션이 FastAPI 결과를 받아 가고, LSTM 결과를 피지컬 AI 시뮬레이션·스마트
글래스 에이전트로 넘기는 범위가 정해졌다. 연동 설계는 입력 형태가 확정돼야 시작할 수 있어서, 팀원에게
물어볼 것(백엔드 6건·시뮬레이션/글래스 3건)을 정리하고 각 항목에 기본안을 붙였다.

- [x] 현재 API 계약 확인 — `POST /predict` multipart CSV + `?method`, 응답 필드 9종, 그 외 엔드포인트 5개
- [x] 팀원용 설명 페이지 작성 — 구조·판정 흐름·API 계약·샘플 응답·현재 상태·질문 9건
      https://claude.ai/artifact/8tnqvDk52d6QkJDR3dTCni (비공개 — Share 로 팀원에게 열어 줘야 보임)
- [ ] 입력 형태 답 받은 뒤: JSON 엔드포인트 추가 여부 결정 → 설계 → 구현

## 팀 연동 1차 답변 (2026-09-23)

팀원이 계약 확정 답변 + 실행 증거 요청으로 회신. 스프링 연결은 이미 검증된 상태(기존
frozen LSTM CSV API 재사용), 이번엔 fault·guide·versions·모델 버전 관리만 맞추면 됨.
새 기능 요청 없음.

- [x] commit 확인 — main HEAD `2e2bdfe`, CI 그린(83eb685)과 02-cnc-machining 아래 diff 0 → 검증 상태 그대로 적용됨
- [x] 실제 서버 기동(로컬 uvicorn, champion v1) 후 `/health` + 정상(exp12)·이상(exp07)·경계(exp22, 문서화된 FP 사례) predict 재실행, RAG on/off 각각 측정
      - contributions 40개 확인(S_SystemInertia 랭킹 제외) — 팀원 지적이 맞음
      - RAG off: 0.09~0.5초 / RAG on: exp07 9.0초, exp22 2.8초, exp12 0.28초(good은 LLM 호출 없음)
      - 09-15에 측정한 4.4초는 exp07을 60배 이어 붙인 33,900행 스트레스 입력 — 정상 크기 입력과는 다른 수치임을 확인해 회신
- [x] MLflow 레지스트리 조회 — champion 여전히 v1(원본), v2~v6은 과거 실데이터 리허설의 재학습·게이트 산출물(하나는 promoted였다가 원상복구), 재학습 배관이 실데이터에서도 실제로 돈 증거로 제시
- [x] serving 최소 파일셋 확인(코드 근거로 필수/폴백/불필요 구분) → `data.tar.gz`(23M, dataset·processed·model·mlflow·rag) + predict 샘플 3개 + README 로 패키징, `/home/sure/cnc-serving-bundle-2e2bdfe/`, SHA256SUMS.txt 포함, 비밀값 없음 확인
- [x] `CNC_DATA_ROOT` 절대경로 자동 보정 테스트로 근거 제시(다른 머신에 복사해도 됨)
- [x] 핸드오버 문서의 "41개" 표기(feature_contributions 개수) 40개로 수정 — 로컬 파일만(코드 변경 없음, 팀원도 요청 안 함)
- 미해결: 이전에 공유한 claude.ai 아티팩트 링크는 계정 전환으로 이 세션에서 더 이상 못 고침(팀원 회신에 필요 없다고 함 — 급하지 않음)

## 경계 샘플 교체: exp22 → exp06 (2026-09-23)

사용자 지적으로 경계 샘플을 exp22에서 exp06으로 교체. 다시 실행해 보니 exp06이 실제로 더 나은
경계 사례였다: score 0.8584 vs threshold 0.8566(비율 1.002) — 임계값을 간신히 넘겨 판정이 뒤집히기
직전인 진짜 경계. `fault.verdict`도 세 단계 중 가장 약한 `weak`(약한 신호, top_z 4.18)로 exp07·exp22와
다른 검증 tier를 보여준다. exp22(비율 2.3, 정상 라벨을 불량으로 오판)는 임계값 근처가 아니라 오분류
사례였다.

- [x] exp06 predict 재실행(RAG on/off) → RAG off 0.16초, RAG on 3.27초
- [x] 번들 predict-samples 교체(`boundary_experiment06.json`), README·SHA256SUMS 갱신
