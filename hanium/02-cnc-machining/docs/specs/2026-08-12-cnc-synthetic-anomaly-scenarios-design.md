# CNC 합성 이상 시나리오 생성 설계

- 날짜: 2026-08-12
- 상태: 설계 완료, 구현 전

## 배경

프로젝트의 최종 목표는 "모델 판정 + 피처별 재구성오차 → RAG 기반 원인 설명 →
현장 조치 가이드"로 이어지는 데모다. 이 흐름을 시연하려면 `/predict`에 넣을
입력이 필요한데, 실제 eval의 불량 실험 11개만으로는 시나리오가 제한적이다
(공구마모/이송축 문제/진동 등 서로 다른 이상 유형을 골고루 보여주기 어려움).

실제 CNC 장비로 추가 실험은 불가능(공개 데이터셋 전용, 기존 확인됨)하므로,
실제 정상 실험 위에 도메인 지식으로 이상 패턴을 주입한 **합성 이상 데이터**를
만들어 데모 입력으로 쓴다. **목적은 데모/시연이며, 모델 성능 개선이나 통계적
검증이 아니다** — 이 전제가 아래 모든 설계 결정의 기준이다.

CNC 가이드북(`data/guide/03. Guidebook_CNC.pdf`)을 다시 확인했으나 "원인/조치/
불량유형" 관련 내용이 전혀 없음을 확인했다(키워드 검색 결과 "원인" 0회, "조치"
1회, "불량 유형" 0회 — CN7 가이드북과 마찬가지로 공정 설명·실습 위주). 따라서
시나리오 자체는 이 문서가 아니라 CNC 가공 도메인 지식으로 직접 설계한다.

## 목표 / 비목표

**목표**
- 실제 정상 실험 1개를 기준으로, 서로 다른 원인을 가진 이상 시나리오 3개를
  합성한다.
- 같은 시나리오 3개를 **진폭만 작게** 적용한 "정상 변형" 3개도 1:1로 함께
  만든다 — "이 정도까지는 정상, 이만큼 넘어가면 불량"이라는 경계를 데모에서
  함께 보여주기 위함.
- 각 시나리오는 원본 실험 CSV와 완전히 같은 형식(48컬럼)이라 `/predict`에
  수정 없이 바로 업로드할 수 있다.
- 각 시나리오가 실제로 모델에 의해 의도한 대로("이상" 3개는 불량, "정상 변형"
  3개는 정상으로) 판정되고, 이상 3개는 `feature_contributions` 상위권에 의도한
  피처가 나오는지 **직접 검증**한다(임의로 만든 값이 실제로 의도대로 작동하는지
  확인 없이 넘기지 않는다).

**비목표**
- 모델 재학습이나 성능 개선 (이 데이터는 학습에 쓰지 않는다)
- OOD 게이트 검증 (별개 트랙, 이미 LOOCV로 결론 낸 사안)
- RAG 연결 자체 (다음 서브프로젝트)

## 시나리오 3개

공통: 기준(baseline) = `experiment_01.csv`(feedrate=6, unworn, 정상, train에
쓰인 실험). `Machining_Process`/`M_sequence_number`/`M_CURRENT_PROGRAM_NUMBER`
등 메타 컬럼은 그대로 두고, 아래 대상 피처만 변형한다. 모든 변형은 **피처
자체의 스케일에 상대적인 비율**로 정의해(절대 매직넘버 없이) 피처마다 원본
단위가 크게 다른 문제(Position ~150 vs OutputPower ~0.001)를 자연스럽게 처리한다.

### ① 공구마모 — 전류/파워 점진적 증가
- 대상 피처: `S_OutputCurrent, S_OutputPower, S_CurrentFeedback, X_OutputPower, Y_OutputPower`
  (지난 세션에 실제 exp22 worn 사례에서 재구성오차가 가장 컸던 피처들과 동일 —
  근거 있는 선택)
- 패턴: 실험 시작(배율 1.0)부터 끝(배율 `1.0 + amplitude`)까지 선형 램프를
  곱한다 — 공구가 서서히 마모되는 그림.

### ② 이송축 부하 급증 — chip 막힘
- 대상 피처: `X_OutputCurrent, X_OutputPower, Y_OutputCurrent, Y_OutputPower`
  (실험 구간 30~50% 지점에서 배율 `1.0 + amplitude`로 스텝 증가 후 복귀) +
  `X_ActualVelocity, Y_ActualVelocity`(같은 구간에서 SetVelocity 대비 처짐을
  표현하려고 배율 `1.0 - min(amplitude * 0.1, 0.5)`로 감소, 최대 50% 저하로 캡)
- 패턴: 국소적(전체 구간이 아니라 일부 구간) 이벤트라는 점이 ①과의 핵심 차이.

### ③ 진동/백래쉬 증가 — position 노이즈
- 대상 피처: `X_ActualPosition, Y_ActualPosition, Z_ActualPosition,
  X_ActualVelocity, Y_ActualVelocity, Z_ActualVelocity`
- 패턴: `SetPosition`/`SetVelocity`는 그대로 두고, `Actual*`에 그 피처 자체의
  표준편차 × `amplitude`를 표준편차로 하는 가우시안 노이즈를 더한다(평균은
  안 바꾸고 흔들림만 키움 — 추종오차의 노이즈 증가를 표현). 시드 고정(42)으로
  재현 가능.

## 진폭(amplitude) 결정 — 자동 보정 루프

정확한 배율을 미리 못 박지 않는다. 대신:
1. 초기 진폭(①·②는 1.0, ③은 0.5)으로 생성
2. 생성된 CSV를 `serving.inference.predict_experiment()`(champion 모델을
   `serving.app.load_model_state()`로 로드, HTTP 서버 없이 직접 호출)에 넣어
   `predicted_label_text == "bad"`인지 확인
3. `"bad"`가 아니면 진폭을 2배로 올려 재시도, 최대 5회
4. 5회 안에 "bad"가 안 나오면 에러로 중단하고 시나리오 설계를 재검토한다
   (조용히 넘어가지 않는다)

이 루프가 이 서브프로젝트의 "검증" 자체이기도 하다 — 생성만 하고 끝내지 않고,
각 시나리오가 의도대로 모델에 이상으로 인식되는지 실제로 확인한다.

## 정상 변형 3개 — 같은 함수, 반대 방향 보정 루프

이상 시나리오 3개와 **완전히 같은 perturbation 함수**를 재사용하되(새 로직
없음), 시작 진폭을 작게 잡고 보정 방향을 뒤집는다:

1. 초기 진폭을 이상 시나리오의 1/10로 시작(①·②는 0.1, ③은 0.05)
2. `predict_experiment()`에 넣어 `predicted_label_text == "good"`인지 확인
3. `"bad"`가 나오면(진폭이 여전히 너무 크면) 진폭을 절반으로 줄여 재시도,
   최대 5회
4. 5회 안에 "good"으로 안 남으면 에러로 중단

이렇게 하면 같은 방향의 변형이 작을 땐 정상으로, 클 땐 불량으로 판정되는
경계를 실제로 확인하면서 만들게 된다.

## 산출물

```
02-cnc-machining/synthetic/
  generate_synthetic.py         # 6개(이상 3 + 정상변형 3) 생성 + 자동 보정 루프 + 검증
  scenarios/
    tool_wear.csv                        # 시나리오 ① (이상)
    tool_wear_predict_result.json
    tool_wear_normal.csv                 # 시나리오 ①의 정상 변형
    tool_wear_normal_predict_result.json
    feed_overload.csv                    # 시나리오 ② (이상)
    feed_overload_predict_result.json
    feed_overload_normal.csv             # 시나리오 ②의 정상 변형
    feed_overload_normal_predict_result.json
    vibration_backlash.csv               # 시나리오 ③ (이상)
    vibration_backlash_predict_result.json
    vibration_backlash_normal.csv        # 시나리오 ③의 정상 변형
    vibration_backlash_normal_predict_result.json
```

`generate_synthetic.py`는 `src/`에 추가하지 않는다 — 이 프로젝트의 기존
일회성 스크립트 관례(`loocv/run_loocv.py` 등)를 따른다. 다만 재실행 가능하게
만들어(데모 준비할 때마다 다시 돌릴 수 있게) 완전한 일회성 스크립트는 아니고,
`02-cnc-machining/synthetic/`에 계속 남겨두는 재사용 가능한 데모 자산이다.

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `02-cnc-machining/synthetic/generate_synthetic.py` | 신규 |
| `02-cnc-machining/synthetic/scenarios/*.csv`, `*.json` | 신규 (생성 산출물) |
| 기존 `src/` 코드 | 변경 없음 — `serving.inference.predict_experiment()`,
  `serving.app.load_model_state()`를 그대로 재사용 |

## 테스트 범위

정식 pytest 단위테스트는 만들지 않는다 — 위 두 자동 보정 루프 자체가 "생성된
6개 시나리오가 각각 의도한 대로(bad/good) 판정되는지" 검증하는 런타임 체크
역할을 한다(이 프로젝트의 일회성 스크립트 관례와 동일).

## 검증 방법

1. `generate_synthetic.py` 실행 → 이상 시나리오 3개는 5회 이내에 `"bad"`,
   정상 변형 3개는 5회 이내에 `"good"` 판정을 받는지 확인(로그로 몇 번째
   시도에서 성공했는지, 최종 진폭이 얼마였는지 출력).
2. 각 이상 시나리오의 `*_predict_result.json`에서 `feature_contributions`
   상위 3개가 의도된 대상 피처와 겹치는지 육안 확인 — 안 겹치면(엉뚱한 피처가
   1등이면) 시나리오 설계 자체를 재검토(다음 세션에서 논의).
3. 결과를 사용자에게 보고 — RAG 연결(다음 서브프로젝트)로 넘어갈지 결정.
