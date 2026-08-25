# 섀도우 배포 (2026-08-25)

## 목적

멘토 요구가 아니라 사용자가 "MLOps 포트폴리오로 발전시키고 싶다"는 목적에서
시작한 기능. 기존 게이트(G1+G2)로는 못 채우는 구멍을 메운다.

지금 구조의 한계: 게이트 통과 즉시 100% 승격한다. 그런데 G2가 확인하는 건
"라벨이 이미 도착한 **과거** 구간에서 candidate가 champion보다 나았다"는
것뿐이다. 이건 retrospective(사후 재현) 검증이라, **candidate가 미래의
새 트래픽에서도 계속 나은지는 한 번도 확인되지 않은 채** 정식 투입된다.
G1은 이미 실측으로 매번 통과해버려 무력화된 것으로 확인돼 있다
(`2026-08-19-cnc-drift-triggered-retraining-design.md` 참고).

섀도우 배포는 게이트를 통과한 candidate를 즉시 승격하는 대신, 실제 운영
트래픽에 **관찰자로 병행 투입**해 미래 데이터로 한 번 더 검증하는 단계를
추가한다. 실서비스 응답에는 영향을 주지 않는다 — 이게 "섀도우"라는 이름의
표준 정의다.

## 데이터 흐름

```
트리거(연속 3회 flagged) → 재학습 → 게이트(G1+G2, 트리거 시점 데이터)
  통과 → 섬도 시작 (POST /start-shadow)
    이후 /predict 요청마다:
      champion 추론 → 사용자 응답 (기존과 동일, 섬도는 응답에 영향 없음)
      candidate 추론 → shadow_predictions 테이블에 기록만
    워커가 매 tick마다 "섬도 시작 이후 새로 도착한 라벨" 개수 확인
    20건 도착 시 → 그 구간 champion/candidate 정확도 재비교
      candidate 승 → 진짜 승격 (기존 promote_to_champion + swap_with_rollback)
      candidate 패 → 폐기, champion 유지
    섬도 해제, 다음 트리거 감시 재개
  거부 → 태그 기록 (기존과 동일, 섬도로 안 감)

섬도 진행 중에는 새 트리거를 억제한다 — 한 번에 후보 하나만 검증.
```

## Part A — 섀도우 예측 로그 (`src/monitoring/shadow_log.py`, 신규)

`labels.py`와 같은 패턴(순수 SQLite 래퍼, 부작용 없는 함수).

```python
def record_shadow_prediction(
    batch_id: str, champion_label: str, candidate_label: str, db_path: Path
) -> None: ...

def get_shadow_predictions(batch_ids: list[str], db_path: Path) -> dict[str, dict]:
    """batch_id -> {"champion_label": str, "candidate_label": str}."""
```

스키마:
```sql
CREATE TABLE shadow_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    champion_label TEXT NOT NULL,
    candidate_label TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
```

`batch_id`는 업로드 파일명에서 확장자를 뗀 값(`day37_2.csv` → `day37_2`).
`qc_labels.batch_id`와 형식이 같아야 나중에 라벨과 조인할 수 있다.

DB 경로는 `data/monitoring/shadow.db` — 기존 `labels.db`/`requests.db`와
같은 디렉토리, 별도 파일(라벨/요청 로그와 섞이지 않게).

## Part B — 서빙 앱 확장 (`src/serving/app.py`, 기존 파일 수정)

**섀도우 상태**: `_state`(champion)와 별개로 전역 `_shadow_state:
ModelState | None`을 둔다. 서버 재시작 시 사라진다 — champion 상태와
동일한 기존 한계이지 이번에 새로 생기는 문제가 아니다.

```python
@app.post("/start-shadow")
def start_shadow(payload: dict) -> dict:
    """payload: {"model_version": str}. candidate를 MLflow 레지스트리에서
    alias 없이 특정 버전으로 직접 로드한다 — champion alias는 아직
    바뀌지 않았으므로."""
    global _shadow_state
    version = payload["model_version"]
    _shadow_state = load_candidate_state(version)  # load_model_state()의 버전 파라미터화
    return {"status": "shadow_started", "candidate_version": version}


@app.post("/stop-shadow")
def stop_shadow() -> dict:
    global _shadow_state
    _shadow_state = None
    return {"status": "shadow_stopped"}
```

`/predict` 확장 — champion 추론 이후, `_shadow_state`가 있으면 candidate로도
추론해 `shadow_predictions`에 기록한다(응답 바디에는 안 넣는다):

```python
if _shadow_state is not None:
    shadow_result = predict_experiment(df=df, model=_shadow_state.model, ...)
    record_shadow_prediction(
        batch_id=Path(file.filename).stem,
        champion_label=result["predicted_label_text"],
        candidate_label=shadow_result["predicted_label_text"],
        db_path=SHADOW_DB,
    )
```

candidate 추론이 실패해도(예: 컬럼 안 맞음) 사용자 응답에는 영향 주면 안
된다 — `try/except`로 감싸고 실패는 로그만 남긴다.

## Part C — 워커 확장 (`monitoring/drift_worker.py`, 기존 파일 수정)

**`WorkerState`에 섀도우 상태 추가**:
```python
@dataclass
class ShadowState:
    candidate_version: str
    run_id: str
    start_day: int
    labels_seen_at_start: int   # 섬도 시작 시점까지 이미 도착해 있던 라벨 수
                                  # (이후 "새로 도착한" 라벨만 세기 위한 기준점)

@dataclass
class WorkerState:
    flag_history: list[bool] = field(default_factory=list)
    cooldown_remaining: int = 0
    champion_missed: int = 1
    champion_accuracy: float = 0.0
    shadow: ShadowState | None = None   # 신규
```

**`tick()` 변경**: 섀도우 활성 중이면 새 트리거 판단 자체를 건너뛰고 섀도우
종료 조건만 확인한다.

```python
def tick(client, state, current_day, scenario) -> dict:
    status = client.get("/drift-status").json()
    flagged = is_drift_flagged(status)
    state.flag_history.append(flagged)
    ratio = status.get("output_drift", {}).get("ratio_to_threshold", 0.0)

    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1

    if state.shadow is not None:
        action = _check_shadow(client, state, current_day, scenario)
        return {"ratio": ratio, "flagged": flagged, "action": action}

    if not should_retrain(state.flag_history, CONSECUTIVE_K, state.cooldown_remaining):
        return {"ratio": ratio, "flagged": flagged, "action": "none"}

    # ... 기존 재학습 → 게이트(G1+G2) ...
    # 통과 시: _start_shadow(client, state, result, current_day) 호출,
    #          action = "shadow_started" (더 이상 여기서 바로 승격 안 함)
```

**`_check_shadow()`** (신규): 섀도우 시작 이후 새로 도착한 라벨이
`GATE_SAMPLE_SIZE`(20건)에 찼는지 확인. 안 찼으면 `"shadow_pending"`.
찼으면 `shadow_predictions`와 `qc_labels`를 batch_id로 조인해 champion/
candidate 정확도를 다시 계산하고, `evaluate_shadow()`로 최종 판정한다.

**`evaluate_shadow()`** (신규, `src/retraining/gate.py`에 추가): G1은 섀도우
단계에서 재확인하지 않는다(이미 트리거 시점에 확인했고, 원본 eval 자체는
시간이 지나도 안 바뀌므로 재확인할 이유가 없다). 단순 비교만 한다:

```python
def evaluate_shadow(candidate_accuracy: float, champion_accuracy: float) -> dict:
    promoted = candidate_accuracy > champion_accuracy
    return {
        "decision": "promoted" if promoted else "rejected",
        "accuracy_delta": candidate_accuracy - champion_accuracy,
    }
```

판정 후: 승격이면 기존 `swap_with_rollback` + `promote_to_champion` +
`/reload-model` 그대로 재사용. 거부든 승격이든 `POST /stop-shadow` 호출해
서버의 섀도우 상태를 해제하고, `state.shadow = None`.

## Part D — 검증 방법

`monitoring/simulate_timeline.py`/`sweep_drift_constants.py`는 안 건드린다
(기존 데이터 생성 로직 재사용).

**시나리오 A(온도) 재현으로 섀도우 매커니즘 자체를 확인한다.** 기존 40일
로는 부족하다 — Day 37에 게이트(트리거 시점) 통과 → 섀도우 시작이고, 라벨
지연이 7일이라 그 이후 생산분의 라벨은 Day 44부터 도착하기 시작한다.
20건(4일치)을 모으려면 최소 Day 47까지 필요하다. **`--days 55`** 정도로
넉넉히 돌려 섀도우 시작부터 최종 승격/폐기까지 전 구간을 확인한다.

**섀도우가 폐기되는 경로는 이번에 강제로 만들지 않는다.** 시나리오 A/B
둘 다 자연스럽게 그 경로를 재현하지 않을 수 있다(A는 제품이 계속 정상이라
섀도우 기간에도 candidate가 계속 우세할 가능성이 높다). 실행해서 실제로
어느 쪽이 나오는지 관찰하고, 예상과 다르면 **값을 조정해 통과시키지 않고
관찰값을 그대로 보고한다** — 이 프로젝트의 기존 검증 원칙과 동일.

## 알려진 한계 (미리 문서화)

- 섀도우 상태(`_shadow_state`, `WorkerState.shadow`)는 인메모리라 서버/워커
  재시작 시 사라진다. champion 상태도 이미 같은 한계를 갖고 있어 이번에
  새로 생기는 문제는 아니지만, 재시작 시 진행 중이던 섀도우 검증은
  처음부터 다시 시작해야 한다.
- 섀도우 후보 추론이 매 `/predict` 요청마다 champion 추론에 얹혀 도므로,
  섀도우가 활성 중인 동안은 요청당 지연 시간이 늘어난다(모델 두 개 추론).
  이 프로젝트 규모(CPU, 소형 LSTM-AE)에서는 무시할 수준으로 보이나,
  실측으로 확인하지는 않는다 — 규모가 커지면 실제 운영에서 고려해야 할
  트레이드오프로만 남겨둔다.

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `src/monitoring/shadow_log.py` | 신규 |
| `src/serving/app.py` | `/start-shadow`, `/stop-shadow` 신규, `/predict`에 섀도우 병행 추론 추가 |
| `src/retraining/gate.py` | `evaluate_shadow()` 추가 |
| `monitoring/drift_worker.py` | `ShadowState` 추가, `tick()`/`_decide_and_promote()` 분기, `_check_shadow()` 신규 |
| `docs/specs/2026-08-25-cnc-shadow-deployment-design.md` | 이 문서 |

`src/retraining/{trigger,runner,promotion}.py`, `src/monitoring/labels.py`는
변경 없음 — 트리거·재학습·백업/롤백 로직은 그대로 재사용한다.
