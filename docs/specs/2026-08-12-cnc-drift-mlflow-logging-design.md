# CNC 드리프트 지표 MLflow 기록 설계

- 날짜: 2026-08-12
- 상태: 설계 완료, 구현 전

## 배경

방금 완료한 드리프트 모니터링(`GET /drift-status`)이 MLflow와 완전히 분리돼
있어서, 이 프로젝트의 핵심 MLOps 스토리(MLflow 실험추적+모델레지스트리)와
연결이 안 된다는 한계가 있었다. 드리프트 체크 결과를 champion 모델의 학습
run에 metric으로 같이 기록해, MLflow UI에서 시간에 따른 드리프트 추이를 볼
수 있게 한다.

**사전 검증(스파이크) 완료**: 이미 `FINISHED` 상태인 run에도
`MlflowClient().log_metric(run_id, key, value)`로 직접 기록이 잘 됨을
실제로 확인했다(run 상태 안 바뀜, 값 정상 조회됨). `mlflow.start_run()`으로
재개할 필요 없음.

## 목표 / 비목표

**목표**
- `GET /drift-status`가 `sufficient_data=true`인 실제 값을 낼 때마다,
  champion 모델을 만든 학습 run(`state.mlflow_run_id`)에 드리프트 관련
  metric 3개를 기록한다.
- MLflow 기록이 실패해도 `/drift-status` 응답 자체는 항상 정상 반환한다
  (RAG의 에러 처리와 동일 원칙).

**비목표**
- throttling(중복 호출 시 생략) — 이번엔 매번 호출마다 기록하기로 확정함.
- 새 MLflow run 생성 — 기존 champion run에 이어붙이는 방식만 쓴다.

## 설계

### `src/monitoring/mlflow_logging.py` (신규)

```python
def log_drift_metrics(status: dict, run_id: str, client=None) -> None:
    if not status["sufficient_data"]:
        return
    client = client or MlflowClient()
    try:
        client.log_metric(
            run_id, "drift_output_ratio_to_threshold",
            status["output_drift"]["ratio_to_threshold"],
        )
        client.log_metric(
            run_id, "drift_input_flagged_count",
            len(status["input_drift"]["flagged_features"]),
        )
        client.log_metric(
            run_id, "drift_avg_score_recent",
            status["output_drift"]["avg_score_recent"],
        )
    except Exception as exc:
        print(f"드리프트 metric MLflow 기록 실패: {exc}")
```

`client` 파라미터를 주입 가능하게 열어둔 이유는 순전히 테스트 때문이다 —
가짜 client 객체로 실제 MLflow 저장소 없이 "3개 metric이 올바른 이름/값으로
호출됐는지"를 검증할 수 있다. 실패를 통째로 삼키는 이유는 두 가지다:
(1) 모니터링 부가기능 하나가 핵심 응답(`/drift-status`)을 막으면 안 됨
(2) 테스트에서 가짜 `run_id`("fake-run-id" 등)를 쓸 때 실제 MLflow 저장소에
없는 run이라 에러가 나는데, 이게 조용히 처리돼야 기존 `test_app.py`의
드리프트 테스트들이 실제 MLflow 연결 없이도 그대로 통과한다.

### `app.py` 통합

`/drift-status` 라우트에서 `compute_drift_status()` 호출 직후, 응답을
만들기 전에 한 줄 추가:
```python
log_drift_metrics(status, state.mlflow_run_id)
```

### metric 이름 — 기존 metric과 충돌 없음
champion run에는 이미 `mean_precision`, `mean_threshold` 등이 있다. 새
metric은 전부 `drift_` 접두사를 붙여 구분한다: `drift_output_ratio_to_threshold`,
`drift_input_flagged_count`, `drift_avg_score_recent`.

## 코드 변경 요약

| 파일 | 변경 |
|---|---|
| `02-cnc-machining/src/monitoring/mlflow_logging.py` | 신규 |
| `02-cnc-machining/src/serving/app.py` | 수정 — `/drift-status`에 로깅 한 줄 추가 |

## 테스트 범위

`src/`에 들어가는 정식 코드라 pytest 단위테스트 작성:
- `mlflow_logging.py`: 가짜 client(스텁, RAG의 OpenAI 클라이언트 스텁 테스트와
  같은 패턴)로 (1) `sufficient_data=false`면 아무 것도 호출 안 하는지
  (2) `sufficient_data=true`면 정확히 3개 metric이 올바른 이름/값으로
  호출됐는지 검증. 실제 MLflow 저장소 불필요.
- `app.py` 기존 드리프트 테스트들(`test_app.py`)은 수정 없이 그대로
  통과해야 한다 — 가짜 `run_id`로 인한 MLflow 에러가 내부에서 삼켜지므로.

## 검증 방법

1. pytest 전체 통과(기존 94개 + 신규).
2. 실제 서버를 띄우고 `/predict`를 10번 호출해 `/drift-status`가
   `sufficient_data=true`를 내게 만든 뒤, MLflow UI 또는
   `MlflowClient().get_run(run_id).data.metrics`로 champion run에
   `drift_*` metric 3개가 실제로 기록됐는지 확인.
3. 결과를 사용자에게 보고.
