"""드리프트 감시 워커 — 폴링, 트리거, 재학습, 게이트, 승격을 엮는다.

서빙과 별도 루프로 도는 구조라 학습이 추론 응답을 지연시키지 않는다.
판정 로직은 전부 src/retraining/ 에 있고 이 파일은 호출만 한다.

핵심: 트리거는 "센서만 틀어진 것"과 "설비가 실제로 망가진 것"을 구분하지
못한다. 두 경우 다 재학습이 발동되며, 구분은 게이트가 사후에 한다.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlflow.tracking import MlflowClient  # noqa: E402

from lstm_ae.tracking import (  # noqa: E402
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    promote_to_champion,
)
from retraining.gate import evaluate_gate  # noqa: E402
from retraining.promotion import swap_with_rollback, verify_serving_contract  # noqa: E402
from retraining.runner import run_retraining  # noqa: E402
from retraining.trigger import days_to_process, is_drift_flagged, should_retrain  # noqa: E402

LABELS_DB = ROOT / "data" / "monitoring" / "labels.db"
REQUESTS_DB = ROOT / "data" / "monitoring" / "requests.db"
MODEL_DIR = ROOT / "data" / "model"
SCALER_PATH = ROOT / "data" / "processed" / "scaler.json"
BACKUP_ROOT = ROOT / "data" / "model_backup"
COOLDOWN_DAYS = 5
CONSECUTIVE_K = 3
# G2 평가에 쓸 최근 라벨 도착 배치 수(= 최근 4일치). 두 모델을 전부 재추론하므로
# 상한을 둔다.
#
# 60으로 올려봤다가 20으로 되돌린 값이다. 표본이 작다는 문제의식으로 넓혔더니
# 시나리오 A 가 승격에서 거부로 뒤집혔다 — Day 37 기준 G2 가 0.95 vs 0.60 에서
# 0.48 vs 0.50 이 됐다.
#
# 이유: 재학습 모델은 최근 환경에 맞춰 학습되므로 최근 구간에서 강하다. 창을
# 과거로 넓히면 그 강점이 희석되는 반면 champion 은 자기가 잘하던 옛 구간
# 점수를 벌어들인다. **드리프트 상황에서 평가 창 확대는 중립적이지 않고
# 구모델에 유리하다.** G2 의 질문은 "지금 현장에서 더 나은가"인데 12일치
# 평균은 그 "지금"이 아니다.
#
# 표본 20건이 통계적으로 넉넉해서가 아니라, 넓히면 측정 대상 자체가
# 바뀌기 때문에 이 값을 쓴다. 이 민감도는 알려진 한계다.
GATE_SAMPLE_SIZE = 20


@dataclass
class WorkerState:
    flag_history: list[bool] = field(default_factory=list)
    cooldown_remaining: int = 0
    champion_missed: int = 1                # 현 champion 실측 — 불량 11개 중 1개 놓침
    champion_accuracy: float = 0.0          # 첫 게이트 평가 시 측정값으로 대체


def tick(client, state: WorkerState, current_day: int, scenario: str) -> dict:
    status = client.get("/drift-status").json()
    flagged = is_drift_flagged(status)
    state.flag_history.append(flagged)
    ratio = status.get("output_drift", {}).get("ratio_to_threshold", 0.0)

    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1

    if not should_retrain(state.flag_history, CONSECUTIVE_K, state.cooldown_remaining):
        return {"ratio": ratio, "flagged": flagged, "action": "none"}

    print(f"  [Day {current_day}] 트리거 발동 — 재학습 시작", flush=True)
    result = run_retraining(
        timeline_dir=ROOT / "data" / "timeline" / scenario,
        labels_db=LABELS_DB,
        current_day=current_day,
        root=ROOT,
    )
    state.cooldown_remaining = COOLDOWN_DAYS

    decision = _decide_and_promote(client, state, result, current_day, scenario)
    return {"ratio": ratio, "flagged": flagged, "action": decision}


def _decide_and_promote(client, state, result, current_day, scenario) -> str:
    mlflow_client = MlflowClient()
    run = mlflow_client.get_run(result["run_id"])

    # 계약 확인을 파일 교체보다 앞에 둔다 — 위반이면 롤백할 것 자체가 생기지 않는다.
    missing = verify_serving_contract(run.data.metrics, run.data.params)
    if missing:
        _tag(mlflow_client, result["run_id"], scenario, current_day,
             decision="rejected", reason=f"서빙 계약 미충족: {missing}")
        print(f"  거부 — 서빙 계약 미충족: {missing}", flush=True)
        return "rejected"

    champion_accuracy, retrained_accuracy, sample_size = _gate_accuracies(
        result, current_day, scenario
    )
    verdict = evaluate_gate(
        retrained_missed=result["missed"],
        champion_missed=state.champion_missed,
        retrained_accuracy=retrained_accuracy,
        champion_accuracy=champion_accuracy,
    )
    _tag(mlflow_client, result["run_id"], scenario, current_day,
         decision=verdict["decision"], reason=verdict["reject_reason"],
         extra={"gate_g1_missed": verdict["g1_missed"],
                "gate_g2_accuracy_delta": verdict["g2_accuracy_delta"],
                "gate_g2_sample_size": sample_size})

    print(f"  게이트: G1 놓침={verdict['g1_missed']}건 (champion {state.champion_missed}건, "
          f"허용 {state.champion_missed + 1}건) / "
          f"G2 {retrained_accuracy:.2f} vs {champion_accuracy:.2f} "
          f"(표본 {sample_size}건)", flush=True)

    if verdict["decision"] == "rejected":
        print(f"  거부 — {verdict['reject_reason']}  (champion 유지, 사람 확인 필요)", flush=True)
        return "rejected"

    previous_version = mlflow_client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS
    ).version

    def _promote() -> None:
        promote_to_champion(result["model_version"])
        client.post("/reload-model").raise_for_status()

    def _verify() -> None:
        health = client.get("/health").json()
        if health["model_version"] != str(result["model_version"]):
            raise RuntimeError(f"리로드 후 버전 불일치: {health['model_version']}")

    try:
        swap_with_rollback(
            result["retrain_dir"], MODEL_DIR, SCALER_PATH, BACKUP_ROOT,
            promote=_promote, verify=_verify,
        )
    except Exception as exc:
        # 파일은 swap_with_rollback 이 되돌렸다. alias 와 서빙 상태만 마저 되돌린다.
        print(f"  승격 실패, 롤백 중: {exc}", flush=True)
        promote_to_champion(previous_version)
        client.post("/reload-model")
        raise

    state.champion_missed = result["missed"]
    print(f"  승격 완료 — version {result['model_version']}", flush=True)
    return "promoted"


def _predict_labels(batch_paths, model, scaler_dict, threshold, baseline, window_size):
    """배치들을 주어진 모델로 판정한다. HTTP를 타지 않는다 — /predict 를 부르면
    요청 로그에 게이트 평가용 가짜 트래픽이 쌓여 드리프트 윈도우가 오염된다."""
    import pandas as pd

    from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
    from serving.inference import predict_experiment

    labels = []
    for path in batch_paths:
        result = predict_experiment(
            pd.read_csv(path), model, FEATURE_COLUMNS, scaler_dict, window_size,
            threshold, "mean", baseline, SETUP_CONSTANT_COLUMNS,
        )
        labels.append(result["predicted_label_text"])
    return labels


def _gate_accuracies(result: dict, current_day: int, scenario: str) -> tuple[float, float, int]:
    """G2 입력 — 라벨 도착 구간에서 champion과 재학습 모델의 정확도.

    두 모델을 같은 배치에 대고 직접 돌려 비교한다. champion 판정을 predict_log
    에서 꺼내오지 않는 이유는 배치 식별자가 로그에 없어 짝을 맞출 수 없기
    때문이고, /predict 를 다시 부르지 않는 이유는 위 _predict_labels 주석과 같다.
    """
    import json

    import torch

    from lstm_ae.model import LSTMAutoencoder
    from monitoring.labels import get_arrived_labels
    from preprocessing.columns import FEATURE_COLUMNS
    from retraining.gate import accuracy_from_pairs
    from retraining.runner import TRAINING_CONFIG
    from serving.app import load_model_state

    arrived = get_arrived_labels(current_day, LABELS_DB)[-GATE_SAMPLE_SIZE:]
    if not arrived:
        return 0.0, 0.0, 0

    timeline_dir = ROOT / "data" / "timeline" / scenario
    batch_paths = [timeline_dir / f"{r['batch_id']}.csv" for r in arrived]
    truths = [r["label"] for r in arrived]

    champion = load_model_state()
    champion_preds = _predict_labels(
        batch_paths, champion.model, champion.scaler_dict,
        champion.thresholds["mean"], champion.feature_baseline, champion.window_size,
    )

    retrain_dir = Path(result["retrain_dir"])
    model = LSTMAutoencoder(
        num_features=len(FEATURE_COLUMNS),
        hidden_size=TRAINING_CONFIG["hidden_size"],
        latent_dim=TRAINING_CONFIG["latent_dim"],
    )
    model.load_state_dict(torch.load(retrain_dir / "model.pt"))
    model.eval()
    retrained_preds = _predict_labels(
        batch_paths,
        model,
        json.loads((retrain_dir / "scaler.json").read_text()),
        result["thresholds"]["mean"],
        json.loads((retrain_dir / "feature_baseline.json").read_text()),
        TRAINING_CONFIG["window_size"],
    )

    return (
        accuracy_from_pairs(truths, champion_preds),
        accuracy_from_pairs(truths, retrained_preds),
        len(truths),
    )


def main() -> None:
    """실제 서버를 상대로 도는 독립 프로세스 진입점.

    서빙(uvicorn)과 배치를 흘려보내는 feeder(simulate_timeline.py --serve-url)를
    각각 별도 프로세스로 띄운 뒤, 이 스크립트를 세 번째 프로세스로 돌린다.
    "오늘이 며칠째인지"는 자체 카운터가 아니라 labels.db 에 feeder 가 기록한
    최신 produced_day 를 읽어서 안다 — 두 프로세스가 서로 다른 날짜를 셀
    위험이 없다.
    """
    import argparse
    import time

    import httpx2

    from monitoring.labels import get_latest_produced_day

    parser = argparse.ArgumentParser(description="드리프트 감시 워커 (독립 프로세스)")
    # simulate_timeline.py 의 PERTURBATIONS 키와 동일 — feeder 가 생성하는 시나리오.
    parser.add_argument("scenario", choices=["temperature", "tool_wear"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="폴링 주기(초)")
    args = parser.parse_args()

    state = WorkerState()
    last_day = 0

    with httpx2.Client(base_url=args.base_url, timeout=30.0) as client:
        print(f"감시 시작 — {args.base_url} 폴링 (주기 {args.poll_interval}초)", flush=True)
        try:
            while True:
                latest_day = get_latest_produced_day(LABELS_DB)
                for day in days_to_process(last_day, latest_day):
                    result = tick(client, state, current_day=day, scenario=args.scenario)
                    print(
                        f"Day {day:02d}  score/threshold={result['ratio']:.2f}  "
                        f"flagged={result['flagged']}  action={result['action']}",
                        flush=True,
                    )
                    last_day = day
                time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            print("감시 워커 종료", flush=True)


def _tag(mlflow_client, run_id, scenario, day, decision, reason, extra=None) -> None:
    tags = {
        "scenario": scenario,
        "trigger_day": day,
        "gate_decision": decision,
        "gate_reject_reason": reason,
        **(extra or {}),
    }
    for key, value in tags.items():
        mlflow_client.set_tag(run_id, key, value)


if __name__ == "__main__":
    main()
