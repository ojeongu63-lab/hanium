from mlflow.tracking import MlflowClient


def log_drift_metrics(status: dict, run_id: str, step: int = 0, client=None) -> None:
    if not status["sufficient_data"]:
        return
    client = client or MlflowClient()
    try:
        client.log_metric(
            run_id,
            "drift_output_ratio_to_threshold",
            status["output_drift"]["ratio_to_threshold"],
            step=step,
        )
        client.log_metric(
            run_id,
            "drift_input_flagged_count",
            len(status["input_drift"]["flagged_features"]),
            step=step,
        )
        client.log_metric(
            run_id,
            "drift_avg_score_recent",
            status["output_drift"]["avg_score_recent"],
            step=step,
        )
    except Exception as exc:
        print(f"드리프트 metric MLflow 기록 실패: {exc}")
