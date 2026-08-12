from mlflow.tracking import MlflowClient


def log_drift_metrics(status: dict, run_id: str, client=None) -> None:
    if not status["sufficient_data"]:
        return
    client = client or MlflowClient()
    try:
        client.log_metric(
            run_id,
            "drift_output_ratio_to_threshold",
            status["output_drift"]["ratio_to_threshold"],
        )
        client.log_metric(
            run_id,
            "drift_input_flagged_count",
            len(status["input_drift"]["flagged_features"]),
        )
        client.log_metric(
            run_id,
            "drift_avg_score_recent",
            status["output_drift"]["avg_score_recent"],
        )
    except Exception as exc:
        print(f"드리프트 metric MLflow 기록 실패: {exc}")
