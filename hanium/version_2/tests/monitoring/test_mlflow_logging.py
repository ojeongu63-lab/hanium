from monitoring.mlflow_logging import log_drift_metrics


class _FakeMlflowClient:
    def __init__(self, raise_on_call: bool = False):
        self.logged = []
        self._raise_on_call = raise_on_call

    def log_metric(self, run_id, key, value):
        if self._raise_on_call:
            raise RuntimeError("mlflow 저장소에 연결할 수 없음")
        self.logged.append((run_id, key, value))


_SUFFICIENT_STATUS = {
    "sufficient_data": True,
    "output_drift": {"ratio_to_threshold": 1.2, "avg_score_recent": 0.9},
    "input_drift": {"flagged_features": [{"feature": "f0", "avg_scaled_mean": 3.0}]},
}


def test_log_drift_metrics_skips_when_insufficient_data():
    client = _FakeMlflowClient()

    log_drift_metrics({"sufficient_data": False}, "run123", client=client)

    assert client.logged == []


def test_log_drift_metrics_logs_three_metrics_when_sufficient():
    client = _FakeMlflowClient()

    log_drift_metrics(_SUFFICIENT_STATUS, "run123", client=client)

    logged_keys = {key for _, key, _ in client.logged}
    assert logged_keys == {
        "drift_output_ratio_to_threshold",
        "drift_input_flagged_count",
        "drift_avg_score_recent",
    }
    assert ("run123", "drift_input_flagged_count", 1) in client.logged
    assert ("run123", "drift_output_ratio_to_threshold", 1.2) in client.logged
    assert ("run123", "drift_avg_score_recent", 0.9) in client.logged


def test_log_drift_metrics_swallows_exceptions():
    client = _FakeMlflowClient(raise_on_call=True)

    # 예외가 밖으로 안 나가야 한다 - 호출 자체가 실패 없이 끝나면 통과
    log_drift_metrics(_SUFFICIENT_STATUS, "run123", client=client)
