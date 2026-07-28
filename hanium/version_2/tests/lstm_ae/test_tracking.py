import json

import mlflow
import mlflow.pytorch
import torch.nn as nn

from lstm_ae.tracking import (
    EXPERIMENT_NAME,
    build_run_metrics,
    build_run_params,
    configure_tracking,
    promote_to_champion,
)


def test_configure_tracking_creates_db_and_experiment(tmp_path):
    configure_tracking(tmp_path)

    assert (tmp_path / "mlflow.db").exists()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None


def test_configure_tracking_is_idempotent(tmp_path):
    configure_tracking(tmp_path)
    configure_tracking(tmp_path)

    client = mlflow.tracking.MlflowClient()
    matches = [e for e in client.search_experiments() if e.name == EXPERIMENT_NAME]
    assert len(matches) == 1


def test_build_run_params_flattens_config_and_split():
    training_config = {"window_size": 20, "hidden_size": 64}
    manifest = {
        "experiment_split": {
            "train": {"experiment_ids": [1, 2, 3]},
            "eval_good": {"experiment_ids": [12, 18]},
            "eval_bad": {"experiment_ids": [4, 5]},
        }
    }

    params = build_run_params(training_config, manifest)

    assert params["window_size"] == 20
    assert params["hidden_size"] == 64
    assert json.loads(params["train_experiment_ids"]) == [1, 2, 3]
    assert json.loads(params["eval_good_experiment_ids"]) == [12, 18]
    assert json.loads(params["eval_bad_experiment_ids"]) == [4, 5]


def test_build_run_metrics_flattens_thresholds_and_results():
    thresholds = {"mean": 0.85, "max": 5.0, "p95": 2.8}
    results = {
        "mean": {"precision": 0.9, "recall": 0.9, "tp": 10, "fp": 1, "fn": 1, "tn": 2},
        "max": {"precision": 0.8, "recall": 0.9, "tp": 10, "fp": 2, "fn": 1, "tn": 1},
        "p95": {"precision": 0.9, "recall": 0.8, "tp": 9, "fp": 1, "fn": 2, "tn": 2},
    }

    metrics = build_run_metrics(thresholds, results)

    assert metrics["mean_threshold"] == 0.85
    assert metrics["mean_precision"] == 0.9
    assert metrics["mean_tp"] == 10
    assert metrics["max_fp"] == 2
    assert metrics["p95_tn"] == 2
    assert len(metrics) == 3 * 7


def test_promote_to_champion_sets_alias(tmp_path):
    configure_tracking(tmp_path)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, x):
            return self.lin(x)

    with mlflow.start_run():
        result = mlflow.pytorch.log_model(
            Tiny(),
            artifact_path="model",
            registered_model_name="cnc-lstm-ae",
            serialization_format="pickle",
        )

    promote_to_champion(result.registered_model_version, tmp_path)

    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias("cnc-lstm-ae", "champion")
    assert mv.version == result.registered_model_version
