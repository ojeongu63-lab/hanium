import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent.parent
MLFLOW_DIR = ROOT / "data" / "mlflow"
EXPERIMENT_NAME = "cnc-lstm-ae"
REGISTERED_MODEL_NAME = "cnc-lstm-ae"
CHAMPION_ALIAS = "champion"


def configure_tracking(mlflow_dir: Path = MLFLOW_DIR) -> None:
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")

    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        artifacts_dir = mlflow_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        client.create_experiment(EXPERIMENT_NAME, artifact_location=f"file://{artifacts_dir}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def build_run_params(training_config: dict, manifest: dict) -> dict:
    split = manifest["experiment_split"]
    return {
        **training_config,
        "train_experiment_ids": json.dumps(split["train"]["experiment_ids"]),
        "eval_good_experiment_ids": json.dumps(split["eval_good"]["experiment_ids"]),
        "eval_bad_experiment_ids": json.dumps(split["eval_bad"]["experiment_ids"]),
    }


def build_run_metrics(thresholds: dict, results: dict) -> dict:
    metrics = {}
    for method in ["mean", "max", "p95"]:
        metrics[f"{method}_threshold"] = thresholds[method]
        r = results[method]
        metrics[f"{method}_precision"] = r["precision"]
        metrics[f"{method}_recall"] = r["recall"]
        metrics[f"{method}_tp"] = r["tp"]
        metrics[f"{method}_fp"] = r["fp"]
        metrics[f"{method}_fn"] = r["fn"]
        metrics[f"{method}_tn"] = r["tn"]
    return metrics
