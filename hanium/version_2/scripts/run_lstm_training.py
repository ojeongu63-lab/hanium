import json
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
from mlflow.tracking import MlflowClient

from lstm_ae.pipeline import run_lstm_pipeline
from lstm_ae.tracking import (
    REGISTERED_MODEL_NAME,
    build_run_metrics,
    build_run_params,
    configure_tracking,
    log_evaluation_plots,
)
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS

ROOT = Path(__file__).resolve().parent.parent

TRAINING_CONFIG = {
    "window_size": 20,
    "hidden_size": 64,
    "latent_dim": 16,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "random_seed": 42,
    "threshold_percentile": 95.0,
}


def main() -> None:
    if len(sys.argv) > 1:
        TRAINING_CONFIG["random_seed"] = int(sys.argv[1])

    configure_tracking()
    manifest = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())

    with mlflow.start_run():
        mlflow.log_params(build_run_params(TRAINING_CONFIG, manifest))

        summary = run_lstm_pipeline(
            train_csv_path=str(ROOT / "data" / "processed" / "train.csv"),
            eval_csv_path=str(ROOT / "data" / "processed" / "eval.csv"),
            feature_columns=FEATURE_COLUMNS,
            output_dir=str(ROOT / "data" / "model"),
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
            **TRAINING_CONFIG,
        )

        mlflow.log_metrics(build_run_metrics(summary["thresholds"], summary["results"]))

        model_info = mlflow.pytorch.log_model(
            summary["model"],
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )

        experiment_scores = pd.read_csv(ROOT / "data" / "model" / "experiment_scores.csv")
        feature_error_scores = pd.read_csv(ROOT / "data" / "model" / "eval_feature_errors.csv")
        feature_baseline = json.loads(
            (ROOT / "data" / "model" / "feature_baseline.json").read_text()
        )
        timeline_errors = pd.read_csv(ROOT / "data" / "model" / "eval_timeline_errors.csv")
        reconstruction_overlay = pd.read_csv(
            ROOT / "data" / "model" / "eval_reconstruction_overlay.csv"
        )
        rankable_columns = [c for c in FEATURE_COLUMNS if c not in SETUP_CONSTANT_COLUMNS]
        log_evaluation_plots(
            MlflowClient(),
            model_info.model_id,
            summary["results"],
            summary["thresholds"],
            experiment_scores,
            feature_error_scores,
            rankable_columns,
            feature_baseline,
            timeline_errors,
            reconstruction_overlay,
        )

    print(f"train_windows: {summary['train_windows']}")
    print(f"eval_windows: {summary['eval_windows']}")
    print(f"final_train_loss: {summary['final_train_loss']:.6f}")
    print(f"thresholds: {summary['thresholds']}")
    for method in ["mean", "max", "p95"]:
        r = summary["results"][method]
        print(
            f"[{method}] precision={r['precision']:.4f} recall={r['recall']:.4f} "
            f"tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}"
        )


if __name__ == "__main__":
    main()
