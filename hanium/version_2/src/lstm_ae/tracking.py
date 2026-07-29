import json
import sqlite3
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent.parent
MLFLOW_DIR = ROOT / "data" / "mlflow"
EXPERIMENT_NAME = "cnc-lstm-ae"
REGISTERED_MODEL_NAME = "cnc-lstm-ae"
CHAMPION_ALIAS = "champion"

_STALE_PATH_COLUMNS = [
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("experiments", "artifact_location"),
    ("model_versions", "storage_location"),
]


def _file_uri_root(path: Path) -> str:
    """path를 file:// URI에 쓸 수 있는 POSIX 형태 절대경로 문자열로 만든다.
    Windows에서는 그냥 str(Path)를 쓰면 백슬래시가 섞여 URI가 깨지므로
    항상 슬래시 형태로 통일하고, 드라이브 문자(C:)는 앞에 '/'를 붙인다."""
    posix = path.resolve().as_posix()
    if len(posix) > 1 and posix[1] == ":":
        posix = "/" + posix
    return posix


def _repair_stale_artifact_paths(mlflow_dir: Path) -> None:
    """mlflow.db는 artifact 위치를 절대경로로 기록한다. 다른 머신에서 만든
    mlflow.db를 그대로 복사해오면 그 절대경로가 이 머신에는 없는 경로라
    모델을 못 찾는다 — 저장된 경로의 루트를 현재 mlflow_dir 기준으로 고쳐준다."""
    db_path = mlflow_dir / "mlflow.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT artifact_location FROM experiments WHERE name = ?", (EXPERIMENT_NAME,)
        )
        row = cur.fetchone()
        if row is None:
            return

        current_root = f"file://{_file_uri_root(mlflow_dir)}/artifacts"
        stored_root = row[0]
        if stored_root == current_root:
            return

        old_root = stored_root.removeprefix("file://").removesuffix("/artifacts")
        new_root = _file_uri_root(mlflow_dir)

        for table, col in _STALE_PATH_COLUMNS:
            cur.execute(f"SELECT rowid, {col} FROM {table} WHERE {col} LIKE ?", (f"%{old_root}%",))
            for rowid, val in cur.fetchall():
                cur.execute(
                    f"UPDATE {table} SET {col} = ? WHERE rowid = ?",
                    (val.replace(old_root, new_root), rowid),
                )
        conn.commit()
    finally:
        conn.close()


def configure_tracking(mlflow_dir: Path = MLFLOW_DIR) -> None:
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    _repair_stale_artifact_paths(mlflow_dir)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")

    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        artifacts_dir = mlflow_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        client.create_experiment(
            EXPERIMENT_NAME, artifact_location=f"file://{_file_uri_root(artifacts_dir)}"
        )
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


def promote_to_champion(version: str, mlflow_dir: Path = MLFLOW_DIR) -> None:
    configure_tracking(mlflow_dir)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
