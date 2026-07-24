import json
from pathlib import Path

import pandas as pd

from .cleaning import normalize_machining_process
from .columns import (
    DEAD_SENSOR_COLUMNS,
    FEATURE_COLUMNS,
    METADATA_EXCLUDED_COLUMNS,
    select_features,
)
from .labels import add_labels
from .manifest import build_manifest
from .scaling import fit_scaler, scaler_to_dict, transform_features

EVAL_METADATA_COLUMNS = [
    "experiment_id",
    "label",
    "tool_condition",
    "feedrate",
    "clamp_pressure",
    "material",
    "Machining_Process",
    "M_sequence_number",
    "M_CURRENT_PROGRAM_NUMBER",
]


def _load_experiment(experiment_dir: str, experiment_id: int) -> pd.DataFrame:
    path = Path(experiment_dir) / f"experiment_{experiment_id:02d}.csv"
    df = pd.read_csv(path)
    df = normalize_machining_process(df)
    df["experiment_id"] = experiment_id
    return df


def run_pipeline(
    experiment_index_path: str,
    experiment_dir: str,
    output_dir: str,
    train_experiment_ids: list[int],
    eval_good_experiment_ids: list[int],
    eval_bad_experiment_ids: list[int],
) -> dict:
    index = pd.read_csv(experiment_index_path)
    index = add_labels(index)
    index_by_id = index.set_index("No")

    eval_experiment_ids = set(eval_good_experiment_ids) | set(eval_bad_experiment_ids)

    train_frames = []
    eval_frames = []
    for experiment_id in train_experiment_ids:
        train_frames.append(_load_experiment(experiment_dir, experiment_id))
    for experiment_id in sorted(eval_experiment_ids):
        ts = _load_experiment(experiment_dir, experiment_id)
        meta = index_by_id.loc[experiment_id]
        ts["label"] = int(meta["label"])
        ts["tool_condition"] = meta["tool_condition"]
        ts["feedrate"] = meta["feedrate"]
        ts["clamp_pressure"] = meta["clamp_pressure"]
        ts["material"] = meta["material"]
        eval_frames.append(ts)

    train_raw = pd.concat(train_frames, ignore_index=True)
    eval_raw = pd.concat(eval_frames, ignore_index=True)

    scaler = fit_scaler(train_raw, FEATURE_COLUMNS)
    train_scaled = transform_features(train_raw, FEATURE_COLUMNS, scaler)
    eval_scaled = transform_features(eval_raw, FEATURE_COLUMNS, scaler)

    train_out = pd.concat(
        [select_features(train_scaled), train_scaled[["experiment_id"]]], axis=1
    )
    eval_out = pd.concat(
        [select_features(eval_scaled), eval_scaled[EVAL_METADATA_COLUMNS]], axis=1
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(out_dir / "train.csv", index=False)
    eval_out.to_csv(out_dir / "eval.csv", index=False)

    scaler_dict = scaler_to_dict(scaler, FEATURE_COLUMNS)
    (out_dir / "scaler.json").write_text(json.dumps(scaler_dict, indent=2, ensure_ascii=False))

    eval_good_rows = int((eval_out["label"] == 0).sum())
    eval_bad_rows = int((eval_out["label"] == 1).sum())

    manifest = build_manifest(
        total_rows=len(train_out) + len(eval_out),
        train_rows=len(train_out),
        eval_rows=len(eval_out),
        eval_good_rows=eval_good_rows,
        eval_bad_rows=eval_bad_rows,
        train_experiment_ids=list(train_experiment_ids),
        eval_good_experiment_ids=list(eval_good_experiment_ids),
        eval_bad_experiment_ids=list(eval_bad_experiment_ids),
        feature_columns=FEATURE_COLUMNS,
        dead_sensor_columns=DEAD_SENSOR_COLUMNS,
        metadata_excluded_columns=METADATA_EXCLUDED_COLUMNS,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
