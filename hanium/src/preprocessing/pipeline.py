import json
from pathlib import Path

import pandas as pd

from .columns import (
    DEAD_SENSOR_COLUMNS,
    DISABLED_SENSOR_COLUMNS,
    FEATURE_COLUMNS,
    select_features,
)
from .data_io import load_csv
from .dedup import remove_exact_duplicates
from .labels import encode_labels
from .manifest import build_manifest
from .outliers import remove_outliers
from .scaling import fit_scaler, scaler_to_dict, transform_features

CONTAMINATION = 0.01
RANDOM_STATE = 42
EVAL_METADATA_COLUMNS = ["PassOrFail", "Reason", "TimeStamp", "label"]


def run_pipeline(labeled_path: str, unlabeled_path: str, output_dir: str) -> dict:
    raw_labeled = load_csv(labeled_path)
    raw_unlabeled = load_csv(unlabeled_path)

    labeled = remove_exact_duplicates(raw_labeled)
    labeled = encode_labels(labeled)

    train_clean, removed = remove_outliers(
        raw_unlabeled, FEATURE_COLUMNS, contamination=CONTAMINATION, random_state=RANDOM_STATE
    )

    scaler = fit_scaler(train_clean, FEATURE_COLUMNS)
    train_scaled = transform_features(train_clean, FEATURE_COLUMNS, scaler)
    eval_scaled = transform_features(labeled, FEATURE_COLUMNS, scaler)

    train_out = select_features(train_scaled)
    eval_out = pd.concat(
        [select_features(eval_scaled), eval_scaled[EVAL_METADATA_COLUMNS]], axis=1
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out.to_csv(out_dir / "train.csv", index=False)
    eval_out.to_csv(out_dir / "eval.csv", index=False)
    removed.to_csv(out_dir / "removed_outliers.csv", index=False)

    scaler_dict = scaler_to_dict(scaler, FEATURE_COLUMNS)
    (out_dir / "scaler.json").write_text(json.dumps(scaler_dict, indent=2, ensure_ascii=False))

    eval_label_counts = {
        "normal": int((eval_out["label"] == 0).sum()),
        "gas": int(((eval_out["label"] == 1) & (eval_out["Reason"] == "가스")).sum()),
        "misform": int(((eval_out["label"] == 1) & (eval_out["Reason"] == "미성형")).sum()),
    }
    manifest = build_manifest(
        raw_labeled_rows=len(raw_labeled),
        labeled_rows_after_dedup=len(labeled),
        raw_unlabeled_rows=len(raw_unlabeled),
        train_rows_after_cleaning=len(train_clean),
        removed_outlier_rows=len(removed),
        contamination=CONTAMINATION,
        feature_columns=FEATURE_COLUMNS,
        dead_sensor_columns=DEAD_SENSOR_COLUMNS,
        disabled_sensor_columns=DISABLED_SENSOR_COLUMNS,
        eval_label_counts=eval_label_counts,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return manifest
