from pathlib import Path

import pandas as pd

from lstm_ae.pipeline import run_lstm_pipeline
from preprocessing.columns import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    output_dir = ROOT / "data" / "model"
    summary = run_lstm_pipeline(
        train_csv_path=str(ROOT / "data" / "processed" / "train.csv"),
        eval_csv_path=str(ROOT / "data" / "processed" / "eval.csv"),
        feature_columns=FEATURE_COLUMNS,
        output_dir=str(output_dir),
    )
    print(f"train_shots: {summary['train_shots']}")
    print(f"eval_shots: {summary['eval_shots']}")
    print(f"final_train_loss: {summary['final_train_loss']:.6f}")
    print(f"threshold: {summary['threshold']:.6f}")
    print(f"precision: {summary['precision']:.4f}  recall: {summary['recall']:.4f}")
    print(f"tp={summary['tp']} fp={summary['fp']} fn={summary['fn']} tn={summary['tn']}")

    eval_errors = pd.read_csv(output_dir / "eval_reconstruction_errors.csv")
    mean_normal = eval_errors.loc[eval_errors["label"] == 0, "shot_error"].mean()
    mean_defect = eval_errors.loc[eval_errors["label"] == 1, "shot_error"].mean()
    print(f"mean shot_error (label=0, 정상): {mean_normal:.6f}")
    print(f"mean shot_error (label=1, 불량): {mean_defect:.6f}")


if __name__ == "__main__":
    main()
