import json
from pathlib import Path

from lstm_ae.pipeline import run_lstm_pipeline
from preprocessing.columns import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    summary = run_lstm_pipeline(
        train_csv_path=str(ROOT / "data" / "processed" / "train.csv"),
        eval_csv_path=str(ROOT / "data" / "processed" / "eval.csv"),
        feature_columns=FEATURE_COLUMNS,
        output_dir=str(ROOT / "data" / "model"),
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
