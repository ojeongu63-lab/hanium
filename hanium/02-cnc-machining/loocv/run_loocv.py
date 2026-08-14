import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lstm_ae.pipeline import run_lstm_pipeline
from preprocessing.columns import FEATURE_COLUMNS
from preprocessing.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209"
LOOCV_DIR = Path(__file__).resolve().parent

GOOD_EXPERIMENT_IDS = [1, 2, 3, 11, 12, 13, 14, 15, 17, 18, 22]
BAD_EXPERIMENT_IDS = [4, 5, 6, 7, 8, 9, 10, 16, 20, 21, 23]
METHODS = ["mean", "max", "p95"]


def run_fold(held_out: int) -> dict:
    fold_train_ids = [i for i in GOOD_EXPERIMENT_IDS if i != held_out]
    assert len(fold_train_ids) == 10, f"fold {held_out}: train ids != 10 ({len(fold_train_ids)})"

    fold_dir = LOOCV_DIR / "folds" / f"fold_{held_out:02d}"
    processed_dir = fold_dir / "processed"
    model_dir = fold_dir / "model"

    run_pipeline(
        experiment_index_path=str(DATASET_DIR / "train.csv"),
        experiment_dir=str(DATASET_DIR / "CNC Virtual Data set _v2"),
        output_dir=str(processed_dir),
        train_experiment_ids=fold_train_ids,
        eval_good_experiment_ids=[held_out],
        eval_bad_experiment_ids=BAD_EXPERIMENT_IDS,
    )

    summary = run_lstm_pipeline(
        train_csv_path=str(processed_dir / "train.csv"),
        eval_csv_path=str(processed_dir / "eval.csv"),
        feature_columns=FEATURE_COLUMNS,
        output_dir=str(model_dir),
        window_size=20,
        hidden_size=64,
        latent_dim=16,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        random_seed=42,
        threshold_percentile=95.0,
    )

    results = summary["results"]
    for method in METHODS:
        r = results[method]
        n_good = r["tn"] + r["fp"]
        n_bad = r["tp"] + r["fn"]
        assert n_good == 1, f"fold {held_out}/{method}: n_good != 1 ({n_good})"
        assert n_bad == 11, f"fold {held_out}/{method}: n_bad != 11 ({n_bad})"

    return {"held_out_experiment_id": held_out, "results": results}


def build_summary(fold_results: list[dict]) -> dict:
    good_side = {}
    bad_side = {}
    for method in METHODS:
        correct_tn = sum(1 for fr in fold_results if fr["results"][method]["tn"] == 1)
        misclassified_fp = sum(1 for fr in fold_results if fr["results"][method]["fp"] == 1)
        assert correct_tn + misclassified_fp == 11, (
            f"{method}: correct_tn+misclassified_fp != 11 ({correct_tn}+{misclassified_fp})"
        )
        fp_ids = [
            fr["held_out_experiment_id"]
            for fr in fold_results
            if fr["results"][method]["fp"] == 1
        ]
        good_side[method] = {
            "n": 11,
            "correct_tn": correct_tn,
            "misclassified_fp": misclassified_fp,
            "fp_experiment_ids": fp_ids,
        }

        recalls = [fr["results"][method]["recall"] for fr in fold_results]
        bad_side[method] = {
            "recall_per_fold": recalls,
            "min": min(recalls),
            "max": max(recalls),
            "mean": sum(recalls) / len(recalls),
        }

    fixed_split_path = ROOT / "data" / "model" / "evaluation_report.json"
    fixed_split_result = (
        json.loads(fixed_split_path.read_text()) if fixed_split_path.exists() else None
    )

    return {
        "good_side_loocv": {
            "note": "정상 실험 11개, 각각 정확히 1번씩 완전 홀드아웃 평가 (진짜 LOOCV)",
            **good_side,
        },
        "bad_side_stability": {
            "note": (
                "불량 실험 11개는 폴드마다 매번 재평가됨(학습 구성이 다른 모델로) - "
                "독립 표본 아님, recall이 학습 구성에 얼마나 민감한지 보는 안정성 체크"
            ),
            **bad_side,
        },
        "comparison_to_fixed_split": {"fixed_split_result": fixed_split_result},
    }


def write_summary_csv(fold_results: list[dict]) -> None:
    csv_path = LOOCV_DIR / "summary.csv"
    fieldnames = ["held_out_experiment_id"]
    for method in METHODS:
        fieldnames += [
            f"{method}_correctly_classified",
            f"{method}_bad_recall",
            f"{method}_precision",
        ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fr in fold_results:
            row = {"held_out_experiment_id": fr["held_out_experiment_id"]}
            for method in METHODS:
                r = fr["results"][method]
                row[f"{method}_correctly_classified"] = r["tn"] == 1
                row[f"{method}_bad_recall"] = r["recall"]
                row[f"{method}_precision"] = r["precision"]
            writer.writerow(row)
    print(f"저장: {csv_path}")


def main() -> None:
    fold_results = []
    for held_out in GOOD_EXPERIMENT_IDS:
        print(f"=== fold: held_out={held_out} ===", flush=True)
        fold_results.append(run_fold(held_out))

    assert len(fold_results) == 11, f"expected 11 folds, got {len(fold_results)}"

    write_summary_csv(fold_results)

    summary = build_summary(fold_results)
    json_path = LOOCV_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"저장: {json_path}")

    print()
    print("=== good_side_loocv (mean) ===")
    print(json.dumps(summary["good_side_loocv"]["mean"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
