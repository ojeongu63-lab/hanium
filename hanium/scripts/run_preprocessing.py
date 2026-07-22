from pathlib import Path

from preprocessing.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    manifest = run_pipeline(
        labeled_path=str(ROOT / "data" / "dataset" / "cn7_labeled.csv"),
        unlabeled_path=str(ROOT / "data" / "dataset" / "cn7_unlabeled.csv"),
        output_dir=str(ROOT / "data" / "processed"),
    )
    print(f"labeled: {manifest['labeled']}")
    print(f"unlabeled: {manifest['unlabeled']}")
    print(f"eval_label_counts: {manifest['eval_label_counts']}")


if __name__ == "__main__":
    main()
