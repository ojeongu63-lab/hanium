import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing.columns import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = (
    ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
)
INDEX_PATH = ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "train.csv"
OUT_DIR = Path(__file__).resolve().parent / "combined_dataset"
COMBINED_INDEX_PATH = Path(__file__).resolve().parent / "combined_index.csv"
AMPLITUDE = 0.05

# parent_id -> [(new_id, seed), ...]
VARIANTS = {
    2: [(2001, 101), (2002, 102), (2003, 103)],
    22: [(2201, 221), (2202, 222), (2203, 223)],
}


def jitter(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    for col in FEATURE_COLUMNS:
        std = df[col].std()
        df[col] = df[col] + rng.normal(0, std * AMPLITUDE, size=len(df))
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original_files = sorted(DATASET_DIR.glob("experiment_*.csv"))
    assert len(original_files) == 25, f"원본 실험 CSV가 25개가 아님: {len(original_files)}"
    for src in original_files:
        shutil.copy(src, OUT_DIR / src.name)

    index = pd.read_csv(INDEX_PATH)
    new_rows = []
    for parent_id, variants in VARIANTS.items():
        parent_path = DATASET_DIR / f"experiment_{parent_id:02d}.csv"
        parent_df = pd.read_csv(parent_path)
        parent_row = index[index["No"] == parent_id].iloc[0]
        meta_cols = [c for c in parent_df.columns if c not in FEATURE_COLUMNS]

        for new_id, seed in variants:
            augmented_df = jitter(parent_df, seed)
            assert len(augmented_df) == len(parent_df), (
                f"증강본 행 수가 원본과 다름: {new_id}"
            )
            assert (augmented_df[meta_cols] == parent_df[meta_cols]).all().all(), (
                f"메타데이터 컬럼이 원본과 달라짐: {new_id}"
            )
            augmented_df.to_csv(OUT_DIR / f"experiment_{new_id}.csv", index=False)

            new_row = parent_row.copy()
            new_row["No"] = new_id
            new_rows.append(new_row)

    combined_index = pd.concat([index, pd.DataFrame(new_rows)], ignore_index=True)
    assert len(combined_index) == 25 + 6, f"combined_index 행 수 이상: {len(combined_index)}"
    combined_index.to_csv(COMBINED_INDEX_PATH, index=False)

    total_variants = sum(len(v) for v in VARIANTS.values())
    print(f"원본 {len(original_files)}개 복사 + 증강 {total_variants}개 생성")
    print(f"저장: {OUT_DIR}, {COMBINED_INDEX_PATH}")


if __name__ == "__main__":
    main()
