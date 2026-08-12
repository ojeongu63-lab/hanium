# CNC 희귀 정상 샘플 증강 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `feedrate=20` 정상 샘플(exp2, exp22)에 소규모 변주를 만들어 학습 데이터에 추가하고, 정보 누수 없는 LOOCV로 증강 전/후를 비교해 실제 효과(오탐 감소·폴드 간 편차 감소)가 있는지 검증한다.

**Architecture:** 데모 시나리오 생성(`synthetic/generate_synthetic.py`)과 같은 가우시안 jittering 메커니즘을 재사용해 exp2/exp22 각 3개씩 변주를 만들고(Part A), 기존 LOOCV 스크립트(`loocv/run_loocv.py`)와 같은 구조를 재사용하되 "held-out 자신의 증강판은 제외"하는 로직만 추가한 새 스크립트로 재검증한다(Part B).

**Tech Stack:** Python (uv run), pandas, numpy — 기존 의존성 그대로, 새 패키지 없음.

## Global Constraints

- 기존 `src/`, `loocv/run_loocv.py`, `synthetic/generate_synthetic.py`는 전혀 수정하지 않는다.
- 증강 진폭은 0.05(전체 41개 `FEATURE_COLUMNS`, 피처 자체 표준편차 대비) — 데모 정상변형에서 이미 "good" 유지가 검증된 값.
- 새 실험 ID: exp2 파생 `[2001,2002,2003]`(시드 101/102/103), exp22 파생 `[2201,2202,2203]`(시드 221/222/223).
- LOOCV 폴드 구성 시 held-out 실험 **자신의** 증강판은 그 폴드에서 제외한다(정보 누수 방지) — 다른 실험의 증강판은 포함한다.
- `bad_side_stability`에 표준편차(`std`)를 새로 계산해 추가한다.
- `version_2/augmentation/combined_dataset/`, `combined_index.csv`, `loocv/augmented_folds/`는 재생성 가능한 산출물이라 `.gitignore` 대상. `loocv/summary_augmented.json`, `summary_augmented.csv`는 결과 증거 자료라 git 포함.
- 정식 pytest 단위테스트는 만들지 않는다 — `loocv`/`synthetic`과 동일한 관례(런타임 assertion으로 검증).

---

## File Structure

- Create: `version_2/augmentation/generate_augmented.py`
- Create: `version_2/augmentation/.gitignore` (`combined_dataset/`, `combined_index.csv` 제외)
- Create: `version_2/loocv/run_loocv_augmented.py`
- Create: `version_2/loocv/.gitignore`에 `augmented_folds/` 추가(기존 `.gitignore` 수정)

## Task 1: `generate_augmented.py` 작성 + 실행 + 검증

**Files:**
- Create: `version_2/augmentation/generate_augmented.py`
- Create: `version_2/augmentation/.gitignore`

**Interfaces:**
- Consumes: `preprocessing.columns.FEATURE_COLUMNS`
- Produces: `version_2/augmentation/combined_dataset/experiment_{01..25,2001..2003,2201..2203}.csv`,
  `version_2/augmentation/combined_index.csv`(31행)

- [ ] **Step 1: `.gitignore` 작성**

`version_2/augmentation/.gitignore`:
```
combined_dataset/
combined_index.csv
```

- [ ] **Step 2: `generate_augmented.py` 작성**

```python
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
```

- [ ] **Step 3: 문법 확인**

Run: `cd version_2 && uv run python -m py_compile augmentation/generate_augmented.py`
Expected: 에러 없이 종료

- [ ] **Step 4: 실행**

Run: `cd version_2 && uv run python augmentation/generate_augmented.py`
Expected: `원본 25개 복사 + 증강 6개 생성`, assertion 에러 없음

- [ ] **Step 5: 산출물 확인**

Run:
```bash
cd version_2
ls augmentation/combined_dataset/ | wc -l   # 31이어야 함
python3 -c "
import pandas as pd
idx = pd.read_csv('augmentation/combined_index.csv')
assert len(idx) == 31, len(idx)
assert set(idx['No']) >= {2001,2002,2003,2201,2202,2203}
print('OK:', len(idx), '행')
"
```
Expected: `31`, `OK: 31 행`

- [ ] **Step 6: Commit**

```bash
git add version_2/augmentation/generate_augmented.py version_2/augmentation/.gitignore
git commit -m "Add rare-sample augmentation generator for exp2/exp22"
```

## Task 2: `run_loocv_augmented.py` 작성 + 실행 + 검증

**Files:**
- Create: `version_2/loocv/run_loocv_augmented.py`
- Modify: `version_2/loocv/.gitignore`

**Interfaces:**
- Consumes: Task 1의 `version_2/augmentation/combined_dataset/`, `combined_index.csv`;
  `preprocessing.pipeline.run_pipeline()`, `lstm_ae.pipeline.run_lstm_pipeline()`(기존 loocv와 동일 시그니처)
- Produces: `version_2/loocv/summary_augmented.json`, `summary_augmented.csv`

- [ ] **Step 1: `.gitignore`에 `augmented_folds/` 추가**

`version_2/loocv/.gitignore`에 `folds/` 다음 줄에 추가:
```
folds/
augmented_folds/
```

- [ ] **Step 2: `run_loocv_augmented.py` 작성**

```python
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lstm_ae.pipeline import run_lstm_pipeline
from preprocessing.columns import FEATURE_COLUMNS
from preprocessing.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
AUGMENTATION_DIR = ROOT / "augmentation"
INDEX_PATH = AUGMENTATION_DIR / "combined_index.csv"
DATASET_DIR = AUGMENTATION_DIR / "combined_dataset"
LOOCV_DIR = Path(__file__).resolve().parent

GOOD_EXPERIMENT_IDS = [1, 2, 3, 11, 12, 13, 14, 15, 17, 18, 22]
BAD_EXPERIMENT_IDS = [4, 5, 6, 7, 8, 9, 10, 16, 20, 21, 23]
METHODS = ["mean", "max", "p95"]

AUGMENTED_VARIANTS = {
    2: [2001, 2002, 2003],
    22: [2201, 2202, 2203],
}


def variants_for(experiment_ids: list[int]) -> list[int]:
    result = []
    for exp_id in experiment_ids:
        result.extend(AUGMENTED_VARIANTS.get(exp_id, []))
    return result


def run_fold(held_out: int) -> dict:
    base_train_ids = [i for i in GOOD_EXPERIMENT_IDS if i != held_out]
    fold_train_ids = base_train_ids + variants_for(base_train_ids)

    expected_variants = sum(
        len(v) for k, v in AUGMENTED_VARIANTS.items() if k != held_out
    )
    assert len(fold_train_ids) == 10 + expected_variants, (
        f"fold {held_out}: train ids != {10 + expected_variants} ({len(fold_train_ids)})"
    )

    fold_dir = LOOCV_DIR / "augmented_folds" / f"fold_{held_out:02d}"
    processed_dir = fold_dir / "processed"
    model_dir = fold_dir / "model"

    run_pipeline(
        experiment_index_path=str(INDEX_PATH),
        experiment_dir=str(DATASET_DIR),
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
        mean_recall = sum(recalls) / len(recalls)
        variance = sum((r - mean_recall) ** 2 for r in recalls) / len(recalls)
        bad_side[method] = {
            "recall_per_fold": recalls,
            "min": min(recalls),
            "max": max(recalls),
            "mean": mean_recall,
            "std": variance ** 0.5,
        }

    fixed_split_path = ROOT / "data" / "model" / "evaluation_report.json"
    fixed_split_result = (
        json.loads(fixed_split_path.read_text()) if fixed_split_path.exists() else None
    )
    original_loocv_path = LOOCV_DIR / "summary.json"
    original_loocv_result = (
        json.loads(original_loocv_path.read_text()) if original_loocv_path.exists() else None
    )

    return {
        "good_side_loocv": {
            "note": (
                "정상 실험 11개, 각각 정확히 1번씩 완전 홀드아웃 평가 (증강 반영, "
                "held-out 자신의 증강판은 그 폴드에서 제외해 정보 누수 방지)"
            ),
            **good_side,
        },
        "bad_side_stability": {
            "note": (
                "불량 실험 11개는 폴드마다 매번 재평가됨(학습 구성이 다른 모델로) - "
                "독립 표본 아님, recall이 학습 구성에 얼마나 민감한지 보는 안정성 체크. "
                "std가 작을수록 폴드 간 편차가 작다는 뜻(신뢰도 지표)"
            ),
            **bad_side,
        },
        "comparison_to_fixed_split": {"fixed_split_result": fixed_split_result},
        "comparison_to_original_loocv": {"original_loocv_result": original_loocv_result},
    }


def write_summary_csv(fold_results: list[dict]) -> None:
    csv_path = LOOCV_DIR / "summary_augmented.csv"
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
    json_path = LOOCV_DIR / "summary_augmented.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"저장: {json_path}")

    print()
    print("=== good_side_loocv (mean) ===")
    print(json.dumps(summary["good_side_loocv"]["mean"], indent=2, ensure_ascii=False))
    print()
    print("=== bad_side_stability (mean) ===")
    print(json.dumps(summary["bad_side_stability"]["mean"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 문법 확인**

Run: `cd version_2 && uv run python -m py_compile loocv/run_loocv_augmented.py`
Expected: 에러 없이 종료

- [ ] **Step 4: 1개 폴드로 스모크 테스트**

Run:
```bash
cd version_2
who && top -bn1 | head -6
nice -n 19 uv run python -c "
import sys; sys.path.insert(0, 'loocv')
from run_loocv_augmented import run_fold
result = run_fold(2)
print(result['held_out_experiment_id'])
print(result['results']['mean'])
"
```
Expected: assertion 없이 통과, `held_out=2` 폴드가 13개(10 real + exp22 변주 3개)로
학습됐다는 게 스크립트 내부 assert로 이미 확인됨. `{'precision':..., 'recall':..., 'tp':..., 'fp':..., 'fn':..., 'tn':...}` 출력.

- [ ] **Step 5: 스모크 테스트 산출물 정리 후 전체 11개 폴드 실행**

Run:
```bash
cd version_2
rm -rf loocv/augmented_folds/fold_02
who && top -bn1 | head -6
nice -n 19 uv run python loocv/run_loocv_augmented.py
```
Expected: 11개 폴드 로그 전부 출력, assertion 에러 없음, 마지막에
`good_side_loocv (mean)`과 `bad_side_stability (mean)` 요약 출력(수 분 소요 예상 —
폴드당 학습 데이터가 기존보다 커져서 기존 LOOCV보다 조금 더 걸릴 수 있음).

- [ ] **Step 6: 산출물 검증**

Run:
```bash
cd version_2
python3 -c "import csv; rows=list(csv.DictReader(open('loocv/summary_augmented.csv'))); assert len(rows)==11, len(rows); print('OK, rows =', len(rows))"
python3 -c "
import json
s = json.load(open('loocv/summary_augmented.json'))
g = s['good_side_loocv']['mean']
b = s['bad_side_stability']['mean']
assert g['n'] == 11
assert g['correct_tn'] + g['misclassified_fp'] == 11
assert 'std' in b
print('good_side_loocv (mean):', g)
print('bad_side_stability (mean) std:', b['std'])
"
```
Expected: 두 체크 모두 통과 출력.

- [ ] **Step 7: Commit**

```bash
git add version_2/loocv/run_loocv_augmented.py version_2/loocv/.gitignore \
        version_2/loocv/summary_augmented.json version_2/loocv/summary_augmented.csv
git commit -m "Run augmented LOOCV, compare rare-sample augmentation effect"
```

## Task 3: 증강 전/후 비교 분석 + 보고

**Files:** 없음 (기존 `loocv/summary.json` vs `loocv/summary_augmented.json` 비교)

- [ ] **Step 1: 비교 스크립트로 4가지 지표 산출**

Run:
```bash
cd version_2
python3 -c "
import json
before = json.load(open('loocv/summary.json'))
after = json.load(open('loocv/summary_augmented.json'))

for method in ['mean', 'max', 'p95']:
    b, a = before['good_side_loocv'][method], after['good_side_loocv'][method]
    print(f'[{method}] 오탐(misclassified_fp): {b[\"misclassified_fp\"]} -> {a[\"misclassified_fp\"]} (fp_ids: {b[\"fp_experiment_ids\"]} -> {a[\"fp_experiment_ids\"]})')

for method in ['mean', 'max', 'p95']:
    b, a = before['bad_side_stability'][method], after['bad_side_stability'][method]
    print(f'[{method}] recall 평균: {b[\"mean\"]:.3f} -> {a[\"mean\"]:.3f}, std: {a[\"std\"]:.3f}')
"
```
Expected: 실행 결과 그대로(가정하지 않고) 사용자에게 보고할 4가지 숫자 확보:
mean 방식 오탐 개수 변화, fp_experiment_ids 변화, recall 평균 변화, std(신규 지표).

- [ ] **Step 2: 결과를 사용자에게 있는 그대로 보고**

오탐이 줄었으면 그 폴드(2 또는 22)가 어느 쪽인지 구체적으로 보고. 안 줄었거나
다른 폴드에 부작용이 생겼으면 그것도 숨기지 않고 보고. std가 줄었는지도 함께
보고(신뢰도 지표). 다음 서브프로젝트(앙상블) 착수 여부는 이 결과를 보고 사용자가
결정.

---

## Self-Review 완료 사항

- 스펙 커버리지: Part A(진폭 0.05, 새 ID, combined_index/dataset)는 Task 1, Part B(정보
  누수 방지 fold 구성, std 계산)는 Task 2, "신뢰도 관점 비교"는 Task 3에 매핑됨.
- 플레이스홀더 없음: 전 태스크 실행 가능한 완성 코드.
- 타입/시그니처 일관성: `run_fold()`가 Task 2 Step 4(스모크 테스트)에서 그대로
  재사용하는 반환 키(`held_out_experiment_id`, `results`)가 정의와 동일.
  `AUGMENTED_VARIANTS` 딕셔너리 키(2, 22)가 Task 1의 `VARIANTS` 딕셔너리 키(2, 22)와
  일치.
