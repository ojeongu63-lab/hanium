# CNC 정상 실험 LOOCV 검증 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 정상 실험 11개에 대해 leave-one-out 교차검증을 돌려, 지금의 8/3 고정 분할 결과(precision=0.909, recall=0.909)가 대표값인지, exp22(worn×feedrate20) 오탐이 우연인지 데이터의 근본 한계인지를 통계적으로 확정한다.

**Architecture:** 기존 `preprocessing.pipeline.run_pipeline()`과 `lstm_ae.pipeline.run_lstm_pipeline()`을 수정 없이 재사용해, 정상 11개를 1개씩 held-out하며 11번 반복 호출하는 오케스트레이션 스크립트 하나만 새로 작성한다. 결과는 `02-cnc-machining/loocv/`에 기존 산출물과 완전히 분리해서 저장한다.

**Tech Stack:** Python (uv run), pandas, PyTorch(CPU) — 전부 기존 `02-cnc-machining` 의존성 그대로, 새 패키지 추가 없음.

## Global Constraints

- 기존 `src/` 코드는 전혀 수정하지 않는다 (스펙 확인: 두 파이프라인 함수 모두 실험 ID 리스트/CSV 경로를 인자로 받아 수정 없이 재사용 가능).
- 학습 설정은 현재 champion 모델과 동일하게 고정: `window_size=20, hidden_size=64, latent_dim=16, epochs=50, batch_size=64, learning_rate=1e-3, random_seed=42, threshold_percentile=95.0`.
- MLflow에 등록하지 않는다 — 진단용 일회성 실행, champion 승격 로직과 무관.
- `experiment_index_path`/`experiment_dir`: `data/dataset/CNC 비식별화 원본데이터_1209/train.csv`, `data/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2/` (기존 `scripts/run_preprocessing.py`와 동일 경로).
- 정식 pytest 단위테스트는 만들지 않는다 — 스펙에서 이미 결정: 일회성 진단 스크립트는 이 프로젝트의 기존 관례(`baseline_isolation_forest.py` 등)를 따라 런타임 assertion으로 검증한다.
- 11번의 전체 학습을 순차 실행하므로, 실행 전 `who`/`top`으로 서버 상태를 확인하고 `nice -n 19`로 실행한다(공유 서버 예의).

---

## File Structure

- Create: `02-cnc-machining/loocv/run_loocv.py` — 오케스트레이션 스크립트 (fold 반복 실행 + 집계 + summary 저장)
- Create: `02-cnc-machining/loocv/.gitignore` — `folds/`(재생성 가능한 대용량 산출물) 제외
- Generated at runtime (git 제외): `02-cnc-machining/loocv/folds/fold_<id>/processed/`, `02-cnc-machining/loocv/folds/fold_<id>/model/`
- Generated at runtime (git 포함): `02-cnc-machining/loocv/summary.csv`, `02-cnc-machining/loocv/summary.json`

## Task 1: `run_loocv.py` 작성 + 문법 확인

**Files:**
- Create: `02-cnc-machining/loocv/run_loocv.py`
- Create: `02-cnc-machining/loocv/.gitignore`

**Interfaces:**
- Consumes: `preprocessing.pipeline.run_pipeline(experiment_index_path, experiment_dir, output_dir, train_experiment_ids, eval_good_experiment_ids, eval_bad_experiment_ids) -> manifest dict`; `preprocessing.columns.FEATURE_COLUMNS`; `lstm_ae.pipeline.run_lstm_pipeline(train_csv_path, eval_csv_path, feature_columns, output_dir, window_size, hidden_size, latent_dim, epochs, batch_size, learning_rate, random_seed, threshold_percentile) -> {"model", "train_windows", "eval_windows", "final_train_loss", "thresholds", "results": {"mean"|"max"|"p95": {"precision","recall","tp","fp","fn","tn"}}}`
- Produces: `02-cnc-machining/loocv/summary.csv` (11행), `02-cnc-machining/loocv/summary.json` (`good_side_loocv`, `bad_side_stability`, `comparison_to_fixed_split` 키)

- [ ] **Step 1: `02-cnc-machining/loocv/.gitignore` 작성**

```
folds/
```

- [ ] **Step 2: `02-cnc-machining/loocv/run_loocv.py` 작성**

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
```

- [ ] **Step 3: 문법 확인 (실행 전 빠른 체크, 무거운 계산 없음)**

Run: `cd 02-cnc-machining && uv run python -m py_compile loocv/run_loocv.py`
Expected: 에러 없이 종료 (exit code 0)

- [ ] **Step 4: Commit**

```bash
git add 02-cnc-machining/loocv/run_loocv.py 02-cnc-machining/loocv/.gitignore
git commit -m "Add LOOCV validation script for CNC good experiments"
```

## Task 2: 1개 폴드로 스모크 테스트 (전체 실행 전 파이프라인 연결 확인)

**Files:** 없음 (Task 1의 `run_fold()` 함수를 인터랙티브하게 호출)

**Interfaces:**
- Consumes: Task 1의 `run_fold(held_out: int) -> dict`

- [ ] **Step 1: 폴드 하나만 돌려서 assertion과 산출물 구조를 먼저 확인**

Run:
```bash
cd 02-cnc-machining && who && top -bn1 | head -15
nice -n 19 uv run python -c "
import sys; sys.path.insert(0, 'loocv')
from run_loocv import run_fold
result = run_fold(1)
print(result['held_out_experiment_id'])
print(result['results']['mean'])
"
```
Expected: `who`/`top`으로 서버 여유 확인 후, assertion 없이 통과하고 `{'precision': ..., 'recall': ..., 'tp': ..., 'fp': ..., 'fn': ..., 'tn': ...}` 형태 출력. `02-cnc-machining/loocv/folds/fold_01/`에 `processed/`, `model/` 디렉토리가 생성됨을 `ls`로 확인.

- [ ] **Step 2: 이상 없으면 스모크 테스트로 생긴 `folds/fold_01/` 산출물은 그대로 둔다 (Task 3의 전체 실행이 fold_01을 다시 덮어쓰므로 별도 정리 불필요)**

## Task 3: 전체 11개 폴드 실행 + 결과 확인

**Files:** 없음 (Task 1 스크립트 실행)

- [ ] **Step 1: 서버 상태 재확인 후 전체 실행**

Run: `cd 02-cnc-machining && who && top -bn1 | head -15 && nice -n 19 uv run python loocv/run_loocv.py`
Expected: 11개 fold 로그(`=== fold: held_out=... ===`)가 순서대로 출력되고, 중간에 AssertionError 없이 끝까지 실행. 마지막에 `summary.csv`, `summary.json` 저장 로그와 `good_side_loocv (mean)` 요약이 출력됨.

- [ ] **Step 2: 산출물 검증**

Run:
```bash
cd 02-cnc-machining
python3 -c "import csv; rows=list(csv.DictReader(open('loocv/summary.csv'))); assert len(rows)==11, len(rows); print('OK, rows =', len(rows))"
python3 -c "
import json
s = json.load(open('loocv/summary.json'))
g = s['good_side_loocv']['mean']
assert g['n'] == 11
assert g['correct_tn'] + g['misclassified_fp'] == 11
print('good_side_loocv (mean):', g)
"
```
Expected: 두 체크 모두 통과 출력. `misclassified_fp`/`fp_experiment_ids`에 22가 포함되는지(가설 검증) 확인.

- [ ] **Step 3: Commit (summary만 — `folds/`는 .gitignore로 이미 제외됨)**

```bash
cd 02-cnc-machining && git add loocv/summary.csv loocv/summary.json
git commit -m "Run CNC good-experiment LOOCV, record fold results"
```

---

## Self-Review 완료 사항

- 스펙 커버리지: 배경/목표의 "정상 11개 각각 1번씩 홀드아웃", "불량 11개 recall 안정성", "기존 champion/MLflow 미접촉", "sanity check 4종" 전부 Task 1~3에 매핑됨.
- 플레이스홀더 없음: 코드 전체 실제 실행 가능한 완성 코드.
- 타입/시그니처 일관성: `run_fold`가 Task 2에서 그대로 재사용하는 함수명과 반환 키(`held_out_experiment_id`, `results`)가 Task 1 정의와 동일.
