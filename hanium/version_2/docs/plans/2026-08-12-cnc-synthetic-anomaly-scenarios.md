# CNC 합성 이상/정상 시나리오 생성 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 실제 정상 실험(`experiment_01.csv`) 위에 도메인 지식 기반 perturbation을 적용해, 서로 다른 원인의 합성 이상 시나리오 3개와 그 정상 변형 3개(총 6개)를 생성하고, 각각이 실제 champion 모델에 의해 의도한 대로(bad/good) 판정되는지 직접 검증한다.

**Architecture:** 3개의 순수 perturbation 함수(공구마모/이송축부하/진동)를 정의하고, 하나의 공용 자동 보정 루프(`calibrate()`)가 진폭을 조절해가며 목표 판정(bad 또는 good)이 나올 때까지 재시도한다. 기존 `serving.inference.predict_experiment()`와 `serving.app.load_model_state()`를 그대로 재사용해 실제 champion 모델로 검증한다.

**Tech Stack:** Python (uv run), pandas, numpy — 기존 `version_2` 의존성 그대로, 새 패키지 추가 없음.

## Global Constraints

- 기존 `src/` 코드는 전혀 수정하지 않는다 — `serving.inference.predict_experiment()`, `serving.app.load_model_state()`, `preprocessing.columns.FEATURE_COLUMNS`/`SETUP_CONSTANT_COLUMNS`를 그대로 재사용.
- 기준(baseline) 실험: `data/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2/experiment_01.csv`.
- 판정은 `method="mean"` 고정 (프로젝트 전체에서 쓰는 기본 방식과 동일).
- 모든 perturbation은 피처 자체 값/표준편차에 대한 **상대 비율**로 정의한다(절대 매직넘버 금지).
- 정식 pytest 단위테스트는 만들지 않는다 — 자동 보정 루프 자체가 런타임 검증 역할을 한다 (스펙에서 이미 결정, `loocv/run_loocv.py`와 동일 관례).
- 생성된 시나리오 CSV/JSON은 재사용 가능한 데모 자산이므로 `.gitignore` 대상이 아니다 — git에 커밋한다 (loocv의 `folds/`와 다른 점).

---

## File Structure

- Create: `version_2/synthetic/generate_synthetic.py` — perturbation 함수 3개 + `calibrate()` + `main()`
- Generated (git 포함): `version_2/synthetic/scenarios/*.csv`, `version_2/synthetic/scenarios/*_predict_result.json` (6개 시나리오 × 2파일 = 12개 파일)

## Task 1: `generate_synthetic.py` 작성 + 문법 확인

**Files:**
- Create: `version_2/synthetic/generate_synthetic.py`

**Interfaces:**
- Consumes: `serving.app.load_model_state() -> ModelState`(필드: `model, scaler_dict, thresholds, window_size, model_version, mlflow_run_id, feature_baseline`); `serving.inference.predict_experiment(df, model, feature_columns, scaler_dict, window_size, threshold, method, feature_baseline, exclude_from_ranking=None) -> dict`(키: `predicted_label, predicted_label_text, score, threshold, method, feature_contributions`); `preprocessing.columns.FEATURE_COLUMNS`, `preprocessing.columns.SETUP_CONSTANT_COLUMNS`
- Produces: `tool_wear(df, amplitude) -> pd.DataFrame`, `feed_overload(df, amplitude) -> pd.DataFrame`, `vibration_backlash(df, amplitude, seed=42) -> pd.DataFrame`, `calibrate(perturb_fn, base_df, state, initial_amplitude, target, step_factor, max_attempts=5) -> tuple[pd.DataFrame, dict, float, int]`(마지막 원소는 성공한 시도 횟수)

- [ ] **Step 1: `version_2/synthetic/generate_synthetic.py` 작성**

```python
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
from serving.app import load_model_state
from serving.inference import predict_experiment

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = (
    ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
)
OUT_DIR = Path(__file__).resolve().parent / "scenarios"


def tool_wear(df: pd.DataFrame, amplitude: float) -> pd.DataFrame:
    """공구마모: 스핀들/절삭축 전류·파워가 실험 시작~끝까지 선형으로 증가."""
    df = df.copy()
    cols = ["S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback", "X_OutputPower", "Y_OutputPower"]
    ramp = np.linspace(0, 1, len(df))
    multiplier = 1 + amplitude * ramp
    for col in cols:
        df[col] = df[col] * multiplier
    return df


def feed_overload(df: pd.DataFrame, amplitude: float) -> pd.DataFrame:
    """이송축 부하 급증(chip 막힘): 실험 구간 30~50%에서 X/Y 전류·파워가 스텝 증가,
    같은 구간에서 ActualVelocity가 SetVelocity 대비 처짐."""
    df = df.copy()
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    mask[int(n * 0.3) : int(n * 0.5)] = True

    for col in ["X_OutputCurrent", "X_OutputPower", "Y_OutputCurrent", "Y_OutputPower"]:
        df.loc[mask, col] = df.loc[mask, col] * (1 + amplitude)

    drop = min(amplitude * 0.1, 0.5)
    for col in ["X_ActualVelocity", "Y_ActualVelocity"]:
        df.loc[mask, col] = df.loc[mask, col] * (1 - drop)
    return df


def vibration_backlash(df: pd.DataFrame, amplitude: float, seed: int = 42) -> pd.DataFrame:
    """진동/백래쉬 증가: Set*는 그대로 두고 Actual*에 그 피처 자체 표준편차 비례
    가우시안 노이즈를 더해 추종오차의 흔들림만 키움."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    cols = [
        "X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition",
        "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity",
    ]
    for col in cols:
        std = df[col].std()
        df[col] = df[col] + rng.normal(0, std * amplitude, size=len(df))
    return df


def calibrate(
    perturb_fn,
    base_df: pd.DataFrame,
    state,
    initial_amplitude: float,
    target: str,
    step_factor: float,
    max_attempts: int = 5,
) -> tuple[pd.DataFrame, dict, float, int]:
    amplitude = initial_amplitude
    for attempt in range(1, max_attempts + 1):
        synthetic_df = perturb_fn(base_df, amplitude)
        result = predict_experiment(
            df=synthetic_df,
            model=state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=state.scaler_dict,
            window_size=state.window_size,
            threshold=state.thresholds["mean"],
            method="mean",
            feature_baseline=state.feature_baseline,
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
        )
        print(
            f"  시도 {attempt}: amplitude={amplitude:.4f} -> {result['predicted_label_text']}",
            flush=True,
        )
        if result["predicted_label_text"] == target:
            return synthetic_df, result, amplitude, attempt
        amplitude *= step_factor
    raise RuntimeError(
        f"{max_attempts}번 시도해도 '{target}'가 안 나옴 "
        f"(마지막 amplitude={amplitude:.4f}) - 시나리오 설계 재검토 필요"
    )


def save_scenario(name: str, df: pd.DataFrame, result: dict, amplitude: float, attempts: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    result_path = OUT_DIR / f"{name}_predict_result.json"
    payload = {**result, "final_amplitude": amplitude, "attempts": attempts}
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"저장: {csv_path}, {result_path}")


SCENARIOS = [
    # (name, perturb_fn, 이상_초기진폭, 정상변형_초기진폭)
    ("tool_wear", tool_wear, 1.0, 0.1),
    ("feed_overload", feed_overload, 1.0, 0.1),
    ("vibration_backlash", vibration_backlash, 0.5, 0.05),
]


def main() -> None:
    base_df = pd.read_csv(DATASET_DIR / "experiment_01.csv")
    state = load_model_state()

    for name, perturb_fn, anomaly_amp, normal_amp in SCENARIOS:
        print(f"=== {name} (이상) ===")
        df, result, amp, attempts = calibrate(
            perturb_fn, base_df, state, anomaly_amp, target="bad", step_factor=2.0
        )
        save_scenario(name, df, result, amp, attempts)

        print(f"=== {name} (정상 변형) ===")
        df, result, amp, attempts = calibrate(
            perturb_fn, base_df, state, normal_amp, target="good", step_factor=0.5
        )
        save_scenario(f"{name}_normal", df, result, amp, attempts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 문법 확인**

Run: `cd version_2 && uv run python -m py_compile synthetic/generate_synthetic.py`
Expected: 에러 없이 종료 (exit code 0)

- [ ] **Step 3: Commit**

```bash
git add version_2/synthetic/generate_synthetic.py
git commit -m "Add synthetic anomaly/normal scenario generator for CNC demo"
```

## Task 2: 시나리오 1개로 스모크 테스트 (모델 로딩 + predict_experiment 연결 확인)

**Files:** 없음 (Task 1의 `calibrate()`/`tool_wear()`를 인터랙티브하게 호출)

**Interfaces:**
- Consumes: Task 1의 `calibrate()`, `tool_wear()`, `load_model_state()`

- [ ] **Step 1: `tool_wear` 이상 시나리오 하나만 돌려서 champion 모델 로딩과 판정 루프가 실제로 동작하는지 확인**

Run:
```bash
cd version_2
uv run python -c "
import sys; sys.path.insert(0, 'synthetic')
import pandas as pd
from generate_synthetic import calibrate, tool_wear, DATASET_DIR
from serving.app import load_model_state

base_df = pd.read_csv(DATASET_DIR / 'experiment_01.csv')
state = load_model_state()
df, result, amp, attempts = calibrate(tool_wear, base_df, state, 1.0, target='bad', step_factor=2.0)
print('최종 진폭:', amp, '시도 횟수:', attempts)
print('predicted_label_text:', result['predicted_label_text'])
print('상위 3개 feature_contributions:', [c['feature'] for c in result['feature_contributions'][:3]])
"
```
Expected: `predicted_label_text: bad`, 몇 번의 시도 로그, 상위 3개 피처에 `S_OutputCurrent`/`S_OutputPower`/`S_CurrentFeedback`/`X_OutputPower`/`Y_OutputPower` 중 일부가 나옴(정확히 어떤 3개인지는 실행해봐야 확정 — Task 3에서 전체 결과로 최종 확인).

## Task 3: 전체 6개 시나리오 생성 + 결과 검증 + 커밋

**Files:** 없음 (Task 1 스크립트 실행)

- [ ] **Step 1: 전체 실행**

Run: `cd version_2 && uv run python synthetic/generate_synthetic.py`
Expected: 6개 시나리오(`tool_wear`, `tool_wear_normal`, `feed_overload`, `feed_overload_normal`, `vibration_backlash`, `vibration_backlash_normal`) 각각 5회 이내에 목표 판정(이상=bad, 정상변형=good) 도달, `저장: ...` 로그 12개(csv+json ×6).

- [ ] **Step 2: 산출물 개수 확인**

Run: `ls version_2/synthetic/scenarios/ | wc -l`
Expected: `12`

- [ ] **Step 3: 이상 시나리오 3개의 feature_contributions 상위권이 의도한 피처와 겹치는지 확인**

Run:
```bash
cd version_2
for name in tool_wear feed_overload vibration_backlash; do
  echo "=== $name ==="
  python3 -c "
import json
r = json.load(open('synthetic/scenarios/${name}_predict_result.json'))
print('top3:', [c['feature'] for c in r['feature_contributions'][:3]])
print('label:', r['predicted_label_text'], 'amplitude:', r['final_amplitude'], 'attempts:', r['attempts'])
"
done
```
Expected: `tool_wear`는 `S_OutputCurrent/S_OutputPower/S_CurrentFeedback/X_OutputPower/Y_OutputPower` 중 다수가 상위, `feed_overload`는 `X_OutputCurrent/X_OutputPower/Y_OutputCurrent/Y_OutputPower` 중 다수, `vibration_backlash`는 `X/Y/Z_ActualPosition`, `ActualVelocity` 계열이 상위. 안 겹치면 원인 파악 후 시나리오 설계 재검토(사용자에게 보고).

- [ ] **Step 4: 정상 변형 3개가 실제로 "good"인지 재확인**

Run:
```bash
cd version_2
for name in tool_wear_normal feed_overload_normal vibration_backlash_normal; do
  python3 -c "
import json
r = json.load(open('synthetic/scenarios/${name}_predict_result.json'))
assert r['predicted_label_text'] == 'good', r
print('$name OK, amplitude=' + str(r['final_amplitude']))
"
done
```
Expected: 3개 다 `OK` 출력, assertion 없이 통과.

- [ ] **Step 5: Commit**

```bash
git add version_2/synthetic/scenarios/
git commit -m "Generate CNC synthetic anomaly/normal demo scenarios"
```

---

## Self-Review 완료 사항

- 스펙 커버리지: 시나리오 3개 perturbation 정의, 자동 보정 루프(이상/정상 양방향), 산출물 구조(csv+json ×6), 검증 방법(feature_contributions 겹침 확인) 전부 Task 1~3에 매핑됨.
- 플레이스홀더 없음: 코드 전체 실제 실행 가능한 완성 코드.
- 타입/시그니처 일관성: `calibrate()`가 Task 2/3에서 그대로 재사용하는 반환값 순서(`df, result, amplitude, attempts`)와 `save_scenario()` 인자 순서가 Task 1 정의와 동일.
