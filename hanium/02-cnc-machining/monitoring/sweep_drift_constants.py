"""Day 40 시점의 변형 폭이 실측 대역 안에 들어오게 상수를 정한다.

목표 score/threshold — temperature: 1.5~2.0, tool_wear: 3.0.
실측 근거는 synthetic/real_anomaly_reference.json (GOOD 0.43~1.30, BAD 1.00~3.79).

과거에 진폭 상한이 없어 z=29790 같은 비현실적 값을 만든 전례가 있어
(2026-08-12-cnc-realistic-synthetic-data-design.md), 대역을 벗어나지 않는 것이
이 스윕의 목적이다.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "monitoring"))

import simulate_timeline as st  # noqa: E402
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS  # noqa: E402
from serving.app import load_model_state  # noqa: E402
from serving.inference import predict_experiment  # noqa: E402

GRID = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]


def ratio_for(df: pd.DataFrame, state) -> float:
    result = predict_experiment(
        df,
        state.model,
        FEATURE_COLUMNS,
        state.scaler_dict,
        state.window_size,
        state.thresholds["mean"],
        "mean",
        state.feature_baseline,
        SETUP_CONSTANT_COLUMNS,
    )
    return result["score"] / state.thresholds["mean"]


def main() -> None:
    state = load_model_state()
    base = pd.read_csv(st.DATASET_DIR / "experiment_01.csv")
    print(f"champion version={state.model_version} threshold={state.thresholds['mean']:.4f}")
    print(f"기준(무변형) ratio={ratio_for(base, state):.2f}\n")

    print("=== temperature (POS_DRIFT = CUR_DRIFT = v) ===")
    for v in GRID:
        st.POS_DRIFT, st.CUR_DRIFT = v, v
        print(f"  v={v:<5} ratio={ratio_for(st.apply_temperature(base, 1.0), state):.2f}")
    st.POS_DRIFT, st.CUR_DRIFT = 0.0, 0.0

    print("=== tool_wear (WEAR_RATE = v) ===")
    for v in GRID:
        st.WEAR_RATE = v
        print(f"  v={v:<5} ratio={ratio_for(st.apply_tool_wear(base, 1.0), state):.2f}")

    print("=== fixture_loosening (VIBRATION_RATE = v) ===")
    for v in GRID:
        st.VIBRATION_RATE = v
        print(f"  v={v:<5} ratio={ratio_for(st.apply_fixture_loosening(base, 1.0), state):.2f}")


if __name__ == "__main__":
    main()
