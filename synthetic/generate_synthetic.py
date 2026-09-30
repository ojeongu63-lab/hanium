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
# 이상 시나리오는 "불량 판정"만으로 멈추지 않고 점수/임계값 배율이 이 값 이상이 되는 첫 진폭에서
# 멈춘다. 실제 불량 실험의 배율 대역이 1.0~3.8이라, 작은 진폭에서 1.5배씩 올리면 그 대역 안에
# 떨어진다(예전에는 진폭 1.0에서 시작해 배율 56~597배짜리 비현실적 배치가 나왔다).
MIN_BAD_RATIO = 2.0


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
        df[col] = df[col].astype(float)
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
        ratio = None
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
        ratio = result["score"] / result["threshold"]
        print(
            f"  시도 {attempt}: amplitude={amplitude:.4f} -> {result['predicted_label_text']} "
            f"(배율 {ratio:.2f})",
            flush=True,
        )
        reached = result["predicted_label_text"] == target
        if target == "bad":
            reached = reached and ratio >= MIN_BAD_RATIO
        if reached:
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
    payload = {
        **result, "final_amplitude": amplitude, "attempts": attempts,
        "final_ratio": round(result["score"] / result["threshold"], 3),
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"저장: {csv_path}, {result_path}")


SCENARIOS = [
    # (name, perturb_fn, 이상_초기진폭, 정상변형_초기진폭)
    # 이상 초기진폭은 작게 잡고 1.5배씩 올린다(MIN_BAD_RATIO 참고).
    ("tool_wear", tool_wear, 0.05, 0.1),
    ("feed_overload", feed_overload, 0.02, 0.1),
    ("vibration_backlash", vibration_backlash, 0.5, 0.05),
]


def main() -> None:
    base_df = pd.read_csv(DATASET_DIR / "experiment_01.csv")
    state = load_model_state()

    for name, perturb_fn, anomaly_amp, normal_amp in SCENARIOS:
        print(f"=== {name} (이상) ===")
        df, result, amp, attempts = calibrate(
            perturb_fn, base_df, state, anomaly_amp, target="bad", step_factor=1.5, max_attempts=12
        )
        save_scenario(name, df, result, amp, attempts)

        print(f"=== {name} (정상 변형) ===")
        df, result, amp, attempts = calibrate(
            perturb_fn, base_df, state, normal_amp, target="good", step_factor=0.5
        )
        save_scenario(f"{name}_normal", df, result, amp, attempts)


if __name__ == "__main__":
    main()
