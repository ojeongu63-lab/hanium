"""가상 운영 타임라인을 만들어 /predict 로 흘려보낸다.

보유 데이터에는 시간축이 없다(실험 25개는 서로 순서가 없는 독립 샘플).
이 스크립트는 train 실험 8개를 재료로 "날짜가 지날수록 조금씩 더 틀어진"
스트림을 만들어, 드리프트가 서서히 심해지는 상황을 재현한다.

시나리오 2종:
  temperature — 계절·온도 변화. 제품 품질은 그대로인데 센서값만 이동한다.
                재학습이 정답인 경우.
  tool_wear   — 공구 마모. 어느 시점부터 실제로 불량품이 나온다.
                재학습하면 안 되는 경우 (게이트가 막아야 한다).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from monitoring.labels import record_label  # noqa: E402
from preprocessing.split import TRAIN_EXPERIMENT_IDS  # noqa: E402

# monitoring/simulate_drift.py 와 동일한 경로 — 원본 CSV 는 두 단계 더 깊다.
DATASET_DIR = (
    ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
)
LABELS_DB = ROOT / "data" / "monitoring" / "labels.db"

TOTAL_DAYS = 40
BATCHES_PER_DAY = 5
DRIFT_START_DAY = 10          # Day 1~10 은 변형 없는 baseline 구간
LABEL_DELAY_DAYS = 7
WEAR_LABEL_FLIP_DAY = 21      # 시나리오 B에서 QC 불합격이 시작되는 날

# sweep_drift_constants.py 로 확정한 값. champion v1 (threshold 0.8566) 기준.
# 목표 Day 40 score/threshold — temperature 1.5~2.0, tool_wear 3.0
# (실측 대역: GOOD 0.43~1.30, BAD 1.00~3.79).
#
#   temperature(v)            tool_wear(v)
#   0.02 → 1.80  ← 채택       0.02 → 0.75
#   0.05 → 6.65               0.05 → 0.88
#   0.10 → 26.76              0.10 → 1.36
#   0.20 → 111.31             0.20 → 3.08  ← 채택
#   0.35 → 349.48             0.35 → 7.54
#   (이후 급격히 발산)         0.50 → 14.51
#
# 무변형 기준 ratio 는 0.72. 날짜별 진행:
#   day   10    15    20    25    30    35    40
#   temp  0.72  0.74  0.83  0.95  1.18  1.49  1.80
#   wear  0.72  0.80  1.00  1.36  1.83  2.41  3.08
#
# 모델이 재학습되면 이 상수는 낡는다. simulate_timeline 이 매일 실제 비율을
# 출력하므로 조용히 틀리지는 않는다.
POS_DRIFT = 0.02
CUR_DRIFT = 0.02
WEAR_RATE = 0.2

TEMP_POSITION_COLUMNS = ["X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition"]
TEMP_CURRENT_COLUMNS = [
    "X_OutputCurrent", "Y_OutputCurrent", "X_OutputPower", "Y_OutputPower",
]
WEAR_COLUMNS = [
    "S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback",
    "X_OutputPower", "Y_OutputPower",
]


def progress_for(day: int) -> float:
    return max(0.0, (day - DRIFT_START_DAY) / (TOTAL_DAYS - DRIFT_START_DAY))


def apply_temperature(df: pd.DataFrame, progress: float) -> pd.DataFrame:
    """온도 상승: 열변위로 Actual 위치가 지령 대비 벌어지고, 서보 권선 저항이
    올라가 같은 토크에 전류·파워가 더 든다. SetPosition 계열은 건드리지 않는다 —
    지령값은 그대로인데 실제값만 벌어지는 것이 열변위의 본질이다."""
    out = df.copy()
    for col in TEMP_POSITION_COLUMNS:
        out[col] = out[col] + out[col].std() * POS_DRIFT * progress
    for col in TEMP_CURRENT_COLUMNS:
        out[col] = out[col] * (1.0 + CUR_DRIFT * progress)
    return out


def apply_tool_wear(df: pd.DataFrame, progress: float) -> pd.DataFrame:
    """공구마모: 절삭이 진행될수록 주축 부하가 선형으로 커진다
    (synthetic/generate_synthetic.py 의 tool_wear 패턴과 동일한 램프)."""
    out = df.copy()
    ramp = 1.0 + (WEAR_RATE * progress) * (
        pd.Series(range(len(out))) / max(len(out) - 1, 1)
    )
    for col in WEAR_COLUMNS:
        out[col] = out[col] * ramp.to_numpy()
    return out


PERTURBATIONS = {"temperature": apply_temperature, "tool_wear": apply_tool_wear}


def true_label(scenario: str, day: int) -> str:
    """제품이 실제로 불량이냐. 온도는 제품 품질을 바꾸지 않는다."""
    if scenario == "temperature":
        return "good"
    return "bad" if day >= WEAR_LABEL_FLIP_DAY else "good"


def generate_batch(day: int, index: int, scenario: str) -> pd.DataFrame:
    experiment_id = TRAIN_EXPERIMENT_IDS[
        (day * BATCHES_PER_DAY + index) % len(TRAIN_EXPERIMENT_IDS)
    ]
    df = pd.read_csv(DATASET_DIR / f"experiment_{experiment_id:02d}.csv")
    return PERTURBATIONS[scenario](df, progress_for(day))


def main() -> None:
    parser = argparse.ArgumentParser(description="가상 운영 타임라인 생성 및 주입")
    parser.add_argument("scenario", choices=list(PERTURBATIONS))
    parser.add_argument("--days", type=int, default=TOTAL_DAYS)
    args = parser.parse_args()

    from fastapi.testclient import TestClient
    from serving.app import app

    from drift_worker import WorkerState, tick  # 같은 폴더

    out_dir = ROOT / "data" / "timeline" / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    state = WorkerState()

    for day in range(1, args.days + 1):
        for index in range(BATCHES_PER_DAY):
            batch_id = f"day{day:02d}_{index}"
            batch = generate_batch(day, index, args.scenario)
            csv_path = out_dir / f"{batch_id}.csv"
            batch.to_csv(csv_path, index=False)

            with csv_path.open("rb") as fh:
                response = client.post(
                    "/predict", files={"file": (csv_path.name, fh, "text/csv")}
                )
            response.raise_for_status()

            record_label(
                batch_id=batch_id,
                produced_day=day,
                arrived_day=day + LABEL_DELAY_DAYS,
                label=true_label(args.scenario, day),
                db_path=LABELS_DB,
            )

        result = tick(client, state, current_day=day, scenario=args.scenario)
        print(
            f"Day {day:02d}  score/threshold={result['ratio']:.2f}  "
            f"flagged={result['flagged']}  action={result['action']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
