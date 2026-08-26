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

import numpy as np
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
VIBRATION_LABEL_FLIP_DAY = 21  # WEAR_LABEL_FLIP_DAY 와 동일 — 고정구 풀림도 같은 날부터 QC 불합격 시작

# sweep_drift_constants.py 로 확정한 값. champion v1 (threshold 0.8566) 기준.
# 목표 Day 40 score/threshold — temperature 1.5~2.0, tool_wear 3.0, fixture_loosening 3.0
# (실측 대역: GOOD 0.43~1.30, BAD 1.00~3.79).
#
#   temperature(v)            tool_wear(v)              fixture_loosening(v)
#   0.02 → 1.80  ← 채택       0.02 → 0.75               0.02 → 0.72
#   0.05 → 6.65               0.05 → 0.88               0.05 → 0.72
#   0.10 → 26.76              0.10 → 1.36               0.10 → 0.72
#   0.20 → 111.31             0.20 → 3.08  ← 채택        0.20 → 0.73
#   0.35 → 349.48             0.35 → 7.54               0.35 → 0.74
#   (이후 급격히 발산)         0.50 → 14.51              0.50 → 0.76
#                                                        1.0  → 0.89
#                                                        2.0  → 1.40  (GRID 최대, 대역 미달)
#                                                        3.65 → 2.99  ← 채택 (GRID 밖, 수동 탐색)
#
# fixture_loosening 은 위치·속도 컬럼에 곱하지 않고 더하는 가산 노이즈라
# tool_wear 의 곱셈 램프보다 점수 민감도가 훨씬 낮다 — GRID(0.02~2.0) 안에서는
# 목표 대역(2.5~3.5)에 못 미쳐(최대 1.40) GRID 밖 값을 수동으로 추가 탐색했다.
#
# 무변형 기준 ratio 는 0.72. 날짜별 진행:
#   day   10    15    20    25    30    35    40
#   temp  0.72  0.74  0.83  0.95  1.18  1.49  1.80
#   wear  0.72  0.80  1.00  1.36  1.83  2.41  3.08
#   vib   0.72  0.78  0.97  1.29  1.73  2.29  2.99
#
# 모델이 재학습되면 이 상수는 낡는다. simulate_timeline 이 매일 실제 비율을
# 출력하므로 조용히 틀리지는 않는다.
POS_DRIFT = 0.02
CUR_DRIFT = 0.02
WEAR_RATE = 0.2
VIBRATION_RATE = 3.65          # 가산 노이즈라 GRID 밖 — 위 주석 참고

TEMP_POSITION_COLUMNS = ["X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition"]
TEMP_CURRENT_COLUMNS = [
    "X_OutputCurrent", "Y_OutputCurrent", "X_OutputPower", "Y_OutputPower",
]
WEAR_COLUMNS = [
    "S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback",
    "X_OutputPower", "Y_OutputPower",
]
VIBRATION_COLUMNS = [
    "X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition",
    "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity",
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


def apply_fixture_loosening(df: pd.DataFrame, progress: float) -> pd.DataFrame:
    """고정구/척 풀림: 진행될수록 위치·속도 추종의 흔들림(분산)이 커진다.
    apply_tool_wear와 달리 평균은 그대로 두고 노이즈만 키운다."""
    out = df.copy()
    rng = np.random.default_rng(43)  # tool_wear 계열과 겹치지 않는 고정 시드
    for col in VIBRATION_COLUMNS:
        out[col] = out[col] + rng.normal(
            0, out[col].std() * VIBRATION_RATE * progress, size=len(out)
        )
    return out


PERTURBATIONS = {
    "temperature": apply_temperature,
    "tool_wear": apply_tool_wear,
    "fixture_loosening": apply_fixture_loosening,
}


def true_label(scenario: str, day: int) -> str:
    """제품이 실제로 불량이냐. 온도는 제품 품질을 바꾸지 않는다."""
    if scenario == "temperature":
        return "good"
    if scenario == "fixture_loosening":
        return "bad" if day >= VIBRATION_LABEL_FLIP_DAY else "good"
    return "bad" if day >= WEAR_LABEL_FLIP_DAY else "good"


def generate_batch(day: int, index: int, scenario: str) -> pd.DataFrame:
    experiment_id = TRAIN_EXPERIMENT_IDS[
        (day * BATCHES_PER_DAY + index) % len(TRAIN_EXPERIMENT_IDS)
    ]
    df = pd.read_csv(DATASET_DIR / f"experiment_{experiment_id:02d}.csv")
    return PERTURBATIONS[scenario](df, progress_for(day))


def feed_day(client, day: int, scenario: str, out_dir: Path) -> None:
    """하루치 배치를 생성해 /predict 로 흘려보내고 라벨을 기록한다.
    감시는 하지 않는다 — --serve-url 모드에서는 drift_worker.py 가 별도
    프로세스로 폴링하며 감시한다."""
    for index in range(BATCHES_PER_DAY):
        batch_id = f"day{day:02d}_{index}"
        batch = generate_batch(day, index, scenario)
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
            label=true_label(scenario, day),
            db_path=LABELS_DB,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="가상 운영 타임라인 생성 및 주입")
    parser.add_argument("scenario", choices=list(PERTURBATIONS))
    parser.add_argument("--days", type=int, default=TOTAL_DAYS)
    parser.add_argument(
        "--serve-url",
        default=None,
        help="지정하면 이 주소의 실제 서버로 배치를 쏜다(진짜 HTTP, 별도 프로세스로 "
        "떠 있는 uvicorn 대상). 감시는 이 프로세스가 하지 않고 별도로 띄운 "
        "drift_worker.py 가 폴링한다. 생략하면 기존처럼 TestClient로 이 "
        "프로세스 안에서 감시까지 함께 한다.",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=0.0,
        help="--serve-url 과 함께 쓸 때, 하루치 배치를 보낸 뒤 이만큼 쉰다. "
        "0(기본값)이면 최대한 빨리 다 쏴버리는데, 그러면 워커가 재학습(수십 초"
        "~분)으로 뒤처지는 사이 feeder 가 이미 끝나버려 섀도우가 관찰할 미래 "
        "트래픽이 하나도 안 남는다(실측으로 확인된 문제) — 섀도우 검증 시에는 "
        "워커의 트리거 간격(재학습 4회 기준 실측 약 2분)보다 feeder 완주 시간이 "
        "짧지 않도록 넉넉히 줘야 한다.",
    )
    args = parser.parse_args()

    out_dir = ROOT / "data" / "timeline" / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.serve_url:
        import time

        import httpx2

        with httpx2.Client(base_url=args.serve_url, timeout=30.0) as client:
            for day in range(1, args.days + 1):
                feed_day(client, day, args.scenario, out_dir)
                print(f"Day {day:02d}  {BATCHES_PER_DAY}개 배치 전송 완료", flush=True)
                if args.pace_seconds:
                    time.sleep(args.pace_seconds)
        return

    from fastapi.testclient import TestClient
    from serving.app import app

    from drift_worker import WorkerState, tick  # 같은 폴더

    state = WorkerState()

    # with 블록이어야 lifespan 이 돌아 champion 모델이 로드된다
    # (simulate_drift.py 와 같은 관례). 없으면 /predict 가 503 을 낸다.
    with TestClient(app) as client:
        for day in range(1, args.days + 1):
            feed_day(client, day, args.scenario, out_dir)

            result = tick(client, state, current_day=day, scenario=args.scenario)
            print(
                f"Day {day:02d}  score/threshold={result['ratio']:.2f}  "
                f"flagged={result['flagged']}  action={result['action']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
