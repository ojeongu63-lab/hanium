"""KAMP CNC 데이터셋과 같은 모양의 합성 데이터셋 — 통합 테스트용.

목적은 배관 검증이지 모델 품질이 아니다. 값의 의미는 없고 형식만 같다:
인덱스 train.csv(No, material, feedrate, clamp_pressure, tool_condition, machining_finalized,
passed_visual_inspection)와 experiment_XX.csv(48컬럼). 실험 번호는 preprocessing/split.py 의
22개 그대로라 전처리·학습·승격 스크립트가 코드 변경 없이 돈다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.columns import DEAD_SENSOR_COLUMNS, FEATURE_COLUMNS
from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)

EXPERIMENT_SUBDIR = "CNC Virtual Data set _v2"
ROWS = 150
# 임계값은 train 실험 8개 점수의 p95 라 사실상 최댓값이다. 8개가 비슷하면 정상 배치의
# 점수/임계값 비율이 1.0 근처가 되어 변형 전에도 출력 드리프트(비율 > 0.8)가 켜진다.
# 그래서 실험 하나의 잡음을 키워 임계값을 끌어올린다(스펙 §2). 하네스의 2 에폭 champion 은
# 사실상 미학습이라(점수 ≈ 스케일된 값의 제곱 평균) ×3 으로는 모자랐다: 실측 정상 배치
# 0.90~0.92, 17번 1.05, Day 1 부터 flagged. ×10 이면 변형 전 창이 0.8 아래에 머문다
# (실측 Day 1 0.62 — 17번 포함 창, Day 2 0.45).
NOISY_TRAIN_ID = 17
NOISE_FACTOR = 10.0
BAD_SCALE_COLUMNS = ["S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback"]
BAD_FACTOR = 3.0
CONSTANT_COLUMNS = {"S_SystemInertia": 12.0, "M_CURRENT_FEEDRATE": 6.0}  # 실험당 상수(설정값)


def write_dataset(dataset_dir: Path, seed: int = 0) -> None:
    dataset_dir = Path(dataset_dir)
    experiment_dir = dataset_dir / EXPERIMENT_SUBDIR
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment_id in sorted(TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS + EVAL_BAD_EXPERIMENT_IDS):
        bad = experiment_id in EVAL_BAD_EXPERIMENT_IDS
        rows.append({
            "No": experiment_id,
            "material": "wax",
            "feedrate": int(CONSTANT_COLUMNS["M_CURRENT_FEEDRATE"]),
            "clamp_pressure": 4,
            "tool_condition": "unworn",
            "machining_finalized": "yes",
            "passed_visual_inspection": "no" if bad else "yes",
        })
        _write_experiment(experiment_dir / f"experiment_{experiment_id:02d}.csv", experiment_id, bad, seed)

    pd.DataFrame(rows).to_csv(dataset_dir / "train.csv", index=False)


def _write_experiment(path: Path, experiment_id: int, bad: bool, seed: int) -> None:
    rng = np.random.default_rng([seed, experiment_id])
    t = np.arange(ROWS)
    noise = 0.5 * (NOISE_FACTOR if experiment_id == NOISY_TRAIN_ID else 1.0)

    data: dict[str, np.ndarray] = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        if col in CONSTANT_COLUMNS:
            data[col] = np.full(ROWS, CONSTANT_COLUMNS[col])
            continue
        phase = rng.uniform(0.0, 2.0 * np.pi)
        base = 100.0 + 10.0 * (i % 7)  # 컬럼마다 다른 수준
        data[col] = base + 5.0 * np.sin(2.0 * np.pi * t / 50.0 + phase) + rng.normal(0.0, noise, ROWS)
    if bad:
        for col in BAD_SCALE_COLUMNS:
            data[col] = data[col] * BAD_FACTOR
    for col in DEAD_SENSOR_COLUMNS:
        data[col] = np.zeros(ROWS)
    data["Machining_Process"] = np.array(["Layer 1 Up"] * ROWS)
    data["M_sequence_number"] = t
    data["M_CURRENT_PROGRAM_NUMBER"] = np.ones(ROWS, dtype=int)

    pd.DataFrame(data).to_csv(path, index=False)
