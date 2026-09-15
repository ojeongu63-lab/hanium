"""실행 환경 설정 — 데이터 루트와 루프 상수를 환경변수에서 읽는다.

기본값은 이 모듈이 생기기 전에 각 파일에 박혀 있던 값과 같다. 환경변수가 없으면 동작은 이전과
같다. 통합 테스트가 임시 데이터 루트와 며칠짜리 루프로 서버·워커를 띄우기 위해 만들었다
(docs/specs/2026-09-15-cnc-loop-integration-test-design.md §1).

값이 있는데 숫자로 읽히지 않으면 import 시점에 ValueError로 죽는다 — 조용히 기본값으로
떨어지면 오타 하나가 40일 루프를 엉뚱하게 돌린다.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 02-cnc-machining/


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: 정수가 아닙니다") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: 실수가 아닙니다") from exc


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return _env_float(name, 0.0)


# 데이터 위치. 원본 CSV는 인덱스 폴더 아래 한 단계 더 들어간다.
DATA_ROOT = Path(os.environ.get("CNC_DATA_ROOT") or PROJECT_ROOT / "data")
DATASET_DIR = DATA_ROOT / "dataset" / "CNC 비식별화 원본데이터_1209"  # 인덱스 train.csv
EXPERIMENT_DIR = DATASET_DIR / "CNC Virtual Data set _v2"  # experiment_XX.csv

# 서빙 — 드리프트 판정에 쓰는 최근 요청 수
DRIFT_WINDOW_SIZE = _env_int("CNC_DRIFT_WINDOW_SIZE", 10)

# 워커 — 트리거·게이트 (근거는 monitoring/drift_worker.py 주석)
CONSECUTIVE_K = _env_int("CNC_CONSECUTIVE_K", 3)
COOLDOWN_DAYS = _env_int("CNC_COOLDOWN_DAYS", 5)
GATE_SAMPLE_SIZE = _env_int("CNC_GATE_SAMPLE_SIZE", 20)

# feeder — 가상 운영 타임라인 (근거는 monitoring/simulate_timeline.py 주석)
TOTAL_DAYS = _env_int("CNC_TOTAL_DAYS", 40)
BATCHES_PER_DAY = _env_int("CNC_BATCHES_PER_DAY", 5)
DRIFT_START_DAY = _env_int("CNC_DRIFT_START_DAY", 10)
LABEL_DELAY_DAYS = _env_int("CNC_LABEL_DELAY_DAYS", 7)
LABEL_FLIP_DAY = _env_int("CNC_LABEL_FLIP_DAY", 21)  # 고장 시나리오에서 QC 불합격이 시작되는 날
DRIFT_MAX_PROGRESS = _env_optional_float("CNC_DRIFT_MAX_PROGRESS")  # None이면 상한 없음
POS_DRIFT = _env_float("CNC_POS_DRIFT", 0.02)
CUR_DRIFT = _env_float("CNC_CUR_DRIFT", 0.02)
WEAR_RATE = _env_float("CNC_WEAR_RATE", 0.2)
VIBRATION_RATE = _env_float("CNC_VIBRATION_RATE", 3.65)

# 학습·재학습 epochs (나머지 하이퍼파라미터는 바꾸지 않는다)
TRAIN_EPOCHS = _env_int("CNC_TRAIN_EPOCHS", 50)
