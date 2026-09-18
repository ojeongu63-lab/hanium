"""monitoring/drift_worker.py 의 순수 함수. 루프 전체는 tests/integration/ 이 실제 프로세스로 검증한다."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "monitoring"))

import drift_worker as dw  # noqa: E402


def test_champion_missed_from_metrics_reads_mean_fn():
    assert dw.champion_missed_from_metrics({"mean_fn": 1.0, "mean_recall": 0.909}) == 1


def test_champion_missed_from_metrics_fails_without_metric():
    with pytest.raises(KeyError, match="mean_fn"):
        dw.champion_missed_from_metrics({"mean_recall": 0.909})


def test_worker_state_requires_champion_missed():
    # 하드코딩 1 이 사라졌다 — 워커는 champion run 에서 읽은 값을 넘겨야 한다.
    with pytest.raises(TypeError):
        dw.WorkerState()
