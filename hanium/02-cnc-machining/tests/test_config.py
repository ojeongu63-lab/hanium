"""src/config.py — 환경변수가 없으면 코드에 박혀 있던 값, 있으면 그 값. 파싱 실패는 즉시 죽는다."""
import importlib
import os
from pathlib import Path

import pytest


def _reload(monkeypatch, **env):
    """CNC_* 환경변수를 전부 지운 뒤 주어진 것만 넣고 config를 다시 읽는다."""
    for key in list(os.environ):
        if key.startswith("CNC_"):
            monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import config

    return importlib.reload(config)


@pytest.fixture(autouse=True)
def _restore_config():
    """reload가 남긴 상태를 되돌린다 — 다른 테스트가 import config 를 다시 할 수 있다."""
    yield
    for key in list(os.environ):
        if key.startswith("CNC_"):
            os.environ.pop(key, None)
    import config

    importlib.reload(config)


def test_defaults_match_previous_hardcoded_values(monkeypatch):
    cfg = _reload(monkeypatch)

    assert cfg.DATA_ROOT == cfg.PROJECT_ROOT / "data"
    assert cfg.DATASET_DIR == cfg.DATA_ROOT / "dataset" / "CNC 비식별화 원본데이터_1209"
    assert cfg.EXPERIMENT_DIR == cfg.DATASET_DIR / "CNC Virtual Data set _v2"
    assert (cfg.DRIFT_WINDOW_SIZE, cfg.CONSECUTIVE_K, cfg.COOLDOWN_DAYS, cfg.GATE_SAMPLE_SIZE) == (
        10, 3, 5, 20,
    )
    assert (
        cfg.TOTAL_DAYS, cfg.BATCHES_PER_DAY, cfg.DRIFT_START_DAY, cfg.LABEL_DELAY_DAYS, cfg.LABEL_FLIP_DAY,
    ) == (40, 5, 10, 7, 21)
    assert cfg.DRIFT_MAX_PROGRESS is None
    assert (cfg.POS_DRIFT, cfg.CUR_DRIFT, cfg.WEAR_RATE, cfg.VIBRATION_RATE) == (0.02, 0.02, 0.2, 3.65)
    assert cfg.TRAIN_EPOCHS == 50


def test_env_overrides_are_parsed(monkeypatch, tmp_path):
    cfg = _reload(
        monkeypatch,
        CNC_DATA_ROOT=str(tmp_path),
        CNC_GATE_SAMPLE_SIZE="5",
        CNC_DRIFT_MAX_PROGRESS="1.0",
        CNC_POS_DRIFT="8",
    )

    assert cfg.DATA_ROOT == tmp_path
    assert cfg.DATASET_DIR == tmp_path / "dataset" / "CNC 비식별화 원본데이터_1209"
    assert cfg.GATE_SAMPLE_SIZE == 5 and isinstance(cfg.GATE_SAMPLE_SIZE, int)
    assert cfg.DRIFT_MAX_PROGRESS == 1.0
    assert cfg.POS_DRIFT == 8.0


def test_unparsable_value_fails_loudly(monkeypatch):
    with pytest.raises(ValueError, match="CNC_COOLDOWN_DAYS"):
        _reload(monkeypatch, CNC_COOLDOWN_DAYS="five")


import json
import subprocess
import sys

SRC_WIRING = """
import json
import lstm_ae.tracking as tracking
import retraining.runner as runner
import serving.app as app
print(json.dumps({
    "mlflow_dir": str(tracking.MLFLOW_DIR),
    "db_path": str(app.DB_PATH),
    "shadow_db": str(app.SHADOW_DB),
    "dataset_dir": str(app.DATASET_DIR),
    "drift_window": app.DRIFT_WINDOW_SIZE,
    "batches_per_day": app.TIMELINE_BATCHES_PER_DAY,
    "epochs": runner.TRAINING_CONFIG["epochs"],
}))
"""


def _run_snippet(snippet: str, env: dict) -> dict:
    """config는 import 시 환경을 읽으므로, 배선은 새 인터프리터에서 확인한다."""
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        env={**os.environ, **env}, capture_output=True, text=True, check=True, timeout=180,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_src_modules_follow_config(tmp_path):
    got = _run_snippet(SRC_WIRING, {
        "CNC_DATA_ROOT": str(tmp_path),
        "CNC_DRIFT_WINDOW_SIZE": "4",
        "CNC_BATCHES_PER_DAY": "2",
        "CNC_TRAIN_EPOCHS": "1",
    })

    root = str(tmp_path)
    assert got["mlflow_dir"] == f"{root}/mlflow"
    assert got["db_path"] == f"{root}/monitoring/requests.db"
    assert got["shadow_db"] == f"{root}/monitoring/shadow.db"
    assert got["dataset_dir"] == f"{root}/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2"
    assert got["drift_window"] == 4
    assert got["batches_per_day"] == 2
    assert got["epochs"] == 1


MONITORING_DIR = str(Path(__file__).resolve().parent.parent / "monitoring")

FEEDER_WIRING = f"""
import json, sys
sys.path.insert(0, {MONITORING_DIR!r})
import simulate_timeline as st
print(json.dumps({{
    "labels_db": str(st.LABELS_DB),
    "dataset_dir": str(st.DATASET_DIR),
    "pos_drift": st.POS_DRIFT,
    "flip": st.WEAR_LABEL_FLIP_DAY,
    "progress_9": st.progress_for(9),
}}))
"""


def test_feeder_follows_config(tmp_path):
    got = _run_snippet(FEEDER_WIRING, {
        "CNC_DATA_ROOT": str(tmp_path),
        "CNC_POS_DRIFT": "8",
        "CNC_LABEL_FLIP_DAY": "3",
        "CNC_DRIFT_START_DAY": "2",
        "CNC_TOTAL_DAYS": "3",
        "CNC_DRIFT_MAX_PROGRESS": "1.0",
    })

    root = str(tmp_path)
    assert got["labels_db"] == f"{root}/monitoring/labels.db"
    assert got["dataset_dir"] == f"{root}/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2"
    assert got["pos_drift"] == 8.0
    assert got["flip"] == 3
    assert got["progress_9"] == 1.0
