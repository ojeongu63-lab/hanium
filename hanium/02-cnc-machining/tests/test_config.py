"""src/config.py — 환경변수가 없으면 코드에 박혀 있던 값, 있으면 그 값. 파싱 실패는 즉시 죽는다."""
import importlib
import os

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
