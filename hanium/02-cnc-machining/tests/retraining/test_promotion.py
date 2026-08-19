import json

import pytest

from retraining.promotion import (
    backup_artifacts,
    install_artifacts,
    restore_backup,
    swap_with_rollback,
    verify_serving_contract,
)


def _full_metrics():
    return {"mean_threshold": 0.85, "max_threshold": 1.2, "p95_threshold": 1.0}


def test_contract_satisfied_returns_empty_list():
    assert verify_serving_contract(_full_metrics(), {"window_size": "20"}) == []


def test_contract_detects_missing_metric():
    metrics = _full_metrics()
    del metrics["mean_threshold"]

    missing = verify_serving_contract(metrics, {"window_size": "20"})

    assert missing == ["mean_threshold"]


def test_contract_detects_missing_param():
    missing = verify_serving_contract(_full_metrics(), {})

    assert missing == ["window_size"]


def _make_current(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("OLD_MODEL")
    (model_dir / "feature_baseline.json").write_text(json.dumps({"mean": {}, "std": {}}))
    scaler_path = tmp_path / "processed" / "scaler.json"
    scaler_path.parent.mkdir()
    scaler_path.write_text(json.dumps({"old": True}))
    return model_dir, scaler_path


def _make_retrain(tmp_path):
    retrain_dir = tmp_path / "retrain"
    retrain_dir.mkdir()
    (retrain_dir / "model.pt").write_text("NEW_MODEL")
    (retrain_dir / "feature_baseline.json").write_text(json.dumps({"mean": {}, "std": {}}))
    (retrain_dir / "scaler.json").write_text(json.dumps({"old": False}))
    return retrain_dir


def test_backup_then_install_then_restore_round_trip(tmp_path):
    model_dir, scaler_path = _make_current(tmp_path)
    retrain_dir = _make_retrain(tmp_path)
    (retrain_dir / "train.csv").write_text("should_not_be_installed")

    backup_dir = backup_artifacts(model_dir, scaler_path, tmp_path / "backup")
    install_artifacts(retrain_dir, model_dir, scaler_path)

    assert (model_dir / "model.pt").read_text() == "NEW_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": False}
    assert not (model_dir / "train.csv").exists()  # 학습 입력은 옮기지 않는다

    restore_backup(backup_dir, model_dir, scaler_path)

    assert (model_dir / "model.pt").read_text() == "OLD_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": True}


def test_swap_with_rollback_keeps_new_artifacts_on_success(tmp_path):
    model_dir, scaler_path = _make_current(tmp_path)
    retrain_dir = _make_retrain(tmp_path)

    swap_with_rollback(
        retrain_dir, model_dir, scaler_path, tmp_path / "backup",
        promote=lambda: None, verify=lambda: None,
    )

    assert (model_dir / "model.pt").read_text() == "NEW_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": False}


def test_swap_with_rollback_restores_when_promote_fails(tmp_path):
    model_dir, scaler_path = _make_current(tmp_path)
    retrain_dir = _make_retrain(tmp_path)

    def _promote_boom():
        raise RuntimeError("alias 교체 실패")

    with pytest.raises(RuntimeError, match="alias 교체 실패"):
        swap_with_rollback(
            retrain_dir, model_dir, scaler_path, tmp_path / "backup",
            promote=_promote_boom, verify=lambda: None,
        )

    # 정본이 원래대로 복원돼야 한다
    assert (model_dir / "model.pt").read_text() == "OLD_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": True}


def test_swap_with_rollback_restores_when_verify_fails(tmp_path):
    model_dir, scaler_path = _make_current(tmp_path)
    retrain_dir = _make_retrain(tmp_path)

    def _verify_boom():
        raise RuntimeError("리로드 후 버전 불일치")

    with pytest.raises(RuntimeError, match="버전 불일치"):
        swap_with_rollback(
            retrain_dir, model_dir, scaler_path, tmp_path / "backup",
            promote=lambda: None, verify=_verify_boom,
        )

    assert (model_dir / "model.pt").read_text() == "OLD_MODEL"
    assert json.loads(scaler_path.read_text()) == {"old": True}
