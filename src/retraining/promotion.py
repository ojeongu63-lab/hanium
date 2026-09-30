import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

# 서빙이 champion run에서 직접 읽는 값들 (src/serving/app.py:68-71).
# 이게 없으면 승격은 성공하는데 모델 로드가 KeyError로 죽는다.
REQUIRED_METRICS = ["mean_threshold", "max_threshold", "p95_threshold"]
REQUIRED_PARAMS = ["window_size"]

# 학습 입력이라 정본 자리로 옮기지 않는 파일들
NOT_INSTALLED = {"train.csv", "eval.csv", "scaler.json"}


def verify_serving_contract(metrics: dict, params: dict) -> list[str]:
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    missing += [key for key in REQUIRED_PARAMS if key not in params]
    return missing


def backup_artifacts(model_dir: Path, scaler_path: Path, backup_root: Path) -> Path:
    backup_dir = Path(backup_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, backup_dir / "model", dirs_exist_ok=True)
    shutil.copy2(scaler_path, backup_dir / "scaler.json")
    return backup_dir


def install_artifacts(retrain_dir: Path, model_dir: Path, scaler_path: Path) -> None:
    for item in Path(retrain_dir).iterdir():
        if item.is_dir() or item.name in NOT_INSTALLED:
            continue
        shutil.copy2(item, Path(model_dir) / item.name)
    shutil.copy2(Path(retrain_dir) / "scaler.json", scaler_path)


def restore_backup(backup_dir: Path, model_dir: Path, scaler_path: Path) -> None:
    shutil.rmtree(model_dir, ignore_errors=True)
    shutil.copytree(Path(backup_dir) / "model", model_dir)
    shutil.copy2(Path(backup_dir) / "scaler.json", scaler_path)


def swap_with_rollback(
    retrain_dir: Path,
    model_dir: Path,
    scaler_path: Path,
    backup_root: Path,
    promote: Callable[[], None],
    verify: Callable[[], None],
) -> Path:
    """백업 → 파일 교체 → alias 교체 → 검증. 어느 단계든 실패하면 정본을 되돌린다.

    모델(MLflow alias)과 동반 파일(디스크)이 서로 다른 저장소에 있어, 중간에
    실패하면 짝이 어긋난 상태로 남는다. 그 상태는 에러 없이 조용히 틀린
    스케일로 추론하므로 반드시 롤백해야 한다.

    promote/verify를 콜러블로 받는 이유는 MLflow와 HTTP 호출을 이 함수에서
    떼어내 실패 주입 테스트를 가능하게 하기 위함이다.
    """
    backup_dir = backup_artifacts(model_dir, scaler_path, backup_root)
    try:
        install_artifacts(retrain_dir, model_dir, scaler_path)
        promote()
        verify()
    except Exception:
        restore_backup(backup_dir, model_dir, scaler_path)
        raise
    return backup_dir
