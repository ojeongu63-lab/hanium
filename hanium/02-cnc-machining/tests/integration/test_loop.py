"""재학습 루프 통합 테스트 — 서버·워커는 실제 서브프로세스, feeder 는 하루씩 동기화.
스펙 docs/specs/2026-09-15-cnc-loop-integration-test-design.md §3. 실데이터 불필요.
실행: uv run pytest -m integration -q (공유 서버에서는 nice -n 19)."""
import pytest

pytestmark = pytest.mark.integration


def test_harness_bootstraps_champion_and_feeds_one_day(loop_factory):
    loop = loop_factory("temperature")

    assert loop.health()["model_version"] == "1"

    loop.run_days(1)

    days = loop.days()
    assert [d for d, *_ in days] == [1]
    assert days[0][3] == "none"
    assert (loop.data_root / "monitoring" / "labels.db").exists()
    assert (loop.data_root / "timeline" / "temperature" / "day01_0.csv").exists()


import hashlib
from pathlib import Path

from mlflow.tracking import MlflowClient

from monitoring.shadow_log import get_shadow_predictions

MODEL_NAME = "cnc-lstm-ae"


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _mlflow(loop) -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{loop.data_root / 'mlflow' / 'mlflow.db'}")


def _scenario_runs(client: MlflowClient, scenario: str) -> list:
    experiment = client.get_experiment_by_name(MODEL_NAME)
    return client.search_runs([experiment.experiment_id], filter_string=f"tags.scenario = '{scenario}'")


def _all_batch_ids(days: int, batches_per_day: int = 5) -> list[str]:
    return [f"day{day:02d}_{index}" for day in range(1, days + 1) for index in range(batches_per_day)]


def test_shift_scenario_reaches_shadow_and_promotion(loop_factory):
    """라벨은 전부 정상, Day 3 부터 위치 3축이 8σ 계단 이동 → 트리거 → 재학습 → 게이트 통과 →
    섀도우 → 승격. 판정의 옳고 그름이 아니라 배관이 승격 끝까지 닿는지 본다(스펙 §3)."""
    loop = loop_factory("temperature", {"CNC_POS_DRIFT": "8", "CNC_CUR_DRIFT": "1.0"})
    model_before = _md5(loop.data_root / "model" / "model.pt")
    scaler_before = _md5(loop.data_root / "processed" / "scaler.json")

    loop.run_days(12)

    days = loop.days()
    assert [d for d, *_ in days] == list(range(1, 13)), "날짜가 빠지거나 순서가 어긋남"
    actions = [action for *_, action in days]
    first_action_day = next(d for d, *_, action in days if action != "none")
    assert first_action_day >= 5, f"연속 3회 전에 트리거: {days}"
    assert "shadow_started" in actions and "promoted" in actions, actions
    assert actions.index("shadow_started") < actions.index("promoted")

    client = _mlflow(loop)
    runs = _scenario_runs(client, "temperature")
    assert runs, "재학습 run 이 MLflow 에 없음"
    tags = [run.data.tags for run in runs]
    assert any(t.get("gate_decision") == "shadow_promoted" for t in tags), tags
    promoted = next(t for t in tags if t.get("gate_decision") == "shadow_promoted")
    assert {"gate_g2_n_good", "gate_g2_fa_delta", "shadow_n_good", "shadow_fa_delta"} <= set(promoted)

    # MLflow SQL 저장소는 version 을 int 로 준다. serving 처럼 str 로 맞춰야 "1"·/health 와 비교가 성립한다.
    champion = str(client.get_model_version_by_alias(MODEL_NAME, "champion").version)
    assert champion != "1"
    assert loop.health()["model_version"] == champion

    model_after = _md5(loop.data_root / "model" / "model.pt")
    assert model_after != model_before
    assert _md5(loop.data_root / "processed" / "scaler.json") != scaler_before
    retrain_models = {_md5(d / "model.pt") for d in (loop.data_root / "retrain").iterdir() if (d / "model.pt").exists()}
    assert model_after in retrain_models, "디스크 정본이 어느 재학습 산출물과도 같지 않음"
    assert list((loop.data_root / "model_backup").iterdir()), "승격 전 백업이 없음"

    shadow = get_shadow_predictions(_all_batch_ids(12), loop.data_root / "monitoring" / "shadow.db")
    assert len(shadow) >= 5, "섀도우가 관찰한 배치가 5건 미만"
