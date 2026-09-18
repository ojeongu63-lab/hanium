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
