import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# monitoring/simulate_timeline.py 는 패키지가 아니라 독립 스크립트다(src/monitoring
# 이 이미 동명의 실제 패키지라 `monitoring.simulate_timeline`으로는 임포트할 수
# 없다) — sweep_drift_constants.py 와 같은 방식으로 접근한다.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "monitoring"))

from simulate_timeline import apply_fixture_loosening  # noqa: E402
import simulate_timeline as st  # noqa: E402


def _position_df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "X_ActualPosition": rng.normal(100, 5, size=n),
        "Y_ActualPosition": rng.normal(100, 5, size=n),
        "Z_ActualPosition": rng.normal(100, 5, size=n),
        "X_ActualVelocity": rng.normal(0, 1, size=n),
        "Y_ActualVelocity": rng.normal(0, 1, size=n),
        "Z_ActualVelocity": rng.normal(0, 1, size=n),
    })


def test_apply_fixture_loosening_no_change_at_zero_progress():
    df = _position_df()

    out = apply_fixture_loosening(df, progress=0.0)

    pd.testing.assert_frame_equal(out, df)


def test_apply_fixture_loosening_keeps_mean_but_increases_spread():
    df = _position_df()

    out = apply_fixture_loosening(df, progress=1.0)

    for col in df.columns:
        assert out[col].mean() == pytest.approx(df[col].mean(), abs=df[col].std() * 0.5)
        assert out[col].std() > df[col].std()


def test_progress_is_unbounded_without_cap(monkeypatch):
    monkeypatch.setattr(st, "DRIFT_START_DAY", 10)
    monkeypatch.setattr(st, "TOTAL_DAYS", 40)
    monkeypatch.setattr(st, "DRIFT_MAX_PROGRESS", None)

    assert st.progress_for(5) == 0.0
    assert st.progress_for(40) == pytest.approx(1.0)
    assert st.progress_for(70) == pytest.approx(2.0)  # 08-25 스펙의 알려진 한계 그대로


def test_progress_is_capped_when_configured(monkeypatch):
    monkeypatch.setattr(st, "DRIFT_START_DAY", 2)
    monkeypatch.setattr(st, "TOTAL_DAYS", 3)
    monkeypatch.setattr(st, "DRIFT_MAX_PROGRESS", 1.0)

    assert st.progress_for(2) == 0.0
    assert st.progress_for(3) == pytest.approx(1.0)
    assert st.progress_for(9) == pytest.approx(1.0)  # 계단


def test_serve_url_mode_feeds_only_requested_day_range(monkeypatch, tmp_path):
    fed = []
    monkeypatch.setattr(st, "feed_day", lambda client, day, scenario, out_dir: fed.append(day))
    monkeypatch.setattr(st, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "simulate_timeline.py", "temperature", "--serve-url", "http://127.0.0.1:1",
        "--start-day", "5", "--days", "6",
    ])

    st.main()

    assert fed == [5, 6]
    assert (tmp_path / "timeline" / "temperature").is_dir()


def test_feed_day_labels_the_day_only_after_all_batches_are_posted(monkeypatch, tmp_path):
    # 워커는 labels.db 의 최신 produced_day 를 시계로 쓴다. 배치마다 라벨을 적으면 그날 첫 배치
    # 직후 그날을 처리해 버려, 드리프트 창이 그날 배치 일부만으로 계산된다.
    events = []

    class FakeClient:
        def post(self, url, files):
            events.append(("post", files["file"][0]))
            return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(st, "BATCHES_PER_DAY", 3)
    monkeypatch.setattr(st, "generate_batch", lambda day, index, scenario: pd.DataFrame({"x": [index]}))
    monkeypatch.setattr(st, "record_label", lambda batch_id, **kwargs: events.append(("label", batch_id)))

    st.feed_day(FakeClient(), day=1, scenario="temperature", out_dir=tmp_path)

    assert events == [
        ("post", "day01_0.csv"), ("post", "day01_1.csv"), ("post", "day01_2.csv"),
        ("label", "day01_0"), ("label", "day01_1"), ("label", "day01_2"),
    ]
