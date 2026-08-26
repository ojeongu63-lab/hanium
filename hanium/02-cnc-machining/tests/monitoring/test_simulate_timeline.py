import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# monitoring/simulate_timeline.py 는 패키지가 아니라 독립 스크립트다(src/monitoring
# 이 이미 동명의 실제 패키지라 `monitoring.simulate_timeline`으로는 임포트할 수
# 없다) — sweep_drift_constants.py 와 같은 방식으로 접근한다.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "monitoring"))

from simulate_timeline import apply_fixture_loosening  # noqa: E402


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
