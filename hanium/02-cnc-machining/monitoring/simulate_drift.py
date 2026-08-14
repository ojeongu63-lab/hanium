import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing.columns import FEATURE_COLUMNS
from serving.app import DRIFT_WINDOW_SIZE, app

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = (
    ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
)
AMPLITUDES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def shift(df: pd.DataFrame, amplitude: float) -> pd.DataFrame:
    """평균이 이동하는 진짜 드리프트를 시뮬레이션한다. 평균 0인 노이즈(jitter)는
    행이 많으면 평균 낼 때 상쇄돼 버려서 drift-status가 추적하는 '실험 평균'이
    거의 안 움직인다 - 그래서 모든 행에 같은 방향(+)의 상수 오프셋(그 피처 자체
    표준편차 x amplitude)을 더해 평균 자체를 옮긴다. 방향을 요청마다 바꾸면
    같은 윈도우 안에서 서로 상쇄되므로 항상 같은 방향(+)으로 고정한다.
    """
    df = df.copy()
    for col in FEATURE_COLUMNS:
        std = df[col].std()
        df[col] = df[col] + std * amplitude
    return df


def main() -> None:
    base_df = pd.read_csv(DATASET_DIR / "experiment_01.csv")

    with TestClient(app) as client:
        for amplitude in AMPLITUDES:
            print(f"=== amplitude={amplitude} ===", flush=True)
            perturbed = shift(base_df, amplitude)
            csv_bytes = perturbed.to_csv(index=False).encode()
            for _ in range(DRIFT_WINDOW_SIZE):
                response = client.post(
                    "/predict",
                    files={"file": ("experiment.csv", csv_bytes, "text/csv")},
                )
                assert response.status_code == 200, response.text

            status = client.get("/drift-status").json()
            print(f"  sufficient_data={status['sufficient_data']}")
            if status["sufficient_data"]:
                top = sorted(
                    status["input_drift"]["all_feature_avg_scaled_means"].items(),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )[:3]
                print(f"  상위 3개 피처 평균 편차: {top}")
                print(f"  output_drift: {status['output_drift']}")
                print(
                    f"  flagged_features 개수: "
                    f"{len(status['input_drift']['flagged_features'])}"
                )


if __name__ == "__main__":
    main()
