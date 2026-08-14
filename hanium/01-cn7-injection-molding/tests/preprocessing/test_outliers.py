import numpy as np
import pandas as pd

from preprocessing.outliers import remove_outliers


def test_remove_outliers_flags_extreme_row():
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    df = pd.DataFrame(normal, columns=["a", "b"])
    df["_id"] = [f"id{i}" for i in range(len(df))]
    df.loc[0, ["a", "b"]] = [100.0, 100.0]

    cleaned, removed = remove_outliers(
        df, feature_columns=["a", "b"], contamination=0.01, random_state=42
    )

    assert "id0" in removed["_id"].tolist()
    assert "id0" not in cleaned["_id"].tolist()
    assert len(cleaned) + len(removed) == len(df)
    assert list(removed.columns) == ["_id", "outlier_score"]
