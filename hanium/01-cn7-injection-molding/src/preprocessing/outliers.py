import pandas as pd
from sklearn.ensemble import IsolationForest


def remove_outliers(
    df: pd.DataFrame,
    feature_columns: list[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    X = df[feature_columns]
    predictions = model.fit_predict(X)
    scores = model.score_samples(X)

    is_outlier = predictions == -1
    cleaned = df.loc[~is_outlier].reset_index(drop=True)
    removed = pd.DataFrame({
        "_id": df.loc[is_outlier, "_id"].values,
        "outlier_score": scores[is_outlier],
    }).reset_index(drop=True)
    return cleaned, removed
