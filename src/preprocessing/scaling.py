import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(df: pd.DataFrame, feature_columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_columns])
    return scaler


def transform_features(
    df: pd.DataFrame, feature_columns: list[str], scaler: StandardScaler
) -> pd.DataFrame:
    df = df.copy()
    df[feature_columns] = scaler.transform(df[feature_columns])
    return df


def scaler_to_dict(scaler: StandardScaler, feature_columns: list[str]) -> dict:
    return {
        col: {"mean": float(mean), "std": float(std)}
        for col, mean, std in zip(feature_columns, scaler.mean_, scaler.scale_)
    }
