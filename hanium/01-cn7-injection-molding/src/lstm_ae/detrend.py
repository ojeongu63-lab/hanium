import pandas as pd


def apply_rolling_zscore(
    df: pd.DataFrame,
    feature_columns: list[str],
    window: int = 200,
    min_periods: int = 30,
) -> pd.DataFrame:
    result = df.copy()
    for col in feature_columns:
        series = df[col]
        global_mean = series.mean()
        global_std = series.std()

        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()

        rolling_mean = rolling_mean.fillna(global_mean)
        rolling_std = rolling_std.fillna(global_std).replace(0, global_std)

        result[col] = (series - rolling_mean) / rolling_std
    return result
