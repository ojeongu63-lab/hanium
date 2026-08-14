import pandas as pd


def normalize_machining_process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Machining_Process"] = df["Machining_Process"].replace({"end": "End"})
    return df
