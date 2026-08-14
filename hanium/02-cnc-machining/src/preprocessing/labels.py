import pandas as pd


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_good = (df["machining_finalized"] == "yes") & (df["passed_visual_inspection"] == "yes")
    df["label"] = (~is_good).astype(int)
    return df
