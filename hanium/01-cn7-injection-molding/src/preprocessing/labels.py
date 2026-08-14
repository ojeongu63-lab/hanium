import pandas as pd

INITIAL_STARTUP_DEFECT_REASON = "초기허용불량"


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    label = (df["PassOrFail"] == "N").astype(int)
    label = label.where(df["Reason"] != INITIAL_STARTUP_DEFECT_REASON, 0)
    df["label"] = label
    return df
