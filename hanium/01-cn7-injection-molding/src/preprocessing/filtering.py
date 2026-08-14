import pandas as pd


def remove_non_plasticizing_shots(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Max_Screw_RPM"] != 0].reset_index(drop=True)
