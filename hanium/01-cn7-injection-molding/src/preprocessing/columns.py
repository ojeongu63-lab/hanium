import pandas as pd

DEAD_SENSOR_COLUMNS = [
    "Mold_Temperature_1",
    "Mold_Temperature_2",
    "Mold_Temperature_5",
    "Mold_Temperature_6",
    "Mold_Temperature_7",
    "Mold_Temperature_8",
    "Mold_Temperature_9",
    "Mold_Temperature_10",
    "Mold_Temperature_11",
    "Mold_Temperature_12",
]

DISABLED_SENSOR_COLUMNS = ["Switch_Over_Position", "Barrel_Temperature_7"]

FEATURE_COLUMNS = [
    "Injection_Time",
    "Filling_Time",
    "Plasticizing_Time",
    "Cycle_Time",
    "Clamp_Close_Time",
    "Cushion_Position",
    "Plasticizing_Position",
    "Clamp_Open_Position",
    "Max_Injection_Speed",
    "Max_Screw_RPM",
    "Average_Screw_RPM",
    "Max_Injection_Pressure",
    "Max_Switch_Over_Pressure",
    "Max_Back_Pressure",
    "Average_Back_Pressure",
    "Barrel_Temperature_1",
    "Barrel_Temperature_2",
    "Barrel_Temperature_3",
    "Barrel_Temperature_4",
    "Barrel_Temperature_5",
    "Barrel_Temperature_6",
    "Hopper_Temperature",
    "Mold_Temperature_3",
    "Mold_Temperature_4",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()
