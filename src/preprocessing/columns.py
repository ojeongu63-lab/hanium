import pandas as pd

DEAD_SENSOR_COLUMNS = [
    "Z_CurrentFeedback",
    "Z_DCBusVoltage",
    "Z_OutputCurrent",
    "Z_OutputVoltage",
]

METADATA_EXCLUDED_COLUMNS = [
    "M_CURRENT_PROGRAM_NUMBER",
    "M_sequence_number",
    "Machining_Process",
]

SETUP_CONSTANT_COLUMNS = [
    # 실험(작업)마다 하나의 상수값만 갖는 셋업 변수 — 공구/공작물 조합에 따른 토크 관성이라
    # 샷 내내 변하지 않는다. 시계열 신호가 아니므로 이상 원인 랭킹(feature_contributions,
    # 히트맵, 재구성 오버레이)에서는 제외한다 — train에 없던 조합이면 늘 크게 벗어나 보여서
    # 진짜 공정 이상과 구분이 안 된다.
    "S_SystemInertia",
]

FEATURE_COLUMNS = [
    "X_ActualPosition",
    "X_ActualVelocity",
    "X_ActualAcceleration",
    "X_SetPosition",
    "X_SetVelocity",
    "X_SetAcceleration",
    "X_CurrentFeedback",
    "X_DCBusVoltage",
    "X_OutputCurrent",
    "X_OutputVoltage",
    "X_OutputPower",
    "Y_ActualPosition",
    "Y_ActualVelocity",
    "Y_ActualAcceleration",
    "Y_SetPosition",
    "Y_SetVelocity",
    "Y_SetAcceleration",
    "Y_CurrentFeedback",
    "Y_DCBusVoltage",
    "Y_OutputCurrent",
    "Y_OutputVoltage",
    "Y_OutputPower",
    "Z_ActualPosition",
    "Z_ActualVelocity",
    "Z_ActualAcceleration",
    "Z_SetPosition",
    "Z_SetVelocity",
    "Z_SetAcceleration",
    "S_ActualPosition",
    "S_ActualVelocity",
    "S_ActualAcceleration",
    "S_SetPosition",
    "S_SetVelocity",
    "S_SetAcceleration",
    "S_CurrentFeedback",
    "S_DCBusVoltage",
    "S_OutputCurrent",
    "S_OutputVoltage",
    "S_OutputPower",
    "S_SystemInertia",
    "M_CURRENT_FEEDRATE",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()
