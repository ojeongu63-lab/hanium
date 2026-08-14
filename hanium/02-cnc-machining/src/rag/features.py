FEATURE_DESCRIPTIONS = {
    "S_OutputCurrent": "스핀들 출력 전류",
    "S_OutputPower": "스핀들 출력 파워",
    "S_CurrentFeedback": "스핀들 전류 피드백",
    "X_OutputCurrent": "X축 출력 전류",
    "X_OutputPower": "X축 출력 파워",
    "Y_OutputCurrent": "Y축 출력 전류",
    "Y_OutputPower": "Y축 출력 파워",
    "X_ActualPosition": "X축 실제 위치",
    "Y_ActualPosition": "Y축 실제 위치",
    "Z_ActualPosition": "Z축 실제 위치",
    "X_ActualVelocity": "X축 실제 속도",
    "Y_ActualVelocity": "Y축 실제 속도",
    "Z_ActualVelocity": "Z축 실제 속도",
}


def describe_feature(code: str) -> str:
    return FEATURE_DESCRIPTIONS.get(code, code)
