TOOL_WEAR_FEATURES = {
    "S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback",
    "X_OutputPower", "Y_OutputPower",
}
VIBRATION_BACKLASH_FEATURES = {
    "X_ActualPosition", "Y_ActualPosition", "Z_ActualPosition",
    "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity",
}


def estimate_cause(feature_contributions_batches: list[list[dict]]) -> str:
    """챔피언 모델의 최근 N건 feature_contributions(배치별 피처-zscore 리스트)를
    받아, 두 피처 그룹 중 어느 쪽이 누적으로 더 크게 벗어났는지로 원인을
    추정한다. 반환값은 코퍼스의 fault_category 값과 동일하게 맞춘다
    ("tool_wear" / "vibration_backlash") — RAG 필터링에 그대로 쓰기 위함.

    simulate_timeline.py의 시나리오 이름은 "fixture_loosening"(물리적 원인
    이름)이지만, 여기서는 기존 Sandvik 코퍼스가 이미 쓰는 카테고리 값인
    "vibration_backlash"를 반환한다 — 시뮬레이션과 지식 코퍼스가 같은 현상을
    각자의 기존 명명 체계로 부르기 때문이다."""
    tool_wear_score = 0.0
    vibration_score = 0.0
    for contributions in feature_contributions_batches:
        for c in contributions:
            if c["feature"] in TOOL_WEAR_FEATURES:
                tool_wear_score += c["z_score"]
            elif c["feature"] in VIBRATION_BACKLASH_FEATURES:
                vibration_score += c["z_score"]
    return "tool_wear" if tool_wear_score >= vibration_score else "vibration_backlash"
