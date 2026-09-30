from monitoring.cause_estimation import estimate_cause


def test_estimate_cause_tool_wear_when_spindle_load_dominates():
    batches = [
        [
            {"feature": "S_OutputCurrent", "z_score": 10.0},
            {"feature": "X_ActualPosition", "z_score": 1.0},
        ]
        for _ in range(20)
    ]

    assert estimate_cause(batches) == "tool_wear"


def test_estimate_cause_vibration_backlash_when_position_variance_dominates():
    batches = [
        [
            {"feature": "S_OutputCurrent", "z_score": 1.0},
            {"feature": "X_ActualVelocity", "z_score": 10.0},
        ]
        for _ in range(20)
    ]

    assert estimate_cause(batches) == "vibration_backlash"


def test_estimate_cause_ignores_features_outside_both_groups():
    batches = [[{"feature": "M_CURRENT_FEEDRATE", "z_score": 999.0}]]

    assert estimate_cause(batches) == "tool_wear"  # 양쪽 다 0점 -> 동점 기본값


def test_estimate_cause_ties_default_to_tool_wear():
    batches = [[
        {"feature": "S_OutputCurrent", "z_score": 5.0},
        {"feature": "X_ActualPosition", "z_score": 5.0},
    ]]

    assert estimate_cause(batches) == "tool_wear"
