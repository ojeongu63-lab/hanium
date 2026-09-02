import pytest

from retraining.gate import evaluate_gate, evaluate_two_sided

# 현 champion 은 eval 불량 11개 중 10개를 잡는다 → 1건 놓침.
CHAMPION_MISSED = 1

# 정상 20건 창에서 후보가 오탐을 5 → 2로 줄인 두 방향 결과 (통과).
PROMOTED_G2 = {
    "n_good": 20, "n_bad": 0,
    "champion_false_alarms": 5, "candidate_false_alarms": 2,
    "champion_misses": 0, "candidate_misses": 0,
    "decision": "promoted", "reject_reason": "",
}
# 같은 창에서 오탐이 그대로인 결과 (거부).
REJECTED_G2 = {
    **PROMOTED_G2,
    "candidate_false_alarms": 5,
    "decision": "rejected",
    "reject_reason": "G2 개선 없음: 오탐·놓침 모두 champion과 동일",
}


def test_promoted_when_both_conditions_pass():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED, champion_missed=CHAMPION_MISSED, g2=PROMOTED_G2
    )

    assert result["decision"] == "promoted"
    assert result["g1_pass"] is True
    assert result["g2_pass"] is True
    assert result["reject_reason"] == ""


def test_g1_boundary_one_extra_miss_passes():
    # champion 1건 놓침 + 허용 1건 = 2건까지 통과
    result = evaluate_gate(retrained_missed=2, champion_missed=CHAMPION_MISSED, g2=PROMOTED_G2)

    assert result["g1_pass"] is True
    assert result["decision"] == "promoted"


def test_g1_boundary_two_extra_misses_rejects():
    result = evaluate_gate(retrained_missed=3, champion_missed=CHAMPION_MISSED, g2=PROMOTED_G2)

    assert result["g1_pass"] is False
    assert result["decision"] == "rejected"
    assert "G1" in result["reject_reason"]


def test_g1_passes_when_model_catches_everything():
    # 모든 것을 불량이라 판정하는 모델은 놓친 개수 0 이라 G1 을 통과한다.
    # 이것이 G2 가 반드시 필요한 이유다 — 실제 실행에서 벌어진 상황이기도 하다.
    g2 = evaluate_two_sided(["good"] * 4, ["good"] * 4, ["bad"] * 4)  # 후보 오탐 4 vs 0
    result = evaluate_gate(retrained_missed=0, champion_missed=CHAMPION_MISSED, g2=g2)

    assert result["g1_pass"] is True
    assert result["g2_pass"] is False
    assert result["decision"] == "rejected"


def test_g2_rejects_when_no_improvement():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED, champion_missed=CHAMPION_MISSED, g2=REJECTED_G2
    )

    assert result["g2_pass"] is False
    assert result["decision"] == "rejected"
    assert "G2" in result["reject_reason"]


def test_reject_reason_lists_both_violations():
    result = evaluate_gate(retrained_missed=9, champion_missed=CHAMPION_MISSED, g2=REJECTED_G2)

    assert "G1" in result["reject_reason"]
    assert "G2" in result["reject_reason"]


def test_g2_result_is_passed_through():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED, champion_missed=CHAMPION_MISSED, g2=PROMOTED_G2
    )

    assert result["g2"] is PROMOTED_G2
    assert "g2_accuracy_delta" not in result


def test_removed_accuracy_helpers_are_gone():
    import retraining.gate as gate

    assert not hasattr(gate, "accuracy_from_pairs")
    assert not hasattr(gate, "evaluate_shadow")


# ---- evaluate_two_sided: 오탐·놓침을 따로 세는 두 방향 규칙 -------------------

def test_all_good_window_promotes_when_false_alarms_drop():
    truths = ["good"] * 4
    g2 = evaluate_two_sided(
        truths,
        ["bad", "bad", "good", "good"],   # champion 오탐 2
        ["bad", "good", "good", "good"],  # 후보 오탐 1
    )
    assert g2["n_good"] == 4 and g2["n_bad"] == 0
    assert g2["champion_false_alarms"] == 2 and g2["candidate_false_alarms"] == 1
    assert g2["decision"] == "promoted"
    assert g2["reject_reason"] == ""


def test_all_good_window_rejects_when_false_alarms_equal():
    truths = ["good"] * 3
    g2 = evaluate_two_sided(truths, ["bad", "good", "good"], ["good", "bad", "good"])
    assert g2["decision"] == "rejected"
    assert "개선 없음" in g2["reject_reason"]


def test_all_bad_window_rejects_as_unmeasurable():
    # 09-02 fixture_loosening Day 34 — 후보가 놓침을 줄여도 오탐을 잴 수 없으면 거부.
    truths = ["bad"] * 3
    g2 = evaluate_two_sided(truths, ["good", "good", "good"], ["bad", "bad", "bad"])
    assert g2["n_good"] == 0
    assert g2["decision"] == "rejected"
    assert "정상 라벨 없음" in g2["reject_reason"]


def test_mixed_window_rejects_trading_misses_for_false_alarms():
    # 09-02 fixture Day 29 형태: 후보가 놓침은 줄이고 오탐은 늘림.
    truths = ["good", "good", "bad", "bad"]
    champion = ["good", "good", "good", "bad"]   # 오탐 0, 놓침 1
    candidate = ["bad", "good", "bad", "bad"]    # 오탐 1, 놓침 0
    g2 = evaluate_two_sided(truths, champion, candidate)
    assert g2["decision"] == "rejected"
    assert "오탐 회귀" in g2["reject_reason"]


def test_mixed_window_rejects_trading_false_alarms_for_misses():
    # 09-02 tool_wear Day 30 형태: 후보가 오탐은 줄이고 놓침은 늘림.
    truths = ["good", "good", "bad", "bad"]
    champion = ["bad", "good", "bad", "bad"]     # 오탐 1, 놓침 0
    candidate = ["good", "good", "good", "bad"]  # 오탐 0, 놓침 1
    g2 = evaluate_two_sided(truths, champion, candidate)
    assert g2["decision"] == "rejected"
    assert "놓침 회귀" in g2["reject_reason"]


def test_mixed_window_promotes_when_no_regression_and_one_side_improves():
    truths = ["good", "good", "bad", "bad"]
    champion = ["bad", "good", "good", "bad"]    # 오탐 1, 놓침 1
    candidate = ["good", "good", "good", "bad"]  # 오탐 0, 놓침 1
    g2 = evaluate_two_sided(truths, champion, candidate)
    assert g2["decision"] == "promoted"
    assert g2["reject_reason"] == ""


def test_empty_window_rejects_as_unmeasurable():
    g2 = evaluate_two_sided([], [], [])
    assert g2["decision"] == "rejected"
    assert "정상 라벨 없음" in g2["reject_reason"]


def test_two_sided_rejects_length_mismatch():
    with pytest.raises(ValueError, match="길이가 다릅니다"):
        evaluate_two_sided(["good", "bad"], ["good"], ["good", "bad"])
