import pytest

from retraining.gate import evaluate_gate

CHAMPION_RECALL = 10 / 11  # 0.9091 — 현 champion 실측


def test_promoted_when_both_conditions_pass():
    result = evaluate_gate(
        retrained_recall=CHAMPION_RECALL,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["decision"] == "promoted"
    assert result["g1_pass"] is True
    assert result["g2_pass"] is True
    assert result["reject_reason"] == ""


def test_g1_boundary_one_extra_miss_passes():
    # 9/11 = 0.8182, 허용선 0.9091 - 0.10 = 0.8091 → 통과
    result = evaluate_gate(
        retrained_recall=9 / 11,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["g1_pass"] is True
    assert result["decision"] == "promoted"


def test_g1_boundary_two_extra_misses_rejects():
    # 8/11 = 0.7273 < 0.8091 → 거부
    result = evaluate_gate(
        retrained_recall=8 / 11,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["g1_pass"] is False
    assert result["decision"] == "rejected"
    assert "G1" in result["reject_reason"]


def test_g2_rejects_when_no_improvement():
    result = evaluate_gate(
        retrained_recall=CHAMPION_RECALL,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.70,
        champion_accuracy=0.70,
    )

    assert result["g2_pass"] is False
    assert result["decision"] == "rejected"
    assert "G2" in result["reject_reason"]


def test_reject_reason_lists_both_violations():
    result = evaluate_gate(
        retrained_recall=0.10,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.10,
        champion_accuracy=0.70,
    )

    assert "G1" in result["reject_reason"]
    assert "G2" in result["reject_reason"]


def test_accuracy_delta_is_reported():
    result = evaluate_gate(
        retrained_recall=CHAMPION_RECALL,
        champion_recall=CHAMPION_RECALL,
        retrained_accuracy=0.85,
        champion_accuracy=0.70,
    )

    assert result["g2_accuracy_delta"] == pytest.approx(0.15)


def test_accuracy_from_pairs_counts_matches():
    from retraining.gate import accuracy_from_pairs

    assert accuracy_from_pairs(["good", "bad", "good"], ["good", "bad", "bad"]) == pytest.approx(2 / 3)


def test_accuracy_from_pairs_all_correct():
    from retraining.gate import accuracy_from_pairs

    assert accuracy_from_pairs(["good", "bad"], ["good", "bad"]) == 1.0


def test_accuracy_from_pairs_rejects_length_mismatch():
    from retraining.gate import accuracy_from_pairs

    with pytest.raises(ValueError, match="길이가 다릅니다"):
        accuracy_from_pairs(["good", "bad"], ["good"])
