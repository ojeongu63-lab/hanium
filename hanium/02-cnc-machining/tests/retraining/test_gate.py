import pytest

from retraining.gate import accuracy_from_pairs, evaluate_gate, evaluate_shadow

# 현 champion 은 eval 불량 11개 중 10개를 잡는다 → 1건 놓침.
CHAMPION_MISSED = 1


def test_promoted_when_both_conditions_pass():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED,
        champion_missed=CHAMPION_MISSED,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["decision"] == "promoted"
    assert result["g1_pass"] is True
    assert result["g2_pass"] is True
    assert result["reject_reason"] == ""


def test_g1_boundary_one_extra_miss_passes():
    # champion 1건 놓침 + 허용 1건 = 2건까지 통과
    result = evaluate_gate(
        retrained_missed=2,
        champion_missed=CHAMPION_MISSED,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["g1_pass"] is True
    assert result["decision"] == "promoted"


def test_g1_boundary_two_extra_misses_rejects():
    result = evaluate_gate(
        retrained_missed=3,
        champion_missed=CHAMPION_MISSED,
        retrained_accuracy=0.90,
        champion_accuracy=0.70,
    )

    assert result["g1_pass"] is False
    assert result["decision"] == "rejected"
    assert "G1" in result["reject_reason"]


def test_g1_passes_when_model_catches_everything():
    # 모든 것을 불량이라 판정하는 모델은 놓친 개수 0 이라 G1 을 통과한다.
    # 이것이 G2 가 반드시 필요한 이유다 — 실제 실행에서 벌어진 상황이기도 하다.
    result = evaluate_gate(
        retrained_missed=0,
        champion_missed=CHAMPION_MISSED,
        retrained_accuracy=0.40,
        champion_accuracy=1.00,
    )

    assert result["g1_pass"] is True
    assert result["g2_pass"] is False
    assert result["decision"] == "rejected"


def test_g2_rejects_when_no_improvement():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED,
        champion_missed=CHAMPION_MISSED,
        retrained_accuracy=0.70,
        champion_accuracy=0.70,
    )

    assert result["g2_pass"] is False
    assert result["decision"] == "rejected"
    assert "G2" in result["reject_reason"]


def test_reject_reason_lists_both_violations():
    result = evaluate_gate(
        retrained_missed=9,
        champion_missed=CHAMPION_MISSED,
        retrained_accuracy=0.10,
        champion_accuracy=0.70,
    )

    assert "G1" in result["reject_reason"]
    assert "G2" in result["reject_reason"]


def test_accuracy_delta_is_reported():
    result = evaluate_gate(
        retrained_missed=CHAMPION_MISSED,
        champion_missed=CHAMPION_MISSED,
        retrained_accuracy=0.85,
        champion_accuracy=0.70,
    )

    assert result["g2_accuracy_delta"] == pytest.approx(0.15)


def test_accuracy_from_pairs_counts_matches():
    assert accuracy_from_pairs(
        ["good", "bad", "good"], ["good", "bad", "bad"]
    ) == pytest.approx(2 / 3)


def test_accuracy_from_pairs_all_correct():
    assert accuracy_from_pairs(["good", "bad"], ["good", "bad"]) == 1.0


def test_accuracy_from_pairs_rejects_length_mismatch():
    with pytest.raises(ValueError, match="길이가 다릅니다"):
        accuracy_from_pairs(["good", "bad"], ["good"])


def test_shadow_promotes_when_candidate_better():
    verdict = evaluate_shadow(candidate_accuracy=0.9, champion_accuracy=0.7)
    assert verdict["decision"] == "promoted"
    assert verdict["accuracy_delta"] == pytest.approx(0.2)


def test_shadow_rejects_when_candidate_not_better():
    verdict = evaluate_shadow(candidate_accuracy=0.7, champion_accuracy=0.7)
    assert verdict["decision"] == "rejected"
    assert verdict["accuracy_delta"] == pytest.approx(0.0)
