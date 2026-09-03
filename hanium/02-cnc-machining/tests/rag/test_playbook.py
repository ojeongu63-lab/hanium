from collections import Counter
from pathlib import Path

import pytest

from preprocessing.columns import FEATURE_COLUMNS
from rag.playbook import PLAYBOOK_SOURCE, parse_playbook

ROOT = Path(__file__).resolve().parent.parent.parent
PLAYBOOK_TEXT = (ROOT / "rag" / "sources" / "scenario_playbook.md").read_text()

_MINI = """# 제목
## 1. 스핀들 부하 상승
### 공구 마모 — 부하 상승
관련 센서: S_OutputCurrent, S_OutputPower
증상: 스핀들 부하 상승.
조치: 교체.
## 4. 고장이 아닌 변화
### 온도 드리프트 — 서서히 이동
관련 센서: 없음
증상: 여러 센서 이동.
"""


def test_parse_playbook_reads_sections_signatures_and_names():
    chunks = parse_playbook(_MINI)

    assert [c["name"] for c in chunks] == ["공구 마모", "온도 드리프트"]
    assert chunks[0]["heading"] == "공구 마모 — 부하 상승"
    assert chunks[0]["fault_category"] == "tool_wear"
    assert chunks[0]["content_type"] == "cause"
    assert chunks[0]["signature"] == ["S_OutputCurrent", "S_OutputPower"]
    assert chunks[0]["source"] == PLAYBOOK_SOURCE
    assert chunks[0]["text"].startswith("관련 센서: S_OutputCurrent")
    assert chunks[1]["fault_category"] == "general"
    assert chunks[1]["content_type"] == "context"
    assert chunks[1]["signature"] == []


def test_parse_playbook_rejects_entry_without_signature_line():
    text = "## 1. 스핀들 부하 상승\n### 공구 마모 — x\n증상: 없음\n"
    with pytest.raises(ValueError, match="관련 센서"):
        parse_playbook(text)


def test_parse_playbook_rejects_unknown_feature_code():
    with pytest.raises(ValueError, match="S_OutputPower"):
        parse_playbook(_MINI, known_features={"S_OutputCurrent"})


def test_parse_playbook_rejects_unknown_section():
    with pytest.raises(ValueError, match="구역"):
        parse_playbook("## 9. 없는 구역\n### a — b\n관련 센서: 없음\n증상: x\n")


def test_real_playbook_has_16_entries_with_valid_codes():
    chunks = parse_playbook(PLAYBOOK_TEXT, known_features=set(FEATURE_COLUMNS))

    assert len(chunks) == 16
    assert Counter(c["fault_category"] for c in chunks) == {
        "tool_wear": 5, "feed_overload": 3, "vibration_backlash": 4, "general": 4,
    }
    firsts = {}
    for c in chunks:
        firsts.setdefault(c["fault_category"], c["name"])
    assert firsts == {
        "tool_wear": "공구 마모", "feed_overload": "이송축 과부하",
        "vibration_backlash": "고정구 풀림·채터", "general": "온도 드리프트",
    }
    assert all(len(c["name"]) < len(c["heading"]) for c in chunks)


import json

from rag.playbook import (
    NO_FAULT, TOP_N, WEAK_Z, coverage, match_playbook,
)

_RULE_CORPUS = [
    {"name": "공구 마모", "fault_category": "tool_wear", "source": PLAYBOOK_SOURCE,
     "signature": ["S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback"]},
    {"name": "스핀들 베어링 손상", "fault_category": "tool_wear", "source": PLAYBOOK_SOURCE,
     "signature": ["S_OutputCurrent", "S_ActualVelocity"]},
    {"name": "이송축 과부하", "fault_category": "feed_overload", "source": PLAYBOOK_SOURCE,
     "signature": ["X_OutputCurrent", "Y_OutputCurrent", "X_OutputPower", "Y_OutputPower"]},
    {"name": "윤활 불량", "fault_category": "feed_overload", "source": PLAYBOOK_SOURCE,
     "signature": ["X_OutputCurrent", "Y_OutputCurrent"]},
    {"name": "온도 드리프트", "fault_category": "general", "source": PLAYBOOK_SOURCE,
     "signature": []},
    {"heading": "플랭크 마모", "fault_category": "tool_wear", "content_type": "cause",
     "text": "Sandvik 청크 — signature 없음"},
]


def _contribs(*features: str, top_z: float = 100.0) -> list[dict]:
    """z를 순위대로 내림차순으로 만든다. 첫 피처의 z가 top_z."""
    return [
        {"feature": f, "error": 1.0, "z_score": top_z / (i + 1)}
        for i, f in enumerate(features)
    ]


def test_coverage_weights_by_rank_over_top_n():
    contribs = _contribs("a", "b", "c", "d", "e", "f")
    assert coverage(["a", "b", "c"], contribs) == pytest.approx(0.80)   # (1+1/2+1/3)/(1+..+1/5)
    assert coverage(["a", "b"], contribs) == pytest.approx(0.66)
    assert coverage(["f"], contribs) == 0.0                              # 6위는 대조 밖
    assert coverage(["z"], contribs) == 0.0
    assert coverage(["a"], []) == 0.0
    assert TOP_N == 5


def test_match_confirmed_picks_best_signature_and_lists_alternatives():
    result = match_playbook(_contribs("S_OutputCurrent", "S_CurrentFeedback", "S_OutputPower"), _RULE_CORPUS)

    assert result["verdict"] == "confirmed"
    assert result["verdict_ko"] == "확정"
    assert result["situation"] == "공구 마모"
    assert result["category"] == "tool_wear"
    assert result["coverage"] == pytest.approx(1.0)
    assert result["matched_features"] == ["S_OutputCurrent", "S_CurrentFeedback", "S_OutputPower"]
    assert result["alternatives"] == ["스핀들 베어링 손상"]
    assert result["other_group"] is None          # 다른 그룹 점수 0
    assert result["top_z"] == pytest.approx(100.0)


def test_match_tie_goes_to_earlier_entry():
    result = match_playbook(_contribs("X_OutputCurrent", "Y_OutputCurrent"), _RULE_CORPUS)

    assert result["situation"] == "이송축 과부하"       # 윤활 불량과 동점(1.0), 앞선 항목
    assert result["alternatives"] == ["윤활 불량"]


def test_match_weak_when_top_z_below_threshold():
    result = match_playbook(_contribs("S_OutputCurrent", "S_OutputPower", top_z=WEAK_Z - 0.1), _RULE_CORPUS)

    assert result["verdict"] == "weak"
    assert result["verdict_ko"] == "약한 신호"
    assert result["situation"] == "공구 마모"           # 참고로 채움


def test_match_composite_when_other_group_is_half_of_best():
    contribs = _contribs("S_OutputCurrent", "X_OutputCurrent", "Y_OutputCurrent", "S_OutputPower", "S_CurrentFeedback")
    result = match_playbook(contribs, _RULE_CORPUS)

    assert result["verdict"] == "composite"            # 공구 마모 0.64 vs 이송축 과부하 0.36
    assert result["situation"] == "공구 마모"
    assert result["other_group"] == {"situation": "이송축 과부하", "category": "feed_overload",
                                     "coverage": pytest.approx(0.36)}


def test_match_unknown_when_nothing_matches():
    result = match_playbook(_contribs("X_DCBusVoltage", "Y_DCBusVoltage"), _RULE_CORPUS)

    assert result["verdict"] == "unknown"
    assert result["situation"] is None
    assert result["category"] is None
    assert result["coverage"] == 0.0
    assert result["top_z"] == pytest.approx(100.0)


def test_match_returns_none_without_playbook_entries():
    assert match_playbook(_contribs("S_OutputCurrent"), [_RULE_CORPUS[-1]]) is None
    assert match_playbook(_contribs("S_OutputCurrent"), []) is None


def test_no_fault_shape_matches_match_result_keys():
    result = match_playbook(_contribs("S_OutputCurrent"), _RULE_CORPUS)
    assert set(NO_FAULT) == set(result)
    assert NO_FAULT["verdict"] == "none"
    assert NO_FAULT["verdict_ko"] == "이상 없음"


_RECORDED = {
    "synthetic/scenarios/tool_wear_predict_result.json": "공구 마모",
    "synthetic/scenarios/feed_overload_predict_result.json": "이송축 과부하",
    "synthetic/scenarios/vibration_backlash_predict_result.json": "고정구 풀림·채터",
    "docs/examples/predict_response_experiment_07.json": "이송축 과부하",
}


@pytest.mark.parametrize("path,expected", list(_RECORDED.items()))
def test_real_playbook_matches_recorded_cases(path, expected):
    corpus = parse_playbook(PLAYBOOK_TEXT, known_features=set(FEATURE_COLUMNS))
    recorded = json.loads((ROOT / path).read_text())

    result = match_playbook(recorded["feature_contributions"], corpus)

    assert result["situation"] == expected
    assert result["verdict"] == "confirmed"
