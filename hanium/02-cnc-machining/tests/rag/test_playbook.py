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
