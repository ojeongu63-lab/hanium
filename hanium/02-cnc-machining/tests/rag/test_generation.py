import json

from rag.generation import generate_guide


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeChatCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


_PAYLOAD = {
    "cause_estimate": "공구 마모 가능성이 있습니다",
    "confidence_note": "간접 추정입니다",
    "recommended_actions": ["점검하세요"],
    "safety_notes": ["전원을 차단하세요"],
    "sources": [{"title": "Sandvik", "url": "https://x"}],
}

_PREDICT_RESULT = {
    "predicted_label_text": "bad",
    "score": 5.0,
    "threshold": 1.0,
    "feature_contributions": [
        {"feature": "S_OutputCurrent", "error": 1.0, "z_score": 36.1}
    ],
}


def test_generate_guide_parses_json_response():
    client = _FakeClient(json.dumps(_PAYLOAD))

    guide = generate_guide(_PREDICT_RESULT, [], client)

    assert guide == _PAYLOAD


def test_generate_guide_includes_confidence_principle_in_system_prompt():
    client = _FakeClient(json.dumps(_PAYLOAD))

    generate_guide(_PREDICT_RESULT, [], client)

    messages = client.chat.completions.last_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "단정하지" in messages[0]["content"]
    assert "통계적 이상" in messages[0]["content"]   # 옛 tool_condition 문장은 공구 마모를 유도해 제거함


def test_generate_guide_includes_retrieved_chunk_text_in_user_prompt():
    client = _FakeClient(json.dumps(_PAYLOAD))
    chunks = [{
        "title": "Sandvik", "url": "https://x", "content_type": "cause",
        "text": "Reduce cutting speed",
    }]

    generate_guide(_PREDICT_RESULT, chunks, client)

    messages = client.chat.completions.last_kwargs["messages"]
    assert "Reduce cutting speed" in messages[1]["content"]


from rag.generation import generate_cause_guide


def test_generate_cause_guide_parses_json_response():
    client = _FakeClient(json.dumps(_PAYLOAD))

    guide = generate_cause_guide("tool_wear", [], client)

    assert guide == _PAYLOAD


def test_generate_cause_guide_includes_confidence_principle_in_system_prompt():
    client = _FakeClient(json.dumps(_PAYLOAD))

    generate_cause_guide("tool_wear", [], client)

    messages = client.chat.completions.last_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "단정하지" in messages[0]["content"]


def test_generate_cause_guide_includes_cause_and_chunk_text_in_user_prompt():
    client = _FakeClient(json.dumps(_PAYLOAD))
    chunks = [{
        "title": "Sandvik", "url": "https://x", "content_type": "cause",
        "text": "Reduce cutting speed",
    }]

    generate_cause_guide("tool_wear", chunks, client)

    messages = client.chat.completions.last_kwargs["messages"]
    assert "tool_wear" in messages[1]["content"]
    assert "Reduce cutting speed" in messages[1]["content"]


from rag.generation import SYSTEM_PROMPT, describe_fault

_CHUNK = {"title": "팀 시나리오 플레이북(자체 작성)", "url": "rag/sources/scenario_playbook.md",
          "content_type": "cause", "text": "관련 센서: S_OutputCurrent\n조치: 교체"}

_CONFIRMED = {
    "verdict": "confirmed", "verdict_ko": "높은 패턴 일치", "situation": "공구 마모", "category": "tool_wear",
    "coverage": 0.8, "matched_features": ["S_OutputCurrent", "S_OutputPower"],
    "alternatives": ["스핀들 베어링 손상"], "other_group": None, "top_z": 36.1,
}


def test_system_prompt_no_longer_primes_tool_wear():
    assert "tool_condition" not in SYSTEM_PROMPT
    assert "센서 패턴 대조" in SYSTEM_PROMPT


def test_system_prompt_forbids_confirmation_wording():
    # 멘토 피드백: AI가 원인을 "확정"한 것처럼 읽히는 표현을 쓰지 않는다
    assert "확정" in SYSTEM_PROMPT and "쓰지 마세요" in SYSTEM_PROMPT
    assert "확률이 아니" in SYSTEM_PROMPT


def test_describe_fault_per_verdict():
    assert describe_fault(_CONFIRMED) == (
        "센서 패턴 대조: 높은 패턴 일치 — 공구 마모 (일치도 0.80, 일치 센서: S_OutputCurrent, S_OutputPower)\n"
        "같은 구역의 다른 후보(현장 확인으로 구분): 스핀들 베어링 손상"
    )
    composite = {**_CONFIRMED, "verdict": "composite", "coverage": 0.66,
                 "other_group": {"situation": "이송축 과부하", "category": "feed_overload", "coverage": 0.44}}
    assert describe_fault(composite).startswith(
        "센서 패턴 대조: 복합 패턴 — 공구 마모(0.66)와 이송축 과부하(0.44)가 함께 나타남"
    )
    weak = {**_CONFIRMED, "verdict": "weak", "top_z": 4.4}
    assert describe_fault(weak) == (
        "센서 패턴 대조: 약한 신호 — 상위 센서 z 4.4 (기준 10 미만). 보류·재확인을 권할 것. 참고 상황: 공구 마모"
    )
    unknown = {**_CONFIRMED, "verdict": "unknown", "situation": None}
    assert describe_fault(unknown) == "센서 패턴 대조: 일치 패턴 없음 — 서명이 일치하는 상황 없음. 현장 확인을 권할 것."


def test_generate_guide_includes_fault_line_only_when_given():
    client = _FakeClient(json.dumps(_PAYLOAD))
    generate_guide(_PREDICT_RESULT, [_CHUNK], client, fault=_CONFIRMED)
    with_fault = client.chat.completions.last_kwargs["messages"][1]["content"]
    assert "센서 패턴 대조: 높은 패턴 일치 — 공구 마모" in with_fault
    assert with_fault.index("상위 이상 피처") < with_fault.index("센서 패턴 대조") < with_fault.index("참고 문서")

    generate_guide(_PREDICT_RESULT, [_CHUNK], client)
    without = client.chat.completions.last_kwargs["messages"][1]["content"]
    assert "센서 패턴 대조" not in without
