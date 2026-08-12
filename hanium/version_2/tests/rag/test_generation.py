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
    assert "tool_condition" in messages[0]["content"]


def test_generate_guide_includes_retrieved_chunk_text_in_user_prompt():
    client = _FakeClient(json.dumps(_PAYLOAD))
    chunks = [{
        "title": "Sandvik", "url": "https://x", "content_type": "cause",
        "text": "Reduce cutting speed",
    }]

    generate_guide(_PREDICT_RESULT, chunks, client)

    messages = client.chat.completions.last_kwargs["messages"]
    assert "Reduce cutting speed" in messages[1]["content"]
