from rag.guide import build_guide, build_cause_guide


def test_build_guide_returns_fixed_message_for_good():
    result = build_guide(
        {"predicted_label_text": "good"}, rag_corpus=None, rag_index=None,
        openai_client=None,
    )

    assert result["cause_estimate"] == "이상 없음"
    assert result["recommended_actions"] == []


def test_build_guide_returns_none_when_rag_unavailable():
    result = build_guide(
        {"predicted_label_text": "bad", "feature_contributions": []},
        rag_corpus=None, rag_index=None, openai_client=None,
    )

    assert result is None


def test_build_guide_returns_none_on_pipeline_exception():
    class _BrokenClient:
        pass  # .embeddings 속성이 없어 AttributeError 발생

    result = build_guide(
        {
            "predicted_label_text": "bad",
            "feature_contributions": [
                {"feature": "f", "error": 1.0, "z_score": 1.0}
            ],
        },
        rag_corpus=[{"text": "x"}],
        rag_index=object(),
        openai_client=_BrokenClient(),
    )

    assert result is None


_CAUSE_CORPUS = [
    {"title": "Sandvik", "url": "https://x", "fault_category": "tool_wear",
     "content_type": "cause", "text": "공구 마모 조치"},
    {"title": "Sandvik", "url": "https://x", "fault_category": "vibration_backlash",
     "content_type": "cause", "text": "진동 조치"},
    {"title": "OSHA", "url": "https://y", "fault_category": "general",
     "content_type": "safety", "text": "안전 수칙"},
]


def test_build_cause_guide_returns_none_when_rag_unavailable():
    result = build_cause_guide("tool_wear", rag_corpus=None, openai_client=None)

    assert result is None


def test_build_cause_guide_filters_corpus_by_cause_and_includes_safety():
    captured = {}

    def fake_generate(cause, chunks, client):
        captured["cause"] = cause
        captured["chunks"] = chunks
        return {"cause_estimate": "x", "recommended_actions": []}

    import rag.guide as guide_module
    original = guide_module.generate_cause_guide
    guide_module.generate_cause_guide = fake_generate
    try:
        build_cause_guide("tool_wear", _CAUSE_CORPUS, openai_client=object())
    finally:
        guide_module.generate_cause_guide = original

    assert captured["cause"] == "tool_wear"
    texts = {c["text"] for c in captured["chunks"]}
    assert texts == {"공구 마모 조치", "안전 수칙"}


def test_build_cause_guide_returns_none_on_pipeline_exception():
    class _BrokenClient:
        pass

    result = build_cause_guide("tool_wear", _CAUSE_CORPUS, openai_client=_BrokenClient())

    assert result is None
