from rag.guide import build_guide


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
