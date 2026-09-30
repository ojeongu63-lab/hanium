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


import faiss
import numpy as np

from rag.guide import select_chunks
from rag.playbook import PLAYBOOK_SOURCE

_PB = lambda name, cat, ctype="cause": {  # noqa: E731
    "name": name, "heading": f"{name} — x", "text": name, "fault_category": cat,
    "content_type": ctype, "source": PLAYBOOK_SOURCE, "signature": [],
    "title": "팀 시나리오 플레이북(자체 작성)", "url": "rag/sources/scenario_playbook.md",
}
_SELECT_CORPUS = [
    _PB("공구 마모", "tool_wear"), _PB("스핀들 베어링 손상", "tool_wear"),
    _PB("이송축 과부하", "feed_overload"),
    _PB("온도 드리프트", "general", "context"), _PB("소재 변경", "general", "context"),
    {"heading": "플랭크 마모", "text": "s1", "fault_category": "tool_wear", "content_type": "cause",
     "title": "Sandvik", "url": "https://x"},
    {"heading": "크레이터 마모", "text": "s2", "fault_category": "tool_wear", "content_type": "cause",
     "title": "Sandvik", "url": "https://x"},
    {"heading": "약한 고정구", "text": "s3", "fault_category": "vibration_backlash", "content_type": "cause",
     "title": "Sandvik", "url": "https://x"},
    {"heading": "안전 1", "text": "o1", "fault_category": "general", "content_type": "safety",
     "title": "OSHA", "url": "https://y"},
]
# 인덱스: 청크 i 의 벡터 = 단위벡터 e_i 를 살짝 섞어, 질의 [0,..,1(6번),..] 이 크레이터(6) > 플랭크(5)
_VECTORS = np.eye(len(_SELECT_CORPUS), dtype=np.float32)
_VECTORS[5, 6] = 0.5
_INDEX = faiss.IndexFlatIP(_VECTORS.shape[1])
_INDEX.add(_VECTORS)
_PREDICT = {"feature_contributions": [{"feature": "S_OutputCurrent", "error": 1.0, "z_score": 30.0}]}


def _fault(verdict, situation="공구 마모", category="tool_wear", alternatives=("스핀들 베어링 손상",), other=None):
    return {"verdict": verdict, "verdict_ko": "", "situation": situation, "category": category,
            "coverage": 0.8, "matched_features": ["S_OutputCurrent"], "alternatives": list(alternatives),
            "other_group": other, "top_z": 30.0}


def _names(chunks):
    return [c.get("name") or c["heading"] for c in chunks]


def test_select_chunks_confirmed_uses_embedding_once_for_same_category_external_chunks():
    calls = []

    def embed_fn(text):
        calls.append(text)
        return _VECTORS[6]

    chunks = select_chunks(_fault("confirmed"), _SELECT_CORPUS, _INDEX, embed_fn, _PREDICT)

    assert _names(chunks) == ["공구 마모", "스핀들 베어링 손상", "크레이터 마모", "플랭크 마모", "안전 1"]
    assert len(calls) == 1


def test_select_chunks_composite_adds_other_group_and_general_without_embedding():
    def embed_fn(_text):
        raise AssertionError("임베딩을 부르면 안 됨")

    other = {"situation": "이송축 과부하", "category": "feed_overload", "coverage": 0.44}
    chunks = select_chunks(_fault("composite", other=other), _SELECT_CORPUS, None, embed_fn, _PREDICT)

    assert _names(chunks) == ["공구 마모", "이송축 과부하", "온도 드리프트", "소재 변경", "안전 1"]


def test_select_chunks_weak_and_unknown():
    def embed_fn(_text):
        raise AssertionError("임베딩을 부르면 안 됨")

    weak = select_chunks(_fault("weak"), _SELECT_CORPUS, None, embed_fn, _PREDICT)
    assert _names(weak) == ["공구 마모", "온도 드리프트", "소재 변경", "안전 1"]

    unknown = select_chunks(_fault("unknown", situation=None, category=None, alternatives=()),
                            _SELECT_CORPUS, None, embed_fn, _PREDICT)
    assert _names(unknown) == ["온도 드리프트", "소재 변경", "안전 1"]


def test_build_guide_passes_fault_to_generation(monkeypatch):
    import rag.guide as guide_module

    captured = {}

    def fake_generate(predict_result, chunks, client, fault=None):
        captured["fault"] = fault
        captured["names"] = _names(chunks)
        return {"cause_estimate": "ok"}

    monkeypatch.setattr(guide_module, "generate_guide", fake_generate)
    monkeypatch.setattr(guide_module, "embed_text", lambda client, text: _VECTORS[6])

    result = build_guide(
        {"predicted_label_text": "bad", **_PREDICT, "fault": _fault("confirmed")},
        rag_corpus=_SELECT_CORPUS, rag_index=_INDEX, openai_client=object(),
    )

    assert result == {"cause_estimate": "ok"}
    assert captured["fault"]["situation"] == "공구 마모"
    assert captured["names"][0] == "공구 마모"


def test_build_guide_without_fault_keeps_top3_search_path(monkeypatch):
    import rag.guide as guide_module

    captured = {}

    def fake_generate(predict_result, chunks, client, fault=None):
        captured["fault"] = fault
        captured["n"] = len(chunks)
        return {"cause_estimate": "ok"}

    monkeypatch.setattr(guide_module, "generate_guide", fake_generate)
    monkeypatch.setattr(guide_module, "embed_text", lambda client, text: _VECTORS[6])

    build_guide({"predicted_label_text": "bad", **_PREDICT},
                rag_corpus=_SELECT_CORPUS, rag_index=_INDEX, openai_client=object())

    assert captured["fault"] is None
    assert captured["n"] == 3
