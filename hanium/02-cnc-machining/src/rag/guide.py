from .generation import generate_guide
from .query import build_query
from .retrieval import embed_text, search

GOOD_GUIDE = {
    "cause_estimate": "이상 없음",
    "confidence_note": None,
    "recommended_actions": [],
    "safety_notes": [],
    "sources": [],
}


def build_guide(
    predict_result: dict, rag_corpus, rag_index, openai_client
) -> dict | None:
    if predict_result["predicted_label_text"] == "good":
        return dict(GOOD_GUIDE)

    if rag_corpus is None or rag_index is None or openai_client is None:
        return None

    try:
        query = build_query(predict_result["feature_contributions"])
        chunks = search(
            query,
            rag_corpus,
            rag_index,
            embed_fn=lambda text: embed_text(openai_client, text),
        )
        return generate_guide(predict_result, chunks, openai_client)
    except Exception as exc:
        print(f"RAG 가이드 생성 실패: {exc}")
        return None
