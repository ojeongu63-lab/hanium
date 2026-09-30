from .generation import generate_cause_guide, generate_guide
from .playbook import PLAYBOOK_SOURCE
from .query import build_query
from .retrieval import embed_text, search

SAFETY_CHUNKS = 3
EXTERNAL_CHUNKS = 2  # 확정 판정에서 같은 카테고리의 비플레이북 cause 청크 수

GOOD_GUIDE = {
    "cause_estimate": "이상 없음",
    "confidence_note": None,
    "recommended_actions": [],
    "safety_notes": [],
    "sources": [],
}


def select_chunks(fault: dict, corpus: list[dict], index, embed_fn, predict_result: dict) -> list[dict]:
    """판정별 참고 문서. 임베딩은 확정 판정에서 Sandvik 등 외부 청크를 고를 때 한 번만 쓴다."""
    playbook = {c["name"]: c for c in corpus if c.get("source") == PLAYBOOK_SOURCE}
    general = [c for c in playbook.values() if c["fault_category"] == "general"]
    safety = [c for c in corpus if c["content_type"] == "safety"][:SAFETY_CHUNKS]
    verdict = fault["verdict"]

    if verdict == "unknown":
        return general + safety
    selected = playbook[fault["situation"]]
    if verdict == "weak":
        return [selected] + general + safety
    if verdict == "composite":
        other = [playbook[fault["other_group"]["situation"]]] if fault["other_group"] else []
        return [selected] + other + general + safety

    alternatives = [playbook[name] for name in fault["alternatives"]]
    hits = search(build_query(predict_result["feature_contributions"]), corpus, index, embed_fn, top_k=len(corpus))
    external = [
        h for h in hits
        if h.get("source") != PLAYBOOK_SOURCE
        and h["fault_category"] == fault["category"]
        and h["content_type"] == "cause"
    ][:EXTERNAL_CHUNKS]
    return [selected] + alternatives + external + safety


def build_guide(
    predict_result: dict, rag_corpus, rag_index, openai_client
) -> dict | None:
    if predict_result["predicted_label_text"] == "good":
        return dict(GOOD_GUIDE)

    if rag_corpus is None or rag_index is None or openai_client is None:
        return None

    try:
        embed_fn = lambda text: embed_text(openai_client, text)  # noqa: E731
        fault = predict_result.get("fault")
        if fault is None:
            chunks = search(
                build_query(predict_result["feature_contributions"]), rag_corpus, rag_index, embed_fn
            )
        else:
            chunks = select_chunks(fault, rag_corpus, rag_index, embed_fn, predict_result)
        return generate_guide(predict_result, chunks, openai_client, fault)
    except Exception as exc:
        print(f"RAG 가이드 생성 실패: {exc}")
        return None


def build_cause_guide(cause: str, rag_corpus: list[dict] | None, openai_client) -> dict | None:
    if rag_corpus is None or openai_client is None:
        return None
    try:
        chunks = [c for c in rag_corpus if c["fault_category"] == cause]
        chunks += [c for c in rag_corpus if c["content_type"] == "safety"]
        return generate_cause_guide(cause, chunks, openai_client)
    except Exception as exc:
        print(f"원인 설명 생성 실패: {exc}")
        return None
