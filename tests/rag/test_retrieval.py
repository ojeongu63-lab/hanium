import faiss
import numpy as np
import pytest

from rag.retrieval import normalize_vector, search


def _build_index(vectors: np.ndarray):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def test_normalize_vector_has_unit_length():
    vector = np.array([3.0, 4.0], dtype=np.float32)
    normalized = normalize_vector(vector)
    assert np.linalg.norm(normalized) == pytest.approx(1.0)


def test_search_returns_top_k_closest_chunks_in_order():
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32
    )
    index = _build_index(vectors)
    corpus = [
        {"text": "chunk about tool wear", "fault_category": "tool_wear"},
        {"text": "chunk about vibration", "fault_category": "vibration_backlash"},
        {"text": "chunk about tool wear 2", "fault_category": "tool_wear"},
    ]

    def embed_fn(_query: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    results = search("query", corpus, index, embed_fn, top_k=2)

    assert [r["fault_category"] for r in results] == ["tool_wear", "tool_wear"]
    assert all("score" in r for r in results)
