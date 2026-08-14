import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def embed_text(client, text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    response = client.embeddings.create(model=model, input=text)
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    return normalize_vector(vector)


def search(
    query: str, corpus: list[dict], index, embed_fn, top_k: int = 3
) -> list[dict]:
    query_vector = embed_fn(query).astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        chunk = dict(corpus[idx])
        chunk["score"] = float(distances[0][rank])
        results.append(chunk)
    return results
