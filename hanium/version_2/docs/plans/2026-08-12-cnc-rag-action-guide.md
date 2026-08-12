# CNC RAG 기반 현장 조치 가이드 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/predict`가 "불량" 판정을 낼 때, 실제 공개 문서(Sandvik Coromant 트러블슈팅 + OSHA 안전문서)를 FAISS로 검색하고 OpenAI로 현장 조치 가이드를 생성해 응답에 포함시킨다.

**Architecture:** `src/rag/`에 4개 순수 모듈(피처 설명 → 질의 생성 → FAISS 검색 → OpenAI 생성)을 계층적으로 쌓고, `guide.py`가 이를 오케스트레이션하며 good/bad 분기와 에러 처리를 담당한다. `serving/inference.py`의 `predict_experiment()`가 이 오케스트레이터를 호출해 `guide` 필드를 응답에 추가한다. 코퍼스는 오프라인 스크립트(`rag/build_corpus.py`)가 로컬 마크다운(이미 확보됨) → 청크 → OpenAI 임베딩 → FAISS 인덱스로 1회 구축한다.

**Tech Stack:** Python (uv run), `openai`(신규), `faiss-cpu`(신규), 기존 FastAPI/pytest.

## Global Constraints

- 임베딩 모델 `text-embedding-3-small`, 생성 모델은 환경변수 `OPENAI_CHAT_MODEL`(기본값 `gpt-4o-mini`).
- `predicted_label_text == "good"`이면 LLM 호출 없이 고정 응답, `"bad"`일 때만 풀 RAG 실행.
- RAG 실패(코퍼스 미구축/API 키 없음/API 에러) 시 `guide: null`, 판정 자체는 절대 막지 않는다.
- 시스템 프롬프트에 "`tool_condition`을 학습하지 않았으니 단정하지 말고 확신도를 낮춘 표현 사용" 원칙 반드시 포함.
- `version_2/data/rag/`(코퍼스 산출물)는 `.gitignore` 대상(`data/` 전체 관례). `version_2/rag/sources/*.md`(원문)는 git 추적됨, 이미 존재함.
- 이번 서브프로젝트는 `src/rag/`뿐 아니라 `src/serving/`도 수정하는 서빙 파이프라인 정식 변경이라, 기존 70개 pytest 관례를 따라 정식 단위테스트를 작성한다.
- 신규 의존성(`openai`, `faiss-cpu`)은 `version_2/pyproject.toml`에 추가.

---

## File Structure

- Create: `version_2/src/rag/__init__.py`(빈 파일), `features.py`, `query.py`, `retrieval.py`, `generation.py`, `guide.py`
- Create: `version_2/tests/rag/__init__.py`(빈 파일), `test_features.py`, `test_query.py`, `test_retrieval.py`, `test_generation.py`, `test_guide.py`
- Create: `version_2/rag/build_corpus.py`
- Modify: `version_2/src/serving/inference.py`, `version_2/src/serving/app.py`, `version_2/tests/serving/test_inference.py`, `version_2/tests/serving/test_app.py`, `version_2/pyproject.toml`

## Task 1: 의존성 추가 + `src/rag/features.py`(피처 설명 사전)

**Files:**
- Modify: `version_2/pyproject.toml`
- Create: `version_2/src/rag/__init__.py`, `version_2/src/rag/features.py`
- Create: `version_2/tests/rag/__init__.py`, `version_2/tests/rag/test_features.py`

**Interfaces:**
- Produces: `rag.features.describe_feature(code: str) -> str`

- [ ] **Step 1: `pyproject.toml`에 의존성 추가**

`version_2/pyproject.toml`의 `dependencies` 배열에 다음 두 줄 추가(기존 항목들 뒤,
알파벳 순서 유지 안 해도 됨 — 기존 파일 순서를 따름):
```toml
"faiss-cpu>=1.9.0",
"openai>=1.50.0",
```

- [ ] **Step 2: 의존성 설치**

Run: `cd version_2 && uv sync`
Expected: 에러 없이 종료, `faiss` / `openai` 패키지가 `.venv`에 설치됨

- [ ] **Step 3: 빈 패키지 초기화 파일 생성**

`version_2/src/rag/__init__.py`: 빈 파일.
`version_2/tests/rag/__init__.py`: 빈 파일.

- [ ] **Step 4: 실패하는 테스트 작성**

`version_2/tests/rag/test_features.py`:
```python
from rag.features import describe_feature


def test_describe_feature_known():
    assert describe_feature("S_OutputCurrent") == "스핀들 출력 전류"


def test_describe_feature_unknown_falls_back_to_code():
    assert describe_feature("Z_SetVelocity") == "Z_SetVelocity"
```

- [ ] **Step 5: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_features.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'rag.features'` 또는 `ImportError`)

- [ ] **Step 6: `features.py` 구현**

`version_2/src/rag/features.py`:
```python
FEATURE_DESCRIPTIONS = {
    "S_OutputCurrent": "스핀들 출력 전류",
    "S_OutputPower": "스핀들 출력 파워",
    "S_CurrentFeedback": "스핀들 전류 피드백",
    "X_OutputCurrent": "X축 출력 전류",
    "X_OutputPower": "X축 출력 파워",
    "Y_OutputCurrent": "Y축 출력 전류",
    "Y_OutputPower": "Y축 출력 파워",
    "X_ActualPosition": "X축 실제 위치",
    "Y_ActualPosition": "Y축 실제 위치",
    "Z_ActualPosition": "Z축 실제 위치",
    "X_ActualVelocity": "X축 실제 속도",
    "Y_ActualVelocity": "Y축 실제 속도",
    "Z_ActualVelocity": "Z축 실제 속도",
}


def describe_feature(code: str) -> str:
    return FEATURE_DESCRIPTIONS.get(code, code)
```

- [ ] **Step 7: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_features.py -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Commit**

```bash
git add version_2/pyproject.toml version_2/uv.lock version_2/src/rag/ version_2/tests/rag/
git commit -m "Add RAG dependencies and feature description mapping"
```

## Task 2: `src/rag/query.py`(질의 생성)

**Files:**
- Create: `version_2/src/rag/query.py`
- Create: `version_2/tests/rag/test_query.py`

**Interfaces:**
- Consumes: Task 1의 `rag.features.describe_feature`
- Produces: `rag.query.build_query(feature_contributions: list[dict], top_n: int = 3) -> str`
  (`feature_contributions`는 `serving.inference.rank_feature_contributions()`가 만드는
  `{"feature": str, "error": float, "z_score": float}` 리스트, z_score 내림차순)

- [ ] **Step 1: 실패하는 테스트 작성**

`version_2/tests/rag/test_query.py`:
```python
from rag.query import build_query


def test_build_query_uses_top_n_sorted_contributions():
    contributions = [
        {"feature": "S_OutputCurrent", "error": 1.0, "z_score": 36.1},
        {"feature": "S_CurrentFeedback", "error": 0.9, "z_score": 35.9},
        {"feature": "S_OutputPower", "error": 0.8, "z_score": 21.1},
        {"feature": "X_OutputPower", "error": 0.1, "z_score": 1.0},
    ]

    query = build_query(contributions, top_n=3)

    assert "스핀들 출력 전류(z=36.1)" in query
    assert "스핀들 전류 피드백(z=35.9)" in query
    assert "스핀들 출력 파워(z=21.1)" in query
    assert "X_OutputPower" not in query


def test_build_query_falls_back_to_code_for_unknown_feature():
    contributions = [{"feature": "Z_SetVelocity", "error": 0.5, "z_score": 5.0}]
    query = build_query(contributions, top_n=1)
    assert "Z_SetVelocity(z=5.0)" in query
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_query.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: `query.py` 구현**

`version_2/src/rag/query.py`:
```python
from .features import describe_feature


def build_query(feature_contributions: list[dict], top_n: int = 3) -> str:
    top = feature_contributions[:top_n]
    parts = [
        f"{describe_feature(c['feature'])}(z={c['z_score']:.1f})" for c in top
    ]
    return "다음 센서 값이 정상 대비 크게 벗어났습니다: " + ", ".join(parts)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_query.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add version_2/src/rag/query.py version_2/tests/rag/test_query.py
git commit -m "Add RAG query builder"
```

## Task 3: `src/rag/retrieval.py`(FAISS 검색)

**Files:**
- Create: `version_2/src/rag/retrieval.py`
- Create: `version_2/tests/rag/test_retrieval.py`

**Interfaces:**
- Produces: `rag.retrieval.normalize_vector(vector: np.ndarray) -> np.ndarray`,
  `rag.retrieval.embed_text(client, text: str, model: str = "text-embedding-3-small") -> np.ndarray`,
  `rag.retrieval.search(query: str, corpus: list[dict], index, embed_fn, top_k: int = 3) -> list[dict]`
  (`embed_fn`은 `str -> np.ndarray`인 콜러블 — 테스트에서 스텁으로 교체 가능하게 주입)

- [ ] **Step 1: 실패하는 테스트 작성**

`version_2/tests/rag/test_retrieval.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_retrieval.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: `retrieval.py` 구현**

`version_2/src/rag/retrieval.py`:
```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_retrieval.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add version_2/src/rag/retrieval.py version_2/tests/rag/test_retrieval.py
git commit -m "Add FAISS-based RAG retrieval"
```

## Task 4: `src/rag/generation.py`(OpenAI 호출 + 프롬프트)

**Files:**
- Create: `version_2/src/rag/generation.py`
- Create: `version_2/tests/rag/test_generation.py`

**Interfaces:**
- Produces: `rag.generation.SYSTEM_PROMPT: str`,
  `rag.generation.generate_guide(predict_result: dict, retrieved_chunks: list[dict], client) -> dict`
  (`predict_result`는 최소 `predicted_label_text`, `score`, `threshold`,
  `feature_contributions` 키를 가짐. `client`는 OpenAI SDK 클라이언트 —
  `.chat.completions.create(model=..., messages=..., response_format=...)` 인터페이스)

- [ ] **Step 1: 실패하는 테스트 작성**

`version_2/tests/rag/test_generation.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_generation.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: `generation.py` 구현**

`version_2/src/rag/generation.py`:
```python
import json
import os

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "당신은 CNC 가공 현장의 이상탐지 결과를 설명하는 어시스턴트입니다.\n"
    "이 모델은 tool_condition(공구 마모 여부)을 입력으로 학습한 적이 없습니다 "
    "- 간접적인 센서 신호(전류/파워 등)로만 추정하는 것이므로, 원인을 단정하지 "
    "말고 '~일 가능성이 있습니다', '~로 추정됩니다' 같은 확신도를 낮춘 표현을 "
    "쓰세요.\n"
    "아래 JSON 스키마로만 답하세요:\n"
    '{"cause_estimate": str, "confidence_note": str, '
    '"recommended_actions": [str], "safety_notes": [str], '
    '"sources": [{"title": str, "url": str}]}'
)


def _build_user_prompt(predict_result: dict, retrieved_chunks: list[dict]) -> str:
    lines = [
        f"판정: {predict_result['predicted_label_text']}, "
        f"점수: {predict_result['score']:.3f} "
        f"(임계값 {predict_result['threshold']:.3f})"
    ]
    top3 = predict_result["feature_contributions"][:3]
    lines.append(
        "상위 이상 피처: "
        + ", ".join(f"{c['feature']}(z={c['z_score']:.1f})" for c in top3)
    )
    lines.append("\n참고 문서:")
    for chunk in retrieved_chunks:
        lines.append(
            f"- [{chunk['title']}]({chunk['url']}) "
            f"{chunk['content_type']}: {chunk['text']}"
        )
    return "\n".join(lines)


def generate_guide(
    predict_result: dict, retrieved_chunks: list[dict], client
) -> dict:
    model = os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(predict_result, retrieved_chunks),
            },
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_generation.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add version_2/src/rag/generation.py version_2/tests/rag/test_generation.py
git commit -m "Add OpenAI-based RAG guide generation"
```

## Task 5: `src/rag/guide.py`(오케스트레이터 + good/bad 분기 + 에러 처리)

**Files:**
- Create: `version_2/src/rag/guide.py`
- Create: `version_2/tests/rag/test_guide.py`

**Interfaces:**
- Consumes: Task 2~4의 `build_query`, `embed_text`, `search`, `generate_guide`
- Produces: `rag.guide.build_guide(predict_result: dict, rag_corpus: list[dict] | None, rag_index, openai_client) -> dict | None`

- [ ] **Step 1: 실패하는 테스트 작성**

`version_2/tests/rag/test_guide.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_guide.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: `guide.py` 구현**

`version_2/src/rag/guide.py`:
```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/rag/test_guide.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add version_2/src/rag/guide.py version_2/tests/rag/test_guide.py
git commit -m "Add RAG guide orchestration with good/bad branching and error handling"
```

## Task 6: `version_2/rag/build_corpus.py` — 코퍼스 구축 + 실행

**Files:**
- Create: `version_2/rag/build_corpus.py`

**Interfaces:**
- Consumes: `version_2/rag/sources/sandvik_milling_troubleshooting.md`,
  `version_2/rag/sources/osha_machine_guarding_lockout.md`(이미 존재)
- Produces: `version_2/data/rag/corpus.json`, `version_2/data/rag/corpus.index`
  (런타임 코드가 이 두 파일을 읽음 — Task 7에서 소비)

이 태스크는 정식 pytest 테스트를 만들지 않는다(OpenAI API를 실제로 호출하는
1회성 빌드 스크립트 — `loocv`/`synthetic`과 같은 관례). 대신 스크립트 자체에
assertion을 넣고 실행 결과로 검증한다.

- [ ] **Step 1: `build_corpus.py` 작성**

`version_2/rag/build_corpus.py`:
```python
import json
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = Path(__file__).resolve().parent / "sources"
OUT_DIR = ROOT / "data" / "rag"
EMBEDDING_MODEL = "text-embedding-3-small"

SANDVIK_SECTION_CATEGORY = {
    "1. Vibration": "vibration_backlash",
    "2. Insert Wear (Tool Wear)": "tool_wear",
    "3. Chip Issues (Feed / Overload)": "feed_overload",
}
SANDVIK_META = {
    "title": "Sandvik Coromant Milling Troubleshooting",
    "url": "https://www.sandvik.coromant.com/en-us/knowledge/milling/troubleshooting-milling",
}
OSHA_META = {
    "title": "OSHA Machine Guarding & Lockout/Tagout",
    "url": "https://www.osha.gov/etools/machine-guarding/introduction/general-requirements",
}


def parse_sandvik(text: str) -> list[dict]:
    chunks = []
    category = None
    heading = None
    body_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if heading and body_lines:
                chunks.append({
                    "heading": heading,
                    "text": "\n".join(body_lines).strip(),
                    "fault_category": category,
                    "content_type": "cause",
                })
            heading, body_lines = None, []
            category = SANDVIK_SECTION_CATEGORY.get(line[3:].strip())
        elif line.startswith("### "):
            if heading and body_lines:
                chunks.append({
                    "heading": heading,
                    "text": "\n".join(body_lines).strip(),
                    "fault_category": category,
                    "content_type": "cause",
                })
            heading = line[4:].strip()
            body_lines = []
        elif heading is not None and line.strip():
            body_lines.append(line.strip())

    if heading and body_lines:
        chunks.append({
            "heading": heading,
            "text": "\n".join(body_lines).strip(),
            "fault_category": category,
            "content_type": "cause",
        })
    return chunks


def parse_osha(text: str) -> list[dict]:
    chunks = []
    heading = None
    body_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if heading and body_lines:
                chunks.append({
                    "heading": heading,
                    "text": "\n".join(body_lines).strip(),
                    "fault_category": "general",
                    "content_type": "safety",
                })
            heading = line[3:].strip()
            body_lines = []
        elif heading is not None and line.strip():
            body_lines.append(line.strip())

    if heading and body_lines:
        chunks.append({
            "heading": heading,
            "text": "\n".join(body_lines).strip(),
            "fault_category": "general",
            "content_type": "safety",
        })
    return chunks


def build_corpus() -> list[dict]:
    sandvik_text = (SOURCES_DIR / "sandvik_milling_troubleshooting.md").read_text()
    osha_text = (SOURCES_DIR / "osha_machine_guarding_lockout.md").read_text()

    corpus = []
    for chunk in parse_sandvik(sandvik_text):
        corpus.append({**chunk, **SANDVIK_META})
    for chunk in parse_osha(osha_text):
        corpus.append({**chunk, **OSHA_META})
    return corpus


def main() -> None:
    corpus = build_corpus()
    assert len(corpus) > 0, "코퍼스가 비어있음 - 소스 마크다운 파싱 실패"
    assert all(c["fault_category"] is not None for c in corpus), (
        "fault_category가 None인 청크가 있음 - SANDVIK_SECTION_CATEGORY 매핑 확인 필요"
    )

    client = OpenAI()
    texts = [f"{c['heading']}\n{c['text']}" for c in corpus]
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    assert index.ntotal == len(corpus), "인덱스에 들어간 벡터 수가 청크 수와 다름"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "corpus.json").write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False)
    )
    faiss.write_index(index, str(OUT_DIR / "corpus.index"))

    print(f"청크 {len(corpus)}개 저장됨")
    print("fault_category 분포:", Counter(c["fault_category"] for c in corpus))
    print("content_type 분포:", Counter(c["content_type"] for c in corpus))
    print(f"저장: {OUT_DIR / 'corpus.json'}, {OUT_DIR / 'corpus.index'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `OPENAI_API_KEY` 환경변수 확인 후 실행**

Run: `cd version_2 && echo ${OPENAI_API_KEY:+설정됨} && uv run python rag/build_corpus.py`
Expected: `청크 N개 저장됨`(N ≈ 20~25), `fault_category 분포`에
`vibration_backlash`/`tool_wear`/`feed_overload`/`general` 4종류가 모두 찍힘,
assertion 에러 없음, `version_2/data/rag/corpus.json`과 `corpus.index` 파일 생성.

- [ ] **Step 3: Commit**

```bash
git add version_2/rag/build_corpus.py
git commit -m "Add RAG corpus build script"
```

## Task 7: `/predict`에 `guide` 필드 통합

**Files:**
- Modify: `version_2/src/serving/inference.py`
- Modify: `version_2/src/serving/app.py`
- Modify: `version_2/tests/serving/test_inference.py`
- Modify: `version_2/tests/serving/test_app.py`

**Interfaces:**
- Consumes: Task 5의 `rag.guide.build_guide(predict_result, rag_corpus, rag_index, openai_client) -> dict | None`
- Produces: `predict_experiment()` 반환 dict에 `"guide"` 키 추가 (다른 키는 기존과 동일)

- [ ] **Step 1: `test_inference.py`의 기존 shape 테스트를 `guide` 키 포함하도록 수정**

`version_2/tests/serving/test_inference.py`의 `test_predict_experiment_returns_expected_shape`
함수에서 아래 두 줄을 찾아:
```python
    assert set(result.keys()) == {
        "predicted_label", "predicted_label_text", "score", "threshold", "method",
        "feature_contributions",
    }
```
다음으로 교체:
```python
    assert set(result.keys()) == {
        "predicted_label", "predicted_label_text", "score", "threshold", "method",
        "feature_contributions", "guide",
    }
```
같은 함수 맨 끝에 아래 줄 추가(guide는 rag 인자 없이 호출했으므로 good이면
고정 dict, bad면 None):
```python
    if result["predicted_label_text"] == "good":
        assert result["guide"]["cause_estimate"] == "이상 없음"
    else:
        assert result["guide"] is None
```

- [ ] **Step 2: `test_inference.py`에 RAG 파라미터 전달 테스트 추가**

파일 맨 아래에 추가:
```python
def test_predict_experiment_forwards_rag_state_for_bad_prediction():
    torch.manual_seed(0)
    np.random.seed(0)
    df = _raw_df(20)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)

    # threshold를 매우 낮게 잡아 무조건 bad가 나오게 강제
    result = predict_experiment(
        df=df, model=model, feature_columns=FEATURE_COLUMNS,
        scaler_dict=_scaler_dict(), window_size=6, threshold=-999.0, method="mean",
        feature_baseline=_feature_baseline(),
        rag_corpus=None, rag_index=None, openai_client=None,
    )

    assert result["predicted_label_text"] == "bad"
    assert result["guide"] is None  # rag_corpus 등이 없으니 None
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd version_2 && uv run pytest tests/serving/test_inference.py -v`
Expected: FAIL(`TypeError: predict_experiment() got an unexpected keyword argument 'rag_corpus'` 등)

- [ ] **Step 4: `inference.py` 수정**

`version_2/src/serving/inference.py` 상단 import에 추가:
```python
from rag.guide import build_guide
```
`predict_experiment()` 함수 시그니처를:
```python
def predict_experiment(
    df: pd.DataFrame,
    model: torch.nn.Module,
    feature_columns: list[str],
    scaler_dict: dict,
    window_size: int,
    threshold: float,
    method: str,
    feature_baseline: dict,
    exclude_from_ranking: list[str] | None = None,
) -> dict:
```
다음으로 교체:
```python
def predict_experiment(
    df: pd.DataFrame,
    model: torch.nn.Module,
    feature_columns: list[str],
    scaler_dict: dict,
    window_size: int,
    threshold: float,
    method: str,
    feature_baseline: dict,
    exclude_from_ranking: list[str] | None = None,
    rag_corpus: list[dict] | None = None,
    rag_index=None,
    openai_client=None,
) -> dict:
```
함수 맨 끝의 `return { ... }` 블록을:
```python
    return {
        "predicted_label": predicted_label,
        "predicted_label_text": predicted_label_text,
        "score": score,
        "threshold": threshold,
        "method": method,
        "feature_contributions": feature_contributions,
    }
```
다음으로 교체:
```python
    guide = build_guide(
        {
            "predicted_label_text": predicted_label_text,
            "score": score,
            "threshold": threshold,
            "feature_contributions": feature_contributions,
        },
        rag_corpus,
        rag_index,
        openai_client,
    )

    return {
        "predicted_label": predicted_label,
        "predicted_label_text": predicted_label_text,
        "score": score,
        "threshold": threshold,
        "method": method,
        "feature_contributions": feature_contributions,
        "guide": guide,
    }
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd version_2 && uv run pytest tests/serving/test_inference.py -v`
Expected: PASS (모든 테스트 통과)

- [ ] **Step 6: `app.py` 수정 — `ModelState`에 RAG 필드 추가 + 로딩 + `/predict`에 전달**

`version_2/src/serving/app.py` 상단 import에 추가:
```python
import os

import faiss
from openai import OpenAI
```
`ModelState` 데이터클래스를:
```python
@dataclass
class ModelState:
    model: torch.nn.Module
    scaler_dict: dict
    thresholds: dict
    window_size: int
    model_version: str
    mlflow_run_id: str
    feature_baseline: dict
```
다음으로 교체:
```python
@dataclass
class ModelState:
    model: torch.nn.Module
    scaler_dict: dict
    thresholds: dict
    window_size: int
    model_version: str
    mlflow_run_id: str
    feature_baseline: dict
    rag_corpus: list[dict] | None = None
    rag_index: object | None = None
    openai_client: object | None = None
```
`load_model_state()` 함수 바로 위에 새 함수 추가:
```python
def load_rag_state() -> tuple[list[dict] | None, object | None, object | None]:
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=api_key) if api_key else None

    corpus_path = ROOT / "data" / "rag" / "corpus.json"
    index_path = ROOT / "data" / "rag" / "corpus.index"
    if not corpus_path.exists() or not index_path.exists():
        return None, None, openai_client

    rag_corpus = json.loads(corpus_path.read_text())
    rag_index = faiss.read_index(str(index_path))
    return rag_corpus, rag_index, openai_client
```
`load_model_state()` 함수 안의 `return ModelState(...)` 호출을:
```python
    return ModelState(
        model=model,
        scaler_dict=scaler_dict,
        thresholds=thresholds,
        window_size=window_size,
        model_version=str(mv.version),
        mlflow_run_id=mv.run_id,
        feature_baseline=feature_baseline,
    )
```
다음으로 교체:
```python
    rag_corpus, rag_index, openai_client = load_rag_state()
    return ModelState(
        model=model,
        scaler_dict=scaler_dict,
        thresholds=thresholds,
        window_size=window_size,
        model_version=str(mv.version),
        mlflow_run_id=mv.run_id,
        feature_baseline=feature_baseline,
        rag_corpus=rag_corpus,
        rag_index=rag_index,
        openai_client=openai_client,
    )
```
`/predict` 엔드포인트 안의 `predict_experiment(...)` 호출을:
```python
        result = predict_experiment(
            df=df,
            model=state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=state.scaler_dict,
            window_size=state.window_size,
            threshold=state.thresholds[method],
            method=method,
            feature_baseline=state.feature_baseline,
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
        )
```
다음으로 교체:
```python
        result = predict_experiment(
            df=df,
            model=state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=state.scaler_dict,
            window_size=state.window_size,
            threshold=state.thresholds[method],
            method=method,
            feature_baseline=state.feature_baseline,
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
            rag_corpus=state.rag_corpus,
            rag_index=state.rag_index,
            openai_client=state.openai_client,
        )
```

- [ ] **Step 7: `test_app.py`에 `guide` 관련 테스트 추가**

`version_2/tests/serving/test_app.py` 맨 아래에 추가:
```python
def test_predict_response_includes_guide_field():
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "guide" in body
    if body["predicted_label_text"] == "good":
        assert body["guide"]["cause_estimate"] == "이상 없음"
    else:
        assert body["guide"] is None  # _fake_state는 rag_corpus 등을 안 채움
```

- [ ] **Step 8: 전체 테스트 통과 확인**

Run: `cd version_2 && uv run pytest -q`
Expected: 기존 70개 + 이번에 추가한 테스트 전부 통과(에러/실패 0건)

- [ ] **Step 9: Commit**

```bash
git add version_2/src/serving/inference.py version_2/src/serving/app.py \
        version_2/tests/serving/test_inference.py version_2/tests/serving/test_app.py
git commit -m "Integrate RAG guide into /predict response"
```

## Task 8: 실제 champion 모델 + 합성 시나리오로 통합 검증

**Files:** 없음 (수동 검증)

- [ ] **Step 1: `who`/`top`으로 서버 상태 확인 후 FastAPI 서버 기동**

Run: `cd version_2 && who && top -bn1 | head -6 && nice -n 19 uv run uvicorn serving.app:app --port 8899 &`
Expected: 정상 기동 로그, `GET /health` 응답에 champion 버전 표시

- [ ] **Step 2: 합성 이상 시나리오로 `/predict` 호출, `guide` 내용 확인**

Run:
```bash
curl -s -X POST "http://127.0.0.1:8899/predict" \
  -F "file=@synthetic/scenarios/tool_wear.csv" | python3 -m json.tool
```
Expected: `predicted_label_text: "bad"`, `guide.cause_estimate`에 공구마모 관련 내용,
`guide.confidence_note`에 "간접"/"추정" 같은 확신도를 낮춘 표현 포함,
`guide.sources`에 실제 URL(Sandvik/OSHA) 포함. `feed_overload.csv`,
`vibration_backlash.csv`도 동일하게 확인.

- [ ] **Step 3: 정상 시나리오로 고정 응답 확인**

Run: `curl -s -X POST "http://127.0.0.1:8899/predict" -F "file=@synthetic/scenarios/tool_wear_normal.csv" | python3 -m json.tool`
Expected: `predicted_label_text: "good"`, `guide.cause_estimate == "이상 없음"`, LLM 호출 없이 즉시 응답(지연 거의 없음).

- [ ] **Step 4: `OPENAI_API_KEY` 없을 때 폴백 확인**

Run:
```bash
kill %1  # Step 1에서 백그라운드로 띄운 서버 종료
cd version_2 && OPENAI_API_KEY="" nice -n 19 uv run uvicorn serving.app:app --port 8899 &
sleep 3
curl -s -X POST "http://127.0.0.1:8899/predict" -F "file=@synthetic/scenarios/tool_wear.csv" | python3 -m json.tool
kill %1
```
Expected: `predicted_label_text: "bad"`(판정 자체는 정상), `guide: null`(RAG만 비활성화).

- [ ] **Step 5: 결과를 사용자에게 보고**

6개 합성 시나리오(이상 3 + 정상 3) 전부에 대한 `guide` 실제 내용, 확신도 표현이
실제로 들어갔는지, API 키 없을 때 폴백이 의도대로 동작하는지 요약해서 보고.

---

## Self-Review 완료 사항

- 스펙 커버리지: Part A(코퍼스 구축, 로컬 md → 헤딩 청킹 → 태그 → 임베딩 → FAISS)는
  Task 6, Part B(`src/rag/` 4개 모듈 + 오케스트레이터 + `/predict` 통합)는 Task 1~7,
  확신도 원칙은 Task 4의 시스템 프롬프트 테스트로, 에러 처리(`guide: null`)는
  Task 5의 예외 테스트 + Task 8의 API 키 없음 시나리오로 커버됨.
- 플레이스홀더 없음: 전 태스크 코드 실행 가능한 완성 코드.
- 타입/시그니처 일관성: `build_guide(predict_result, rag_corpus, rag_index, openai_client)`
  시그니처가 Task 5 정의, Task 7의 `inference.py` 호출부에서 동일하게 사용됨.
  `search(query, corpus, index, embed_fn, top_k)`의 `embed_fn` 주입 방식이
  Task 3 정의와 Task 5의 `guide.py` 사용부에서 일치.
