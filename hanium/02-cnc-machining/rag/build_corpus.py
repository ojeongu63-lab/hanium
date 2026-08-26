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
KAMP_SECTION_META = {
    "CNC 공구 마모와 가공 정밀도": {"fault_category": "tool_wear", "content_type": "context"},
    "간접 센서 신호를 통한 공구수명 추정": {"fault_category": "tool_wear", "content_type": "cause"},
    "양품/불량 판정 기준": {"fault_category": "general", "content_type": "context"},
}
KAMP_META = {
    "title": "KAMP CNC 머신 AI 데이터셋 가이드북",
    "url": "https://www.kamp-ai.kr/",
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


def parse_kamp_guide(text: str) -> list[dict]:
    chunks = []
    section_meta = None
    heading = None
    body_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if heading and body_lines:
                chunks.append({
                    "heading": heading,
                    "text": "\n".join(body_lines).strip(),
                    **section_meta,
                })
            heading, body_lines = None, []
            section_meta = KAMP_SECTION_META.get(line[3:].strip())
        elif line.startswith("### "):
            if heading and body_lines:
                chunks.append({
                    "heading": heading,
                    "text": "\n".join(body_lines).strip(),
                    **section_meta,
                })
            heading = line[4:].strip()
            body_lines = []
        elif heading is not None and line.strip():
            body_lines.append(line.strip())

    if heading and body_lines:
        chunks.append({
            "heading": heading,
            "text": "\n".join(body_lines).strip(),
            **section_meta,
        })
    return chunks


def build_corpus() -> list[dict]:
    sandvik_text = (SOURCES_DIR / "sandvik_milling_troubleshooting.md").read_text()
    osha_text = (SOURCES_DIR / "osha_machine_guarding_lockout.md").read_text()
    kamp_text = (SOURCES_DIR / "kamp_cnc_dataset_guide.md").read_text()

    corpus = []
    for chunk in parse_sandvik(sandvik_text):
        corpus.append({**chunk, **SANDVIK_META})
    for chunk in parse_osha(osha_text):
        corpus.append({**chunk, **OSHA_META})
    for chunk in parse_kamp_guide(kamp_text):
        corpus.append({**chunk, **KAMP_META})
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
