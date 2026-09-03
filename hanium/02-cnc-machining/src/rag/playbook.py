"""플레이북(팀 작성 상황 문서) 파싱과 센서 서명 대조.

판정은 순수 계산이다 — 임베딩·LLM을 쓰지 않으므로 같은 입력이면 같은 결과가 나오고
OpenAI 키 없이도 돈다. 임베딩은 guide.py가 확정 판정에서 Sandvik 청크를 고를 때만 쓴다.
"""

PLAYBOOK_SOURCE = "playbook"
PLAYBOOK_META = {
    "title": "팀 시나리오 플레이북(자체 작성)",
    "url": "rag/sources/scenario_playbook.md",
}
SECTION_CATEGORY = {
    "1. 스핀들 부하 상승": "tool_wear",
    "2. 이송축 부하 상승": "feed_overload",
    "3. 축 위치·속도 편차": "vibration_backlash",
    "4. 고장이 아닌 변화": "general",
}
SIGNATURE_PREFIX = "관련 센서:"
HEADING_SEPARATOR = " — "


def _parse_signature(value: str, heading: str, known_features: set[str] | None) -> list[str]:
    value = value.strip()
    if value == "없음":
        return []
    codes = [code.strip() for code in value.split(",") if code.strip()]
    if known_features is not None:
        unknown = [code for code in codes if code not in known_features]
        if unknown:
            raise ValueError(f"'{heading}' 항목의 모르는 센서 코드: {unknown}")
    return codes


def parse_playbook(text: str, known_features: set[str] | None = None) -> list[dict]:
    """'## ' 구역(SECTION_CATEGORY로 카테고리 결정), '### ' 항목. 항목 본문에 '관련 센서:'
    줄이 반드시 있어야 하며 그 줄이 signature가 된다(본문 text에도 남긴다)."""
    chunks: list[dict] = []
    category = None
    heading = None
    body: list[str] = []
    signature: list[str] | None = None

    def flush() -> None:
        if heading is None:
            return
        if signature is None:
            raise ValueError(f"'{heading}' 항목에 '{SIGNATURE_PREFIX}' 줄이 없음")
        chunks.append({
            "heading": heading,
            "name": heading.split(HEADING_SEPARATOR)[0].strip(),
            "text": "\n".join(body).strip(),
            "fault_category": category,
            "content_type": "context" if category == "general" else "cause",
            "signature": signature,
            "source": PLAYBOOK_SOURCE,
        })

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading, body, signature = None, [], None
            section = line[3:].strip()
            if section not in SECTION_CATEGORY:
                raise ValueError(f"모르는 구역 '{section}' — SECTION_CATEGORY에 추가 필요")
            category = SECTION_CATEGORY[section]
        elif line.startswith("### "):
            flush()
            heading, body, signature = line[4:].strip(), [], None
        elif heading is not None and line.strip():
            stripped = line.strip()
            if stripped.startswith(SIGNATURE_PREFIX):
                signature = _parse_signature(
                    stripped[len(SIGNATURE_PREFIX):], heading, known_features
                )
            body.append(stripped)
    flush()
    return chunks
