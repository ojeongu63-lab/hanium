"""플레이북(팀 작성 상황 문서) 파싱과 센서 서명 대조.

판정은 순수 계산이다 — 임베딩·LLM을 쓰지 않으므로 같은 입력이면 같은 결과가 나오고
OpenAI 키 없이도 돈다. 임베딩은 guide.py가 높은 패턴 일치(confirmed)에서 Sandvik 청크를 고를 때만 쓴다.
"""

import hashlib

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


def playbook_version(text: str) -> str:
    """플레이북 문서 내용의 sha256 앞 8자리. 코퍼스 빌드 시 corpus_meta.json에 기록되고
    /predict 응답의 versions.playbook 으로 나간다."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


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


TOP_N = 5              # 대조에 쓰는 상위 피처 수
WEAK_Z = 10.0          # 상위 1 피처의 z가 이 미만이면 "약한 신호"
COMPOSITE_RATIO = 0.5  # 다른 그룹 최고 점수가 1위의 이 비율 이상이면 "복합 징후"

# 사용자 화면 문구. 멘토 피드백(2026-09-03): AI가 고장 원인을 "확정"한 것처럼 읽히는 표현을
# 쓰지 않는다 — 내부 값(confirmed 등)은 그대로 두고 한글만 "패턴 일치" 수준으로 표현한다.
VERDICT_KO = {
    "confirmed": "높은 패턴 일치",
    "composite": "복합 패턴",
    "weak": "약한 신호",
    "unknown": "일치 패턴 없음",
    "none": "이상 없음",
}

NO_FAULT = {
    "verdict": "none", "verdict_ko": VERDICT_KO["none"],
    "situation": None, "category": None, "coverage": 0.0,
    "matched_features": [], "alternatives": [], "other_group": None, "top_z": None,
}


def coverage(signature: list[str], contributions: list[dict], top_n: int = TOP_N) -> float:
    """상위 top_n 피처를 1/순위로 가중해, 서명이 설명하는 비율(0~1). 소수 둘째 자리."""
    top = contributions[:top_n]
    if not top:
        return 0.0
    total = sum(1 / (rank + 1) for rank in range(len(top)))
    got = sum(1 / (rank + 1) for rank, c in enumerate(top) if c["feature"] in signature)
    return round(got / total, 2)


def match_playbook(contributions: list[dict], corpus: list[dict]) -> dict | None:
    """플레이북 항목 중 상위 피처를 가장 잘 설명하는 상황을 고르고 세 단계로 판정한다.
    동점이면 코퍼스 순서(구역 대표 우선). 플레이북 항목이 없으면 None."""
    entries = [c for c in corpus if c.get("source") == PLAYBOOK_SOURCE]
    if not entries:
        return None

    scored = [(coverage(entry["signature"], contributions), entry) for entry in entries]
    best_cov, best = max(scored, key=lambda pair: pair[0])   # max는 첫 최댓값을 돌려준다
    top_z = float(contributions[0]["z_score"]) if contributions else None

    if best_cov == 0.0:
        return {
            **NO_FAULT, "verdict": "unknown", "verdict_ko": VERDICT_KO["unknown"], "top_z": top_z,
        }

    others = [(cov, e) for cov, e in scored if e["fault_category"] != best["fault_category"]]
    other_cov, other = max(others, key=lambda pair: pair[0]) if others else (0.0, None)
    same = sorted(
        ((cov, e) for cov, e in scored if e["fault_category"] == best["fault_category"] and e is not best),
        key=lambda pair: -pair[0],
    )

    if top_z is not None and top_z < WEAK_Z:
        verdict = "weak"
    elif other is not None and other_cov >= COMPOSITE_RATIO * best_cov:
        verdict = "composite"
    else:
        verdict = "confirmed"

    return {
        "verdict": verdict,
        "verdict_ko": VERDICT_KO[verdict],
        "situation": best["name"],
        "category": best["fault_category"],
        "coverage": best_cov,
        "matched_features": [
            c["feature"] for c in contributions[:TOP_N] if c["feature"] in best["signature"]
        ],
        "alternatives": [e["name"] for _, e in same[:2]],
        "other_group": (
            {"situation": other["name"], "category": other["fault_category"], "coverage": other_cov}
            if other is not None and other_cov > 0.0 else None
        ),
        "top_z": top_z,
    }
