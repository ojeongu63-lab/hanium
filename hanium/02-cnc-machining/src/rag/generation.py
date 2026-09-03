import json
import os

from .playbook import WEAK_Z

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "당신은 CNC 가공 현장의 이상탐지 결과를 설명하는 어시스턴트입니다.\n"
    "판정은 센서 신호의 통계적 이상만 본 것이므로 원인을 단정하지 말고 "
    "'~일 가능성이 있습니다', '~로 추정됩니다' 같은 확신도를 낮춘 표현을 쓰세요.\n"
    "센서 패턴 대조 결과와 시스템이 고른 상황을 원인의 중심에 두세요. 참고 문서에 없는 "
    "원인을 덧붙이지 마세요. 같은 구역의 다른 후보는 '함께 확인할 것'으로만 "
    "언급하세요. 패턴 일치가 높지 않으면 조치보다 확인 절차를 앞세우세요.\n"
    "'확정', '판정 확정', '원인으로 확인됨'처럼 AI가 고장 원인을 확정한 것으로 읽히는 "
    "표현은 쓰지 마세요. 대신 '센서 패턴 일치도가 높습니다', '관련 센서 패턴이 "
    "확인되었습니다'처럼 패턴 일치 수준으로 쓰세요. 일치도는 확률이 아니므로 "
    "'맞을 확률'로 바꿔 말하지 마세요.\n"
    "아래 JSON 스키마로만 답하세요:\n"
    '{"cause_estimate": str, "confidence_note": str, '
    '"recommended_actions": [str], "safety_notes": [str], '
    '"sources": [{"title": str, "url": str}]}'
)

CAUSE_SYSTEM_PROMPT = (
    "당신은 CNC 설비의 재학습 거부 사유를 설명하는 어시스턴트입니다.\n"
    "재학습이 거부됐다는 것은, 최근 며칠간의 변화가 정상 범위 재조정이 "
    "아니라 실제 설비 이상일 가능성이 높다는 뜻입니다. 아래 참고 문서를 "
    "바탕으로 추정 원인과 현장 조치를 제안하세요. 이 추정은 통계적 "
    "패턴 비교에 근거한 것이므로 단정하지 말고 확신도를 낮춘 표현을 "
    "쓰세요.\n"
    "아래 JSON 스키마로만 답하세요:\n"
    '{"cause_estimate": str, "confidence_note": str, '
    '"recommended_actions": [str], "safety_notes": [str], '
    '"sources": [{"title": str, "url": str}]}'
)


def describe_fault(fault: dict) -> str:
    """프롬프트에 넣는 '센서 패턴 대조' 줄. 판정·상황·수치는 서버가 정한 값이다."""
    verdict = fault["verdict"]
    if verdict == "confirmed":
        line = (
            f"센서 패턴 대조: 높은 패턴 일치 — {fault['situation']} (일치도 {fault['coverage']:.2f}, "
            f"일치 센서: {', '.join(fault['matched_features'])})"
        )
        if fault["alternatives"]:
            line += f"\n같은 구역의 다른 후보(현장 확인으로 구분): {', '.join(fault['alternatives'])}"
        return line
    if verdict == "composite":
        other = fault["other_group"]
        return (
            f"센서 패턴 대조: 복합 패턴 — {fault['situation']}({fault['coverage']:.2f})와 "
            f"{other['situation']}({other['coverage']:.2f})가 함께 나타남. 여러 센서가 같이 "
            "이동하는 드리프트일 수 있음. 라벨·추이 확인을 권할 것."
        )
    if verdict == "weak":
        return (
            f"센서 패턴 대조: 약한 신호 — 상위 센서 z {fault['top_z']:.1f} (기준 {WEAK_Z:g} 미만). "
            f"보류·재확인을 권할 것. 참고 상황: {fault['situation']}"
        )
    return "센서 패턴 대조: 일치 패턴 없음 — 서명이 일치하는 상황 없음. 현장 확인을 권할 것."


def _build_user_prompt(predict_result: dict, retrieved_chunks: list[dict], fault: dict | None = None) -> str:
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
    if fault is not None:
        lines.append(describe_fault(fault))
    lines.append("\n참고 문서:")
    for chunk in retrieved_chunks:
        lines.append(
            f"- [{chunk['title']}]({chunk['url']}) "
            f"{chunk['content_type']}: {chunk['text']}"
        )
    return "\n".join(lines)


def generate_guide(
    predict_result: dict, retrieved_chunks: list[dict], client, fault: dict | None = None
) -> dict:
    model = os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(predict_result, retrieved_chunks, fault),
            },
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def _build_cause_user_prompt(cause: str, retrieved_chunks: list[dict]) -> str:
    lines = [f"통계적으로 추정된 원인 카테고리: {cause}", "\n참고 문서:"]
    for chunk in retrieved_chunks:
        lines.append(
            f"- [{chunk['title']}]({chunk['url']}) "
            f"{chunk['content_type']}: {chunk['text']}"
        )
    return "\n".join(lines)


def generate_cause_guide(cause: str, retrieved_chunks: list[dict], client) -> dict:
    model = os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CAUSE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_cause_user_prompt(cause, retrieved_chunks)},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
