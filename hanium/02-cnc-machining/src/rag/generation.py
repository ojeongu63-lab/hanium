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
