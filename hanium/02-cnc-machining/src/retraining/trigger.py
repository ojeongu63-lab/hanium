def is_drift_flagged(status: dict) -> bool:
    """한 번의 /drift-status 조회가 '드리프트 있음'인지 판정한다."""
    if not status.get("sufficient_data"):
        return False
    output_flagged = bool(status["output_drift"]["flagged"])
    input_flagged = bool(status["input_drift"]["flagged_features"])
    return output_flagged or input_flagged


def should_retrain(
    flag_history: list[bool],
    consecutive_k: int = 3,
    cooldown_remaining: int = 0,
) -> bool:
    """연속 consecutive_k회 드리프트가 잡히면 재학습을 발동한다.

    쿨다운 중에는 발동하지 않는다 — 재학습 직후에는 요청 로그에 옛 데이터가
    남아 있어 즉시 재발동해 버린다.
    """
    if cooldown_remaining > 0:
        return False
    if len(flag_history) < consecutive_k:
        return False
    return all(flag_history[-consecutive_k:])
