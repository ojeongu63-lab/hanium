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


def days_to_process(last_day: int, latest_day: int) -> list[int]:
    """감시 워커가 이번 폴링에서 순서대로 처리해야 할 날짜들.

    폴링 간격보다 feeder 가 배치를 빨리 흘려보내면 여러 날짜가 한 번의
    폴링 사이에 지나가 버린다. 최신 날짜만 보고 건너뛰면 중간 날짜의
    tick 이 통째로 빠져 flag_history 의 연속성이 깨진다(실측으로 확인:
    Day 2 다음 Day 4가 찍히고 Day 3 이 사라짐) — 그래서 하나씩 다 돌려준다.
    """
    return list(range(last_day + 1, latest_day + 1))
