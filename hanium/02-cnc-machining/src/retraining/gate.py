RECALL_TOLERANCE = 0.10


def evaluate_gate(
    retrained_recall: float,
    champion_recall: float,
    retrained_accuracy: float,
    champion_accuracy: float,
    recall_tolerance: float = RECALL_TOLERANCE,
) -> dict:
    """승격 여부를 두 조건의 AND로 판정한다.

    G1 (안전): 원본 실측 eval셋에서 불량 검출력이 champion 대비 유지되는가.
      허용 하락폭 0.10은 불량 11개 기준 1건(0.0909)까지만 봐준다는 뜻이다.
      precision을 보지 않는 이유는, 센서 좌표계가 이동한 환경에서 새 모델을
      옛 좌표계 eval에 적용하면 precision이 좌표계 차이 때문에 떨어지기 때문이다.
    G2 (근거): 라벨이 도착한 최근 구간에서 실제로 나아졌는가.
      G1만으로는 모든 것을 불량이라 판정하는 모델도 recall 1.0으로 통과한다.
    """
    g1_pass = retrained_recall >= champion_recall - recall_tolerance
    g2_pass = retrained_accuracy > champion_accuracy

    reasons = []
    if not g1_pass:
        reasons.append(
            f"G1 recall 회귀: {retrained_recall:.4f} < "
            f"{champion_recall - recall_tolerance:.4f}"
        )
    if not g2_pass:
        reasons.append(
            f"G2 개선 없음: {retrained_accuracy:.4f} <= {champion_accuracy:.4f}"
        )

    return {
        "decision": "promoted" if (g1_pass and g2_pass) else "rejected",
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "g1_recall": retrained_recall,
        "g2_accuracy_delta": retrained_accuracy - champion_accuracy,
        "reject_reason": "; ".join(reasons),
    }
