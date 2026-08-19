EXTRA_MISSES_ALLOWED = 1


def evaluate_gate(
    retrained_missed: int,
    champion_missed: int,
    retrained_accuracy: float,
    champion_accuracy: float,
    extra_misses_allowed: int = EXTRA_MISSES_ALLOWED,
) -> dict:
    """승격 여부를 두 조건의 AND로 판정한다.

    G1 (안전): 원본 실측 eval셋에서 불량 검출력이 champion 대비 유지되는가.
      **놓친 개수(fn)로 비교한다.** eval 불량이 11개뿐이라 recall 소수값은
      실험 1개당 0.0909씩 뚝뚝 끊긴다 — 소수점 임계값은 그 눈금을 가리는
      가짜 정밀도라, 기준을 개수로 직접 표현한다.
      precision을 보지 않는 이유는, 센서 좌표계가 이동한 환경에서 새 모델을
      옛 좌표계 eval에 적용하면 precision이 좌표계 차이 때문에 떨어지기 때문이다.
    G2 (근거): 라벨이 도착한 최근 구간에서 실제로 나아졌는가.
      G1만으로는 모든 것을 불량이라 판정하는 모델도 놓친 개수 0으로 통과한다.
      실제로 두 시나리오 모두에서 판정을 내린 것은 G2였다.
    """
    g1_pass = retrained_missed <= champion_missed + extra_misses_allowed
    g2_pass = retrained_accuracy > champion_accuracy

    reasons = []
    if not g1_pass:
        reasons.append(
            f"G1 불량 검출 회귀: {retrained_missed}건 놓침 > "
            f"허용 {champion_missed + extra_misses_allowed}건"
        )
    if not g2_pass:
        reasons.append(
            f"G2 개선 없음: {retrained_accuracy:.4f} <= {champion_accuracy:.4f}"
        )

    return {
        "decision": "promoted" if (g1_pass and g2_pass) else "rejected",
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "g1_missed": retrained_missed,
        "g2_accuracy_delta": retrained_accuracy - champion_accuracy,
        "reject_reason": "; ".join(reasons),
    }


def accuracy_from_pairs(truths: list[str], predictions: list[str]) -> float:
    """QC 진실 라벨과 모델 판정을 짝지어 정확도를 낸다. 둘 다 "good"/"bad" 문자열."""
    if not truths or len(truths) != len(predictions):
        raise ValueError(
            f"라벨 {len(truths)}개와 판정 {len(predictions)}개의 길이가 다릅니다"
        )
    hits = sum(1 for truth, pred in zip(truths, predictions) if truth == pred)
    return hits / len(truths)
