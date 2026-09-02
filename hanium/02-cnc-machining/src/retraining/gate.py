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


def evaluate_shadow(candidate_accuracy: float, champion_accuracy: float) -> dict:
    """섀도우 기간 종료 후 최종 판정. G1은 트리거 시점에 이미 확인했고
    원본 eval은 시간이 지나도 안 바뀌므로 재확인하지 않는다 — G2에
    해당하는 정확도 비교만 반복한다."""
    promoted = candidate_accuracy > champion_accuracy
    return {
        "decision": "promoted" if promoted else "rejected",
        "accuracy_delta": candidate_accuracy - champion_accuracy,
    }


def evaluate_two_sided(
    truths: list[str], champion_preds: list[str], candidate_preds: list[str]
) -> dict:
    """라벨 창을 정상/불량으로 나눠 오탐(정상→bad)과 놓침(불량→good)을 두 모델
    각각 센 뒤 승격 여부를 낸다.

    정확도 하나로 비교하면 창에 한 클래스만 있을 때 "더 자주 불량이라 하는"
    모델이 무조건 이긴다(09-02 fixture_loosening Day 34에서 실제로 통과됨).
    그래서 두 건수를 따로 보고, 한쪽을 다른 쪽과 맞바꾸는 후보와 오탐을 아예
    잴 수 없는 창(정상 라벨 0건)은 통과시키지 않는다. 정상 라벨만 있으면
    놓침 쪽은 G1이 맡는다 — temperature 시나리오의 기존 경로.

    빈 창은 정상 라벨 0건이므로 거부. 세 리스트 길이가 다르면 ValueError.
    """
    if not (len(truths) == len(champion_preds) == len(candidate_preds)):
        raise ValueError(
            f"라벨 {len(truths)}개, champion 판정 {len(champion_preds)}개, "
            f"후보 판정 {len(candidate_preds)}개의 길이가 다릅니다"
        )
    good = [i for i, t in enumerate(truths) if t == "good"]
    bad = [i for i, t in enumerate(truths) if t == "bad"]

    def false_alarms(preds: list[str]) -> int:
        return sum(1 for i in good if preds[i] == "bad")

    def misses(preds: list[str]) -> int:
        return sum(1 for i in bad if preds[i] == "good")

    counts = {
        "n_good": len(good),
        "n_bad": len(bad),
        "champion_false_alarms": false_alarms(champion_preds),
        "candidate_false_alarms": false_alarms(candidate_preds),
        "champion_misses": misses(champion_preds),
        "candidate_misses": misses(candidate_preds),
    }
    if counts["n_good"] == 0:
        return {
            **counts,
            "decision": "rejected",
            "reject_reason": "G2 판정 불가: 창에 정상 라벨 없음(오탐 회귀 확인 불가)",
        }

    fa_ok = counts["candidate_false_alarms"] <= counts["champion_false_alarms"]
    miss_ok = counts["n_bad"] == 0 or counts["candidate_misses"] <= counts["champion_misses"]
    improved = counts["candidate_false_alarms"] < counts["champion_false_alarms"] or (
        counts["n_bad"] > 0 and counts["candidate_misses"] < counts["champion_misses"]
    )

    reasons = []
    if not fa_ok:
        reasons.append(
            f"G2 오탐 회귀: 후보 {counts['candidate_false_alarms']}건 > "
            f"champion {counts['champion_false_alarms']}건 (정상 {counts['n_good']}건 중)"
        )
    if not miss_ok:
        reasons.append(
            f"G2 놓침 회귀: 후보 {counts['candidate_misses']}건 > "
            f"champion {counts['champion_misses']}건 (불량 {counts['n_bad']}건 중)"
        )
    if fa_ok and miss_ok and not improved:
        reasons.append("G2 개선 없음: 오탐·놓침 모두 champion과 동일")

    promoted = fa_ok and miss_ok and improved
    return {
        **counts,
        "decision": "promoted" if promoted else "rejected",
        "reject_reason": "; ".join(reasons),
    }
