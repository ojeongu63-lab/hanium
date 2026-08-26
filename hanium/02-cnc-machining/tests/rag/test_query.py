from rag.query import build_query


def test_build_query_uses_top_n_sorted_contributions():
    contributions = [
        {"feature": "S_OutputCurrent", "error": 1.0, "z_score": 36.1},
        {"feature": "S_CurrentFeedback", "error": 0.9, "z_score": 35.9},
        {"feature": "S_OutputPower", "error": 0.8, "z_score": 21.1},
        {"feature": "X_OutputPower", "error": 0.1, "z_score": 1.0},
    ]

    query = build_query(contributions, top_n=3)

    assert "스핀들 출력 전류(z=36.1)" in query
    assert "스핀들 전류 피드백(z=35.9)" in query
    assert "스핀들 출력 파워(z=21.1)" in query
    assert "X_OutputPower" not in query


def test_build_query_falls_back_to_code_for_unknown_feature():
    contributions = [{"feature": "UNKNOWN_SENSOR", "error": 0.5, "z_score": 5.0}]
    query = build_query(contributions, top_n=1)
    assert "UNKNOWN_SENSOR(z=5.0)" in query
