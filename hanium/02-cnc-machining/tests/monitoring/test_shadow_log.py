from monitoring.shadow_log import get_shadow_predictions, record_shadow_prediction


def test_record_and_retrieve_roundtrip(tmp_path):
    db_path = tmp_path / "shadow.db"
    record_shadow_prediction("day37_0", "good", "bad", db_path)
    record_shadow_prediction("day37_1", "bad", "bad", db_path)

    result = get_shadow_predictions(["day37_0", "day37_1"], db_path)

    assert result["day37_0"] == {"champion_label": "good", "candidate_label": "bad"}
    assert result["day37_1"] == {"champion_label": "bad", "candidate_label": "bad"}


def test_missing_batch_ids_are_omitted(tmp_path):
    db_path = tmp_path / "shadow.db"
    record_shadow_prediction("day37_0", "good", "good", db_path)

    result = get_shadow_predictions(["day37_0", "day37_1"], db_path)

    assert list(result.keys()) == ["day37_0"]


def test_missing_db_returns_empty_dict(tmp_path):
    assert get_shadow_predictions(["day37_0"], tmp_path / "nope.db") == {}
