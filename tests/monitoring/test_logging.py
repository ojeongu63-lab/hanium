from monitoring.logging import count_requests, get_recent_requests, log_request


def test_log_and_retrieve_roundtrip(tmp_path):
    db_path = tmp_path / "requests.db"
    log_request({"f0": 0.1, "f1": -0.2}, score=0.5, predicted_label_text="good", db_path=db_path)
    log_request({"f0": 0.3, "f1": -0.1}, score=0.9, predicted_label_text="bad", db_path=db_path)

    recent = get_recent_requests(n=10, db_path=db_path)

    assert len(recent) == 2
    assert recent[0]["score"] == 0.9  # 최신이 먼저
    assert recent[0]["predicted_label_text"] == "bad"
    assert recent[1]["feature_means"] == {"f0": 0.1, "f1": -0.2}


def test_get_recent_requests_respects_limit(tmp_path):
    db_path = tmp_path / "requests.db"
    for i in range(5):
        log_request({"f0": float(i)}, score=float(i), predicted_label_text="good", db_path=db_path)

    recent = get_recent_requests(n=3, db_path=db_path)

    assert len(recent) == 3
    assert recent[0]["score"] == 4.0


def test_get_recent_requests_empty_db_returns_empty_list(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    assert get_recent_requests(n=10, db_path=db_path) == []


def test_count_requests_returns_total_row_count(tmp_path):
    db_path = tmp_path / "requests.db"
    for i in range(4):
        log_request({"f0": float(i)}, score=float(i), predicted_label_text="good", db_path=db_path)

    assert count_requests(db_path) == 4


def test_count_requests_empty_db_returns_zero(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    assert count_requests(db_path) == 0
