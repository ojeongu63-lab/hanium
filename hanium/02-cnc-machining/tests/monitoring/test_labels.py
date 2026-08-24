from monitoring.labels import get_arrived_labels, get_latest_produced_day, record_label


def test_record_and_retrieve_roundtrip(tmp_path):
    db_path = tmp_path / "labels.db"
    record_label("day01_0", produced_day=1, arrived_day=8, label="good", db_path=db_path)
    record_label("day02_0", produced_day=2, arrived_day=9, label="bad", db_path=db_path)

    arrived = get_arrived_labels(current_day=9, db_path=db_path)

    assert len(arrived) == 2
    assert arrived[0]["batch_id"] == "day01_0"  # produced_day 오름차순
    assert arrived[1]["label"] == "bad"


def test_future_labels_are_not_returned(tmp_path):
    db_path = tmp_path / "labels.db"
    record_label("day01_0", produced_day=1, arrived_day=8, label="good", db_path=db_path)
    record_label("day05_0", produced_day=5, arrived_day=12, label="good", db_path=db_path)

    arrived = get_arrived_labels(current_day=8, db_path=db_path)

    assert len(arrived) == 1
    assert arrived[0]["batch_id"] == "day01_0"


def test_missing_db_returns_empty_list(tmp_path):
    assert get_arrived_labels(current_day=99, db_path=tmp_path / "nope.db") == []


def test_latest_produced_day_tracks_max(tmp_path):
    db_path = tmp_path / "labels.db"
    record_label("day01_0", produced_day=1, arrived_day=8, label="good", db_path=db_path)
    record_label("day05_0", produced_day=5, arrived_day=12, label="good", db_path=db_path)
    record_label("day03_0", produced_day=3, arrived_day=10, label="good", db_path=db_path)

    assert get_latest_produced_day(db_path) == 5


def test_latest_produced_day_missing_db_returns_zero(tmp_path):
    assert get_latest_produced_day(tmp_path / "nope.db") == 0
