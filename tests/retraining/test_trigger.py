from retraining.trigger import days_to_process, is_drift_flagged, should_retrain


def _status(sufficient=True, output_flagged=False, input_flagged=()):
    return {
        "sufficient_data": sufficient,
        "output_drift": {"flagged": output_flagged},
        "input_drift": {"flagged_features": list(input_flagged)},
    }


def test_not_flagged_when_data_insufficient():
    assert is_drift_flagged(_status(sufficient=False, output_flagged=True)) is False


def test_flagged_on_output_drift():
    assert is_drift_flagged(_status(output_flagged=True)) is True


def test_flagged_on_input_drift():
    assert is_drift_flagged(_status(input_flagged=[{"feature": "X_OutputPower"}])) is True


def test_not_flagged_when_clean():
    assert is_drift_flagged(_status()) is False


def test_no_retrain_before_k_consecutive():
    assert should_retrain([True, True], consecutive_k=3) is False


def test_retrain_after_k_consecutive():
    assert should_retrain([False, True, True, True], consecutive_k=3) is True


def test_no_retrain_when_streak_broken():
    assert should_retrain([True, True, False], consecutive_k=3) is False


def test_no_retrain_during_cooldown():
    assert should_retrain([True, True, True], consecutive_k=3, cooldown_remaining=2) is False


def test_days_to_process_returns_each_day_in_order():
    assert days_to_process(last_day=0, latest_day=3) == [1, 2, 3]


def test_days_to_process_skips_nothing_when_polling_lags_behind():
    # 폴링 간격보다 feeder 가 빨라 2일치가 한꺼번에 앞서 있어도 둘 다 처리해야 한다.
    assert days_to_process(last_day=2, latest_day=4) == [3, 4]


def test_days_to_process_empty_when_no_new_day():
    assert days_to_process(last_day=5, latest_day=5) == []
