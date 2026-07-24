from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)


def test_split_counts_match_spec():
    assert len(TRAIN_EXPERIMENT_IDS) == 8
    assert len(EVAL_GOOD_EXPERIMENT_IDS) == 5
    assert len(EVAL_BAD_EXPERIMENT_IDS) == 12


def test_split_partitions_all_25_experiments_with_no_overlap():
    all_ids = TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS + EVAL_BAD_EXPERIMENT_IDS

    assert sorted(all_ids) == list(range(1, 26))
    assert len(set(all_ids)) == 25


def test_exact_experiment_ids_match_spec():
    assert TRAIN_EXPERIMENT_IDS == [1, 2, 3, 11, 13, 14, 15, 17]
    assert EVAL_GOOD_EXPERIMENT_IDS == [12, 18, 22, 24, 25]
    assert EVAL_BAD_EXPERIMENT_IDS == [4, 5, 6, 7, 8, 9, 10, 16, 19, 20, 21, 23]
