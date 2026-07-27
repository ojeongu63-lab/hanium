from preprocessing.split import (
    EVAL_BAD_EXPERIMENT_IDS,
    EVAL_GOOD_EXPERIMENT_IDS,
    EXCLUDED_DUPLICATE_EXPERIMENT_IDS,
    TRAIN_EXPERIMENT_IDS,
)


def test_split_counts_match_spec():
    assert len(TRAIN_EXPERIMENT_IDS) == 8
    assert len(EVAL_GOOD_EXPERIMENT_IDS) == 3
    assert len(EVAL_BAD_EXPERIMENT_IDS) == 11


def test_split_partitions_25_experiments_minus_excluded_duplicates_with_no_overlap():
    all_ids = TRAIN_EXPERIMENT_IDS + EVAL_GOOD_EXPERIMENT_IDS + EVAL_BAD_EXPERIMENT_IDS

    assert sorted(all_ids + EXCLUDED_DUPLICATE_EXPERIMENT_IDS) == list(range(1, 26))
    assert len(set(all_ids)) == len(all_ids)
    assert set(all_ids).isdisjoint(EXCLUDED_DUPLICATE_EXPERIMENT_IDS)


def test_exact_experiment_ids_match_spec():
    assert TRAIN_EXPERIMENT_IDS == [1, 2, 3, 11, 13, 14, 15, 17]
    assert EVAL_GOOD_EXPERIMENT_IDS == [12, 18, 22]
    assert EVAL_BAD_EXPERIMENT_IDS == [4, 5, 6, 7, 8, 9, 10, 16, 20, 21, 23]
    assert EXCLUDED_DUPLICATE_EXPERIMENT_IDS == [19, 24, 25]
