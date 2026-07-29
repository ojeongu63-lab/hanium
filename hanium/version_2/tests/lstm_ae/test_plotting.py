import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lstm_ae.plotting import (
    build_confusion_matrix_figure,
    build_feature_contribution_heatmap_figure,
    build_reconstruction_overlay_figure,
    build_score_distribution_figure,
    build_timeline_error_figure,
)

RESULTS = {
    "mean": {"precision": 0.9, "recall": 0.9, "tp": 10, "fp": 3, "fn": 1, "tn": 2},
}


def test_build_confusion_matrix_figure_returns_figure_with_counts():
    fig = build_confusion_matrix_figure(RESULTS, method="mean")

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    texts = {t.get_text() for t in ax.texts}
    assert texts == {"10", "3", "1", "2"}
    plt.close(fig)


def test_build_score_distribution_figure_returns_figure():
    experiment_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3],
        "mean_score": [0.2, 0.9, 0.5],
        "label": [0, 1, 0],
    })

    fig = build_score_distribution_figure(experiment_scores, threshold=0.6, method="mean")

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.collections) == 1  # the scatter
    assert len(ax.get_yticklabels()) == 3
    plt.close(fig)


def test_build_feature_contribution_heatmap_figure_returns_figure_with_all_rows():
    feature_columns = ["f0", "f1", "f2"]
    feature_error_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3],
        "f0": [0.1, 0.5, 0.9],
        "f1": [0.2, 0.4, 1.2],
        "f2": [0.3, 0.6, 0.8],
    })
    experiment_scores = pd.DataFrame({
        "experiment_id": [1, 2, 3],
        "label": [0, 0, 1],
        "mean_score": [0.2, 0.5, 0.9],
        "mean_exceeds_threshold": [False, True, True],
    })
    feature_baseline = {
        "mean": {"f0": 0.2, "f1": 0.3, "f2": 0.4},
        "std": {"f0": 0.1, "f1": 0.1, "f2": 0.1},
    }

    fig = build_feature_contribution_heatmap_figure(
        feature_error_scores, experiment_scores, feature_columns, feature_baseline, method="mean"
    )

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == 3
    assert len(ax.get_xticklabels()) == 3
    plt.close(fig)


def test_build_feature_contribution_heatmap_figure_ignores_baseline_columns_not_in_feature_columns():
    # feature_baseline has "f1" but feature_columns (the ones to actually rank/plot) excludes
    # it -- regression test for a pandas DataFrame-Series alignment bug where the missing
    # column silently reappeared as NaN and scrambled the column order (see plotting.py).
    feature_columns = ["f0", "f2"]
    feature_error_scores = pd.DataFrame({
        "experiment_id": [1, 2],
        "f0": [0.1, 0.9],
        "f1": [0.9, 0.9],
        "f2": [0.3, 0.8],
    })
    experiment_scores = pd.DataFrame({
        "experiment_id": [1, 2],
        "label": [0, 1],
        "mean_score": [0.2, 0.9],
        "mean_exceeds_threshold": [False, True],
    })
    feature_baseline = {
        "mean": {"f0": 0.2, "f1": 0.3, "f2": 0.4},
        "std": {"f0": 0.1, "f1": 0.1, "f2": 0.1},
    }

    fig = build_feature_contribution_heatmap_figure(
        feature_error_scores, experiment_scores, feature_columns, feature_baseline, method="mean"
    )

    ax = fig.axes[0]
    data = ax.images[0].get_array()
    assert data.shape == (2, 2)
    assert not np.isnan(data).any()
    assert [t.get_text() for t in ax.get_xticklabels()] == feature_columns
    # row 0 = exp1 (lower score, sorted first): z-scores clipped at 0 -> both 0
    # row 1 = exp2: f0 z=(0.9-0.2)/0.1=7.0, f2 z=(0.8-0.4)/0.1=4.0
    np.testing.assert_allclose(data, [[0.0, 0.0], [7.0, 4.0]])
    plt.close(fig)


def test_build_timeline_error_figure_returns_one_subplot_per_experiment():
    timeline_errors = pd.DataFrame({
        "experiment_id": [1, 1, 1, 2, 2, 2],
        "timestep": [0, 1, 2, 0, 1, 2],
        "error": [0.1, 0.2, 0.15, 0.5, 0.9, 0.6],
    })
    experiment_scores = pd.DataFrame({
        "experiment_id": [1, 2],
        "label": [0, 1],
        "mean_score": [0.2, 0.9],
    })

    fig = build_timeline_error_figure(timeline_errors, experiment_scores, threshold=0.5, method="mean")

    assert isinstance(fig, plt.Figure)
    active_axes = [ax for ax in fig.axes if ax.lines]
    assert len(active_axes) == 2
    plt.close(fig)


def test_build_reconstruction_overlay_figure_returns_one_subplot_per_experiment():
    overlay = pd.DataFrame({
        "experiment_id": [1, 1, 1, 2, 2, 2],
        "feature": ["f0", "f0", "f0", "f1", "f1", "f1"],
        "timestep": [0, 1, 2, 0, 1, 2],
        "actual": [0.1, 0.2, 0.15, 0.5, 0.9, 0.6],
        "reconstructed": [0.12, 0.18, 0.16, 0.4, 0.8, 0.55],
    })
    experiment_scores = pd.DataFrame({
        "experiment_id": [1, 2],
        "label": [0, 1],
        "mean_score": [0.2, 0.9],
    })

    fig = build_reconstruction_overlay_figure(overlay, experiment_scores)

    assert isinstance(fig, plt.Figure)
    active_axes = [ax for ax in fig.axes if ax.lines]
    assert len(active_axes) == 2
    for ax in active_axes:
        assert len(ax.lines) == 2  # actual + reconstructed
    plt.close(fig)
