import matplotlib.pyplot as plt
import pandas as pd

from lstm_ae.plotting import build_confusion_matrix_figure, build_score_distribution_figure

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
