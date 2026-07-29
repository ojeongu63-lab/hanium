import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_GOOD = "#0ca30c"
_CRITICAL = "#d03b3b"
_GOOD_WASH = "#e3f5e3"
_CRITICAL_WASH = "#fbe6e6"
_INK = "#0b0b0b"
_MUTED = "#898781"


def build_confusion_matrix_figure(results: dict, method: str) -> plt.Figure:
    r = results[method]
    cells = [[r["tp"], r["fn"]], [r["fp"], r["tn"]]]
    correct = [[True, False], [False, True]]
    row_labels = ["actual bad", "actual good"]
    col_labels = ["predicted bad", "predicted good"]

    fig, ax = plt.subplots(figsize=(4, 4))
    for i in range(2):
        for j in range(2):
            color = _GOOD_WASH if correct[i][j] else _CRITICAL_WASH
            ax.add_patch(
                plt.Rectangle((j, 1 - i), 1, 1, facecolor=color, edgecolor="white", linewidth=2)
            )
            ax.text(
                j + 0.5, 1 - i + 0.5, str(cells[i][j]),
                ha="center", va="center", fontsize=22, color=_INK, fontweight="bold",
            )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(col_labels, fontsize=10, color=_INK)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(row_labels[::-1], fontsize=10, color=_INK)
    ax.set_title(f"Confusion Matrix ({method})", fontsize=12, color=_INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def build_score_distribution_figure(
    experiment_scores: pd.DataFrame, threshold: float, method: str
) -> plt.Figure:
    score_col = f"{method}_score"
    df = experiment_scores.sort_values(score_col).reset_index(drop=True)
    colors = df["label"].map({0: _GOOD, 1: _CRITICAL})

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        df[score_col], range(len(df)), c=colors, s=80, zorder=3,
        edgecolors="white", linewidths=1,
    )
    ax.axvline(
        threshold, color=_MUTED, linestyle="--", linewidth=1.5, label=f"threshold={threshold:.3f}"
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["experiment_id"])
    ax.set_xlabel(f"{method} reconstruction error score")
    ax.set_ylabel("experiment_id")
    ax.set_title(f"Eval Experiment Scores ({method})", fontsize=12, color=_INK)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
