import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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


def build_feature_contribution_heatmap_figure(
    feature_error_scores: pd.DataFrame,
    experiment_scores: pd.DataFrame,
    feature_columns: list[str],
    feature_baseline: dict,
    method: str = "mean",
) -> plt.Figure:
    merged = feature_error_scores.merge(
        experiment_scores[
            ["experiment_id", "label", f"{method}_score", f"{method}_exceeds_threshold"]
        ],
        on="experiment_id",
    ).sort_values(f"{method}_score").reset_index(drop=True)

    # feature_columns로 먼저 인덱싱해 baseline과 merged[feature_columns]의 컬럼 순서를
    # 강제로 맞춘다 — 그냥 두면 pandas가 DataFrame-Series 연산에서 안 맞는 컬럼을 자동으로
    # 합집합(알파벳순 재정렬)해버려서, feature_columns에서 뺀 컬럼이 NaN으로 다시 끼어들고
    # 전체 컬럼 순서도 xticklabels와 어긋나게 된다.
    baseline_mean = pd.Series(feature_baseline["mean"])[feature_columns]
    baseline_std = pd.Series(feature_baseline["std"])[feature_columns]
    # train(정상) 기준 z-score로 정규화: 만성적으로 오차가 큰 피처(원래 재구성이 어려운 피처)가
    # 항상 상위권을 차지해 진짜 이상 신호를 가리는 문제를 피하기 위함.
    z_matrix = (merged[feature_columns] - baseline_mean) / baseline_std
    matrix = np.clip(z_matrix.to_numpy(), 0, None)

    row_labels = []
    row_colors = []
    for _, r in merged.iterrows():
        label = int(r["label"])
        predicted = int(r[f"{method}_exceeds_threshold"])
        row_labels.append(f"exp {int(r['experiment_id'])} ({'bad' if label == 1 else 'good'})")
        row_colors.append(_CRITICAL if label != predicted else _INK)

    cmap = LinearSegmentedColormap.from_list("error_scale", ["#ffffff", _CRITICAL])

    fig, ax = plt.subplots(
        figsize=(max(8, len(feature_columns) * 0.35), max(4, len(merged) * 0.35))
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=matrix.max())
    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=90, fontsize=7, color=_INK)
    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    ax.set_title(f"Feature Contribution Heatmap ({method})", fontsize=12, color=_INK)
    fig.colorbar(im, ax=ax, label="z-score vs train baseline (clipped at 0)", shrink=0.8)
    fig.tight_layout()
    return fig


def build_timeline_error_figure(
    timeline_errors: pd.DataFrame,
    experiment_scores: pd.DataFrame,
    threshold: float,
    method: str = "mean",
) -> plt.Figure:
    order = experiment_scores.sort_values(f"{method}_score")["experiment_id"].tolist()
    label_by_exp = experiment_scores.set_index("experiment_id")["label"]

    ncols = 4
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.2 * nrows), sharey=True)
    axes = axes.flatten()

    for ax, exp_id in zip(axes, order):
        sub = timeline_errors[timeline_errors["experiment_id"] == exp_id]
        label = int(label_by_exp[exp_id])
        ax.plot(sub["timestep"], sub["error"], color=_CRITICAL if label == 1 else _GOOD, linewidth=1)
        ax.axhline(threshold, color=_MUTED, linestyle="--", linewidth=1)
        ax.set_title(f"exp {exp_id} ({'bad' if label else 'good'})", fontsize=9, color=_INK)
        ax.tick_params(labelsize=6)
    for ax in axes[len(order):]:
        ax.axis("off")

    fig.suptitle(f"Reconstruction Error Timeline within Shot ({method})", fontsize=12, color=_INK)
    fig.tight_layout()
    return fig


def build_reconstruction_overlay_figure(
    overlay: pd.DataFrame,
    experiment_scores: pd.DataFrame,
) -> plt.Figure:
    order = experiment_scores.sort_values("mean_score")["experiment_id"].tolist()
    label_by_exp = experiment_scores.set_index("experiment_id")["label"]

    ncols = 4
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.4 * nrows))
    axes = axes.flatten()

    for ax, exp_id in zip(axes, order):
        sub = overlay[overlay["experiment_id"] == exp_id]
        label = int(label_by_exp[exp_id])
        feature = sub["feature"].iloc[0]
        ax.plot(sub["timestep"], sub["actual"], color=_INK, linewidth=1, label="actual")
        ax.plot(
            sub["timestep"], sub["reconstructed"], color=_CRITICAL if label else _GOOD,
            linewidth=1, linestyle="--", label="reconstructed",
        )
        ax.set_title(f"exp {exp_id} ({'bad' if label else 'good'})\n{feature}", fontsize=8, color=_INK)
        ax.tick_params(labelsize=6)
    axes[0].legend(loc="upper right", fontsize=6, frameon=False)
    for ax in axes[len(order):]:
        ax.axis("off")

    fig.suptitle("Actual vs Reconstructed (top contributing feature per experiment)", fontsize=12, color=_INK)
    fig.tight_layout()
    return fig
