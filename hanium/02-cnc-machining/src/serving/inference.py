import numpy as np
import pandas as pd
import torch

from lstm_ae.pipeline import compute_feature_errors, compute_window_errors
from lstm_ae.scoring import aggregate_window_errors_by_experiment
from lstm_ae.sequencing import make_eval_windows
from rag.guide import build_guide
from rag.playbook import NO_FAULT, match_playbook

DEMO_EXPERIMENT_ID = 0


def validate_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [col for col in feature_columns if col not in df.columns]


def scale_features(
    df: pd.DataFrame, feature_columns: list[str], scaler_dict: dict
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_columns:
        mean = scaler_dict[col]["mean"]
        std = scaler_dict[col]["std"]
        df[col] = (df[col] - mean) / std
    return df


def score_to_label(score: float, threshold: float) -> tuple[int, str]:
    if score > threshold:
        return 1, "bad"
    return 0, "good"


def rank_feature_contributions(
    feature_errors: np.ndarray,
    feature_columns: list[str],
    feature_baseline: dict,
    exclude_from_ranking: list[str] | None = None,
) -> list[dict]:
    """피처별 오차를 train(정상) 기준 z-score로 정규화해 순위를 매긴다. 원본 오차 절댓값만
    쓰면 정상 상태에서도 원래 재구성이 어려운(만성적으로 오차가 큰) 피처가 항상 상위권을
    차지해 진짜 이상 신호를 가린다 — baseline 평균/표준편차 대비 몇 시그마 벗어났는지를 봐야
    이 샷에서 실제로 이상해진 피처를 가려낼 수 있다. exclude_from_ranking은 실험마다
    상수값만 갖는 셋업 변수처럼 시계열 신호가 아닌 피처를 랭킹에서 빼기 위한 것이다."""
    exclude_from_ranking = exclude_from_ranking or []
    contributions = []
    for col, err in zip(feature_columns, feature_errors):
        if col in exclude_from_ranking:
            continue
        z_score = (err - feature_baseline["mean"][col]) / feature_baseline["std"][col]
        contributions.append({"feature": col, "error": float(err), "z_score": float(z_score)})
    return sorted(contributions, key=lambda c: c["z_score"], reverse=True)


def predict_experiment(
    df: pd.DataFrame,
    model: torch.nn.Module,
    feature_columns: list[str],
    scaler_dict: dict,
    window_size: int,
    threshold: float,
    method: str,
    feature_baseline: dict,
    exclude_from_ranking: list[str] | None = None,
    rag_corpus: list[dict] | None = None,
    rag_index=None,
    openai_client=None,
) -> dict:
    missing = validate_columns(df, feature_columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if len(df) < window_size:
        raise ValueError(
            f"experiment has {len(df)} rows, needs at least {window_size} rows "
            f"for window_size={window_size}"
        )

    scaled = scale_features(df, feature_columns, scaler_dict)
    scaled["experiment_id"] = DEMO_EXPERIMENT_ID

    windows, experiment_ids = make_eval_windows(scaled, feature_columns, window_size)
    window_errors = compute_window_errors(model, windows)
    experiment_scores = aggregate_window_errors_by_experiment(window_errors, experiment_ids)

    score = float(experiment_scores.loc[0, f"{method}_score"])
    predicted_label, predicted_label_text = score_to_label(score, threshold)

    feature_errors = compute_feature_errors(model, windows).mean(axis=0)
    feature_contributions = rank_feature_contributions(
        feature_errors, feature_columns, feature_baseline, exclude_from_ranking
    )

    if predicted_label_text == "good":
        fault = dict(NO_FAULT)
    elif rag_corpus:
        fault = match_playbook(feature_contributions, rag_corpus)
    else:
        fault = None

    guide = build_guide(
        {
            "predicted_label_text": predicted_label_text,
            "score": score,
            "threshold": threshold,
            "feature_contributions": feature_contributions,
            "fault": fault,
        },
        rag_corpus,
        rag_index,
        openai_client,
    )

    return {
        "predicted_label": predicted_label,
        "predicted_label_text": predicted_label_text,
        "score": score,
        "threshold": threshold,
        "method": method,
        "feature_contributions": feature_contributions,
        "fault": fault,
        "guide": guide,
    }
