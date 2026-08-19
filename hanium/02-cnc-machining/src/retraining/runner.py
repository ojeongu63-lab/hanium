from pathlib import Path

import pandas as pd


def collect_normal_batches(
    arrived_labels: list[dict],
    timeline_dir: Path,
    current_day: int,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """라벨이 도착했고 정상인 배치만 모아 raw 프레임으로 합친다.

    LSTM-AE는 정상 데이터만 학습하므로 불량 라벨은 제외한다.
    배치 하나가 가공 1회분이므로, 배치마다 다른 experiment_id를 부여한다
    (lstm_ae 파이프라인이 experiment_id로 윈도우를 그룹핑한다).
    """
    cutoff = current_day - lookback_days
    frames = []
    for record in arrived_labels:
        if record["label"] != "good" or record["produced_day"] < cutoff:
            continue
        csv_path = Path(timeline_dir) / f"{record['batch_id']}.csv"
        frames.append(pd.read_csv(csv_path).assign(experiment_id=len(frames)))

    if not frames:
        raise ValueError("재학습에 쓸 정상 라벨 배치가 없습니다")
    return pd.concat(frames, ignore_index=True)


def rescale_eval(
    eval_scaled: pd.DataFrame,
    old_scaler: dict,
    new_scaler: dict,
    feature_columns: list[str],
) -> pd.DataFrame:
    """옛 scaler로 스케일된 eval을 원값으로 되돌린 뒤 새 scaler로 다시 스케일한다.

    data/processed/eval.csv는 이미 옛 scaler로 스케일돼 저장돼 있고
    run_lstm_pipeline은 스케일링을 하지 않는다. 새 scaler로 학습하면서 옛
    eval을 그대로 넣으면 train과 eval의 좌표계가 어긋난다.

    라벨·실험 구성 메타 컬럼은 손대지 않으므로 게이트 기준셋이 불변으로 유지된다.
    """
    out = eval_scaled.copy()
    for col in feature_columns:
        raw = out[col] * old_scaler[col]["std"] + old_scaler[col]["mean"]
        out[col] = (raw - new_scaler[col]["mean"]) / new_scaler[col]["std"]
    return out
