import json
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd

from lstm_ae.pipeline import run_lstm_pipeline
from lstm_ae.tracking import REGISTERED_MODEL_NAME, build_run_metrics, configure_tracking
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
from preprocessing.scaling import fit_scaler, scaler_to_dict

# scripts/run_lstm_training.py 의 TRAINING_CONFIG 와 동일하게 유지한다.
# 재학습은 모델 구조·하이퍼파라미터를 바꾸지 않는다 — 바뀌는 것은 데이터와 scaler뿐이다.
TRAINING_CONFIG = {
    "window_size": 20,
    "hidden_size": 64,
    "latent_dim": 16,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "random_seed": 42,
    "threshold_percentile": 95.0,
}


def build_retrain_params(batch_days: str, batch_count: int) -> dict:
    """재학습 run의 params.

    build_run_params()를 쓰지 않는 이유: 그 함수는 전처리 manifest의
    experiment_split(train/eval 실험 ID 목록)을 요구하는데, 재학습의 학습
    데이터는 실험 ID가 아니라 날짜 배치라 그 개념이 성립하지 않는다.

    window_size가 반드시 들어가야 서빙이 champion을 로드할 수 있다.
    """
    return {
        **TRAINING_CONFIG,
        "source": "auto_retrain",
        "retrain_batch_days": batch_days,
        "retrain_batch_count": batch_count,
    }


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


def run_retraining(
    timeline_dir: Path,
    labels_db: Path,
    current_day: int,
    root: Path,
    lookback_days: int = 30,
) -> dict:
    """라벨 도착분으로 재학습하고 MLflow에 새 run으로 기록한다. 승격은 하지 않는다.

    산출물은 data/retrain/<timestamp>/ 에 격리한다 — 게이트가 거부할 수도 있는데
    정본(data/model/, data/processed/scaler.json)을 먼저 덮어쓰면 champion과
    짝이 어긋난 상태로 남는다.
    """
    from monitoring.labels import get_arrived_labels

    arrived = get_arrived_labels(current_day, labels_db)
    train_raw = collect_normal_batches(arrived, timeline_dir, current_day, lookback_days)

    retrain_dir = root / "data" / "retrain" / datetime.now().strftime("%Y%m%d_%H%M%S")
    retrain_dir.mkdir(parents=True, exist_ok=True)

    scaler = fit_scaler(train_raw, FEATURE_COLUMNS)
    new_scaler_dict = scaler_to_dict(scaler, FEATURE_COLUMNS)
    (retrain_dir / "scaler.json").write_text(
        json.dumps(new_scaler_dict, indent=2, ensure_ascii=False)
    )

    train_scaled = train_raw.copy()
    train_scaled[FEATURE_COLUMNS] = scaler.transform(train_raw[FEATURE_COLUMNS])
    train_scaled[FEATURE_COLUMNS + ["experiment_id"]].to_csv(
        retrain_dir / "train.csv", index=False
    )

    old_scaler_dict = json.loads((root / "data" / "processed" / "scaler.json").read_text())
    eval_old = pd.read_csv(root / "data" / "processed" / "eval.csv")
    rescale_eval(eval_old, old_scaler_dict, new_scaler_dict, FEATURE_COLUMNS).to_csv(
        retrain_dir / "eval.csv", index=False
    )

    used_days = sorted({r["produced_day"] for r in arrived if r["label"] == "good"})
    batch_days = f"{used_days[0]}-{used_days[-1]}" if used_days else "none"

    configure_tracking()
    with mlflow.start_run() as active:
        mlflow.log_params(
            build_retrain_params(batch_days, int(train_raw["experiment_id"].nunique()))
        )
        summary = run_lstm_pipeline(
            train_csv_path=str(retrain_dir / "train.csv"),
            eval_csv_path=str(retrain_dir / "eval.csv"),
            feature_columns=FEATURE_COLUMNS,
            output_dir=str(retrain_dir),
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
            **TRAINING_CONFIG,
        )
        # 서빙 계약: {mean,max,p95}_threshold 를 만든다
        mlflow.log_metrics(build_run_metrics(summary["thresholds"], summary["results"]))
        for name in ["scaler.json", "feature_baseline.json"]:
            mlflow.log_artifact(str(retrain_dir / name), artifact_path="companion")
        model_info = mlflow.pytorch.log_model(
            summary["model"],
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )
        run_id = active.info.run_id

    return {
        "run_id": run_id,
        "model_version": model_info.registered_model_version,
        "retrain_dir": retrain_dir,
        "recall": summary["results"]["mean"]["recall"],
        # G1 은 소수 recall 이 아니라 놓친 개수로 비교한다 (eval 불량이 11개뿐).
        "missed": int(summary["results"]["mean"]["fn"]),
        "thresholds": summary["thresholds"],
    }
