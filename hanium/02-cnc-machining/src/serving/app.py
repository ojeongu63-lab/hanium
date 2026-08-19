import io
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import faiss
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from mlflow.tracking import MlflowClient
from openai import OpenAI

from lstm_ae.tracking import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, configure_tracking
from monitoring.drift import compute_drift_status
from monitoring.logging import count_requests, get_recent_requests, log_request
from monitoring.mlflow_logging import log_drift_metrics
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
from serving.inference import predict_experiment, scale_features

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "monitoring" / "requests.db"
DRIFT_WINDOW_SIZE = 10


@dataclass
class ModelState:
    model: torch.nn.Module
    scaler_dict: dict
    thresholds: dict
    window_size: int
    model_version: str
    mlflow_run_id: str
    feature_baseline: dict
    rag_corpus: list[dict] | None = None
    rag_index: object | None = None
    openai_client: object | None = None


_state: ModelState | None = None


def load_rag_state() -> tuple[list[dict] | None, object | None, object | None]:
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=api_key) if api_key else None

    corpus_path = ROOT / "data" / "rag" / "corpus.json"
    index_path = ROOT / "data" / "rag" / "corpus.index"
    if not corpus_path.exists() or not index_path.exists():
        return None, None, openai_client

    rag_corpus = json.loads(corpus_path.read_text())
    rag_index = faiss.read_index(str(index_path))
    return rag_corpus, rag_index, openai_client


def load_model_state() -> ModelState:
    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    run = client.get_run(mv.run_id)
    thresholds = {
        method: run.data.metrics[f"{method}_threshold"] for method in ["mean", "max", "p95"]
    }
    window_size = int(run.data.params["window_size"])
    scaler_dict = json.loads((ROOT / "data" / "processed" / "scaler.json").read_text())
    feature_baseline = json.loads((ROOT / "data" / "model" / "feature_baseline.json").read_text())
    rag_corpus, rag_index, openai_client = load_rag_state()
    return ModelState(
        model=model,
        scaler_dict=scaler_dict,
        thresholds=thresholds,
        window_size=window_size,
        model_version=str(mv.version),
        mlflow_run_id=mv.run_id,
        feature_baseline=feature_baseline,
        rag_corpus=rag_corpus,
        rag_index=rag_index,
        openai_client=openai_client,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    try:
        _state = load_model_state()
    except Exception as exc:
        print(
            "champion 모델을 로드하지 못했습니다 "
            f"(scripts/promote_model.py를 먼저 실행하세요): {exc}"
        )
        _state = None
    yield


app = FastAPI(lifespan=lifespan)


def get_model_state() -> ModelState:
    if _state is None:
        raise HTTPException(
            status_code=503,
            detail="champion 모델이 로드되지 않았습니다. scripts/promote_model.py를 먼저 실행하세요.",
        )
    return _state


@app.get("/health")
def health(state: ModelState = Depends(get_model_state)) -> dict:
    return {
        "status": "ok",
        "model_version": state.model_version,
        "mlflow_run_id": state.mlflow_run_id,
    }


@app.post("/predict")
async def predict(
    file: UploadFile,
    method: Literal["mean", "max", "p95"] = "mean",
    state: ModelState = Depends(get_model_state),
) -> dict:
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
        result = predict_experiment(
            df=df,
            model=state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=state.scaler_dict,
            window_size=state.window_size,
            threshold=state.thresholds[method],
            method=method,
            feature_baseline=state.feature_baseline,
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
            rag_corpus=state.rag_corpus,
            rag_index=state.rag_index,
            openai_client=state.openai_client,
        )
        scaled = scale_features(df, FEATURE_COLUMNS, state.scaler_dict)
        feature_means = scaled[FEATURE_COLUMNS].mean().to_dict()
        log_request(feature_means, result["score"], result["predicted_label_text"], DB_PATH)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **result,
        "model_version": state.model_version,
        "mlflow_run_id": state.mlflow_run_id,
    }


@app.get("/drift-status")
def drift_status(state: ModelState = Depends(get_model_state)) -> dict:
    recent = get_recent_requests(DRIFT_WINDOW_SIZE, DB_PATH)
    status = compute_drift_status(
        recent, threshold=state.thresholds["mean"], window_size=DRIFT_WINDOW_SIZE
    )
    log_drift_metrics(status, state.mlflow_run_id, step=count_requests(DB_PATH))
    return {**status, "checked_at": datetime.now(timezone.utc).isoformat()}


@app.post("/reload-model")
def reload_model() -> dict:
    """champion alias가 바뀐 뒤 돌고 있는 서버가 새 모델을 집어들게 한다.

    로드에 실패하면 기존 상태를 유지한다 — 교체 실패가 서빙 중단으로
    번지면 안 된다.
    """
    global _state
    previous = _state
    try:
        _state = load_model_state()
    except Exception as exc:
        _state = previous
        raise HTTPException(status_code=500, detail=f"모델 리로드 실패, 기존 모델 유지: {exc}")
    return {"status": "reloaded", "model_version": _state.model_version}
