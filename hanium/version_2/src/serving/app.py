import io
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from mlflow.tracking import MlflowClient

from lstm_ae.tracking import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, configure_tracking
from preprocessing.columns import FEATURE_COLUMNS
from serving.inference import predict_experiment

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelState:
    model: torch.nn.Module
    scaler_dict: dict
    thresholds: dict
    window_size: int
    model_version: str
    mlflow_run_id: str


_state: ModelState | None = None


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
    return ModelState(
        model=model,
        scaler_dict=scaler_dict,
        thresholds=thresholds,
        window_size=window_size,
        model_version=mv.version,
        mlflow_run_id=mv.run_id,
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
    df = pd.read_csv(io.BytesIO(content))
    try:
        result = predict_experiment(
            df=df,
            model=state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=state.scaler_dict,
            window_size=state.window_size,
            threshold=state.thresholds[method],
            method=method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **result,
        "model_version": state.model_version,
        "mlflow_run_id": state.mlflow_run_id,
    }
