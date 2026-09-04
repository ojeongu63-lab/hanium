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
import mlflow.artifacts
import mlflow.pytorch
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from mlflow.tracking import MlflowClient
from openai import OpenAI

from lstm_ae.tracking import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, configure_tracking
from monitoring.drift import compute_drift_status
from monitoring.logging import count_requests, get_recent_requests, log_request
from monitoring.mlflow_logging import log_drift_metrics
from monitoring.shadow_log import record_shadow_prediction
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
from rag.generation import DEFAULT_MODEL
from serving.inference import predict_experiment, scale_features

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "monitoring" / "requests.db"
SHADOW_DB = ROOT / "data" / "monitoring" / "shadow.db"
DRIFT_WINDOW_SIZE = 10
DEMO_INDEX = ROOT / "demo" / "index.html"
DATASET_DIR = ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
DEMO_INPUTS = {
    "tool_wear": ROOT / "synthetic" / "scenarios" / "tool_wear.csv",
    "feed_overload": ROOT / "synthetic" / "scenarios" / "feed_overload.csv",
    "vibration_backlash": ROOT / "synthetic" / "scenarios" / "vibration_backlash.csv",
    "experiment_07": DATASET_DIR / "experiment_07.csv",
    "experiment_12": DATASET_DIR / "experiment_12.csv",
}


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
    rag_versions: dict | None = None  # data/rag/corpus_meta.json (playbook 해시, 빌드 시각)


_state: ModelState | None = None
_shadow_state: ModelState | None = None


def load_rag_state() -> tuple[list[dict] | None, object | None, object | None, dict | None]:
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=api_key) if api_key else None

    corpus_path = ROOT / "data" / "rag" / "corpus.json"
    index_path = ROOT / "data" / "rag" / "corpus.index"
    if not corpus_path.exists() or not index_path.exists():
        return None, None, openai_client, None

    rag_corpus = json.loads(corpus_path.read_text())
    rag_index = faiss.read_index(str(index_path))
    meta_path = ROOT / "data" / "rag" / "corpus_meta.json"
    rag_versions = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return rag_corpus, rag_index, openai_client, rag_versions


def load_companion_json(run_id: str, name: str, fallback: Path) -> dict:
    """모델 run에 붙은 동반 아티팩트를 우선 읽고, 없으면 고정 경로로 폴백한다.

    scaler와 feature_baseline은 모델과 짝이 맞아야 하는데, 원래는 고정 경로에만
    있어 모델 버전과 따로 놀았다. 자동 재학습이 이 파일들을 덮어쓰면 champion과
    짝이 어긋나 조용히 틀린 스케일로 추론하게 된다.

    폴백이 필요한 이유: 최초 학습으로 만들어진 기존 champion run에는 이
    아티팩트가 없다. 자동 재학습으로 만들어진 run만 갖고 있다.
    """
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"companion/{name}"
        )
        return json.loads(Path(local_path).read_text())
    except Exception:
        return json.loads(fallback.read_text())


def _build_model_state(mv, run, model, include_rag: bool) -> ModelState:
    thresholds = {
        method: run.data.metrics[f"{method}_threshold"] for method in ["mean", "max", "p95"]
    }
    window_size = int(run.data.params["window_size"])
    scaler_dict = load_companion_json(
        mv.run_id, "scaler.json", ROOT / "data" / "processed" / "scaler.json"
    )
    feature_baseline = load_companion_json(
        mv.run_id, "feature_baseline.json", ROOT / "data" / "model" / "feature_baseline.json"
    )
    rag_corpus, rag_index, openai_client, rag_versions = (
        load_rag_state() if include_rag else (None, None, None, None)
    )
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
        rag_versions=rag_versions,
    )


def load_model_state() -> ModelState:
    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    run = client.get_run(mv.run_id)
    return _build_model_state(mv, run, model, include_rag=True)


def load_candidate_state(version: str) -> ModelState:
    """섀도우 후보를 champion alias 없이 특정 버전으로 직접 로드한다 —
    승격 전이라 champion alias는 아직 candidate를 안 가리킨다. RAG는
    섀도우 추론(로그 기록용)에는 필요 없어 안 채운다."""
    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version(REGISTERED_MODEL_NAME, version)
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    run = client.get_run(mv.run_id)
    return _build_model_state(mv, run, model, include_rag=False)


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

        if _shadow_state is not None:
            _record_shadow_if_possible(df, method, result["predicted_label_text"], file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **result,
        "model_version": state.model_version,
        "mlflow_run_id": state.mlflow_run_id,
        "versions": _versions(state),
    }


def _versions(state: ModelState) -> dict:
    """fault·guide가 어떤 플레이북·코퍼스·LLM으로 만들어졌는지. RAG 미로드면 전부 None."""
    meta = state.rag_versions or {}
    return {
        "playbook": meta.get("playbook"),
        "corpus": meta.get("built_at"),
        "chat_model": os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL) if state.openai_client else None,
    }


def _record_shadow_if_possible(df, method, champion_label, filename) -> None:
    """섀도우 후보 추론이 실패해도 사용자 응답에는 영향을 주면 안 된다."""
    try:
        shadow_result = predict_experiment(
            df=df,
            model=_shadow_state.model,
            feature_columns=FEATURE_COLUMNS,
            scaler_dict=_shadow_state.scaler_dict,
            window_size=_shadow_state.window_size,
            threshold=_shadow_state.thresholds[method],
            method=method,
            feature_baseline=_shadow_state.feature_baseline,
            exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
        )
        batch_id = Path(filename).stem
        record_shadow_prediction(
            batch_id, champion_label, shadow_result["predicted_label_text"], SHADOW_DB
        )
    except Exception as exc:
        print(f"섀도우 추론 실패(무시하고 계속): {exc}")


@app.get("/drift-status")
def drift_status(state: ModelState = Depends(get_model_state)) -> dict:
    recent = get_recent_requests(DRIFT_WINDOW_SIZE, DB_PATH)
    status = compute_drift_status(
        recent, threshold=state.thresholds["mean"], window_size=DRIFT_WINDOW_SIZE
    )
    log_drift_metrics(status, state.mlflow_run_id, step=count_requests(DB_PATH))
    return {**status, "checked_at": datetime.now(timezone.utc).isoformat()}


@app.get("/demo")
def demo_page() -> FileResponse:
    """미팅 데모 페이지. 모델 로드와 무관하게 열린다 — 페이지가 /health로 실시간 여부를 판단한다."""
    if not DEMO_INDEX.exists():
        raise HTTPException(
            status_code=404,
            detail="demo/index.html 없음 — `uv run python demo/build_demo.py`를 먼저 실행하세요.",
        )
    return FileResponse(DEMO_INDEX, media_type="text/html")


@app.get("/demo/inputs/{key}")
def demo_input(key: str) -> FileResponse:
    """데모 페이지의 '실시간으로 다시 계산'이 /predict에 올릴 예시 CSV."""
    path = DEMO_INPUTS.get(key)
    if path is None:
        raise HTTPException(status_code=404, detail=f"모르는 예시 키 '{key}' — {sorted(DEMO_INPUTS)}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"입력 파일 없음: {path} (README §1-2 데이터 배치 확인)")
    return FileResponse(path, media_type="text/csv", filename=path.name)


TIMELINE_SCENARIOS = ("temperature", "tool_wear", "fixture_loosening")
TIMELINE_BATCHES_PER_DAY = 5


def _generate_timeline_batch(day: int, index: int, scenario: str) -> pd.DataFrame:
    """monitoring/simulate_timeline.py 의 generate_batch 를 그대로 쓴다(수정 금지 파일이라 import만).
    monitoring/ 은 패키지가 아닌 최상위 스크립트 폴더라 sys.path 에 더한다 — rag/eval_playbook.py 와 같은 방식."""
    import sys

    monitoring_dir = str(ROOT / "monitoring")
    if monitoring_dir not in sys.path:
        sys.path.insert(0, monitoring_dir)
    from simulate_timeline import generate_batch  # noqa: PLC0415

    return generate_batch(day, index, scenario)


@app.get("/demo/timeline/{scenario}/{day}/{index}")
def demo_timeline_batch(scenario: str, day: int, index: int) -> Response:
    """시뮬레이션 탭의 '이 배치 지금 계산'이 /predict에 올릴 타임라인 배치 CSV.
    시뮬레이터와 같은 규칙으로 그 자리에서 만든다(원본 데이터셋이 있는 PC에서만 동작)."""
    if scenario not in TIMELINE_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"모르는 시나리오 '{scenario}' — {TIMELINE_SCENARIOS}")
    if day < 1 or not 0 <= index < TIMELINE_BATCHES_PER_DAY:
        raise HTTPException(status_code=404, detail=f"day는 1 이상, index는 0~{TIMELINE_BATCHES_PER_DAY - 1}")
    try:
        df = _generate_timeline_batch(day, index, scenario)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"원본 데이터셋 없음({exc}) — README §1-2 데이터 배치 확인"
        ) from exc
    return Response(
        content=df.to_csv(index=False), media_type="text/csv",
        headers={"Content-Disposition": f'inline; filename="{scenario}_day{day:02d}_{index}.csv"'},
    )


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


@app.post("/start-shadow")
def start_shadow(payload: dict) -> dict:
    global _shadow_state
    _shadow_state = load_candidate_state(payload["model_version"])
    return {"status": "shadow_started", "candidate_version": _shadow_state.model_version}


@app.post("/stop-shadow")
def stop_shadow() -> dict:
    global _shadow_state
    _shadow_state = None
    return {"status": "shadow_stopped"}
