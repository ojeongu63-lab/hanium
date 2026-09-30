"""demo/index.html 생성. 재료: 응답 예시 5개, 코퍼스(플레이북 16항목), 채점 기록, 워커 로그 3개,
대표 가이드(data/rag/demo_guides.json).

  uv run python demo/build_demo.py                              # 페이지 생성
  uv run --env-file .env python demo/build_demo.py --representative   # 대표 가이드 생성 후 페이지 생성

워커 로그는 data/monitoring/ 아래 보관본이라 git에 없다 — 없으면 이벤트 없이 만들고 경고한다.
대표 가이드는 시나리오마다 (상황, 판정) 조합별 첫 불량 배치를 champion + OpenAI 로 다시 추론해
만든다(키 필요). 없으면 기존 파일을 쓰고, 그것도 없으면 가이드 없이 만든다.
생성물 demo/index.html은 커밋한다(pull만 받은 PC에서 바로 열리게)."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "monitoring"))

from demo.build import assemble, pick_representatives, render_html  # noqa: E402
from rag.generation import DEFAULT_MODEL  # noqa: E402

EXAMPLES = [
    ("tool_wear", "합성 · 공구 마모", "predict_response_synthetic_tool_wear.json"),
    ("feed_overload", "합성 · 이송축 과부하", "predict_response_synthetic_feed_overload.json"),
    ("vibration_backlash", "합성 · 진동·고정구 풀림", "predict_response_synthetic_vibration_backlash.json"),
    ("experiment_07", "실제 · experiment_07 (불량)", "predict_response_experiment_07.json"),
    ("experiment_12", "실제 · experiment_12 (정상)", "predict_response_experiment_12.json"),
]
WORKER_LOGS = {
    "temperature": ROOT / "data/monitoring/_temperature_20260902/worker_temperature.log",
    "tool_wear": ROOT / "data/monitoring/_tool_wear_20260902_v2/worker_tool_wear.log",
    "fixture_loosening": ROOT / "data/monitoring/_fixture_loosening_20260902_v2/worker_fixture_loosening.log",
}
OUT = ROOT / "demo" / "index.html"
GUIDES_PATH = ROOT / "data" / "rag" / "demo_guides.json"


def generate_representative_guides(eval_data: dict) -> dict:
    """조합별 첫 불량 배치를 실제 서빙 경로(predict_experiment + RAG)로 다시 추론해 가이드를 만든다."""
    from openai import OpenAI

    from lstm_ae.tracking import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, configure_tracking
    from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS
    from serving.app import _build_model_state
    from serving.inference import predict_experiment
    from simulate_timeline import generate_batch
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient

    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    state = _build_model_state(mv, client.get_run(mv.run_id), model, include_rag=True)
    if state.openai_client is None:
        raise SystemExit("OPENAI_API_KEY 없음 — `uv run --env-file .env python demo/build_demo.py --representative`")
    chat_model = os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL)

    guides: dict[str, dict] = {}
    for scenario, combos in pick_representatives(eval_data["timeline"]).items():
        guides[scenario] = {}
        for key, where in combos.items():
            result = predict_experiment(
                generate_batch(where["day"], where["index"], scenario), state.model, FEATURE_COLUMNS,
                state.scaler_dict, state.window_size, state.thresholds["mean"], "mean", state.feature_baseline,
                exclude_from_ranking=SETUP_CONSTANT_COLUMNS, rag_corpus=state.rag_corpus,
                rag_index=state.rag_index, openai_client=state.openai_client,
            )
            guides[scenario][key] = {
                **where, "guide": result["guide"],
                "fault": result["fault"],
                "versions": {**(state.rag_versions or {}), "chat_model": chat_model},
            }
            print(f"  {scenario} {key} ← day{where['day']:02d}_{where['index']}: "
                  f"{(result['guide'] or {}).get('cause_estimate', '(guide 없음)')[:50]}")
    GUIDES_PATH.write_text(json.dumps(guides, ensure_ascii=False, indent=1))
    print(f"대표 가이드 저장: {GUIDES_PATH} ({sum(len(v) for v in guides.values())}개)")
    return guides


def main() -> None:
    parser = argparse.ArgumentParser(description="데모 페이지 생성")
    parser.add_argument("--representative", action="store_true", help="대표 가이드를 OpenAI로 새로 생성")
    args = parser.parse_args()

    examples = [
        {"key": key, "label": label, "response": json.loads((ROOT / "docs/examples" / filename).read_text())}
        for key, label, filename in EXAMPLES
    ]
    corpus = json.loads((ROOT / "data/rag/corpus.json").read_text())
    eval_data = json.loads((ROOT / "data/rag/eval_playbook.json").read_text())
    meta_path = ROOT / "data/rag/corpus_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    versions = {
        "playbook": meta.get("playbook"), "corpus": meta.get("built_at"),
        "chat_model": os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_MODEL),
    }
    logs = {}
    for name, path in WORKER_LOGS.items():
        if path.exists():
            logs[name] = path.read_text()
        else:
            print(f"경고: 워커 로그 없음 — {path} (이벤트 없이 생성)")

    if args.representative:
        guides = generate_representative_guides(eval_data)
    elif GUIDES_PATH.exists():
        guides = json.loads(GUIDES_PATH.read_text())
    else:
        print(f"경고: 대표 가이드 없음 — {GUIDES_PATH} (--representative 로 생성 가능; 가이드 없이 생성)")
        guides = {}

    data = assemble(examples, corpus, eval_data, logs, versions,
                    datetime.now().astimezone().isoformat(timespec="minutes"), guides=guides)
    engine_js = (ROOT / "demo/sim_engine.js").read_text()
    OUT.write_text(render_html((ROOT / "demo/template.html").read_text(), data, engine_js))
    events = {name: len(sc["events"]) for name, sc in data["scenarios"].items()}
    n_guides = {name: len(sc["guides"]) for name, sc in data["scenarios"].items()}
    print(f"저장: {OUT} ({OUT.stat().st_size / 1024:.0f} KB) — 예시 {len(data['examples'])}개, "
          f"시나리오 {list(data['scenarios'])}, 이벤트 {events}, 대표 가이드 {n_guides}")


if __name__ == "__main__":
    main()
