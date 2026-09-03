"""demo/index.html 생성. 재료: 응답 예시 5개, 코퍼스(플레이북 16항목), 채점 기록, 워커 로그 3개.

  uv run python demo/build_demo.py

워커 로그는 data/monitoring/ 아래 보관본이라 git에 없다 — 없으면 이벤트 없이 만들고 경고한다.
생성물 demo/index.html은 커밋한다(pull만 받은 PC에서 바로 열리게)."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from demo.build import assemble, render_html  # noqa: E402
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


def main() -> None:
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

    data = assemble(examples, corpus, eval_data, logs, versions,
                    datetime.now().astimezone().isoformat(timespec="minutes"))
    OUT.write_text(render_html((ROOT / "demo/template.html").read_text(), data))
    events = {name: len(sc["events"]) for name, sc in data["scenarios"].items()}
    print(f"저장: {OUT} ({OUT.stat().st_size / 1024:.0f} KB) — 예시 {len(data['examples'])}개, "
          f"시나리오 {list(data['scenarios'])}, 이벤트 {events}")


if __name__ == "__main__":
    main()
