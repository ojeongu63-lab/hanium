"""플레이북 서명 대조 오프라인 채점 — LLM 호출 없음, 임베딩 없음.

1) 기록된 4건(합성 3 + experiment_07)의 상황·판정
2) 타임라인 3종 × N일 × 5배치를 champion으로 추론해 구간별 판정 분포
결과는 data/rag/eval_playbook.json 과 표준 출력. 서버·DB는 건드리지 않는다.

  nice -n 19 uv run python rag/eval_playbook.py [--scenarios temperature tool_wear] [--days 40]

여러 번 실행하면 --scenarios 로 지정한 시나리오만 갱신되고 나머지는 유지된다.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "monitoring"))

from lstm_ae.tracking import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, configure_tracking  # noqa: E402
from preprocessing.columns import FEATURE_COLUMNS, SETUP_CONSTANT_COLUMNS  # noqa: E402
from rag.playbook import match_playbook  # noqa: E402
from serving.app import _build_model_state  # noqa: E402
from serving.inference import predict_experiment  # noqa: E402
from simulate_timeline import BATCHES_PER_DAY, generate_batch, true_label  # noqa: E402

CORPUS_PATH = ROOT / "data" / "rag" / "corpus.json"
OUT_PATH = ROOT / "data" / "rag" / "eval_playbook.json"
RECORDED = {
    "합성 tool_wear": ROOT / "synthetic/scenarios/tool_wear_predict_result.json",
    "합성 feed_overload": ROOT / "synthetic/scenarios/feed_overload_predict_result.json",
    "합성 vibration_backlash": ROOT / "synthetic/scenarios/vibration_backlash_predict_result.json",
    "실제 experiment_07": ROOT / "docs/examples/predict_response_experiment_07.json",
}
PHASES = ((1, 10), (11, 20), (21, 30), (31, 40))


def score_recorded(corpus: list[dict]) -> dict:
    out = {}
    for name, path in RECORDED.items():
        fault = match_playbook(json.loads(path.read_text())["feature_contributions"], corpus)
        out[name] = {k: fault[k] for k in ("situation", "category", "verdict", "coverage")}
        print(f"  {name:<24} {fault['verdict_ko']:<6} {fault['situation']}  일치 {fault['coverage']:.2f}")
    return out


def load_champion():
    configure_tracking()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    return _build_model_state(mv, client.get_run(mv.run_id), model, include_rag=False)


def score_timeline(corpus: list[dict], state, scenarios: list[str], days: int) -> dict:
    out = {}
    for scenario in scenarios:
        rows = []
        for day in range(1, days + 1):
            for index in range(BATCHES_PER_DAY):
                result = predict_experiment(
                    generate_batch(day, index, scenario), state.model, FEATURE_COLUMNS,
                    state.scaler_dict, state.window_size, state.thresholds["mean"], "mean",
                    state.feature_baseline, exclude_from_ranking=SETUP_CONSTANT_COLUMNS,
                    rag_corpus=corpus,
                )
                rows.append({
                    "day": day, "index": index, "truth": true_label(scenario, day),
                    "pred": result["predicted_label_text"], "fault": result["fault"],
                    "ratio": round(result["score"] / result["threshold"], 4),
                    "top": [[c["feature"], round(c["z_score"], 1)] for c in result["feature_contributions"][:3]],
                    "top10": [[c["feature"], round(c["z_score"], 1)] for c in result["feature_contributions"][:10]],
                    "score": round(result["score"], 4), "threshold": round(result["threshold"], 4),
                })
        print(f"\n### {scenario} (champion v{state.model_version})")
        summary = {}
        for lo, hi in PHASES:
            if lo > days:
                break
            phase = [r for r in rows if lo <= r["day"] <= hi]
            bad = [r for r in phase if r["pred"] == "bad"]
            verdicts = Counter(r["fault"]["verdict_ko"] for r in bad)
            situations = Counter(
                r["fault"]["situation"] for r in bad if r["fault"]["verdict"] == "confirmed"
            )
            summary[f"day{lo:02d}-{hi:02d}"] = {
                "n": len(phase), "bad": len(bad), "verdicts": dict(verdicts),
                "confirmed_situations": dict(situations),
            }
            print(f"  Day {lo:>2}-{hi:<2} 불량 {len(bad):>2}/{len(phase)}  판정 {dict(verdicts)}  "
                  f"확정 상황 {dict(situations)}")
        out[scenario] = {"summary": summary, "rows": rows}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="플레이북 서명 대조 오프라인 채점")
    parser.add_argument("--scenarios", nargs="*", default=["temperature", "tool_wear", "fixture_loosening"])
    parser.add_argument("--days", type=int, default=40)
    args = parser.parse_args()

    corpus = json.loads(CORPUS_PATH.read_text())
    print("## 기록된 4건")
    recorded = score_recorded(corpus)
    print("\n## 타임라인")
    timeline = score_timeline(corpus, load_champion(), args.scenarios, args.days)

    existing = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    merged = {**existing.get("timeline", {}), **timeline}
    OUT_PATH.write_text(json.dumps({"recorded": recorded, "timeline": merged}, ensure_ascii=False, indent=1))
    print(f"\n저장: {OUT_PATH} (시나리오 {sorted(merged)})")


if __name__ == "__main__":
    main()
