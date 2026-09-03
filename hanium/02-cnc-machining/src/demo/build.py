"""데모 페이지 재료 조립 — 순수 함수. 파일 경로는 demo/build_demo.py가 다룬다."""
import json
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")
DAY_LINE = re.compile(r"^Day (\d+)\s+score/threshold=")
TRIGGER = re.compile(r"^\[Day (\d+)\] 트리거 발동")
GATE = re.compile(r"^게이트: (.+)$")
REJECTED = re.compile(r"^거부 — (.+?)\s*\(champion 유지")
SHADOW_START = re.compile(r"^섀도우 시작 — (.+)$")
SHADOW_END = re.compile(r"^섀도우 종료 — (.+)$")
PROMOTED = re.compile(r"^승격 완료 — (.+)$")
CAUSE = re.compile(r"^추정 원인: (\S+) / 권장 조치: (.+)$")

CATEGORY_KO = {
    "tool_wear": "스핀들 부하 상승", "feed_overload": "이송축 부하 상승",
    "vibration_backlash": "축 위치·속도 편차", "general": "고장이 아닌 변화",
}
SCENARIO_LABELS = {
    "temperature": "온도 드리프트 — 제품은 정상, 재보정이 정답",
    "tool_wear": "공구 마모 — Day 21부터 실제 불량, 재학습 거부가 정답",
    "fixture_loosening": "고정구 풀림 — Day 21부터 실제 불량, 재학습 거부가 정답",
}
FAULT_FROM_DAY = {"temperature": None, "tool_wear": 21, "fixture_loosening": 21}


def parse_worker_log(text: str) -> list[dict]:
    """drift_worker 로그에서 트리거·게이트·거부·섀도우·승격 이벤트를 뽑는다.
    각 이벤트의 day는 그 줄 뒤에 처음 나오는 'Day NN  score/threshold=' 요약 줄의 날짜다
    (워커는 하루 처리를 끝낸 뒤 요약 줄을 찍는다)."""
    lines = [ANSI.sub("", line).strip() for line in text.splitlines()]
    day_after = [None] * len(lines)
    current = None
    for i in range(len(lines) - 1, -1, -1):
        match = DAY_LINE.match(lines[i])
        if match:
            current = int(match.group(1))
        day_after[i] = current

    events: list[dict] = []
    pending_gate = None
    last_trigger_day = None
    for i, line in enumerate(lines):
        day = day_after[i] if day_after[i] is not None else last_trigger_day
        if match := TRIGGER.match(line):
            last_trigger_day = int(match.group(1))
            events.append({"day": last_trigger_day, "kind": "trigger", "text": line})
            pending_gate = None
        elif match := GATE.match(line):
            pending_gate = match.group(1)
        elif match := REJECTED.match(line):
            events.append({"day": day, "kind": "rejected", "text": line, "reason": match.group(1), "gate": pending_gate})
        elif SHADOW_START.match(line):
            events.append({"day": day, "kind": "shadow_started", "text": line, "gate": pending_gate})
        elif SHADOW_END.match(line):
            events.append({"day": day, "kind": "shadow_ended", "text": line})
        elif PROMOTED.match(line):
            events.append({"day": day, "kind": "promoted", "text": line})
        elif (match := CAUSE.match(line)) and events:
            events[-1]["cause"] = match.group(1)
            events[-1]["actions"] = match.group(2)
    return events


def _batch(row: dict) -> dict:
    fault = row["fault"]
    return {
        "index": row["index"], "ratio": round(row["ratio"], 3), "pred": row["pred"],
        "verdict": fault["verdict"], "verdict_ko": fault["verdict_ko"],
        "situation": fault["situation"], "coverage": fault["coverage"], "top": row["top"],
    }


def assemble(examples: list[dict], corpus: list[dict], eval_data: dict, worker_logs: dict[str, str],
             versions: dict, generated_at: str) -> dict:
    playbook = [
        {"name": c["name"], "heading": c["heading"], "category": c["fault_category"],
         "category_ko": CATEGORY_KO[c["fault_category"]], "signature": c["signature"], "text": c["text"]}
        for c in corpus if c.get("source") == "playbook"
    ]
    scenarios = {}
    for name, block in eval_data["timeline"].items():
        by_day: dict[int, list[dict]] = {}
        for row in block["rows"]:
            by_day.setdefault(row["day"], []).append(row)
        days = []
        for day in sorted(by_day):
            rows = sorted(by_day[day], key=lambda r: r["index"])
            days.append({
                "day": day, "truth": rows[0]["truth"],
                "ratio_mean": round(sum(r["ratio"] for r in rows) / len(rows), 3),
                "batches": [_batch(r) for r in rows],
            })
        scenarios[name] = {
            "label": SCENARIO_LABELS.get(name, name), "fault_from_day": FAULT_FROM_DAY.get(name),
            "days_total": len(days), "days": days,
            "events": parse_worker_log(worker_logs[name]) if worker_logs.get(name) else [],
        }
    return {"generated_at": generated_at, "versions": versions, "playbook": playbook,
            "examples": examples, "scenarios": scenarios}


def render_html(template: str, data: dict) -> str:
    """JSON을 <script type="application/json"> 안에 넣는다. '</'는 '<\\/'로 바꿔 스크립트 태그가
    일찍 닫히지 않게 한다(JSON에서 유효한 이스케이프)."""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__DEMO_DATA__", payload)
