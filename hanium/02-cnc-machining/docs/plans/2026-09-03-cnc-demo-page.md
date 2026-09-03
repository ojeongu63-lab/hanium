# 데모 화면 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pull만 받아도 열리는 단일 HTML 데모 페이지(배치 진단 흐름 + 시나리오 타임라인 + 루프 이벤트)를 만들고, 서버로 열면 같은 카드가 실제 `/predict`로 다시 계산되게 한다.

**Architecture:** 순수 함수(`src/demo/build.py`: 로그 파서·데이터 조립·템플릿 주입)와 실행 스크립트(`demo/build_demo.py`)가 `demo/template.html`의 `__DEMO_DATA__` 자리에 JSON을 넣어 `demo/index.html`을 만든다(생성물 커밋). 페이지는 외부 라이브러리 없이 인라인 SVG로 그리고, 같은 origin의 `/health`가 응답할 때만 실시간 UI를 켠다. 서버에는 `GET /demo`, `GET /demo/inputs/{key}` 두 라우트만 더한다.

**Tech Stack:** Python 3.14, FastAPI(`FileResponse`), pytest, 순수 HTML/CSS/JS(ES2020, fetch, AbortController), node 24(문법 검사만).

**Spec:** `docs/specs/2026-09-03-cnc-demo-page-design.md`

## Global Constraints

- 작업 디렉터리 `02-cnc-machining/`. 테스트 `uv run pytest -q`(기준선 200개 통과).
- 판정·가이드·루프 코드(`src/rag/*`, `src/serving/inference.py`, `src/retraining/*`, `monitoring/*`)는 변경 금지. `rag/eval_playbook.py`는 행 필드 2개 추가와 시나리오별 병합만.
- `demo/index.html`은 생성물이지만 커밋한다. 재료가 바뀌면 다시 생성해 함께 커밋한다.
- 외부 CDN·라이브러리 금지(미팅 장소에 인터넷이 없을 수 있다).
- 무거운 실행(채점 1,050건)은 `who`/`uptime` 확인 후 `nice -n 19`, 백그라운드.
- 커밋은 main에 직접, 메시지 끝에 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`와 `Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz`.
- 패키지 `src/demo/`는 편집 설치된 `src/`에 자동 포함된다(`uv sync` 불필요). 최상위 `demo/`(스크립트)와 이름이 같지만 `rag/`·`src/rag/`와 같은 기존 관례다.

---

## 파일 구조

| 파일 | 책임 | 변경 |
|---|---|---|
| `rag/eval_playbook.py` | 행에 `ratio`, `top` 추가, 시나리오별 결과 병합 | 수정 |
| `src/demo/__init__.py`, `src/demo/build.py` | `parse_worker_log`, `assemble`, `render_html` | 신규 |
| `tests/demo/test_build_demo.py` | 위 세 함수 | 신규 |
| `demo/template.html` | 화면 | 신규 |
| `demo/build_demo.py` | 재료 경로, 생성 실행 | 신규 |
| `demo/index.html` | 생성물 | 신규(커밋) |
| `src/serving/app.py` | `/demo`, `/demo/inputs/{key}` | 수정 |
| `tests/serving/test_app.py` | 라우트 테스트 | 수정 |
| `README.md` §2-9, `docs/STRUCTURE.md` | 여는 법, 개인 PC 준비, 폴더 표 | 수정 |
| `docs/specs/2026-09-03-cnc-demo-page-design.md` | 실행 결과 정정 절 | 수정 |

---

### Task 1: 채점 기록에 배율·상위 센서 추가, 시나리오별 병합, 재실행

**Files:**
- Modify: `rag/eval_playbook.py`

**Interfaces:**
- Produces: `data/rag/eval_playbook.json`의 `timeline[<scenario>].rows[*]`에 `ratio: float`(score/threshold, 소수 4자리)와 `top: [[feature, z] ×3]` 추가. 여러 번 실행해도 시나리오별로 덮어써 합쳐진다.

- [ ] **Step 1: 행 필드와 병합 구현**

`score_timeline`의 `rows.append({...})`를 다음으로 바꾼다.

```python
                rows.append({
                    "day": day, "index": index, "truth": true_label(scenario, day),
                    "pred": result["predicted_label_text"], "fault": result["fault"],
                    "ratio": round(result["score"] / result["threshold"], 4),
                    "top": [[c["feature"], round(c["z_score"], 1)] for c in result["feature_contributions"][:3]],
                })
```

`main`의 저장 부분을 다음으로 바꾼다(다른 시나리오 결과는 유지).

```python
    existing = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    merged = {**existing.get("timeline", {}), **timeline}
    OUT_PATH.write_text(json.dumps({"recorded": recorded, "timeline": merged}, ensure_ascii=False, indent=1))
    print(f"\n저장: {OUT_PATH} (시나리오 {sorted(merged)})")
```

모듈 docstring 끝에 한 줄 추가: `여러 번 실행하면 --scenarios 로 지정한 시나리오만 갱신되고 나머지는 유지된다.`

- [ ] **Step 2: 실행 (백그라운드, 약 10분)**

Run:
```bash
uptime
nice -n 19 uv run python rag/eval_playbook.py --scenarios temperature --days 70 > /home/sure/.claude/jobs/6b3a648e/tmp/eval_temp70.log 2>&1 && \
nice -n 19 uv run python rag/eval_playbook.py --scenarios tool_wear fixture_loosening --days 40 > /home/sure/.claude/jobs/6b3a648e/tmp/eval_faults40.log 2>&1
```
Expected: 두 로그 끝에 `저장: ... (시나리오 ['fixture_loosening', 'temperature', 'tool_wear'])`. 확인:
`uv run python -c "import json; d=json.load(open('data/rag/eval_playbook.json')); print({k: (len(v['rows']), 'ratio' in v['rows'][0]) for k, v in d['timeline'].items()})"` → `{'temperature': (350, True), 'tool_wear': (200, True), 'fixture_loosening': (200, True)}`.

- [ ] **Step 3: 커밋** (Task 2·3을 병행하다 실행이 끝난 뒤)

```bash
git add rag/eval_playbook.py
git commit -m "feat(rag): record ratio and top sensors per batch; merge scoring runs per scenario

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 2: 조립 모듈 `src/demo/build.py`

**Files:**
- Create: `src/demo/__init__.py` (빈 파일), `src/demo/build.py`
- Test: `tests/demo/test_build_demo.py`

**Interfaces:**
- Produces: `parse_worker_log(text: str) -> list[dict]` — `{day, kind, text, gate?, reason?, cause?, actions?}`, kind ∈ trigger/rejected/shadow_started/shadow_ended/promoted. `assemble(examples, corpus, eval_data, worker_logs, versions, generated_at) -> dict`(스펙의 내장 JSON). `render_html(template: str, data: dict) -> str`.

- [ ] **Step 1: 테스트 작성**

`tests/demo/test_build_demo.py`:

```python
import json

from demo.build import assemble, parse_worker_log, render_html

LOG = """감시 시작 — http://127.0.0.1:8000 폴링
Day 18  score/threshold=0.80  flagged=True  action=none
  [Day 19] 트리거 발동 — 재학습 시작
\x1b[32mEpoch 1/50 - loss: 0.8\x1b[0m
  게이트: G1 놓침=0건 (champion 1건, 허용 2건) / G2 정상 20건 — 오탐 후보 2 vs champion 2 · 불량 0건
  거부 — G2 개선 없음: 오탐·놓침 모두 champion과 동일  (champion 유지, 사람 확인 필요)
  추정 원인: tool_wear / 권장 조치: ['절삭 속도(vc)를 낮추십시오.', '이송량(fz)을 늘리십시오.']
Day 19  score/threshold=0.91  flagged=True  action=rejected
  [Day 37] 트리거 발동 — 재학습 시작
  게이트: G1 놓침=0건 (champion 1건, 허용 2건) / G2 정상 20건 — 오탐 후보 11 vs champion 18 · 불량 0건
  섀도우 시작 — version 63 (라벨 20건 도착까지 관찰, 관찰 기준일 Day 53)
Day 37  score/threshold=1.60  flagged=True  action=shadow_started
  섀도우 종료 — 정상 20건 — 오탐 후보 19 vs champion 20 · 불량 0건 → promoted
  승격 완료 — version 63
Day 64  score/threshold=1.10  flagged=False  action=promoted
"""


def test_parse_worker_log_extracts_events_with_days_gate_reason_and_cause():
    events = parse_worker_log(LOG)

    assert [(e["day"], e["kind"]) for e in events] == [
        (19, "trigger"), (19, "rejected"), (37, "trigger"), (37, "shadow_started"),
        (64, "shadow_ended"), (64, "promoted"),
    ]
    rejected = events[1]
    assert rejected["reason"] == "G2 개선 없음: 오탐·놓침 모두 champion과 동일"
    assert rejected["gate"].startswith("G1 놓침=0건")
    assert rejected["cause"] == "tool_wear"
    assert "절삭 속도" in rejected["actions"]
    assert events[3]["gate"].startswith("G1 놓침=0건")
    assert "Epoch" not in json.dumps(events, ensure_ascii=False)
    assert "\x1b" not in json.dumps(events, ensure_ascii=False)


def _row(day, index, pred, verdict, situation, ratio):
    return {
        "day": day, "index": index, "truth": "good" if day < 21 else "bad", "pred": pred, "ratio": ratio,
        "top": [["S_OutputCurrent", 30.0], ["S_OutputPower", 5.0], ["S_CurrentFeedback", 4.0]],
        "fault": {
            "verdict": verdict, "verdict_ko": {"confirmed": "확정", "none": "이상 없음"}[verdict],
            "situation": situation, "category": "tool_wear" if situation else None,
            "coverage": 0.8 if situation else 0.0, "matched_features": [], "alternatives": [],
            "other_group": None, "top_z": 30.0,
        },
    }


def test_assemble_groups_days_and_builds_playbook_and_scenarios():
    corpus = [
        {"name": "공구 마모", "heading": "공구 마모 — x", "fault_category": "tool_wear",
         "signature": ["S_OutputCurrent"], "text": "관련 센서: S_OutputCurrent", "source": "playbook"},
        {"heading": "플랭크 마모", "fault_category": "tool_wear", "text": "s"},
    ]
    eval_data = {"timeline": {"tool_wear": {"rows": [
        _row(21, 1, "bad", "confirmed", "공구 마모", 1.4),
        _row(21, 0, "good", "none", None, 0.6),
        _row(22, 0, "bad", "confirmed", "공구 마모", 2.0),
    ]}}}
    examples = [{"key": "tool_wear", "label": "합성", "response": {"predicted_label_text": "bad"}}]

    data = assemble(examples, corpus, eval_data, {"tool_wear": LOG}, {"playbook": "abcd1234"}, "2026-09-03T15:00+09:00")

    assert [p["name"] for p in data["playbook"]] == ["공구 마모"]
    assert data["playbook"][0]["category_ko"] == "스핀들 부하 상승"
    sc = data["scenarios"]["tool_wear"]
    assert sc["fault_from_day"] == 21 and sc["days_total"] == 2
    assert [b["index"] for b in sc["days"][0]["batches"]] == [0, 1]
    assert sc["days"][0]["ratio_mean"] == 1.0 and sc["days"][0]["truth"] == "bad"
    assert sc["days"][0]["batches"][1]["situation"] == "공구 마모"
    assert sc["events"][0]["kind"] == "trigger"
    assert data["examples"][0]["key"] == "tool_wear"
    assert data["versions"]["playbook"] == "abcd1234"
    assert data["generated_at"] == "2026-09-03T15:00+09:00"


def test_assemble_without_log_gives_empty_events():
    eval_data = {"timeline": {"temperature": {"rows": [_row(1, 0, "good", "none", None, 0.7)]}}}
    data = assemble([], [], eval_data, {}, {}, "t")
    assert data["scenarios"]["temperature"]["events"] == []
    assert data["scenarios"]["temperature"]["fault_from_day"] is None


def test_render_html_injects_json_and_escapes_script_close():
    template = '<script id="demo-data" type="application/json">__DEMO_DATA__</script>'
    data = {"note": "</script><b>x"}

    html = render_html(template, data)

    assert "__DEMO_DATA__" not in html
    start = html.index('application/json">') + len('application/json">')
    payload = html[start: html.index("</script>", start)]
    assert json.loads(payload) == data
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/demo -q`
Expected: `ModuleNotFoundError: No module named 'demo'`

- [ ] **Step 3: 구현**

`src/demo/__init__.py`는 빈 파일. `src/demo/build.py`:

```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/demo -q`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/demo/__init__.py src/demo/build.py tests/demo/test_build_demo.py
git commit -m "feat(demo): worker-log parser and data assembly for the demo page

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 3: 화면 템플릿과 생성 스크립트

**Files:**
- Create: `demo/template.html`, `demo/build_demo.py`
- Create(생성물): `demo/index.html`

**Interfaces:**
- Consumes: Task 2의 `assemble`, `render_html`; Task 1의 `data/rag/eval_playbook.json`; `data/rag/corpus.json`, `corpus_meta.json`; `docs/examples/*.json`; 워커 로그 3개.
- Produces: `demo/index.html`. 페이지는 서버의 `/health`, `/predict`, `/demo/inputs/{key}`(Task 4)를 부른다.

- [ ] **Step 1: 템플릿 작성**

`demo/template.html` 전체(아래 그대로). 스크립트 안의 계산식은 `src/rag/playbook.py`와 같다(TOP_N 5, 1/순위 가중, WEAK_Z 10, COMPOSITE_RATIO 0.5).

````html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CNC 이상탐지 · 조치 가이드 데모</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --line:#dfe3e8; --text:#1f2430; --muted:#6b7280; --ok:#2e7d32; --bad:#c62828;
          --confirmed:#c62828; --composite:#ef6c00; --weak:#f9a825; --unknown:#757575; --none:#2e7d32; --accent:#1e56a0; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,"Segoe UI","Malgun Gothic","Apple SD Gothic Neo",sans-serif; background:var(--bg); color:var(--text); }
  header { display:flex; align-items:center; gap:16px; padding:14px 24px; background:#fff; border-bottom:1px solid var(--line); }
  header h1 { font-size:18px; margin:0; }
  .badge { display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; background:#e8eaf0; color:#333; }
  .badge.live { background:#e3f2e6; color:var(--ok); }
  #hint { color:var(--muted); font-size:12px; }
  nav.tabs { display:flex; gap:4px; padding:10px 24px 0; }
  nav.tabs button { border:1px solid var(--line); border-bottom:none; background:#eef0f4; padding:8px 16px; cursor:pointer; border-radius:8px 8px 0 0; font-size:14px; }
  nav.tabs button.active { background:#fff; font-weight:700; }
  main { padding:16px 24px 40px; }
  #tab-batch { display:grid; grid-template-columns:260px 1fr; gap:16px; }
  aside, .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  aside h2, .card h3 { margin:0 0 10px; font-size:14px; color:var(--accent); }
  #examples { list-style:none; margin:0; padding:0; }
  #examples li { padding:8px 10px; border-radius:6px; cursor:pointer; margin-bottom:4px; border:1px solid transparent; }
  #examples li:hover { background:#f0f3f8; }
  #examples li.active { background:#e8effa; border-color:#b9c9e6; font-weight:600; }
  #examples small { display:block; color:var(--muted); font-weight:400; }
  #live-controls { margin-top:14px; padding-top:12px; border-top:1px dashed var(--line); }
  button.primary { background:var(--accent); color:#fff; border:none; padding:8px 12px; border-radius:6px; cursor:pointer; width:100%; font-size:13px; }
  button.primary:disabled { opacity:.5; cursor:default; }
  #upload { margin-top:8px; width:100%; font-size:12px; }
  #live-msg { font-size:12px; color:var(--muted); margin-top:8px; white-space:pre-wrap; }
  #cards { display:flex; flex-direction:column; gap:12px; }
  .meta { color:var(--muted); font-size:12px; font-weight:400; }
  .pill { display:inline-block; padding:2px 10px; border-radius:10px; color:#fff; font-size:12px; font-weight:700; }
  .pill.good { background:var(--ok); } .pill.bad { background:var(--bad); }
  .pill.confirmed { background:var(--confirmed); } .pill.composite { background:var(--composite); }
  .pill.weak { background:var(--weak); color:#333; } .pill.unknown { background:var(--unknown); } .pill.none { background:var(--none); }
  .bar { position:relative; height:14px; background:#eef0f4; border-radius:4px; overflow:visible; }
  .bar > span { position:absolute; left:0; top:0; bottom:0; background:var(--accent); border-radius:4px; }
  .bar.ratio > span { background:var(--bad); }
  .bar .th { position:absolute; top:-3px; bottom:-3px; width:2px; background:#333; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:middle; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  td.num { font-variant-numeric:tabular-nums; text-align:right; }
  tr.selected td { background:#fdecea; font-weight:600; }
  tr.alt td { background:#fff8e1; }
  tr.other td { background:#eef4fb; }
  tr.group td { background:#f3f4f6; color:var(--muted); font-size:12px; }
  .contrib .row { display:grid; grid-template-columns:170px 1fr 70px; gap:8px; align-items:center; font-size:13px; padding:2px 0; }
  .contrib .row.top5 { font-weight:600; }
  .reason { font-size:13px; line-height:1.6; }
  ul.docs, ul.actions { margin:6px 0; padding-left:20px; font-size:13px; line-height:1.6; }
  .kv { display:grid; grid-template-columns:120px 1fr; gap:4px 12px; font-size:13px; }
  .scenario-buttons { display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
  .scenario-buttons button { padding:8px 14px; border:1px solid var(--line); background:#fff; border-radius:6px; cursor:pointer; }
  .scenario-buttons button.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  #scenario-label { margin:4px 0 10px; color:var(--muted); font-size:13px; }
  #chart svg, #band svg { width:100%; height:auto; display:block; background:#fff; border:1px solid var(--line); border-radius:10px; }
  #band { margin-top:8px; }
  .legend { display:flex; gap:14px; font-size:12px; color:var(--muted); margin:8px 0; flex-wrap:wrap; }
  .legend span[style]::before { content:""; display:inline-block; width:10px; height:10px; margin-right:4px; border-radius:2px; background:var(--c); }
  #day-detail { margin-top:12px; }
  .event { border-left:4px solid var(--accent); padding:6px 10px; margin:6px 0; background:#f7f9fc; font-size:13px; }
  .event.rejected { border-color:var(--bad); } .event.promoted { border-color:var(--ok); }
  .event.shadow_started, .event.shadow_ended { border-color:#8e24aa; }
  .note { color:var(--muted); font-size:12px; }
  .error { color:var(--bad); font-size:13px; }
  @media (max-width:900px) { #tab-batch { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>CNC 이상탐지 · 조치 가이드 데모</h1>
  <span id="status" class="badge">기록 모드</span>
  <span id="hint"></span>
  <span id="generated" class="note" style="margin-left:auto"></span>
</header>
<nav class="tabs">
  <button data-tab="batch" class="active">배치 진단 흐름</button>
  <button data-tab="timeline">시나리오 타임라인</button>
</nav>
<main>
  <section id="tab-batch">
    <aside>
      <h2>입력 배치</h2>
      <ul id="examples"></ul>
      <div id="live-controls" hidden>
        <button id="recompute" class="primary">실시간으로 다시 계산</button>
        <input type="file" id="upload" accept=".csv">
        <div id="live-msg"></div>
      </div>
      <p class="note" id="mode-note"></p>
    </aside>
    <div id="cards"></div>
  </section>
  <section id="tab-timeline" hidden>
    <div class="scenario-buttons" id="scenario-buttons"></div>
    <p id="scenario-label"></p>
    <div class="legend">
      <span style="--c:#1e56a0">일 평균 배율</span><span style="--c:#c62828">확정</span><span style="--c:#ef6c00">복합 징후</span>
      <span style="--c:#f9a825">약한 신호</span><span style="--c:#757575">판단 불가</span><span style="--c:#9ccc9c">정상 판정 배치</span><span style="--c:#fdecea">실제 불량 구간</span>
      <span>▲ 트리거 · ✕ 거부 · ◆ 섀도우 시작 · ◇ 섀도우 종료 · ★ 승격</span>
    </div>
    <div id="chart"></div>
    <div id="band"></div>
    <div id="day-detail"><p class="note">그래프나 띠에서 날짜를 클릭하면 그날 배치 5개와 이벤트가 나옵니다.</p></div>
    <p class="note">판정은 champion v1 기준 오프라인 계산(rag/eval_playbook.py), 이벤트는 2026-09-02 라이브 워커 로그. 온도 시나리오는 Day 64 승격 뒤 실제 운영에서는 새 champion이 판정한다.</p>
  </section>
</main>
<script id="demo-data" type="application/json">__DEMO_DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById('demo-data').textContent);
  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (x, d = 2) => (x === null || x === undefined) ? '-' : Number(x).toFixed(d);
  const TOP_N = 5, WEAK_Z = 10, COMPOSITE_RATIO = 0.5;
  const state = { exampleKey: null, live: false, scenario: null, day: null };

  document.querySelectorAll('nav.tabs button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('nav.tabs button').forEach((x) => x.classList.toggle('active', x === b));
    $('#tab-batch').hidden = b.dataset.tab !== 'batch';
    $('#tab-timeline').hidden = b.dataset.tab !== 'timeline';
  }));
  $('#generated').textContent = `생성 ${DATA.generated_at} · 플레이북 ${(DATA.versions && DATA.versions.playbook) || '-'}`;

  // ---------- 탭 1: 예시 목록 ----------
  function renderExamples() {
    const ul = $('#examples'); ul.innerHTML = '';
    DATA.examples.forEach((ex) => {
      const li = document.createElement('li');
      const f = ex.response.fault;
      li.innerHTML = `${esc(ex.label)}<small>${esc(ex.key)} · ${ex.response.predicted_label_text === 'bad' ? '불량' : '정상'} · ${esc(f ? f.verdict_ko : '-')}</small>`;
      li.addEventListener('click', () => selectExample(ex.key));
      ul.appendChild(li);
    });
  }
  function selectExample(key) {
    state.exampleKey = key;
    document.querySelectorAll('#examples li').forEach((li, i) => li.classList.toggle('active', DATA.examples[i].key === key));
    const ex = DATA.examples.find((e) => e.key === key);
    showResponse(ex.response, {source: '기록', label: ex.label});
  }

  // ---------- 탭 1: 카드 ----------
  function coverage(signature, contribs) {
    const top = contribs.slice(0, TOP_N); if (!top.length) return 0;
    let total = 0, got = 0;
    top.forEach((c, i) => { const w = 1 / (i + 1); total += w; if (signature.includes(c.feature)) got += w; });
    return Math.round((got / total) * 100) / 100;
  }
  function chunkNames(fault) {
    const general = DATA.playbook.filter((p) => p.category === 'general').map((p) => `${p.name} (플레이북)`);
    const safety = ['OSHA 안전 수칙 3개'];
    if (!fault || fault.verdict === 'none') return [];
    if (fault.verdict === 'unknown') return [...general, ...safety];
    const sel = `${fault.situation} (플레이북, 선택)`;
    if (fault.verdict === 'weak') return [sel, ...general, ...safety];
    if (fault.verdict === 'composite') {
      const other = fault.other_group ? [`${fault.other_group.situation} (플레이북, 다른 구역 1위)`] : [];
      return [sel, ...other, ...general, ...safety];
    }
    return [sel, ...(fault.alternatives || []).map((a) => `${a} (플레이북, 같은 구역 후보)`), '같은 카테고리 Sandvik 문서 2개 (임베딩 유사도 순)', ...safety];
  }
  function reasonText(fault) {
    if (!fault) return '코퍼스가 없어 fault를 계산하지 못했습니다.';
    const v = fault.verdict;
    if (v === 'none') return '정상 판정이라 대조하지 않습니다.';
    if (v === 'unknown') return `상위 ${TOP_N}개 센서를 설명하는 항목이 없습니다. 상위 센서 z ${fmt(fault.top_z, 1)}.`;
    const og = fault.other_group;
    const ogText = og ? `다른 구역 최고는 ${og.situation} ${fmt(og.coverage)}` : '다른 구역은 0';
    const half = fmt(fault.coverage * COMPOSITE_RATIO);
    if (v === 'weak') return `상위 센서 z ${fmt(fault.top_z, 1)}이 기준 ${WEAK_Z} 미만이라 판단을 보류합니다. 참고 상황 ${fault.situation} (일치 ${fmt(fault.coverage)}).`;
    if (v === 'composite') return `${fault.situation} 일치 ${fmt(fault.coverage)}인데 ${ogText}로 절반(${half}) 이상 → 두 구역이 함께 움직임. 드리프트 가능성, 라벨·추이 확인.`;
    return `일치 센서 ${(fault.matched_features || []).join(', ')} → ${fault.situation} 일치 ${fmt(fault.coverage)}. ${ogText}로 절반(${half}) 미만, 상위 센서 z ${fmt(fault.top_z, 1)} ≥ ${WEAK_Z}.`;
  }
  function showResponse(resp, meta) {
    const fault = resp.fault, guide = resp.guide, contribs = resp.feature_contributions || [];
    const ratio = resp.score / resp.threshold, scale = Math.max(4, ratio);
    const isBad = resp.predicted_label_text === 'bad';
    const cards = [];
    cards.push(`<div class="card"><h3>1. 판정 <span class="meta">${esc(meta.source)}${meta.elapsed ? ` · ${meta.elapsed}s` : ''} · ${esc(meta.label || '')}</span></h3>
      <p><span class="pill ${isBad ? 'bad' : 'good'}">${isBad ? '불량' : '정상'}</span>
      &nbsp; 이상 점수 <b>${fmt(resp.score, 3)}</b> · 판정 기준 <b>${fmt(resp.threshold, 4)}</b> · 기준 대비 배율 <b>${fmt(ratio)}</b> · method ${esc(resp.method)}</p>
      <div class="bar ratio"><span style="width:${Math.min(100, ratio / scale * 100)}%"></span><i class="th" style="left:${(1 / scale) * 100}%"></i></div>
      <p class="meta">검은 선이 판정 기준(배율 1.0) · 모델 v${esc(resp.model_version)}</p></div>`);
    const top10 = contribs.slice(0, 10);
    const maxZ = Math.max(1e-9, ...top10.map((c) => c.z_score));
    cards.push(`<div class="card contrib"><h3>2. 센서 기여도 <span class="meta">정상 대비 z-score 상위 10개 · 굵은 5개가 대조 범위</span></h3>
      ${top10.map((c, i) => `<div class="row ${i < TOP_N ? 'top5' : ''}"><span>${i + 1}. ${esc(c.feature)}</span><div class="bar"><span style="width:${Math.max(1, c.z_score / maxZ * 100)}%"></span></div><span class="num">${fmt(c.z_score, 1)}</span></div>`).join('')}
      ${top10.length ? '' : '<p class="meta">기여도 없음</p>'}</div>`);
    if (fault && fault.verdict !== 'none') {
      let rows = '', lastCat = null;
      DATA.playbook.forEach((p) => {
        if (p.category_ko !== lastCat) { rows += `<tr class="group"><td colspan="4">${esc(p.category_ko)}</td></tr>`; lastCat = p.category_ko; }
        const cov = coverage(p.signature, contribs);
        const isSel = p.name === fault.situation, isAlt = (fault.alternatives || []).includes(p.name), isOther = !!(fault.other_group && p.name === fault.other_group.situation);
        const cls = isSel ? 'selected' : isAlt ? 'alt' : isOther ? 'other' : '';
        const tag = isSel ? ' ← 선택' : isAlt ? ' (같은 구역 후보)' : isOther ? ' (다른 구역 1위)' : '';
        rows += `<tr class="${cls}"><td>${esc(p.name)}${tag}</td><td class="meta">${p.signature.length ? esc(p.signature.join(', ')) : '없음'}</td><td><div class="bar"><span style="width:${cov * 100}%"></span></div></td><td class="num">${fmt(cov)}</td></tr>`;
      });
      cards.push(`<div class="card"><h3>3. 플레이북 대조 <span class="meta">항목의 관련 센서가 상위 ${TOP_N}개 센서(1/순위 가중)를 설명하는 비율</span></h3>
        <table><thead><tr><th>상황</th><th>관련 센서</th><th style="width:30%">일치도</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`);
    } else {
      cards.push(`<div class="card"><h3>3. 플레이북 대조</h3><p class="meta">${fault ? '정상 판정이라 대조하지 않습니다.' : '코퍼스 미로드 — fault 없음'}</p></div>`);
    }
    cards.push(`<div class="card"><h3>4. 판정</h3>
      <p>${fault ? `<span class="pill ${fault.verdict}">${esc(fault.verdict_ko)}</span> &nbsp; ${fault.situation ? `<b>${esc(fault.situation)}</b> (${esc(fault.category || '')})` : ''}` : '<span class="pill unknown">fault 없음</span>'}</p>
      <p class="reason">${esc(reasonText(fault))}</p></div>`);
    const docs = chunkNames(fault);
    cards.push(`<div class="card"><h3>5. LLM에 준 참고 문서 <span class="meta">판정별 선택 규칙을 재현한 목록</span></h3>
      ${docs.length ? `<ul class="docs">${docs.map((d) => `<li>${esc(d)}</li>`).join('')}</ul>` : '<p class="meta">정상 판정 — 문서를 쓰지 않습니다.</p>'}</div>`);
    if (guide) {
      const list = (arr) => `<ul class="actions">${(arr || []).map((a) => `<li>${esc(a)}</li>`).join('') || '<li>-</li>'}</ul>`;
      cards.push(`<div class="card"><h3>6. 조치 가이드 <span class="meta">LLM 작성 · 문장은 호출마다 달라질 수 있음</span></h3>
        <div class="kv"><span>원인 추정</span><span>${esc(guide.cause_estimate)}</span>
        <span>확신도</span><span>${esc(guide.confidence_note || '-')}</span>
        <span>권장 조치</span><span>${list(guide.recommended_actions)}</span>
        <span>안전 수칙</span><span>${list(guide.safety_notes)}</span>
        <span>출처</span><span>${(guide.sources || []).map((s) => esc(s.title)).join(' · ') || '-'}</span></div></div>`);
    } else {
      cards.push(`<div class="card"><h3>6. 조치 가이드</h3><p class="meta">guide가 null — OpenAI 키가 없거나 코퍼스 미로드. fault는 위와 같이 계산됩니다.</p></div>`);
    }
    const ver = resp.versions || {};
    cards.push(`<div class="card"><h3>7. versions</h3><div class="kv"><span>playbook</span><span>${esc(ver.playbook || '-')}</span><span>corpus</span><span>${esc(ver.corpus || '-')}</span><span>chat_model</span><span>${esc(ver.chat_model || '-')}</span><span>mlflow_run_id</span><span>${esc(resp.mlflow_run_id || '-')}</span></div></div>`);
    $('#cards').innerHTML = cards.join('');
  }

  // ---------- 실시간 ----------
  async function fetchWithTimeout(url, opts, ms) {
    const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), ms);
    try { return await fetch(url, Object.assign({}, opts, {signal: ctl.signal})); } finally { clearTimeout(t); }
  }
  async function detectLive() {
    if (location.protocol === 'file:') { $('#mode-note').textContent = '파일로 열었습니다. 서버를 띄우고 /demo 로 열면 실시간 재계산이 가능합니다.'; return; }
    try {
      const r = await fetchWithTimeout('/health', {}, 3000);
      if (!r.ok) throw new Error(r.status);
      const h = await r.json();
      state.live = true;
      $('#status').textContent = `실시간 연결됨 · champion v${h.model_version}`; $('#status').classList.add('live');
      $('#live-controls').hidden = false;
      $('#mode-note').textContent = '예시를 고른 뒤 다시 계산하거나 CSV를 올리면 실제 /predict 결과로 카드가 바뀝니다.';
    } catch (e) {
      $('#hint').textContent = '서버 응답 없음 → 기록 모드';
    }
  }
  async function postPredict(blob, filename, label) {
    const fd = new FormData(); fd.append('file', blob, filename);
    const btn = $('#recompute'); btn.disabled = true; $('#live-msg').textContent = `계산 중… (${filename})`;
    const t0 = performance.now();
    try {
      const r = await fetchWithTimeout('/predict', {method: 'POST', body: fd}, 90000);
      const body = await r.json();
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${body.detail || ''}`);
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      showResponse(body, {source: '실시간', label, elapsed});
      $('#live-msg').textContent = `완료 · ${elapsed}s · ${body.fault ? body.fault.verdict_ko : '-'} ${body.fault && body.fault.situation ? body.fault.situation : ''}`;
    } catch (e) {
      $('#live-msg').innerHTML = `<span class="error">실패: ${esc(e.message || e)}</span>`;
    } finally { btn.disabled = false; }
  }
  $('#recompute').addEventListener('click', async () => {
    if (!state.exampleKey) { $('#live-msg').textContent = '먼저 예시를 고르세요.'; return; }
    try {
      const r = await fetchWithTimeout(`/demo/inputs/${state.exampleKey}`, {}, 30000);
      if (!r.ok) { const b = await r.json().catch(() => ({})); throw new Error(`입력 파일 ${r.status}: ${b.detail || ''}`); }
      const ex = DATA.examples.find((e) => e.key === state.exampleKey);
      await postPredict(await r.blob(), `${state.exampleKey}.csv`, ex ? ex.label : state.exampleKey);
    } catch (e) { $('#live-msg').innerHTML = `<span class="error">실패: ${esc(e.message || e)}</span>`; }
  });
  $('#upload').addEventListener('change', (ev) => { const f = ev.target.files[0]; if (f) postPredict(f, f.name, f.name); });

  // ---------- 탭 2: 타임라인 ----------
  const COLORS = {confirmed: '#c62828', composite: '#ef6c00', weak: '#f9a825', unknown: '#757575'};
  const MARK = {trigger: '▲', rejected: '✕', shadow_started: '◆', shadow_ended: '◇', promoted: '★'};
  const MARK_COLOR = {trigger: '#1e56a0', rejected: '#c62828', shadow_started: '#8e24aa', shadow_ended: '#8e24aa', promoted: '#2e7d32'};
  function renderScenarioButtons() {
    const box = $('#scenario-buttons'); box.innerHTML = '';
    Object.entries(DATA.scenarios).forEach(([name, sc]) => {
      const b = document.createElement('button'); b.textContent = sc.label; b.dataset.name = name;
      b.addEventListener('click', () => renderScenario(name)); box.appendChild(b);
    });
  }
  function renderScenario(name) {
    state.scenario = name; state.day = null;
    document.querySelectorAll('#scenario-buttons button').forEach((b) => b.classList.toggle('active', b.dataset.name === name));
    const sc = DATA.scenarios[name];
    const nBad = sc.days.reduce((s, d) => s + d.batches.filter((b) => b.pred === 'bad').length, 0);
    $('#scenario-label').textContent = `${sc.label} · ${sc.days_total}일 × 5배치 · 불량 판정 ${nBad}건 · 이벤트 ${sc.events.length}건${sc.fault_from_day ? ` · Day ${sc.fault_from_day}부터 실제 불량` : ' · 실제 라벨은 전부 정상'}`;
    const W = 960, H = 280, L = 44, R = 12, T = 16, B = 28, days = sc.days, n = days.length, cw = (W - L - R) / n;
    const ymax = Math.max(2, ...days.flatMap((d) => d.batches.map((b) => b.ratio))) * 1.05;
    const x = (day) => L + (day - 0.5) * cw, y = (v) => T + (1 - v / ymax) * (H - T - B);
    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    if (sc.fault_from_day) s += `<rect x="${x(sc.fault_from_day) - cw / 2}" y="${T}" width="${W - R - x(sc.fault_from_day) + cw / 2}" height="${H - T - B}" fill="#fdecea"/>`;
    for (let v = 0; v <= ymax; v += 0.5) s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#eee"/><text x="${L - 6}" y="${y(v) + 4}" font-size="10" text-anchor="end" fill="#888">${v.toFixed(1)}</text>`;
    s += `<line x1="${L}" x2="${W - R}" y1="${y(1)}" y2="${y(1)}" stroke="#333" stroke-dasharray="4 3"/>`;
    days.forEach((d) => { if (d.day % 5 === 0 || d.day === 1) s += `<text x="${x(d.day)}" y="${H - 8}" font-size="10" text-anchor="middle" fill="#888">D${d.day}</text>`; });
    days.forEach((d) => d.batches.forEach((b) => { s += `<circle cx="${x(d.day)}" cy="${y(b.ratio)}" r="2.5" fill="${b.pred === 'bad' ? (COLORS[b.verdict] || '#c62828') : '#9ccc9c'}" opacity="0.85"/>`; }));
    s += `<polyline fill="none" stroke="#1e56a0" stroke-width="2" points="${days.map((d) => `${x(d.day)},${y(d.ratio_mean)}`).join(' ')}"/>`;
    sc.events.forEach((ev) => {
      if (ev.day == null || ev.day > n) return;
      const col = MARK_COLOR[ev.kind] || '#999';
      s += `<line x1="${x(ev.day)}" x2="${x(ev.day)}" y1="${T}" y2="${H - B}" stroke="${col}" stroke-width="1" opacity="0.5"/><text x="${x(ev.day)}" y="${T + 12}" font-size="12" text-anchor="middle" fill="${col}">${MARK[ev.kind] || '•'}</text>`;
    });
    days.forEach((d) => { s += `<rect class="hit" data-day="${d.day}" x="${x(d.day) - cw / 2}" y="${T}" width="${cw}" height="${H - T - B}" fill="transparent" style="cursor:pointer"><title>Day ${d.day} · 평균 배율 ${d.ratio_mean.toFixed(2)}</title></rect>`; });
    s += '</svg>';
    $('#chart').innerHTML = s;
    const BH = 90, bh = (BH - 20) / 5;
    let b = `<svg viewBox="0 0 ${W} ${BH}" xmlns="http://www.w3.org/2000/svg">`;
    days.forEach((d) => {
      let yy = BH - 16;
      ['unknown', 'weak', 'composite', 'confirmed'].forEach((v) => {
        const k = d.batches.filter((bb) => bb.pred === 'bad' && bb.verdict === v).length;
        if (k) { b += `<rect x="${x(d.day) - cw / 2 + 1}" y="${yy - k * bh}" width="${cw - 2}" height="${k * bh}" fill="${COLORS[v]}"/>`; yy -= k * bh; }
      });
      b += `<rect class="hit" data-day="${d.day}" x="${x(d.day) - cw / 2}" y="0" width="${cw}" height="${BH}" fill="transparent" style="cursor:pointer"/>`;
    });
    b += `<text x="${L}" y="${BH - 3}" font-size="10" fill="#888">일별 불량 판정 배치 수(최대 5)와 판정 종류</text></svg>`;
    $('#band').innerHTML = b;
    document.querySelectorAll('#chart rect.hit, #band rect.hit').forEach((r) => r.addEventListener('click', () => renderDay(name, Number(r.dataset.day))));
    $('#day-detail').innerHTML = '<p class="note">그래프나 띠에서 날짜를 클릭하면 그날 배치 5개와 이벤트가 나옵니다.</p>';
  }
  function renderDay(name, day) {
    state.day = day;
    const sc = DATA.scenarios[name], d = sc.days.find((x) => x.day === day), evs = sc.events.filter((e) => e.day === day);
    let h = `<div class="card"><h3>Day ${day} <span class="meta">실제 라벨 ${d.truth === 'bad' ? '불량' : '정상'} · 평균 배율 ${d.ratio_mean.toFixed(2)}</span></h3>
      <table><thead><tr><th>배치</th><th>배율</th><th>판정</th><th>진단</th><th>상황</th><th>일치</th><th>상위 센서</th></tr></thead><tbody>`;
    d.batches.forEach((b) => {
      h += `<tr><td>day${String(day).padStart(2, '0')}_${b.index}</td><td class="num">${b.ratio.toFixed(2)}</td><td><span class="pill ${b.pred === 'bad' ? 'bad' : 'good'}">${b.pred === 'bad' ? '불량' : '정상'}</span></td>
        <td>${b.pred === 'bad' ? `<span class="pill ${b.verdict}">${esc(b.verdict_ko)}</span>` : '-'}</td><td>${esc(b.situation || '-')}</td><td class="num">${b.pred === 'bad' ? b.coverage.toFixed(2) : '-'}</td>
        <td class="meta">${(b.top || []).map(([f, z]) => `${esc(f)} ${Number(z).toFixed(1)}`).join(' · ')}</td></tr>`;
    });
    h += '</tbody></table>';
    if (evs.length) {
      h += '<h3 style="margin-top:14px">이 날의 루프 이벤트</h3>';
      evs.forEach((e) => {
        h += `<div class="event ${e.kind}"><b>${MARK[e.kind] || '•'} ${esc(e.text)}</b>`;
        if (e.gate) h += `<div class="meta">게이트: ${esc(e.gate)}</div>`;
        if (e.reason) h += `<div>사유: ${esc(e.reason)}</div>`;
        if (e.cause) h += `<div>추정 원인: <b>${esc(e.cause)}</b> · 권장 조치: ${esc(e.actions)}</div>`;
        h += '</div>';
      });
    } else { h += '<p class="note">이 날은 루프 이벤트가 없습니다.</p>'; }
    h += '</div>';
    $('#day-detail').innerHTML = h;
  }

  renderExamples();
  if (DATA.examples.length) selectExample(DATA.examples[0].key);
  renderScenarioButtons();
  const firstScenario = Object.keys(DATA.scenarios)[0]; if (firstScenario) renderScenario(firstScenario);
  detectLive();
})();
</script>
</body>
</html>
````

- [ ] **Step 2: 생성 스크립트 작성**

`demo/build_demo.py`:

```python
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
```

- [ ] **Step 3: 생성과 문법 검사**

Task 1의 채점이 끝난 뒤:
```bash
uv run python demo/build_demo.py
python3 - <<'EOF'
import json, re
html = open("demo/index.html").read()
s = html.index('application/json">') + len('application/json">'); payload = html[s: html.index("</script>", s)]
d = json.loads(payload); print("예시", len(d["examples"]), "플레이북", len(d["playbook"]), {k: (v["days_total"], len(v["events"])) for k, v in d["scenarios"].items()})
open("/home/sure/.claude/jobs/6b3a648e/tmp/demo_script.js", "w").write(re.search(r"<script>\n(.*?)\n</script>", html, re.S).group(1))
EOF
node --check /home/sure/.claude/jobs/6b3a648e/tmp/demo_script.js && echo "JS 문법 OK"
```
Expected: `예시 5 플레이북 16 {'temperature': (70, ...), 'tool_wear': (40, ...), 'fixture_loosening': (40, ...)}`, `JS 문법 OK`. 이벤트 수는 로그 기준(tool_wear·fixture 각 트리거 5 + 거부 5 = 10 안팎, temperature 트리거 5 + 거부 3 + 섀도우 시작 2 + 종료 1 + 승격 1).

- [ ] **Step 4: 브라우저로 열어 눈으로 확인**

Run: `explorer.exe "$(wslpath -w demo/index.html)"` (WSL에서 Windows 브라우저가 열린다.) 확인할 것: 상태 배지 "기록 모드", 예시 5개 클릭 시 카드 7개, 타임라인 탭 3개 시나리오 그래프와 띠, 날짜 클릭 시 표와 이벤트. 콘솔 오류 없음(F12).

- [ ] **Step 5: 커밋**

```bash
git add demo/template.html demo/build_demo.py demo/index.html
git commit -m "feat(demo): single-file demo page (batch diagnosis flow, scenario timeline, loop events)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 4: 서버 라우트와 문서

**Files:**
- Modify: `src/serving/app.py` (import, 상수, 라우트 2개)
- Modify: `README.md`(§2-9 신설, §4 폴더 표), `docs/STRUCTURE.md`(폴더 표)
- Test: `tests/serving/test_app.py`

**Interfaces:**
- Produces: `GET /demo` → `demo/index.html`(text/html), 없으면 404. `GET /demo/inputs/{key}` → CSV(text/csv), key ∉ `DEMO_INPUTS` 또는 파일 없음 → 404. 모듈 상수 `DEMO_INDEX: Path`, `DEMO_INPUTS: dict[str, Path]`(테스트가 monkeypatch).

- [ ] **Step 1: 테스트 작성**

`tests/serving/test_app.py` 끝에:

```python
def test_demo_routes_serve_page_and_inputs(tmp_path, monkeypatch):
    import serving.app as app_module

    index = tmp_path / "index.html"
    index.write_text('<html><script id="demo-data" type="application/json">{}</script></html>')
    csv = tmp_path / "tool_wear.csv"
    csv.write_text("a,b\n1,2\n")
    monkeypatch.setattr(app_module, "DEMO_INDEX", index)
    monkeypatch.setattr(app_module, "DEMO_INPUTS", {"tool_wear": csv, "missing": tmp_path / "nope.csv"})
    client = TestClient(app)

    page = client.get("/demo")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "demo-data" in page.text

    got = client.get("/demo/inputs/tool_wear")
    assert got.status_code == 200 and got.text.startswith("a,b")
    assert client.get("/demo/inputs/unknown").status_code == 404
    assert client.get("/demo/inputs/missing").status_code == 404

    monkeypatch.setattr(app_module, "DEMO_INDEX", tmp_path / "absent.html")
    assert client.get("/demo").status_code == 404
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/serving/test_app.py -q -k demo`
Expected: FAIL — `AttributeError: ... has no attribute 'DEMO_INDEX'`

- [ ] **Step 3: 구현**

`src/serving/app.py`: import에 `from fastapi.responses import FileResponse` 추가. `DRIFT_WINDOW_SIZE = 10` 아래에:

```python
DEMO_INDEX = ROOT / "demo" / "index.html"
DATASET_DIR = ROOT / "data" / "dataset" / "CNC 비식별화 원본데이터_1209" / "CNC Virtual Data set _v2"
DEMO_INPUTS = {
    "tool_wear": ROOT / "synthetic" / "scenarios" / "tool_wear.csv",
    "feed_overload": ROOT / "synthetic" / "scenarios" / "feed_overload.csv",
    "vibration_backlash": ROOT / "synthetic" / "scenarios" / "vibration_backlash.csv",
    "experiment_07": DATASET_DIR / "experiment_07.csv",
    "experiment_12": DATASET_DIR / "experiment_12.csv",
}
```

`/reload-model` 라우트 앞(파일 끝 근처)에:

```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest -q`
Expected: 모두 passed (200 + 4 + 1 = 205)

- [ ] **Step 5: README §2-9와 폴더 표**

`README.md`의 `### 2-8. Docker로 실행` 절 뒤(`## 3.` 앞)에 추가:

```markdown
### 2-9. 미팅 데모 화면

모델 판정 → 센서 근거 → 플레이북 대조 → 조치 가이드(탭 1)와 시나리오 3종 40일
타임라인 + 재학습 루프 이벤트(탭 2)를 한 페이지로 보여 준다. 실제 결과가 파일 안에
내장돼 있어 **서버 없이도 열린다.**

**기록 모드 (pull만 받은 PC):** `demo/index.html`을 브라우저로 연다. 끝.

**실시간 모드 (같은 카드를 실제 `/predict`로 다시 계산):**

1. 회사 PC에서 `data/`를 묶어 옮긴다 — `tar -czf cnc-data.tar.gz --exclude=data/monitoring data`
   (60MB 안팎). 개인 PC의 `02-cnc-machining/` 바로 아래에 풀어 §1-2 구조가 되게 한다.
   MLflow가 저장한 절대경로는 첫 실행 때 `src/lstm_ae/tracking.py`가 이 PC 기준으로
   고치므로 재학습이 필요 없다.
2. `uv sync`
3. `.env`에 `OPENAI_API_KEY`(§1-4). 없어도 판정·`fault`는 되고 `guide`만 `null`.
4. `uv run --env-file .env uvicorn serving.app:app --port 8899` → 브라우저에서
   `http://127.0.0.1:8899/demo`. 상단 배지가 "실시간 연결됨"이면 예시 재계산과 CSV
   업로드가 된다. WSL이면 §3 참고.
5. 미팅 전 점검: 예시 하나를 다시 계산해 가이드가 오는지와 소요 시간(몇 초).

페이지 다시 만들기(재료가 바뀐 경우, 워커 로그가 있는 회사 PC에서):
`uv run python demo/build_demo.py` → `demo/index.html`을 커밋.
```

`README.md` §4 "실행 스크립트 + 결과물" 표와 `docs/STRUCTURE.md` 폴더 표에 한 줄:
`| `demo/` | 미팅 데모 페이지 생성 (§2-9) | `demo/index.html` (커밋됨) |`
(STRUCTURE.md는 그 표의 열 구성에 맞춰 같은 내용으로.)

- [ ] **Step 6: 커밋**

```bash
git add src/serving/app.py tests/serving/test_app.py README.md docs/STRUCTURE.md
git commit -m "feat(serving): serve the demo page and example inputs; document meeting setup

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 5: 라이브 확인, 스펙 정정, push

**Files:**
- Modify: `docs/specs/2026-09-03-cnc-demo-page-design.md` (정정 절)

- [ ] **Step 1: 서버로 열어 실시간 확인**

```bash
T=/home/sure/.claude/jobs/6b3a648e/tmp
nohup nice -n 19 uv run --env-file .env uvicorn serving.app:app --port 8899 > $T/serve_demo.log 2>&1 &
echo $! > $T/serve_demo.pid
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8899/health > /dev/null && break; sleep 1; done
curl -s -o /dev/null -w "/demo %{http_code} %{content_type}\n" http://127.0.0.1:8899/demo
for k in tool_wear feed_overload vibration_backlash experiment_07 experiment_12; do curl -s -o /dev/null -w "$k %{http_code} %{size_download}B\n" http://127.0.0.1:8899/demo/inputs/$k; done
explorer.exe "http://127.0.0.1:8899/demo"
```
브라우저에서: 배지 "실시간 연결됨 · champion v1", 예시 재계산 5건(소요 시간 기록), CSV 업로드 1건. 끝나면 `kill $(cat $T/serve_demo.pid); pkill -f '[u]vicorn serving.app'; rm -f data/monitoring/requests.db`.

- [ ] **Step 2: 키 없이 서버로 열어 폴백 확인**

`env -u OPENAI_API_KEY nohup uv run uvicorn serving.app:app --port 8899 ...` 로 띄우고 예시 재계산 → 카드 6에 "guide가 null" 안내, fault는 표시. 종료 후 `requests.db` 삭제.

- [ ] **Step 3: 스펙 정정 절**

`docs/specs/2026-09-03-cnc-demo-page-design.md` 끝에 "실행 결과에 따른 정정" 절: index.html 크기, 시나리오별 일수·이벤트 수, 실시간 재계산 소요 시간, 계획과 달랐던 점.

- [ ] **Step 4: 커밋·push**

```bash
uv run pytest -q
git add docs/specs/2026-09-03-cnc-demo-page-design.md
git commit -m "docs(spec): record demo page verification results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
git push origin main
```

---

## Self-Review

- **스펙 커버리지**: 내장 데이터(Task 1·2·3), 화면 두 탭과 카드 7개(Task 3), 실시간 감지·재계산·업로드·오류 처리(Task 3 스크립트), 서버 라우트 2개(Task 4), 개인 PC 준비 README(Task 4), 테스트(Task 2·4), 검증 1~6(Task 1 Step 2, Task 3 Step 3~4, Task 5).
- **플레이스홀더**: 정정 절은 실행값으로 채우는 칸. 그 외 없음.
- **타입 일관성**: `assemble` 결과 키(`generated_at, versions, playbook, examples, scenarios`)와 스크립트의 `DATA.*` 접근 일치. 배치 키(`index, ratio, pred, verdict, verdict_ko, situation, coverage, top`)와 `renderDay`의 접근 일치. 이벤트 키(`day, kind, text, gate, reason, cause, actions`)와 `renderDay`·차트 일치. `DEMO_INDEX`·`DEMO_INPUTS` 이름을 테스트·구현이 같이 씀.
