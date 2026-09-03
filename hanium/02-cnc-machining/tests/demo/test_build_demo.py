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
            "verdict": verdict, "verdict_ko": {"confirmed": "확정(옛 기록)", "none": "이상 없음"}[verdict],
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
    # 채점 기록의 옛 verdict_ko("확정")를 쓰지 않고 현재 VERDICT_KO로 다시 붙인다
    assert sc["days"][0]["batches"][1]["verdict_ko"] == "높은 패턴 일치"
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
