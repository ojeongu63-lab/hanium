# 플레이북 기반 조치 가이드 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/predict`의 조치 가이드가 임베딩 유사도 대신 "문서에 적힌 센서 서명과 모델 상위 피처의 일치도"로 16개 상황 중 하나를 결정적으로 고르고, 세 단계 판정(확정/복합 징후/약한 신호)과 함께 `fault` 필드로 응답한다.

**Architecture:** 플레이북 문서(`rag/sources/scenario_playbook.md`)가 상황과 센서 서명을 담고, `src/rag/playbook.py`가 그 문서를 파싱하고 대조하는 순수 함수를 제공한다. `src/serving/inference.py`가 판정 뒤 `match_playbook`으로 `fault`를 만들고, `src/rag/guide.py`가 판정에 따라 청크를 고른 뒤 `generation.py`가 "시스템 판정" 줄이 들어간 프롬프트로 가이드를 쓴다. 임베딩은 확정 판정에서 Sandvik 청크 2개를 고를 때만 쓴다. 재학습·게이트·워커·시뮬레이터·원인 추정 코드는 건드리지 않는다.

**Tech Stack:** Python 3.14, uv, pytest, FAISS(IndexFlatIP), OpenAI(text-embedding-3-small, gpt-4o-mini), FastAPI(TestClient), MLflow(champion 로드).

**Spec:** `docs/specs/2026-09-03-cnc-playbook-guide-design.md` — Part A(문서·파서), B(대조), C(청크 선택·프롬프트·응답). 진단 에이전트는 보류.

## Global Constraints

- 작업 디렉터리는 `02-cnc-machining/`. 테스트는 `uv run pytest -q`(현재 169개 통과가 기준선).
- 상수는 스펙 값 그대로: `TOP_N = 5`, `WEAK_Z = 10.0`, `COMPOSITE_RATIO = 0.5`. 바꾸지 않는다.
- 판정 문자열은 `"confirmed" | "composite" | "weak" | "unknown" | "none"`, 한글은 `확정 | 복합 징후 | 약한 신호 | 판단 불가 | 이상 없음`.
- 플레이북 메타: `title = "팀 시나리오 플레이북(자체 작성)"`, `url = "rag/sources/scenario_playbook.md"`. 청크 식별은 `source == "playbook"`.
- `/predict` 기존 9개 키(`predicted_label`, `predicted_label_text`, `score`, `threshold`, `method`, `feature_contributions`, `model_version`, `mlflow_run_id`, `guide`)와 `guide`의 JSON 스키마는 바꾸지 않는다. `fault`만 더한다.
- `관련 센서` 코드는 `preprocessing.columns.FEATURE_COLUMNS`(41개)에 있는 것만. 빌드 시 검증.
- 변경 금지: `src/retraining/*`, `monitoring/*`, `src/monitoring/*`, `src/lstm_ae/*`, `src/preprocessing/*`, `src/rag/{query,retrieval,features}.py`, `generate_cause_guide`·`build_cause_guide`.
- 무거운 실행(타임라인 채점, 라이브 서버)은 `who`/`uptime` 확인 후 `nice -n 19`. OpenAI 키는 `.env`(`uv run --env-file .env`), 값은 절대 출력하지 않는다.
- 커밋은 main에 직접, 메시지 끝에 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`와 `Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz`.
- 스크립트는 `Path(__file__).resolve().parent.parent`로 루트를 역산하고 `sys.path.insert(0, ROOT / "src")`로 라이브러리를 잡는다(기존 `monitoring/*.py` 관례).

---

## 파일 구조

| 파일 | 책임 | 이번 변경 |
|---|---|---|
| `rag/sources/scenario_playbook.md` | 상황 16개, 구역 4개, 항목마다 `관련 센서` 줄 | 신규 |
| `src/rag/playbook.py` | `parse_playbook`, `coverage`, `match_playbook`, 상수, `NO_FAULT` | 신규 |
| `tests/rag/test_playbook.py` | 위 함수 단위 테스트 + 실제 문서·기록 4건 검증 | 신규 |
| `rag/build_corpus.py` | 플레이북 파싱·검증·메타 추가 | 수정 |
| `src/rag/generation.py` | 시스템 프롬프트 개정, `describe_fault`, `generate_guide(..., fault=None)` | 수정 |
| `tests/rag/test_generation.py` | 프롬프트 검증 3개 추가 | 수정 |
| `src/rag/guide.py` | `select_chunks`, `build_guide`의 `fault` 경로 | 수정 |
| `tests/rag/test_guide.py` | `select_chunks` 4개 + 기존 경로 유지 1개 | 수정 |
| `src/serving/inference.py` | `fault` 계산·반환 | 수정 |
| `tests/serving/test_inference.py`, `tests/serving/test_app.py` | `fault` 필드 검증 | 수정 |
| `rag/eval_playbook.py` | 오프라인 채점(기록 4건 + 타임라인 3종) | 신규 |
| `docs/examples/predict_response_*.json` | 라이브 응답 예 5개 | 갱신 2 + 신규 3 |
| `README.md`, `docs/STRUCTURE.md` | §2-1 `fault`, §2-4 플레이북, 폴더 표 | 수정 |
| `docs/specs/2026-09-03-cnc-playbook-guide-design.md` | 실행 결과 정정 절 | 수정 |

---

### Task 1: 플레이북 문서와 파서

**Files:**
- Create: `rag/sources/scenario_playbook.md`
- Create: `src/rag/playbook.py` (이 태스크에서는 파서와 상수만)
- Test: `tests/rag/test_playbook.py`

**Interfaces:**
- Produces: `parse_playbook(text: str, known_features: set[str] | None = None) -> list[dict]` — 청크 키 `heading, name, text, fault_category, content_type, signature, source`. `PLAYBOOK_SOURCE = "playbook"`, `PLAYBOOK_META`, `SECTION_CATEGORY`.

- [ ] **Step 1: 플레이북 문서 작성**

`rag/sources/scenario_playbook.md`를 아래 내용 그대로 만든다. 항목 제목의 구분자는 " — "(앞뒤 공백 있는 em dash), 각 항목 본문 첫 줄은 `관련 센서:`.

````markdown
# CNC 이상 상황 플레이북 (팀 작성, 데모용)

이 문서는 외부 자료가 아니라 팀이 시나리오에 맞춰 쓴 플레이북이다. 구역은
원인이 아니라 센서가 보여주는 모습으로 나누고, 항목마다 다섯 줄을 고정으로
쓴다: 관련 센서(`/predict` 피처 코드만), 증상, 가능 원인, 현장 확인, 조치.
센서로 구분이 안 되는 원인은 같은 구역에 두고 현장 확인 줄에 구분법을 적는다.
각 구역의 첫 항목이 대표 상황이며 서명 점수가 같으면 앞선 항목이 뽑힌다.

## 1. 스핀들 부하 상승

### 공구 마모 — 스핀들 부하가 서서히 상승
관련 센서: S_OutputCurrent, S_OutputPower, S_CurrentFeedback
증상: 스핀들 출력 전류, 스핀들 출력 파워, 스핀들 전류 피드백이 정상 대비 크게 상승한다. 축 위치·속도는 정상 범위이고, 배치가 진행될수록 상승 폭이 커진다.
가능 원인: 절삭날 마모로 절삭 저항이 커져 같은 회전수를 유지하기 위한 스핀들 모터 부하가 증가한다.
현장 확인: 인서트 플랭크 마모 폭, 가공면 조도 저하, 버(burr) 발생 여부.
조치: 마모 한계에 도달했으면 인서트를 교체한다. 절삭 속도(vc)를 낮추고 내마모성이 높은 등급으로 바꾼다. 교체 후 스핀들 부하가 기준선으로 돌아오는지 확인한다.

### 공구 파손·치핑 — 스핀들 부하가 갑자기 튀거나 떨어짐
관련 센서: S_OutputCurrent, S_OutputPower
증상: 스핀들 출력 전류·출력 파워가 특정 시점에 급격히 튀었다가 이후 비정상적으로 낮아진다(절삭이 되지 않음). 축 속도는 정상이다.
가능 원인: 인서트 치핑, 날끝 파손, 칩 해머링.
현장 확인: 공구 육안 점검, 가공물 치수 불량. 마모와 달리 변화가 한 시점에 급격하다.
조치: 즉시 정지하고 공구를 교체한다. 인성이 높은 등급과 강한 지오메트리를 선택하고 진입 조건을 완화한다.

### 절삭유 부족·과열 — 스핀들 부하 상승과 크레이터 마모
관련 센서: S_OutputPower, S_OutputVoltage
증상: 스핀들 출력 파워가 오르면서 스핀들 출력 전압이 함께 변동한다. 위치·속도는 정상이다.
가능 원인: 절삭유 공급 부족으로 절삭 온도가 올라 크레이터 마모·소성 변형이 생긴다.
현장 확인: 절삭유 노즐 막힘, 유량, 인서트 레이크면 변색.
조치: 절삭유 공급을 복구하고 속도를 낮춘다. Al2O3 코팅 등급을 선택한다.

### 스핀들 베어링 손상 — 부하 상승과 회전 속도 흔들림
관련 센서: S_OutputCurrent, S_ActualVelocity, S_ActualAcceleration
증상: 스핀들 출력 전류 상승과 동시에 스핀들 실제 속도·실제 가속도가 참조값 대비 흔들린다. 공구를 교체해도 부하가 내려가지 않는다.
가능 원인: 스핀들 베어링 손상, 윤활 불량, 벨트·커플링 이상.
현장 확인: 공회전 시 소음·발열·진동, 공구 교체 후 부하 재측정. 공구 마모와 달리 공구를 바꿔도 부하가 남는다.
조치: 공회전 부하를 측정하고 베어링·윤활 상태를 점검한다. 필요하면 베어링을 교체한다.

### 스핀들 전원·인버터 이상 — 전압 변동
관련 센서: S_DCBusVoltage, S_OutputVoltage
증상: 스핀들 DC 버스 전압과 출력 전압이 변동하고 출력 전류가 그에 따라 요동한다. 절삭 여부와 무관하게 나타난다.
가능 원인: 인버터·전원 이상, 전원 품질 문제.
현장 확인: 알람 이력, 전원 전압 측정. 절삭하지 않을 때도 나타나면 공구 문제가 아니다.
조치: 전기 담당자가 잠금·표시(LOTO) 후 인버터와 전원을 점검한다.

## 2. 이송축 부하 상승

### 이송축 과부하 — X·Y축 전류·파워 상승
관련 센서: X_OutputCurrent, Y_OutputCurrent, X_OutputPower, Y_OutputPower
증상: X축·Y축 출력 전류와 출력 파워가 정상 대비 크게 상승한다. 스핀들 부하는 상대적으로 덜 오른다.
가능 원인: 절삭 깊이·이송 과다, 칩 걸림(재밍), 칩 재절삭.
현장 확인: 칩 배출 상태, 서보 부하 미터, 프로그램의 절삭 조건.
조치: 이송량(fz)과 절삭 깊이(ap)를 줄이고 깊은 절삭은 여러 패스로 나눈다. 절삭유·압축 공기로 칩 배출을 개선한다. 서보 알람 이력을 확인한다.

### 볼스크류·가이드 윤활 불량 — 무부하 이송에서도 축 전류 높음
관련 센서: X_OutputCurrent, Y_OutputCurrent
증상: X·Y축 출력 전류가 절삭하지 않는 급속 이송 구간에서도 높다. 축 실제 속도는 참조값을 잘 따라간다.
가능 원인: 윤활 부족, 가이드 오염, 볼스크류 예압 이상.
현장 확인: 윤활유 레벨과 펌프 동작, 급속 이송 시 부하. 이송축 과부하와 달리 절삭하지 않을 때도 전류가 높다.
조치: 윤활유를 보충하고 배관을 점검한다. 가이드를 청소하고 예압을 점검한다.

### 절삭 조건 과다 — 스핀들과 이송축 부하가 함께 상승
관련 센서: S_OutputPower, X_OutputPower, Y_OutputPower
증상: 스핀들 출력 파워와 X·Y축 출력 파워가 같이 크게 상승한다. 위치·속도는 정상이다.
가능 원인: 절삭 깊이·이송·속도 조합이 기계 능력을 초과한다. 새 프로그램 투입 직후 흔하다.
현장 확인: 프로그램 변경 이력, 권장 절삭 조건 대비 실제 조건.
조치: 절삭 조건을 재검토해 파라미터를 낮추고 시험 가공으로 확인한다.

## 3. 축 위치·속도 편차

### 고정구 풀림·채터 — 축 실제 속도·위치가 불규칙하게 흔들림
관련 센서: X_ActualVelocity, Y_ActualVelocity, Z_ActualVelocity, X_ActualPosition, Y_ActualPosition, Z_ActualPosition
증상: X·Y·Z축 실제 속도와 실제 위치가 참조값 대비 불규칙하게 흔들린다. 전류·파워 상승보다 위치·속도 편차가 먼저 나타난다.
가능 원인: 고정구(지그) 풀림, 클램프 토크 부족, 절삭 중 채터.
현장 확인: 가공면 채터 마크, 클램프 토크, 공작물 흔들림, 소음.
조치: 기계를 정지하고 고정구·클램프를 점검해 다시 체결한다. 절삭 깊이를 낮추고 differential pitch 커터를 쓴다.

### 볼스크류 백래시 — 방향 전환 시 위치 오차
관련 센서: X_ActualPosition, Y_ActualPosition, Z_ActualPosition, X_ActualAcceleration, Y_ActualAcceleration, Z_ActualAcceleration
증상: 축 실제 위치가 참조 위치 대비 방향 전환 구간에서 지연·오차를 보이고 실제 가속도가 튄다. 속도 흔들림은 크지 않다.
가능 원인: 볼스크류 백래시, 너트 마모, 커플링 풀림.
현장 확인: 다이얼 게이지로 백래시 측정, 원형 보간 테스트. 고정구 풀림과 달리 방향 전환 순간에만 오차가 난다.
조치: 백래시 보정값을 갱신하고 커플링·너트를 점검한다.

### 공구 돌출 과다·홀더 불량 — 잔진동
관련 센서: S_CurrentFeedback, X_ActualVelocity, Y_ActualVelocity
증상: 스핀들 전류 피드백이 미세하게 요동하고 X·Y축 실제 속도에 잔진동이 실린다. 가공면 조도가 나쁘다.
가능 원인: 공구 돌출 길이 과다, 홀더 런아웃, 콜릿 불량.
현장 확인: 돌출 길이, 런아웃 측정.
조치: 돌출 길이를 최소화하고 런아웃을 0.02mm 이하로 맞춘다. 홀더·콜릿을 교체한다.

### 엔코더·피드백 이상 — 실제값과 참조값의 체계적 불일치
관련 센서: X_ActualPosition, Y_ActualPosition, Z_ActualPosition
증상: 특정 축의 실제 위치·속도가 참조값과 일정한 오프셋으로 어긋나거나 뛴다. 전류·파워는 정상이다.
가능 원인: 엔코더 오염·손상, 케이블 접촉 불량.
현장 확인: 알람 이력, 엔코더 케이블·커넥터. 고정구 풀림과 달리 한 축에서만 일정하게 어긋난다.
조치: 케이블과 커넥터를 점검하고 서비스 담당을 호출한다.

## 4. 고장이 아닌 변화

### 온도 드리프트 — 여러 센서가 서서히 이동, 불량 없음
관련 센서: 없음
증상: 특정 그룹 없이 여러 센서의 평균값이 며칠에 걸쳐 조금씩 이동한다. 스핀들과 이송축이 함께 오르기도 한다. 가공 불량률은 늘지 않는다.
가능 원인: 공장 온도·계절 변화로 센서 기준선이 이동한 것이며 설비 고장이 아니다.
현장 확인: 실내 온도 기록, 불량률 추이, 드리프트 감시 결과.
조치: 최근 정상 데이터로 모델을 재학습(재보정)한다. 불량률이 늘지 않는지 함께 확인한다.

### 이송 속도 설정 변경 — 학습 범위 밖 프로그램
관련 센서: M_CURRENT_FEEDRATE, X_SetVelocity, Y_SetVelocity, Z_SetVelocity
증상: 현재 이송 속도가 학습 데이터에 없던 값이고 축 참조 속도도 함께 다르다. 가공 품질은 정상이다.
가능 원인: 프로그램 변경이나 새 절삭 조건. 모델의 정상 범위 밖이라 오탐일 수 있다.
현장 확인: 프로그램 변경 이력, 가공 품질.
조치: 품질이 정상이면 정상으로 기록한다. 같은 조건이 반복되면 그 데이터로 재학습한다.

### 가공 시작·종료 과도 구간 — 일시적 편차
관련 센서: X_ActualAcceleration, Y_ActualAcceleration, Z_ActualAcceleration
증상: 배치 앞뒤 짧은 구간에서만 실제 가속도와 전류가 튀고 나머지 구간은 정상이다.
가능 원인: 가감속 과도 응답, 공구 교환 구간.
현장 확인: 편차가 나타난 시점이 시작·종료 구간인지.
조치: 반복되지 않으면 조치가 필요 없다. 지속되면 가감속 파라미터를 확인한다.

### 공작물 소재 변경 — 부하 수준 전체가 이동
관련 센서: 없음
증상: 스핀들·이송축 부하가 전체적으로 다른 수준으로 이동하지만 흔들림은 없다. 로트 변경 시점과 일치한다.
가능 원인: 소재 경도·형상 변경.
현장 확인: 소재 로트 변경 기록.
조치: 소재 변경을 확인하고 절삭 조건을 재설정한다. 필요하면 재학습한다.
````

- [ ] **Step 2: 파서 실패 테스트 작성**

`tests/rag/test_playbook.py`:

```python
from collections import Counter
from pathlib import Path

import pytest

from preprocessing.columns import FEATURE_COLUMNS
from rag.playbook import PLAYBOOK_SOURCE, parse_playbook

ROOT = Path(__file__).resolve().parent.parent.parent
PLAYBOOK_TEXT = (ROOT / "rag" / "sources" / "scenario_playbook.md").read_text()

_MINI = """# 제목
## 1. 스핀들 부하 상승
### 공구 마모 — 부하 상승
관련 센서: S_OutputCurrent, S_OutputPower
증상: 스핀들 부하 상승.
조치: 교체.
## 4. 고장이 아닌 변화
### 온도 드리프트 — 서서히 이동
관련 센서: 없음
증상: 여러 센서 이동.
"""


def test_parse_playbook_reads_sections_signatures_and_names():
    chunks = parse_playbook(_MINI)

    assert [c["name"] for c in chunks] == ["공구 마모", "온도 드리프트"]
    assert chunks[0]["heading"] == "공구 마모 — 부하 상승"
    assert chunks[0]["fault_category"] == "tool_wear"
    assert chunks[0]["content_type"] == "cause"
    assert chunks[0]["signature"] == ["S_OutputCurrent", "S_OutputPower"]
    assert chunks[0]["source"] == PLAYBOOK_SOURCE
    assert chunks[0]["text"].startswith("관련 센서: S_OutputCurrent")
    assert chunks[1]["fault_category"] == "general"
    assert chunks[1]["content_type"] == "context"
    assert chunks[1]["signature"] == []


def test_parse_playbook_rejects_entry_without_signature_line():
    text = "## 1. 스핀들 부하 상승\n### 공구 마모 — x\n증상: 없음\n"
    with pytest.raises(ValueError, match="관련 센서"):
        parse_playbook(text)


def test_parse_playbook_rejects_unknown_feature_code():
    with pytest.raises(ValueError, match="S_OutputPower"):
        parse_playbook(_MINI, known_features={"S_OutputCurrent"})


def test_parse_playbook_rejects_unknown_section():
    with pytest.raises(ValueError, match="구역"):
        parse_playbook("## 9. 없는 구역\n### a — b\n관련 센서: 없음\n증상: x\n")


def test_real_playbook_has_16_entries_with_valid_codes():
    chunks = parse_playbook(PLAYBOOK_TEXT, known_features=set(FEATURE_COLUMNS))

    assert len(chunks) == 16
    assert Counter(c["fault_category"] for c in chunks) == {
        "tool_wear": 5, "feed_overload": 3, "vibration_backlash": 4, "general": 4,
    }
    firsts = {}
    for c in chunks:
        firsts.setdefault(c["fault_category"], c["name"])
    assert firsts == {
        "tool_wear": "공구 마모", "feed_overload": "이송축 과부하",
        "vibration_backlash": "고정구 풀림·채터", "general": "온도 드리프트",
    }
    assert all(len(c["name"]) < len(c["heading"]) for c in chunks)
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/rag/test_playbook.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.playbook'`

- [ ] **Step 4: 파서 구현**

`src/rag/playbook.py`:

```python
"""플레이북(팀 작성 상황 문서) 파싱과 센서 서명 대조.

판정은 순수 계산이다 — 임베딩·LLM을 쓰지 않으므로 같은 입력이면 같은 결과가 나오고
OpenAI 키 없이도 돈다. 임베딩은 guide.py가 확정 판정에서 Sandvik 청크를 고를 때만 쓴다.
"""

PLAYBOOK_SOURCE = "playbook"
PLAYBOOK_META = {
    "title": "팀 시나리오 플레이북(자체 작성)",
    "url": "rag/sources/scenario_playbook.md",
}
SECTION_CATEGORY = {
    "1. 스핀들 부하 상승": "tool_wear",
    "2. 이송축 부하 상승": "feed_overload",
    "3. 축 위치·속도 편차": "vibration_backlash",
    "4. 고장이 아닌 변화": "general",
}
SIGNATURE_PREFIX = "관련 센서:"
HEADING_SEPARATOR = " — "


def _parse_signature(value: str, heading: str, known_features: set[str] | None) -> list[str]:
    value = value.strip()
    if value == "없음":
        return []
    codes = [code.strip() for code in value.split(",") if code.strip()]
    if known_features is not None:
        unknown = [code for code in codes if code not in known_features]
        if unknown:
            raise ValueError(f"'{heading}' 항목의 모르는 센서 코드: {unknown}")
    return codes


def parse_playbook(text: str, known_features: set[str] | None = None) -> list[dict]:
    """'## ' 구역(SECTION_CATEGORY로 카테고리 결정), '### ' 항목. 항목 본문에 '관련 센서:'
    줄이 반드시 있어야 하며 그 줄이 signature가 된다(본문 text에도 남긴다)."""
    chunks: list[dict] = []
    category = None
    heading = None
    body: list[str] = []
    signature: list[str] | None = None

    def flush() -> None:
        if heading is None:
            return
        if signature is None:
            raise ValueError(f"'{heading}' 항목에 '{SIGNATURE_PREFIX}' 줄이 없음")
        chunks.append({
            "heading": heading,
            "name": heading.split(HEADING_SEPARATOR)[0].strip(),
            "text": "\n".join(body).strip(),
            "fault_category": category,
            "content_type": "context" if category == "general" else "cause",
            "signature": signature,
            "source": PLAYBOOK_SOURCE,
        })

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading, body, signature = None, [], None
            section = line[3:].strip()
            if section not in SECTION_CATEGORY:
                raise ValueError(f"모르는 구역 '{section}' — SECTION_CATEGORY에 추가 필요")
            category = SECTION_CATEGORY[section]
        elif line.startswith("### "):
            flush()
            heading, body, signature = line[4:].strip(), [], None
        elif heading is not None and line.strip():
            stripped = line.strip()
            if stripped.startswith(SIGNATURE_PREFIX):
                signature = _parse_signature(
                    stripped[len(SIGNATURE_PREFIX):], heading, known_features
                )
            body.append(stripped)
    flush()
    return chunks
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/rag/test_playbook.py -q`
Expected: 5 passed

- [ ] **Step 6: 커밋**

```bash
git add rag/sources/scenario_playbook.md src/rag/playbook.py tests/rag/test_playbook.py
git commit -m "feat(rag): add scenario playbook (16 situations) and parser

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 2: 센서 서명 대조 `coverage` · `match_playbook`

**Files:**
- Modify: `src/rag/playbook.py`
- Test: `tests/rag/test_playbook.py`

**Interfaces:**
- Consumes: Task 1의 청크 형식(`name`, `fault_category`, `signature`, `source`).
- Produces: `TOP_N = 5`, `WEAK_Z = 10.0`, `COMPOSITE_RATIO = 0.5`, `VERDICT_KO: dict[str, str]`, `NO_FAULT: dict`, `coverage(signature, contributions, top_n=TOP_N) -> float`, `match_playbook(contributions, corpus) -> dict | None`. 반환 dict 키: `verdict, verdict_ko, situation, category, coverage, matched_features, alternatives, other_group, top_z`.

- [ ] **Step 1: 대조 테스트 작성**

`tests/rag/test_playbook.py`에 추가:

```python
import json

from rag.playbook import (
    NO_FAULT, TOP_N, WEAK_Z, coverage, match_playbook,
)

_RULE_CORPUS = [
    {"name": "공구 마모", "fault_category": "tool_wear", "source": PLAYBOOK_SOURCE,
     "signature": ["S_OutputCurrent", "S_OutputPower", "S_CurrentFeedback"]},
    {"name": "스핀들 베어링 손상", "fault_category": "tool_wear", "source": PLAYBOOK_SOURCE,
     "signature": ["S_OutputCurrent", "S_ActualVelocity"]},
    {"name": "이송축 과부하", "fault_category": "feed_overload", "source": PLAYBOOK_SOURCE,
     "signature": ["X_OutputCurrent", "Y_OutputCurrent", "X_OutputPower", "Y_OutputPower"]},
    {"name": "윤활 불량", "fault_category": "feed_overload", "source": PLAYBOOK_SOURCE,
     "signature": ["X_OutputCurrent", "Y_OutputCurrent"]},
    {"name": "온도 드리프트", "fault_category": "general", "source": PLAYBOOK_SOURCE,
     "signature": []},
    {"heading": "플랭크 마모", "fault_category": "tool_wear", "content_type": "cause",
     "text": "Sandvik 청크 — signature 없음"},
]


def _contribs(*features: str, top_z: float = 100.0) -> list[dict]:
    """z를 순위대로 내림차순으로 만든다. 첫 피처의 z가 top_z."""
    return [
        {"feature": f, "error": 1.0, "z_score": top_z / (i + 1)}
        for i, f in enumerate(features)
    ]


def test_coverage_weights_by_rank_over_top_n():
    contribs = _contribs("a", "b", "c", "d", "e", "f")
    assert coverage(["a", "b", "c"], contribs) == pytest.approx(0.80)   # (1+1/2+1/3)/(1+..+1/5)
    assert coverage(["a", "b"], contribs) == pytest.approx(0.66)
    assert coverage(["f"], contribs) == 0.0                              # 6위는 대조 밖
    assert coverage(["z"], contribs) == 0.0
    assert coverage(["a"], []) == 0.0
    assert TOP_N == 5


def test_match_confirmed_picks_best_signature_and_lists_alternatives():
    result = match_playbook(_contribs("S_OutputCurrent", "S_CurrentFeedback", "S_OutputPower"), _RULE_CORPUS)

    assert result["verdict"] == "confirmed"
    assert result["verdict_ko"] == "확정"
    assert result["situation"] == "공구 마모"
    assert result["category"] == "tool_wear"
    assert result["coverage"] == pytest.approx(1.0)
    assert result["matched_features"] == ["S_OutputCurrent", "S_CurrentFeedback", "S_OutputPower"]
    assert result["alternatives"] == ["스핀들 베어링 손상"]
    assert result["other_group"] is None          # 다른 그룹 점수 0
    assert result["top_z"] == pytest.approx(100.0)


def test_match_tie_goes_to_earlier_entry():
    result = match_playbook(_contribs("X_OutputCurrent", "Y_OutputCurrent"), _RULE_CORPUS)

    assert result["situation"] == "이송축 과부하"       # 윤활 불량과 동점(1.0), 앞선 항목
    assert result["alternatives"] == ["윤활 불량"]


def test_match_weak_when_top_z_below_threshold():
    result = match_playbook(_contribs("S_OutputCurrent", "S_OutputPower", top_z=WEAK_Z - 0.1), _RULE_CORPUS)

    assert result["verdict"] == "weak"
    assert result["verdict_ko"] == "약한 신호"
    assert result["situation"] == "공구 마모"           # 참고로 채움


def test_match_composite_when_other_group_is_half_of_best():
    contribs = _contribs("S_OutputCurrent", "X_OutputCurrent", "Y_OutputCurrent", "S_OutputPower", "S_CurrentFeedback")
    result = match_playbook(contribs, _RULE_CORPUS)

    assert result["verdict"] == "composite"            # 공구 마모 0.64 vs 이송축 과부하 0.36
    assert result["situation"] == "공구 마모"
    assert result["other_group"] == {"situation": "이송축 과부하", "category": "feed_overload",
                                     "coverage": pytest.approx(0.36)}


def test_match_unknown_when_nothing_matches():
    result = match_playbook(_contribs("X_DCBusVoltage", "Y_DCBusVoltage"), _RULE_CORPUS)

    assert result["verdict"] == "unknown"
    assert result["situation"] is None
    assert result["category"] is None
    assert result["coverage"] == 0.0
    assert result["top_z"] == pytest.approx(100.0)


def test_match_returns_none_without_playbook_entries():
    assert match_playbook(_contribs("S_OutputCurrent"), [_RULE_CORPUS[-1]]) is None
    assert match_playbook(_contribs("S_OutputCurrent"), []) is None


def test_no_fault_shape_matches_match_result_keys():
    result = match_playbook(_contribs("S_OutputCurrent"), _RULE_CORPUS)
    assert set(NO_FAULT) == set(result)
    assert NO_FAULT["verdict"] == "none"
    assert NO_FAULT["verdict_ko"] == "이상 없음"


_RECORDED = {
    "synthetic/scenarios/tool_wear_predict_result.json": "공구 마모",
    "synthetic/scenarios/feed_overload_predict_result.json": "이송축 과부하",
    "synthetic/scenarios/vibration_backlash_predict_result.json": "고정구 풀림·채터",
    "docs/examples/predict_response_experiment_07.json": "이송축 과부하",
}


@pytest.mark.parametrize("path,expected", list(_RECORDED.items()))
def test_real_playbook_matches_recorded_cases(path, expected):
    corpus = parse_playbook(PLAYBOOK_TEXT, known_features=set(FEATURE_COLUMNS))
    recorded = json.loads((ROOT / path).read_text())

    result = match_playbook(recorded["feature_contributions"], corpus)

    assert result["situation"] == expected
    assert result["verdict"] == "confirmed"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/rag/test_playbook.py -q`
Expected: FAIL — `ImportError: cannot import name 'NO_FAULT'`

- [ ] **Step 3: 대조 구현**

`src/rag/playbook.py` 끝에 추가:

```python
TOP_N = 5              # 대조에 쓰는 상위 피처 수
WEAK_Z = 10.0          # 상위 1 피처의 z가 이 미만이면 "약한 신호"
COMPOSITE_RATIO = 0.5  # 다른 그룹 최고 점수가 1위의 이 비율 이상이면 "복합 징후"

VERDICT_KO = {
    "confirmed": "확정",
    "composite": "복합 징후",
    "weak": "약한 신호",
    "unknown": "판단 불가",
    "none": "이상 없음",
}

NO_FAULT = {
    "verdict": "none", "verdict_ko": VERDICT_KO["none"],
    "situation": None, "category": None, "coverage": 0.0,
    "matched_features": [], "alternatives": [], "other_group": None, "top_z": None,
}


def coverage(signature: list[str], contributions: list[dict], top_n: int = TOP_N) -> float:
    """상위 top_n 피처를 1/순위로 가중해, 서명이 설명하는 비율(0~1). 소수 둘째 자리."""
    top = contributions[:top_n]
    if not top:
        return 0.0
    total = sum(1 / (rank + 1) for rank in range(len(top)))
    got = sum(1 / (rank + 1) for rank, c in enumerate(top) if c["feature"] in signature)
    return round(got / total, 2)


def match_playbook(contributions: list[dict], corpus: list[dict]) -> dict | None:
    """플레이북 항목 중 상위 피처를 가장 잘 설명하는 상황을 고르고 세 단계로 판정한다.
    동점이면 코퍼스 순서(구역 대표 우선). 플레이북 항목이 없으면 None."""
    entries = [c for c in corpus if c.get("source") == PLAYBOOK_SOURCE]
    if not entries:
        return None

    scored = [(coverage(entry["signature"], contributions), entry) for entry in entries]
    best_cov, best = max(scored, key=lambda pair: pair[0])   # max는 첫 최댓값을 돌려준다
    top_z = float(contributions[0]["z_score"]) if contributions else None

    if best_cov == 0.0:
        return {
            **NO_FAULT, "verdict": "unknown", "verdict_ko": VERDICT_KO["unknown"], "top_z": top_z,
        }

    others = [(cov, e) for cov, e in scored if e["fault_category"] != best["fault_category"]]
    other_cov, other = max(others, key=lambda pair: pair[0]) if others else (0.0, None)
    same = sorted(
        ((cov, e) for cov, e in scored if e["fault_category"] == best["fault_category"] and e is not best),
        key=lambda pair: -pair[0],
    )

    if top_z is not None and top_z < WEAK_Z:
        verdict = "weak"
    elif other is not None and other_cov >= COMPOSITE_RATIO * best_cov:
        verdict = "composite"
    else:
        verdict = "confirmed"

    return {
        "verdict": verdict,
        "verdict_ko": VERDICT_KO[verdict],
        "situation": best["name"],
        "category": best["fault_category"],
        "coverage": best_cov,
        "matched_features": [
            c["feature"] for c in contributions[:TOP_N] if c["feature"] in best["signature"]
        ],
        "alternatives": [e["name"] for _, e in same[:2]],
        "other_group": (
            {"situation": other["name"], "category": other["fault_category"], "coverage": other_cov}
            if other is not None and other_cov > 0.0 else None
        ),
        "top_z": top_z,
    }
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/rag/test_playbook.py -q`
Expected: 17 passed (기록 4건 파라미터 포함)

- [ ] **Step 5: 커밋**

```bash
git add src/rag/playbook.py tests/rag/test_playbook.py
git commit -m "feat(rag): match playbook situations by sensor-signature coverage with 3-level verdict

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 3: 코퍼스 빌드에 플레이북 포함

**Files:**
- Modify: `rag/build_corpus.py:1-36` (import·메타), `:144-157` (`build_corpus`), `:159-186` (`main` 출력)

**Interfaces:**
- Consumes: `rag.playbook.parse_playbook`, `PLAYBOOK_META`, `PLAYBOOK_SOURCE`; `preprocessing.columns.FEATURE_COLUMNS`.
- Produces: `data/rag/corpus.json`(42청크, 플레이북 청크는 `source == "playbook"`과 `signature` 포함), `data/rag/corpus.index`.

- [ ] **Step 1: import와 소스 경로 추가**

`rag/build_corpus.py` 상단(`import json` 아래)에:

```python
import sys
```

`ROOT = ...` 정의 바로 아래에:

```python
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.columns import FEATURE_COLUMNS  # noqa: E402
from rag.playbook import PLAYBOOK_META, PLAYBOOK_SOURCE, parse_playbook  # noqa: E402
```

- [ ] **Step 2: `build_corpus`에 플레이북 추가**

`build_corpus()`에서 `kamp_text = ...` 아래에 `playbook_text = (SOURCES_DIR / "scenario_playbook.md").read_text()`를 넣고, KAMP 루프 뒤에:

```python
    for chunk in parse_playbook(playbook_text, known_features=set(FEATURE_COLUMNS)):
        corpus.append({**chunk, **PLAYBOOK_META})
```

- [ ] **Step 3: `main` 출력에 플레이북 수 추가**

`print(f"청크 {len(corpus)}개 저장됨")` 아래에:

```python
    playbook_count = sum(c.get("source") == PLAYBOOK_SOURCE for c in corpus)
    print(f"플레이북 항목 {playbook_count}개 (source == '{PLAYBOOK_SOURCE}', signature 포함)")
```

- [ ] **Step 4: 파싱만 드라이런 (키 없이)**

Run: `uv run python -c "import sys; sys.path.insert(0, 'rag'); import build_corpus as b; c = b.build_corpus(); print(len(c), sum(x.get('source') == 'playbook' for x in c))"`
Expected: `42 16`

- [ ] **Step 5: 코퍼스 재빌드 (OpenAI 키 필요)**

Run: `uv run --env-file .env python rag/build_corpus.py`
Expected: `청크 42개 저장됨`, `플레이북 항목 16개`, fault_category 분포에 `tool_wear: 19, vibration_backlash: 10, feed_overload: 5, general: 8`

- [ ] **Step 6: 회귀 확인 후 커밋**

Run: `uv run pytest -q`
Expected: 186 passed (169 + 17)

```bash
git add rag/build_corpus.py
git commit -m "feat(rag): include playbook chunks (with sensor signatures) in the corpus build

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 4: 프롬프트 — 시스템 판정 줄

**Files:**
- Modify: `src/rag/generation.py:6-16` (`SYSTEM_PROMPT`), `:32-49` (`_build_user_prompt`), `:52-67` (`generate_guide`)
- Test: `tests/rag/test_generation.py`

**Interfaces:**
- Consumes: Task 2의 `fault` dict, `WEAK_Z`.
- Produces: `describe_fault(fault: dict) -> str`, `_build_user_prompt(predict_result, retrieved_chunks, fault=None)`, `generate_guide(predict_result, retrieved_chunks, client, fault=None)`. `generate_cause_guide`·`CAUSE_SYSTEM_PROMPT`는 변경 없음.

- [ ] **Step 1: 프롬프트 테스트 작성**

`tests/rag/test_generation.py` 끝에 추가 (파일 상단의 `_FakeClient`, `_PAYLOAD`, `_PREDICT_RESULT` 재사용):

```python
from rag.generation import SYSTEM_PROMPT, describe_fault

_CHUNK = {"title": "팀 시나리오 플레이북(자체 작성)", "url": "rag/sources/scenario_playbook.md",
          "content_type": "cause", "text": "관련 센서: S_OutputCurrent\n조치: 교체"}

_CONFIRMED = {
    "verdict": "confirmed", "verdict_ko": "확정", "situation": "공구 마모", "category": "tool_wear",
    "coverage": 0.8, "matched_features": ["S_OutputCurrent", "S_OutputPower"],
    "alternatives": ["스핀들 베어링 손상"], "other_group": None, "top_z": 36.1,
}


def test_system_prompt_no_longer_primes_tool_wear():
    assert "tool_condition" not in SYSTEM_PROMPT
    assert "시스템 판정" in SYSTEM_PROMPT


def test_describe_fault_per_verdict():
    assert describe_fault(_CONFIRMED) == (
        "시스템 판정: 확정 — 공구 마모 (센서 서명 일치 0.80, 일치 센서: S_OutputCurrent, S_OutputPower)\n"
        "같은 구역의 다른 후보(현장 확인으로 구분): 스핀들 베어링 손상"
    )
    composite = {**_CONFIRMED, "verdict": "composite", "coverage": 0.66,
                 "other_group": {"situation": "이송축 과부하", "category": "feed_overload", "coverage": 0.44}}
    assert describe_fault(composite).startswith(
        "시스템 판정: 복합 징후 — 공구 마모(0.66)와 이송축 과부하(0.44)가 함께 나타남"
    )
    weak = {**_CONFIRMED, "verdict": "weak", "top_z": 4.4}
    assert describe_fault(weak) == (
        "시스템 판정: 약한 신호 — 상위 센서 z 4.4 (기준 10 미만). 보류·재확인을 권할 것. 참고 상황: 공구 마모"
    )
    unknown = {**_CONFIRMED, "verdict": "unknown", "situation": None}
    assert describe_fault(unknown) == "시스템 판정: 판단 불가 — 서명이 일치하는 상황 없음. 현장 확인을 권할 것."


def test_generate_guide_includes_fault_line_only_when_given():
    client = _FakeClient(json.dumps(_PAYLOAD))
    generate_guide(_PREDICT_RESULT, [_CHUNK], client, fault=_CONFIRMED)
    with_fault = client.chat.completions.last_kwargs["messages"][1]["content"]
    assert "시스템 판정: 확정 — 공구 마모" in with_fault
    assert with_fault.index("상위 이상 피처") < with_fault.index("시스템 판정") < with_fault.index("참고 문서")

    generate_guide(_PREDICT_RESULT, [_CHUNK], client)
    without = client.chat.completions.last_kwargs["messages"][1]["content"]
    assert "시스템 판정" not in without
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/rag/test_generation.py -q`
Expected: FAIL — `ImportError: cannot import name 'describe_fault'`

- [ ] **Step 3: 구현**

`src/rag/generation.py`의 `SYSTEM_PROMPT`를 교체:

```python
SYSTEM_PROMPT = (
    "당신은 CNC 가공 현장의 이상탐지 결과를 설명하는 어시스턴트입니다.\n"
    "판정은 센서 신호의 통계적 이상만 본 것이므로 원인을 단정하지 말고 "
    "'~일 가능성이 있습니다', '~로 추정됩니다' 같은 확신도를 낮춘 표현을 쓰세요.\n"
    "시스템 판정과 시스템이 고른 상황을 원인의 중심에 두세요. 참고 문서에 없는 "
    "원인을 덧붙이지 마세요. 같은 구역의 다른 후보는 '함께 확인할 것'으로만 "
    "언급하세요. 판정이 확정이 아니면 조치보다 확인 절차를 앞세우세요.\n"
    "아래 JSON 스키마로만 답하세요:\n"
    '{"cause_estimate": str, "confidence_note": str, '
    '"recommended_actions": [str], "safety_notes": [str], '
    '"sources": [{"title": str, "url": str}]}'
)
```

`from .playbook import WEAK_Z`를 import에 추가하고(파일 상단 `import os` 아래, 같은 패키지라 상대 import), `_build_user_prompt` 앞에:

```python
def describe_fault(fault: dict) -> str:
    """프롬프트에 넣는 '시스템 판정' 줄. 판정·상황·수치는 서버가 정한 값이다."""
    verdict = fault["verdict"]
    if verdict == "confirmed":
        line = (
            f"시스템 판정: 확정 — {fault['situation']} (센서 서명 일치 {fault['coverage']:.2f}, "
            f"일치 센서: {', '.join(fault['matched_features'])})"
        )
        if fault["alternatives"]:
            line += f"\n같은 구역의 다른 후보(현장 확인으로 구분): {', '.join(fault['alternatives'])}"
        return line
    if verdict == "composite":
        other = fault["other_group"]
        return (
            f"시스템 판정: 복합 징후 — {fault['situation']}({fault['coverage']:.2f})와 "
            f"{other['situation']}({other['coverage']:.2f})가 함께 나타남. 여러 센서가 같이 "
            "이동하는 드리프트일 수 있음. 라벨·추이 확인을 권할 것."
        )
    if verdict == "weak":
        return (
            f"시스템 판정: 약한 신호 — 상위 센서 z {fault['top_z']:.1f} (기준 {WEAK_Z:g} 미만). "
            f"보류·재확인을 권할 것. 참고 상황: {fault['situation']}"
        )
    return "시스템 판정: 판단 불가 — 서명이 일치하는 상황 없음. 현장 확인을 권할 것."
```

`_build_user_prompt`의 시그니처를 `(predict_result: dict, retrieved_chunks: list[dict], fault: dict | None = None)`로 바꾸고, `lines.append("상위 이상 피처: ...")` 바로 뒤에:

```python
    if fault is not None:
        lines.append(describe_fault(fault))
```

`generate_guide` 시그니처를 `(predict_result: dict, retrieved_chunks: list[dict], client, fault: dict | None = None)`로 바꾸고 user 메시지 content를 `_build_user_prompt(predict_result, retrieved_chunks, fault)`로.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/rag/test_generation.py -q`
Expected: 모두 passed (기존 + 3)

- [ ] **Step 5: 커밋**

```bash
git add src/rag/generation.py tests/rag/test_generation.py
git commit -m "feat(rag): put the system verdict in the guide prompt, drop tool-wear priming

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 5: 판정별 청크 선택과 `build_guide`

**Files:**
- Modify: `src/rag/guide.py:1-34`
- Test: `tests/rag/test_guide.py`

**Interfaces:**
- Consumes: `match_playbook` 결과 형식, `PLAYBOOK_SOURCE`, `search`, `build_query`, `generate_guide(..., fault)`.
- Produces: `select_chunks(fault, corpus, index, embed_fn, predict_result) -> list[dict]`. `build_guide` 시그니처는 그대로, `predict_result["fault"]`가 있으면 새 경로.

- [ ] **Step 1: 테스트 작성**

`tests/rag/test_guide.py` 끝에 추가:

```python
import faiss
import numpy as np

from rag.guide import select_chunks
from rag.playbook import PLAYBOOK_SOURCE

_PB = lambda name, cat, ctype="cause": {  # noqa: E731
    "name": name, "heading": f"{name} — x", "text": name, "fault_category": cat,
    "content_type": ctype, "source": PLAYBOOK_SOURCE, "signature": [],
    "title": "팀 시나리오 플레이북(자체 작성)", "url": "rag/sources/scenario_playbook.md",
}
_SELECT_CORPUS = [
    _PB("공구 마모", "tool_wear"), _PB("스핀들 베어링 손상", "tool_wear"),
    _PB("이송축 과부하", "feed_overload"),
    _PB("온도 드리프트", "general", "context"), _PB("소재 변경", "general", "context"),
    {"heading": "플랭크 마모", "text": "s1", "fault_category": "tool_wear", "content_type": "cause",
     "title": "Sandvik", "url": "https://x"},
    {"heading": "크레이터 마모", "text": "s2", "fault_category": "tool_wear", "content_type": "cause",
     "title": "Sandvik", "url": "https://x"},
    {"heading": "약한 고정구", "text": "s3", "fault_category": "vibration_backlash", "content_type": "cause",
     "title": "Sandvik", "url": "https://x"},
    {"heading": "안전 1", "text": "o1", "fault_category": "general", "content_type": "safety",
     "title": "OSHA", "url": "https://y"},
]
# 인덱스: 청크 i 의 벡터 = 단위벡터 e_i 를 살짝 섞어, 질의 [0,..,1(6번),..] 이 크레이터(6) > 플랭크(5)
_VECTORS = np.eye(len(_SELECT_CORPUS), dtype=np.float32)
_VECTORS[5, 6] = 0.5
_INDEX = faiss.IndexFlatIP(_VECTORS.shape[1])
_INDEX.add(_VECTORS)
_PREDICT = {"feature_contributions": [{"feature": "S_OutputCurrent", "error": 1.0, "z_score": 30.0}]}


def _fault(verdict, situation="공구 마모", category="tool_wear", alternatives=("스핀들 베어링 손상",), other=None):
    return {"verdict": verdict, "verdict_ko": "", "situation": situation, "category": category,
            "coverage": 0.8, "matched_features": ["S_OutputCurrent"], "alternatives": list(alternatives),
            "other_group": other, "top_z": 30.0}


def _names(chunks):
    return [c.get("name") or c["heading"] for c in chunks]


def test_select_chunks_confirmed_uses_embedding_once_for_same_category_external_chunks():
    calls = []

    def embed_fn(text):
        calls.append(text)
        return _VECTORS[6]

    chunks = select_chunks(_fault("confirmed"), _SELECT_CORPUS, _INDEX, embed_fn, _PREDICT)

    assert _names(chunks) == ["공구 마모", "스핀들 베어링 손상", "크레이터 마모", "플랭크 마모", "안전 1"]
    assert len(calls) == 1


def test_select_chunks_composite_adds_other_group_and_general_without_embedding():
    def embed_fn(_text):
        raise AssertionError("임베딩을 부르면 안 됨")

    other = {"situation": "이송축 과부하", "category": "feed_overload", "coverage": 0.44}
    chunks = select_chunks(_fault("composite", other=other), _SELECT_CORPUS, None, embed_fn, _PREDICT)

    assert _names(chunks) == ["공구 마모", "이송축 과부하", "온도 드리프트", "소재 변경", "안전 1"]


def test_select_chunks_weak_and_unknown():
    def embed_fn(_text):
        raise AssertionError("임베딩을 부르면 안 됨")

    weak = select_chunks(_fault("weak"), _SELECT_CORPUS, None, embed_fn, _PREDICT)
    assert _names(weak) == ["공구 마모", "온도 드리프트", "소재 변경", "안전 1"]

    unknown = select_chunks(_fault("unknown", situation=None, category=None, alternatives=()),
                            _SELECT_CORPUS, None, embed_fn, _PREDICT)
    assert _names(unknown) == ["온도 드리프트", "소재 변경", "안전 1"]


def test_build_guide_passes_fault_to_generation(monkeypatch):
    import rag.guide as guide_module

    captured = {}

    def fake_generate(predict_result, chunks, client, fault=None):
        captured["fault"] = fault
        captured["names"] = _names(chunks)
        return {"cause_estimate": "ok"}

    monkeypatch.setattr(guide_module, "generate_guide", fake_generate)
    monkeypatch.setattr(guide_module, "embed_text", lambda client, text: _VECTORS[6])

    result = build_guide(
        {"predicted_label_text": "bad", **_PREDICT, "fault": _fault("confirmed")},
        rag_corpus=_SELECT_CORPUS, rag_index=_INDEX, openai_client=object(),
    )

    assert result == {"cause_estimate": "ok"}
    assert captured["fault"]["situation"] == "공구 마모"
    assert captured["names"][0] == "공구 마모"


def test_build_guide_without_fault_keeps_top3_search_path(monkeypatch):
    import rag.guide as guide_module

    captured = {}

    def fake_generate(predict_result, chunks, client, fault=None):
        captured["fault"] = fault
        captured["n"] = len(chunks)
        return {"cause_estimate": "ok"}

    monkeypatch.setattr(guide_module, "generate_guide", fake_generate)
    monkeypatch.setattr(guide_module, "embed_text", lambda client, text: _VECTORS[6])

    build_guide({"predicted_label_text": "bad", **_PREDICT},
                rag_corpus=_SELECT_CORPUS, rag_index=_INDEX, openai_client=object())

    assert captured["fault"] is None
    assert captured["n"] == 3
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/rag/test_guide.py -q`
Expected: FAIL — `ImportError: cannot import name 'select_chunks'`

- [ ] **Step 3: 구현**

`src/rag/guide.py`의 import를 다음으로 바꾸고:

```python
from .generation import generate_cause_guide, generate_guide
from .playbook import PLAYBOOK_SOURCE
from .query import build_query
from .retrieval import embed_text, search

SAFETY_CHUNKS = 3
EXTERNAL_CHUNKS = 2  # 확정 판정에서 같은 카테고리의 비플레이북 cause 청크 수
```

`GOOD_GUIDE` 아래에 `select_chunks`를 추가:

```python
def select_chunks(fault: dict, corpus: list[dict], index, embed_fn, predict_result: dict) -> list[dict]:
    """판정별 참고 문서. 임베딩은 확정 판정에서 Sandvik 등 외부 청크를 고를 때 한 번만 쓴다."""
    playbook = {c["name"]: c for c in corpus if c.get("source") == PLAYBOOK_SOURCE}
    general = [c for c in playbook.values() if c["fault_category"] == "general"]
    safety = [c for c in corpus if c["content_type"] == "safety"][:SAFETY_CHUNKS]
    verdict = fault["verdict"]

    if verdict == "unknown":
        return general + safety
    selected = playbook[fault["situation"]]
    if verdict == "weak":
        return [selected] + general + safety
    if verdict == "composite":
        other = [playbook[fault["other_group"]["situation"]]] if fault["other_group"] else []
        return [selected] + other + general + safety

    alternatives = [playbook[name] for name in fault["alternatives"]]
    hits = search(build_query(predict_result["feature_contributions"]), corpus, index, embed_fn, top_k=len(corpus))
    external = [
        h for h in hits
        if h.get("source") != PLAYBOOK_SOURCE
        and h["fault_category"] == fault["category"]
        and h["content_type"] == "cause"
    ][:EXTERNAL_CHUNKS]
    return [selected] + alternatives + external + safety
```

`build_guide`의 `try:` 블록을 교체:

```python
    try:
        embed_fn = lambda text: embed_text(openai_client, text)  # noqa: E731
        fault = predict_result.get("fault")
        if fault is None:
            chunks = search(
                build_query(predict_result["feature_contributions"]), rag_corpus, rag_index, embed_fn
            )
        else:
            chunks = select_chunks(fault, rag_corpus, rag_index, embed_fn, predict_result)
        return generate_guide(predict_result, chunks, openai_client, fault)
    except Exception as exc:
        print(f"RAG 가이드 생성 실패: {exc}")
        return None
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/rag/ -q`
Expected: 모두 passed (기존 guide 6 + 5 신규 포함)

- [ ] **Step 5: 커밋**

```bash
git add src/rag/guide.py tests/rag/test_guide.py
git commit -m "feat(rag): choose guide chunks by verdict; embed only for confirmed external chunks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 6: `/predict` 응답의 `fault` 필드와 문서

**Files:**
- Modify: `src/serving/inference.py:1-10` (import), `:88-114` (`fault` 계산·반환)
- Modify: `README.md:88-94` (§2-1), `:143-157` (§2-4); `docs/STRUCTURE.md:31`
- Test: `tests/serving/test_inference.py`, `tests/serving/test_app.py`

**Interfaces:**
- Consumes: `rag.playbook.NO_FAULT`, `match_playbook`; `build_guide`가 `predict_result["fault"]`를 읽음.
- Produces: `predict_experiment` 반환에 `"fault": dict | None` 추가. good → `NO_FAULT` 사본, bad + 코퍼스 있음 → `match_playbook` 결과, bad + 코퍼스 없음 → `None`.

- [ ] **Step 1: 테스트 작성**

`tests/serving/test_inference.py` 끝에 추가 (파일의 `FEATURE_COLUMNS = ["f0", "f1", "f2"]`, `_scaler_dict`, `_feature_baseline`, `_raw_df` 재사용):

```python
_PLAYBOOK_CORPUS = [
    {"name": "f0 상황", "heading": "f0 상황 — x", "text": "관련 센서: f0", "fault_category": "tool_wear",
     "content_type": "cause", "source": "playbook", "signature": ["f0", "f1", "f2"],
     "title": "팀 시나리오 플레이북(자체 작성)", "url": "rag/sources/scenario_playbook.md"},
]


def _predict(df, threshold, rag_corpus=None):
    np.random.seed(0)
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=3, hidden_size=4, latent_dim=2)
    return predict_experiment(
        df=df, model=model, feature_columns=FEATURE_COLUMNS, scaler_dict=_scaler_dict(),
        window_size=6, threshold=threshold, method="mean", feature_baseline=_feature_baseline(),
        rag_corpus=rag_corpus,
    )


def test_predict_experiment_good_returns_no_fault():
    result = _predict(_raw_df(20), threshold=1e9)

    assert result["predicted_label_text"] == "good"
    assert result["fault"]["verdict"] == "none"
    assert result["fault"]["verdict_ko"] == "이상 없음"


def test_predict_experiment_bad_without_corpus_has_null_fault():
    result = _predict(_raw_df(20), threshold=-1.0)

    assert result["predicted_label_text"] == "bad"
    assert result["fault"] is None
    assert result["guide"] is None


def test_predict_experiment_bad_with_playbook_corpus_matches_situation():
    result = _predict(_raw_df(20), threshold=-1.0, rag_corpus=_PLAYBOOK_CORPUS)

    assert result["fault"]["situation"] == "f0 상황"
    assert result["fault"]["verdict"] in {"confirmed", "weak"}   # z 크기는 난수에 달림
    assert result["fault"]["coverage"] == 1.0
    assert result["guide"] is None                                # 인덱스·클라이언트 없음
```

`tests/serving/test_app.py`의 `test_predict_response_includes_guide_field` 바로 뒤에:

```python
def test_predict_response_includes_fault_field(tmp_path, monkeypatch):
    import serving.app as app_module

    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "requests.db")
    np.random.seed(0)
    app.dependency_overrides[get_model_state] = lambda: _fake_state(window_size=6)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("experiment.csv", io.BytesIO(_raw_csv_bytes(20)), "text/csv")},
    )

    body = response.json()
    assert "fault" in body
    if body["predicted_label_text"] == "good":
        assert body["fault"]["verdict"] == "none"
    else:
        assert body["fault"] is None  # _fake_state는 rag_corpus를 안 채움
    assert {"predicted_label", "predicted_label_text", "score", "threshold", "method",
            "feature_contributions", "model_version", "mlflow_run_id", "guide"} <= set(body)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/serving/ -q`
Expected: FAIL — `KeyError: 'fault'`

- [ ] **Step 3: 구현**

`src/serving/inference.py` import에 `from rag.playbook import NO_FAULT, match_playbook` 추가(`from rag.guide import build_guide` 아래). `predict_experiment`에서 `guide = build_guide(...)` 호출을 다음으로 교체:

```python
    if predicted_label_text == "good":
        fault = dict(NO_FAULT)
    elif rag_corpus:
        fault = match_playbook(feature_contributions, rag_corpus)
    else:
        fault = None

    guide = build_guide(
        {
            "predicted_label_text": predicted_label_text,
            "score": score,
            "threshold": threshold,
            "feature_contributions": feature_contributions,
            "fault": fault,
        },
        rag_corpus,
        rag_index,
        openai_client,
    )
```

반환 dict의 `"guide": guide,` 앞에 `"fault": fault,` 추가.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest -q`
Expected: 모두 passed (169 + 17 + 3 + 5 + 4 = 198)

- [ ] **Step 5: README·STRUCTURE 갱신**

`README.md` §2-1의 `/predict` 항목(`+ \`guide\`: ...` 문단) 뒤에 한 항목 추가:

```markdown
  + `fault`: 불량 판정일 때 플레이북(`rag/sources/scenario_playbook.md`, 팀 작성
  상황 16개)의 센서 서명과 상위 피처를 대조해 고른 상황(`situation`, `category`)과
  판정(`verdict`: `confirmed` 확정 / `composite` 복합 징후 / `weak` 약한 신호 /
  `unknown` 판단 불가, 정상이면 `none`), 근거(`coverage`, `matched_features`,
  `alternatives`, `other_group`, `top_z`). 임베딩·LLM 없이 계산하므로 키 없이도 나오고
  같은 입력이면 같은 값. 코퍼스 미구축이면 `null`
```

§2-4 첫 문단의 소스 나열에 플레이북을 더한다 — "KAMP 데이터셋 가이드북 발췌" 뒤에:

```markdown
, 그리고 팀이 시나리오에 맞춰 직접 쓴 상황 플레이북 `rag/sources/scenario_playbook.md`
(외부 자료 아님 — 출처에 "자체 작성"으로 표시됨. 항목마다 `관련 센서:` 줄의 피처 코드가
`fault` 판정의 근거이며, 빌드 시 코드가 `FEATURE_COLUMNS`에 있는지 검증)
```

그리고 "청크로 쪼개" 앞 문장이 자연스럽게 이어지도록 문장을 정리한다. 마지막 문단의 "코퍼스 소스 문서(`rag/sources/*.md`)를 바꾸지 않는 한" 뒤에 "(플레이북 항목을 더하거나 고치면 다시 돌린다)"를 추가.

`docs/STRUCTURE.md` 31행의 `src/rag/`, `rag/` 설명을:

```markdown
| 서빙 | `src/rag/`, `rag/` | 조치 가이드 생성. 안쪽이 라이브러리(플레이북 서명 대조 포함), 바깥쪽이 코퍼스 구축 스크립트와 원문(`rag/sources/`, 팀 작성 플레이북 포함) |
```

- [ ] **Step 6: 커밋**

```bash
git add src/serving/inference.py tests/serving/test_inference.py tests/serving/test_app.py README.md docs/STRUCTURE.md
git commit -m "feat(serving): add deterministic fault verdict to /predict; document playbook

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 7: 오프라인 채점 `rag/eval_playbook.py`

**Files:**
- Create: `rag/eval_playbook.py`
- Modify: `docs/specs/2026-09-03-cnc-playbook-guide-design.md` (끝에 "실행 결과에 따른 정정" 절)

**Interfaces:**
- Consumes: `data/rag/corpus.json`(Task 3), `predict_experiment(..., rag_corpus=corpus)`의 `fault`(Task 6), `monitoring/simulate_timeline.py`의 `generate_batch`, `true_label`, `BATCHES_PER_DAY`; `serving.app._build_model_state`.
- Produces: `data/rag/eval_playbook.json`, 표준 출력 표.

- [ ] **Step 1: 스크립트 작성**

```python
"""플레이북 서명 대조 오프라인 채점 — LLM 호출 없음, 임베딩 없음.

1) 기록된 4건(합성 3 + experiment_07)의 상황·판정
2) 타임라인 3종 × N일 × 5배치를 champion으로 추론해 구간별 판정 분포
결과는 data/rag/eval_playbook.json 과 표준 출력. 서버·DB는 건드리지 않는다.

  nice -n 19 uv run python rag/eval_playbook.py [--scenarios temperature tool_wear] [--days 40]
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

    OUT_PATH.write_text(json.dumps({"recorded": recorded, "timeline": timeline}, ensure_ascii=False, indent=1))
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 (공유 서버 확인 후)**

Run: `uptime` 확인 뒤 `nice -n 19 uv run python rag/eval_playbook.py`
Expected (스펙의 표와 같아야 함): 기록 4건 모두 `확정`, 상황이 공구 마모 / 이송축 과부하 / 고정구 풀림·채터 / 이송축 과부하. 타임라인 — tool_wear Day 21-40 확정 공구 마모 93건·다른 판정 0, fixture_loosening Day 21-40 확정 고정구 풀림·채터 77건, temperature Day 21-40 복합 징후·약한 신호가 다수(확정은 10건 안팎), Day 1-10은 세 시나리오 모두 약한 신호 6건. 숫자가 다르면 파서의 서명(엔코더·공구 파손 항목이 실험 때와 다름)이 원인일 수 있으니 실제 값을 정정 절에 그대로 적는다.

- [ ] **Step 3: 스펙에 정정 절 추가**

`docs/specs/2026-09-03-cnc-playbook-guide-design.md` 끝에:

```markdown
## 실행 결과에 따른 정정 (2026-09-03, 구현 후)

`rag/eval_playbook.py` 결과(`data/rag/eval_playbook.json`). 기록 4건: <상황·판정 4줄>.

| 시나리오 | 구간 | 불량 판정 | 확정 | 복합 징후 | 약한 신호 | 확정 상황 |
|---|---|---|---|---|---|---|
| tool_wear | Day 21-40 | | | | | |
| fixture_loosening | Day 21-40 | | | | | |
| temperature | Day 21-40 | | | | | |
| 셋 공통 | Day 1-10 | | | | | |

배경 절의 실험 표와 다른 점: <있으면 적고, 없으면 "없음">.
```

표는 실제 출력값으로 채운다.

- [ ] **Step 4: 커밋**

```bash
git add rag/eval_playbook.py docs/specs/2026-09-03-cnc-playbook-guide-design.md
git commit -m "feat(rag): offline playbook scoring over recorded cases and timeline scenarios

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
```

---

### Task 8: 라이브 검증과 응답 예시

**Files:**
- Modify: `docs/examples/predict_response_experiment_07.json`, `docs/examples/predict_response_experiment_12.json`
- Create: `docs/examples/predict_response_synthetic_tool_wear.json`, `..._feed_overload.json`, `..._vibration_backlash.json`
- Modify: `docs/specs/2026-09-03-cnc-playbook-guide-design.md` (정정 절에 라이브 표)

- [ ] **Step 1: 서버 기동 (키 있음)**

Run:
```bash
uptime
nice -n 19 uv run --env-file .env uvicorn serving.app:app --port 8899 > /home/sure/.claude/jobs/6b3a648e/tmp/serve_playbook.log 2>&1 &
echo $! > /home/sure/.claude/jobs/6b3a648e/tmp/serve_playbook.pid
until curl -sf http://127.0.0.1:8899/health > /dev/null; do sleep 1; done; curl -s http://127.0.0.1:8899/health
```
Expected: `{"status": "ok", "model_version": "1", ...}`

- [ ] **Step 2: 5건 요청 저장**

Run:
```bash
E="data/dataset/CNC 비식별화 원본데이터_1209/CNC Virtual Data set _v2"
for s in tool_wear feed_overload vibration_backlash; do
  curl -s -X POST http://127.0.0.1:8899/predict -F "file=@synthetic/scenarios/$s.csv" | python3 -m json.tool --no-ensure-ascii > docs/examples/predict_response_synthetic_$s.json
done
curl -s -X POST http://127.0.0.1:8899/predict -F "file=@$E/experiment_07.csv" | python3 -m json.tool --no-ensure-ascii > docs/examples/predict_response_experiment_07.json
curl -s -X POST http://127.0.0.1:8899/predict -F "file=@$E/experiment_12.csv" | python3 -m json.tool --no-ensure-ascii > docs/examples/predict_response_experiment_12.json
for f in docs/examples/predict_response_*.json; do python3 -c "
import json,sys; d=json.load(open('$f')); f=d['fault']; g=d['guide']
print('$f'.split('/')[-1], d['predicted_label_text'], f['verdict'], f['situation'], '|', (g or {}).get('cause_estimate'))"; done
```
Expected: 합성 3건 + experiment_07이 `confirmed`이고 상황이 공구 마모 / 이송축 과부하 / 고정구 풀림·채터 / 이송축 과부하, `cause_estimate`에 다른 상황이 섞이지 않음. experiment_12는 `good`, `none`, `이상 없음`. 서버 로그에 `RAG 가이드 생성 실패`가 없음(`grep -c "가이드 생성 실패" .../serve_playbook.log` → 0).

- [ ] **Step 3: 서버 종료, 키 없이 재기동해 폴백 확인**

Run:
```bash
kill $(cat /home/sure/.claude/jobs/6b3a648e/tmp/serve_playbook.pid); sleep 2
nice -n 19 uv run uvicorn serving.app:app --port 8899 > /home/sure/.claude/jobs/6b3a648e/tmp/serve_nokey.log 2>&1 &
echo $! > /home/sure/.claude/jobs/6b3a648e/tmp/serve_playbook.pid
until curl -sf http://127.0.0.1:8899/health > /dev/null; do sleep 1; done
curl -s -X POST http://127.0.0.1:8899/predict -F "file=@synthetic/scenarios/tool_wear.csv" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['fault']['verdict'], d['fault']['situation'], d['guide'])"
kill $(cat /home/sure/.claude/jobs/6b3a648e/tmp/serve_playbook.pid)
```
Expected: `confirmed 공구 마모 None` — `fault`는 나오고 `guide`만 `None`. (환경에 `OPENAI_API_KEY`가 이미 export돼 있으면 `env -u OPENAI_API_KEY`를 앞에 붙인다.)

- [ ] **Step 4: 라이브 DB 정리**

Run: `rm -f data/monitoring/requests.db` (라이브 요청 5건이 드리프트 감시 창에 남지 않도록. 시나리오 재현 전 정리 규칙과 같음.)

- [ ] **Step 5: 스펙 정정 절에 라이브 표 추가**

Task 7에서 만든 절 아래에:

```markdown
### 라이브 `/predict` (키 있음)

| 입력 | 판정 | verdict | situation | cause_estimate 요지 | 출처 |
|---|---|---|---|---|---|
| synthetic tool_wear | bad | confirmed | 공구 마모 | | 플레이북 |
| synthetic feed_overload | bad | confirmed | 이송축 과부하 | | |
| synthetic vibration_backlash | bad | confirmed | 고정구 풀림·채터 | | |
| experiment_07 | bad | confirmed | 이송축 과부하 | | |
| experiment_12 | good | none | - | 이상 없음 | - |

키 없이: `fault` 그대로, `guide` null 확인.
```

실제 값으로 채운다.

- [ ] **Step 6: 전체 테스트, 커밋, push**

Run: `uv run pytest -q` → 198 passed. `git status`로 `data/`가 안 잡히는지 확인.

```bash
git add docs/examples/ docs/specs/2026-09-03-cnc-playbook-guide-design.md
git commit -m "docs: record live /predict examples with playbook fault verdicts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QxcZBHAwZBZTz9tnLxoGwz"
git push origin main
```

---

## Self-Review

- **스펙 커버리지**: Part A 문서·파서(Task 1)·빌드(Task 3), Part B 대조·상수·`NO_FAULT`(Task 2), Part C 청크 선택(Task 5)·프롬프트(Task 4)·응답 `fault`(Task 6), 검증 1~5(Task 3 Step 5, Task 7, Task 8 Step 2~3, 회귀는 각 태스크), 문서 갱신(Task 6 Step 5), 알려진 한계·정정 절(Task 7·8). 거부 경로는 코드 변경 없음(스펙대로).
- **플레이스홀더**: 정정 절의 표는 실행값으로 채우는 칸이며 값 자체는 실행이 정한다. 그 외 TBD 없음.
- **타입 일관성**: `match_playbook` 반환 키 9개 = `NO_FAULT` 키 = `describe_fault`·`select_chunks`가 읽는 키. `generate_guide(..., fault=None)`을 `build_guide`가 위치 인자로 넘김. `select_chunks(fault, corpus, index, embed_fn, predict_result)` 순서를 테스트·구현·`build_guide`가 동일하게 씀. 청크 `name`은 플레이북에만 있고, 비플레이북 청크는 `heading`으로 식별(테스트 `_names`).
