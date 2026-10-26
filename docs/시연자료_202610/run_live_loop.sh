#!/usr/bin/env bash
# 드리프트 트리거 재학습 루프 — 라이브 시연용 실행 도우미.
# README §2-7 의 3터미널(서빙·워커·feeder)을 그대로 쓰되,
#   (1) 정본 data/ 대신 복사본 데이터 루트(CNC_DATA_ROOT)에서 돌려 승격이 정본 모델·MLflow 를 바꾸지 않게 하고,
#   (2) 루프 상수(CNC_*)를 화면용으로 줄여 트리거 → 재학습 → 게이트 → 섀도우 → 승격을 몇 분 안에 본다.
# 판정 규칙(트리거·게이트·섀도우)은 그대로다 — 상수만 줄인 것이고, 전부 src/config.py 가 읽는 환경변수다.
#
#   ./run_live_loop.sh reset      # 복사본 데이터 루트를 정본에서 새로 만든다(이전 시연 흔적 제거). 시연 전 1회
#   ./run_live_loop.sh serving    # 터미널 1
#   ./run_live_loop.sh worker     # 터미널 2 (서빙이 뜬 뒤)
#   ./run_live_loop.sh feeder     # 터미널 3 (워커가 뜬 뒤)
#   ./run_live_loop.sh status     # /health 의 모델 버전 + MLflow champion alias
#   ./run_live_loop.sh mlflow-ui  # 복사본 레지스트리의 MLflow UI(:5000) — 승격 뒤 champion alias 가 바뀐 것을 화면에 보일 때
#   ./run_live_loop.sh rehearsal  # 셋을 백그라운드로 띄워 끝까지 돌리고 로그를 rehearsal/ 에 남긴다(리허설용)
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DEMO_DATA="${CNC_DEMO_DATA:-$HOME/project/hanium-demo-data}"
PORT="${CNC_DEMO_PORT:-8000}"
SCENARIO="${CNC_DEMO_SCENARIO:-temperature}"
DAYS="${CNC_DEMO_DAYS:-9}"
# pace(하루치 배치를 보낸 뒤 쉬는 시간)는 재학습(이 PC 에서 50 epoch 약 31초) + 게이트 + 섀도우 시작보다 길어야 한다.
# 짧으면 섀도우가 시작되는 순간 하루가 반쯤 전송된 상태라 그 날 배치 일부에 섀도우 기록이 없고, 워커의 섀도우 판정이
# "기준일 이후 첫 5건에 모두 섀도우 기록이 있어야" 넘어가므로 영원히 shadow_pending 에 머문다(리허설 1회차에서 확인).
PACE="${CNC_DEMO_PACE:-50}"
BASE="http://127.0.0.1:$PORT"

export CNC_DATA_ROOT="$DEMO_DATA"
# 화면용 단축 상수 — 괄호는 설계 기본값. 드리프트 창 5(10) · 연속 3(3) · 쿨다운 2(5) · 게이트 표본 5(20) ·
# 하루 5배치(5) · 변형 시작 3일째(10) · 변형이 최대에 닿는 날 4(40) · 변형 상한 1.0(없음) · 라벨 지연 1일(7).
# 재학습 epoch 는 설계값 50 그대로(실제 재학습). 기대 흐름: Day 4·5·6 flagged → Day 6 트리거·재학습·게이트·섀도우 시작
# → Day 7 배치가 섀도우로 기록 → Day 8 에 라벨 5건 도착 → 승격(v2) → Day 9 는 새 champion 으로 판정.
export CNC_DRIFT_WINDOW_SIZE="${CNC_DRIFT_WINDOW_SIZE:-5}" CNC_CONSECUTIVE_K="${CNC_CONSECUTIVE_K:-3}"
export CNC_COOLDOWN_DAYS="${CNC_COOLDOWN_DAYS:-2}" CNC_GATE_SAMPLE_SIZE="${CNC_GATE_SAMPLE_SIZE:-5}"
export CNC_BATCHES_PER_DAY="${CNC_BATCHES_PER_DAY:-5}" CNC_DRIFT_START_DAY="${CNC_DRIFT_START_DAY:-3}"
export CNC_TOTAL_DAYS="${CNC_TOTAL_DAYS:-4}" CNC_DRIFT_MAX_PROGRESS="${CNC_DRIFT_MAX_PROGRESS:-1.0}"
export CNC_LABEL_DELAY_DAYS="${CNC_LABEL_DELAY_DAYS:-1}" CNC_TRAIN_EPOCHS="${CNC_TRAIN_EPOCHS:-50}"

cd "$REPO"

champion_alias() {
  uv run --no-sync python - "$DEMO_DATA/mlflow/mlflow.db" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
rows = con.execute("select name, alias, version from registered_model_aliases").fetchall()
print("MLflow alias:", rows or "(없음)")
PY
}

case "${1:-}" in
  reset)
    rm -rf "$DEMO_DATA"
    cp -a "$REPO/data" "$DEMO_DATA"
    rm -f "$DEMO_DATA"/monitoring/labels.db "$DEMO_DATA"/monitoring/requests.db "$DEMO_DATA"/monitoring/shadow.db
    rm -rf "$DEMO_DATA"/timeline
    echo "복사본 데이터 루트 준비: $DEMO_DATA (정본 $REPO/data 는 건드리지 않음)"
    ;;
  serving) exec nice -n 19 uv run --no-sync uvicorn src.serving.app:app --app-dir . --port "$PORT" ;;
  worker)  exec nice -n 19 uv run --no-sync python monitoring/drift_worker.py "$SCENARIO" --base-url "$BASE" --poll-interval 2 ;;
  feeder)  exec nice -n 19 uv run --no-sync python monitoring/simulate_timeline.py "$SCENARIO" --days "$DAYS" --serve-url "$BASE" --pace-seconds "$PACE" ;;
  status)
    echo "/health: $(curl -s "$BASE/health" || echo '(서빙 없음)')"
    champion_alias
    ;;
  mlflow-ui)
    # 0.0.0.0 바인딩: WSL 의 127.0.0.1 은 이 PC 의 Windows 브라우저에서 안 보이므로 WSL IP 로 연다(LAN 에는 노출되지 않음).
    echo "Windows 브라우저: http://$(hostname -I | awk '{print $1}'):5000"
    exec uv run --no-sync mlflow ui --backend-store-uri "sqlite:///$DEMO_DATA/mlflow/mlflow.db" --host 0.0.0.0 --port 5000
    ;;
  rehearsal)
    LOGDIR="$HERE/rehearsal"; mkdir -p "$LOGDIR"
    "$0" reset
    setsid "$0" serving > "$LOGDIR/serving.log" 2>&1 & SERVING=$!
    WORKER=""
    # 자식은 setsid 로 띄워 각자 프로세스 그룹이므로 그룹째 내린다. bash 는 trap 없는 신호로 죽을 때 EXIT 트랩을 안 돌리므로
    # TERM/INT 를 잡아 exit 로 보내 EXIT 트랩이 돌게 한다(리허설 1회차에서 서빙·워커가 고아로 남은 원인).
    cleanup() { for p in $WORKER $SERVING; do [ -n "$p" ] && kill -TERM -- -"$p" 2>/dev/null || true; done; }
    trap cleanup EXIT
    trap 'exit 143' TERM INT
    for _ in $(seq 1 90); do curl -sf "$BASE/health" > /dev/null && break; sleep 2; done
    curl -sf "$BASE/health" > /dev/null || { echo "서빙이 뜨지 않음: $LOGDIR/serving.log"; exit 1; }
    echo "[$(date +%T)] 서빙 준비: $(curl -s "$BASE/health")"
    setsid "$0" worker > "$LOGDIR/worker.log" 2>&1 & WORKER=$!
    sleep 3
    T0=$(date +%s)
    echo "[$(date +%T)] feeder 시작 ($SCENARIO, ${DAYS}일, 하루 ${CNC_BATCHES_PER_DAY}배치, pace ${PACE}s)"
    "$0" feeder > "$LOGDIR/feeder.log" 2>&1 || echo "feeder 종료 코드 $? — $LOGDIR/feeder.log"
    echo "[$(date +%T)] feeder 종료 ($(( $(date +%s) - T0 ))초). 워커의 승격을 최대 600초 기다림…"
    for _ in $(seq 1 120); do
      grep -q 'promoted' "$LOGDIR/worker.log" && break
      kill -0 "$WORKER" 2>/dev/null || { echo "워커가 먼저 종료됨 — $LOGDIR/worker.log"; break; }
      sleep 5
    done
    echo "[$(date +%T)] 전체 소요 $(( $(date +%s) - T0 ))초"
    {
      echo "== 워커 로그 중 Day/이벤트 줄 =="
      # -a: 워커 로그에 학습 진행 출력의 NUL 바이트가 섞여 grep 이 바이너리로 취급한다
      grep -a -E '^Day |트리거|게이트|섀도우|승격|거부' "$LOGDIR/worker.log" | cut -c1-220 || true
      echo; echo "== 종료 시점 상태 =="; "$0" status
    } | tee "$LOGDIR/summary.txt"
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
