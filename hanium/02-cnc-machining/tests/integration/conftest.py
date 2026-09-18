"""루프 통합 테스트 하네스.

서버(uvicorn)와 워커(monitoring/drift_worker.py)는 실제 서브프로세스로 띄운다 — 지금까지 찾은
루프 버그(날짜 스킵, 섀도우 시작일, feeder 페이스, 승격 직후 재트리거)가 전부 프로세스 경계에서
나왔기 때문이다. feeder(monitoring/simulate_timeline.py)는 테스트가 하루씩 실행하고, 워커 로그에
그 날이 찍힌 뒤에야 다음 날을 보낸다. 재학습에 몇 초가 걸리든 결과가 타이밍에 의존하지 않는다.
"""

import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib 모드라 형제 모듈이 경로에 없다

from fixture_dataset import write_dataset  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]  # 02-cnc-machining/
SCRIPTS = PROJECT / "scripts"
MONITORING = PROJECT / "monitoring"
DATASET_DIRNAME = "CNC 비식별화 원본데이터_1209"  # config.DATASET_DIR 의 마지막 요소

# 스펙 §3 "루프 상수". 40일 루프를 며칠로 줄인다. DRIFT_START_DAY=2, TOTAL_DAYS=3, 상한 1.0 이라
# Day 3 부터 변형이 계단으로 고정된다. TOTAL_DAYS 는 램프 기울기에만 쓰이고 며칠을 보낼지는
# 테스트가 정한다.
LOOP_ENV = {
    "CNC_BATCHES_PER_DAY": "5",
    "CNC_DRIFT_WINDOW_SIZE": "5",
    "CNC_CONSECUTIVE_K": "3",
    "CNC_COOLDOWN_DAYS": "2",
    "CNC_GATE_SAMPLE_SIZE": "5",
    "CNC_LABEL_DELAY_DAYS": "1",
    "CNC_DRIFT_START_DAY": "2",
    "CNC_TOTAL_DAYS": "3",
    "CNC_DRIFT_MAX_PROGRESS": "1.0",
    "CNC_LABEL_FLIP_DAY": "3",
    "CNC_TRAIN_EPOCHS": "2",
}
DAY_LINE = re.compile(r"^Day (\d{2})\s+score/threshold=([\d.]+)\s+flagged=(True|False)\s+action=(\w+)")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _decoded(output: str | bytes | None) -> str:
    """TimeoutExpired 의 stdout/stderr 는 text=True 여도 bytes 이거나 None 이다."""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output or ""


def run(cmd: list[str], env: dict, timeout: int = 600) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"{timeout}초 초과: {' '.join(cmd)}\n--- stdout\n{_decoded(exc.stdout)[-4000:]}"
            f"\n--- stderr\n{_decoded(exc.stderr)[-4000:]}"
        ) from exc
    if result.returncode != 0:
        raise AssertionError(
            f"실패: {' '.join(cmd)}\n--- stdout\n{result.stdout[-4000:]}\n--- stderr\n{result.stderr[-4000:]}"
        )
    return result


def bootstrap(data_root: Path, extra_env: dict | None = None) -> dict:
    """합성 데이터셋 → 전처리 → 학습 → champion v1. 기존 스크립트 3개를 그대로 쓴다."""
    # 개발자 셸의 CNC_* 가 LOOP_ENV·extra_env 에 없는 상수로 섞여 들지 않게 먼저 걷어 낸다.
    env = {key: value for key, value in os.environ.items() if not key.startswith("CNC_")}
    env.update({**LOOP_ENV, "CNC_DATA_ROOT": str(data_root), "PYTHONUNBUFFERED": "1"})
    env.pop("OPENAI_API_KEY", None)  # 셸에 키가 있어도 통합 테스트는 LLM 을 부르지 않는다
    env.update(extra_env or {})
    write_dataset(data_root / "dataset" / DATASET_DIRNAME)
    run([sys.executable, str(SCRIPTS / "run_preprocessing.py")], env)
    run([sys.executable, str(SCRIPTS / "run_lstm_training.py")], env)
    run([sys.executable, str(SCRIPTS / "promote_model.py"), "1"], env)
    return env


@dataclass
class Loop:
    scenario: str
    data_root: Path
    env: dict
    port: int = field(default_factory=free_port)
    server: subprocess.Popen | None = None
    worker: subprocess.Popen | None = None
    _handles: list = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def server_log(self) -> Path:
        return self.data_root / "server.log"

    @property
    def worker_log(self) -> Path:
        return self.data_root / "worker.log"

    def _open(self, path: Path):
        handle = path.open("w")
        self._handles.append(handle)
        return handle

    def start_server(self) -> None:
        self.server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "serving.app:app", "--port", str(self.port)],
            cwd=PROJECT, env=self.env, stdout=self._open(self.server_log), stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if httpx2.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return
            except httpx2.HTTPError:
                pass
            if self.server.poll() is not None:
                break
            time.sleep(0.5)
        raise AssertionError(
            f"서버가 /health 200 을 내지 않음\n{self.server_log.read_text(errors='replace')[-4000:]}"
        )

    def start_worker(self) -> None:
        self.worker = subprocess.Popen(
            [
                sys.executable, str(MONITORING / "drift_worker.py"), self.scenario,
                "--base-url", self.base_url, "--poll-interval", "0.2",
            ],
            cwd=PROJECT, env=self.env, stdout=self._open(self.worker_log), stderr=subprocess.STDOUT,
        )

    def feed_day(self, day: int) -> None:
        run(
            [
                sys.executable, str(MONITORING / "simulate_timeline.py"), self.scenario,
                "--serve-url", self.base_url, "--start-day", str(day), "--days", str(day),
            ],
            self.env,
        )

    def days(self) -> list[tuple[int, float, bool, str]]:
        out = []
        for line in self.worker_log.read_text(errors="replace").splitlines():
            m = DAY_LINE.match(line)
            if m:
                out.append((int(m[1]), float(m[2]), m[3] == "True", m[4]))
        return out

    def wait_for_day(self, day: int, timeout: int = 180) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(d == day for d, *_ in self.days()):
                return
            if self.worker.poll() is not None:
                raise AssertionError(
                    f"워커가 종료됨 (exit {self.worker.returncode})\n"
                    f"{self.worker_log.read_text(errors='replace')[-4000:]}"
                )
            time.sleep(0.2)
        raise AssertionError(
            f"워커가 {timeout}초 안에 Day {day:02d} 를 처리하지 않음\n"
            f"{self.worker_log.read_text(errors='replace')[-4000:]}"
        )

    def run_days(self, days: int) -> None:
        for day in range(1, days + 1):
            self.feed_day(day)
            self.wait_for_day(day)

    def health(self) -> dict:
        return httpx2.get(f"{self.base_url}/health", timeout=5).json()

    def stop(self) -> None:
        for proc in (self.worker, self.server):
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for handle in self._handles:
            handle.close()


@pytest.fixture
def loop_factory(tmp_path):
    loops: list[Loop] = []

    def make(scenario: str, extra_env: dict | None = None) -> Loop:
        data_root = tmp_path / scenario
        env = bootstrap(data_root, extra_env)
        loop = Loop(scenario=scenario, data_root=data_root, env=env)
        loops.append(loop)
        loop.start_server()
        loop.start_worker()
        return loop

    yield make
    for loop in loops:
        loop.stop()
