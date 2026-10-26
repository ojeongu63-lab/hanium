#!/usr/bin/env python3
"""demo/index.html(기록 재생 페이지)의 탭 1·2·3 을 1440×900 PNG 로 찍는다 — 보고서·발표용 캡처.

서버 없이 파일로 연다(README §2-9 "기록 모드"). 시뮬레이션 탭은 가장 빠른 속도로 몇 초 재생한 뒤 찍어
그래프·이벤트 로그가 채워진 상태를 담는다. Playwright(Chromium)가 필요하다.

사용: python3 capture_demo_page.py <출력 폴더> [index.html 경로]
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1]).resolve()
INDEX = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "demo/index.html"
OUT.mkdir(parents=True, exist_ok=True)


def shot(page, name, full_page=False):
    path = OUT / name
    page.screenshot(path=str(path), full_page=full_page)
    print(f"저장: {path}")


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_context(viewport={"width": 1440, "height": 900}, locale="ko-KR").new_page()
    page.goto(INDEX.as_uri())
    page.wait_for_load_state("networkidle")
    time.sleep(1)
    shot(page, "demo_1_batch.png")
    shot(page, "demo_1_batch_full.png", full_page=True)

    page.click('button[data-tab="timeline"]')
    time.sleep(1.5)
    shot(page, "demo_2_timeline.png")
    shot(page, "demo_2_timeline_full.png", full_page=True)

    page.click('button[data-tab="sim"]')
    time.sleep(1)
    shot(page, "demo_3_sim_start.png")
    options = page.eval_on_selector_all("#sim-speed option", "els => els.map(e => e.value)")
    if options:
        page.select_option("#sim-speed", min(options, key=float))  # 값이 "초/일"이라 가장 작은 값이 가장 빠르다
    page.click("#sim-play")
    time.sleep(12)
    shot(page, "demo_3_sim_running.png")
    shot(page, "demo_3_sim_running_full.png", full_page=True)
    browser.close()
