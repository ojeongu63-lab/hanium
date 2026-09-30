// 시뮬레이션 재생 엔진 — 순수 함수. 브라우저(템플릿에 인라인)와 node 테스트(demo/test_sim.mjs)가 같이 쓴다.
// 상태: {day, playing, pauseOnEvent, speed}. 하루가 한 틱이다.
const SimEngine = (() => {
  function initialState(pauseOnEvent = true, speed = 1) {
    return { day: 0, playing: false, pauseOnEvent, speed };
  }
  function eventsOn(scenario, day) {
    return (scenario.events || []).filter((e) => e.day === day);
  }
  // 한 틱 진행. 반환: {state, reached: 새 날의 데이터 | null, stoppedBy: 'end'|'event'|null}
  function step(state, scenario) {
    const total = scenario.days_total;
    if (state.day >= total) return { state: { ...state, playing: false }, reached: null, stoppedBy: 'end' };
    const day = state.day + 1;
    const reached = scenario.days[day - 1];
    let playing = state.playing, stoppedBy = null;
    if (day >= total) { playing = false; stoppedBy = 'end'; }
    else if (state.pauseOnEvent && eventsOn(scenario, day).length) { playing = false; stoppedBy = 'event'; }
    return { state: { ...state, day, playing }, reached, stoppedBy };
  }
  // 드리프트 감시 안내용: 현재 날짜까지 일 평균 배율이 1.0 이상인 날의 연속 수(최대 3까지 표시)
  function consecutiveFlagged(scenario, day) {
    let n = 0;
    for (let d = day; d >= 1; d--) {
      if (scenario.days[d - 1].ratio_mean >= 1.0) n += 1; else break;
    }
    return n;
  }
  function firstBadIndex(dayData) {
    const i = dayData.batches.findIndex((b) => b.pred === 'bad');
    return i >= 0 ? i : 0;
  }
  return { initialState, step, eventsOn, consecutiveFlagged, firstBadIndex };
})();
if (typeof module !== 'undefined') module.exports = SimEngine;
