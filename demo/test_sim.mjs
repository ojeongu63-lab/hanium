// node demo/test_sim.mjs — 재생 엔진 단위 테스트 (pytest와 별개, 브라우저 없이 실행)
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const E = require('./sim_engine.js');

const day = (d, ratios, preds) => ({
  day: d, truth: 'good', ratio_mean: ratios.reduce((a, b) => a + b, 0) / ratios.length,
  batches: ratios.map((r, i) => ({ index: i, ratio: r, pred: preds[i] })),
});
const scenario = {
  days_total: 4,
  days: [day(1, [0.7, 0.8], ['good', 'good']), day(2, [1.1, 1.2], ['good', 'bad']), day(3, [1.3, 1.4], ['bad', 'bad']), day(4, [0.9, 0.9], ['good', 'good'])],
  events: [{ day: 3, kind: 'trigger', text: 't' }, { day: 3, kind: 'rejected', text: 'r' }],
};

let s = { ...E.initialState(), playing: true };
let r = E.step(s, scenario);
assert.equal(r.state.day, 1); assert.equal(r.reached.day, 1); assert.equal(r.stoppedBy, null); assert.equal(r.state.playing, true);
r = E.step(r.state, scenario);
assert.equal(r.state.day, 2); assert.equal(r.state.playing, true);
r = E.step(r.state, scenario);                       // Day 3: 이벤트 → 자동 정지
assert.equal(r.state.day, 3); assert.equal(r.stoppedBy, 'event'); assert.equal(r.state.playing, false);
assert.deepEqual(E.eventsOn(scenario, 3).map((e) => e.kind), ['trigger', 'rejected']);
r = E.step({ ...r.state, playing: true }, scenario);  // Day 4: 마지막 → end
assert.equal(r.state.day, 4); assert.equal(r.stoppedBy, 'end'); assert.equal(r.state.playing, false);
r = E.step(r.state, scenario);                       // 끝에서 더 진행 안 함
assert.equal(r.state.day, 4); assert.equal(r.reached, null); assert.equal(r.stoppedBy, 'end');

// 자동 정지 끄면 이벤트 날에도 계속
let t = { ...E.initialState(false), playing: true, day: 2 };
t = E.step(t, scenario);
assert.equal(t.state.day, 3); assert.equal(t.stoppedBy, null); assert.equal(t.state.playing, true);

// 연속 감지 수와 첫 불량 배치
assert.equal(E.consecutiveFlagged(scenario, 3), 2);   // Day 2·3 배율 ≥ 1.0
assert.equal(E.consecutiveFlagged(scenario, 4), 0);
assert.equal(E.firstBadIndex(scenario.days[1]), 1);
assert.equal(E.firstBadIndex(scenario.days[0]), 0);
console.log('sim engine: ok');
