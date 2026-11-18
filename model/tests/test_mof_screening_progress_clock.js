'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const clockModule = require(path.join(
  __dirname,
  '..',
  'app',
  'mof_screening',
  'static',
  'progress_clock.js'
));

const clock = clockModule.createElapsedClock();
const jobId = 'a'.repeat(32);
const startedUtc = '2026-11-06T19:23:39.346Z';
const startedWallMs = Date.parse(startedUtc);
const staleSnapshot = {
  job_id: jobId,
  started_utc: startedUtc,
  elapsed_wall_s: 0.766
};
const wallAtAttach = startedWallMs + 26_000;
assert.equal(clock.observe(staleSnapshot, 1000, wallAtAttach), 26.0);
assert.equal(clock.observe(staleSnapshot, 1900, wallAtAttach + 900), 26.9);
assert.equal(clock.observe(staleSnapshot, 2800, wallAtAttach + 1800), 27.8);
assert.equal(
  clock.elapsedSeconds(staleSnapshot, true, 2800, wallAtAttach + 1800),
  0.766
);

const caughtUp = {
  job_id: jobId,
  started_utc: startedUtc,
  elapsed_wall_s: 29.0
};
assert.equal(clock.observe(caughtUp, 3000, wallAtAttach + 2900), 29.0);
assert.equal(clock.observe(caughtUp, 4000, wallAtAttach + 3900), 30.0);

clock.reset(
  'b'.repeat(32),
  7.25,
  5000,
  '2026-11-06T19:00:00.000Z',
  Date.parse('2026-11-06T19:00:07.250Z')
);
assert.equal(
  clock.observe({
    job_id: 'b'.repeat(32),
    started_utc: '2026-11-06T19:00:00.000Z',
    elapsed_wall_s: 7.25
  }, 6000, Date.parse('2026-11-06T19:00:08.250Z')),
  8.25
);

console.log('progress clock tests passed');
