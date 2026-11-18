'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const reattach = require(path.join(
  __dirname,
  '..',
  'app',
  'mof_screening',
  'static',
  'job_reattach.js'
));

assert.equal(reattach.activeJob({
  current: { job_id: 'a'.repeat(32), state: 'runs' }
}), null);
assert.equal(reattach.activeJob({
  current: { job_id: 'a'.repeat(32), state: 'completed' }
}), null);
const active = reattach.activeJob({
  current: { job_id: 'a'.repeat(32), state: 'running' }
});
assert.equal(active.state, 'running');
assert.equal(reattach.hasCapacity({max_concurrent_jobs: 0, active_count: 100}), true);
assert.equal(reattach.hasCapacity({max_concurrent_jobs: 3, active_count: 3}), false);
assert.equal(reattach.hasCapacity({max_concurrent_jobs: 3, active_count: 2}), true);
assert.equal(reattach.hasCapacity({max_concurrent_jobs: 3, active_count: 2}, 1), false);
assert.equal(reattach.hasCapacity({max_concurrent_jobs: 3, active_count: 1}, 1), true);
assert.equal(reattach.hasCapacity({max_concurrent_jobs: 0, active_count: 100}, 100), true);
assert.deepEqual(reattach.activeJobs({active_jobs: [active, {job_id: 'b'.repeat(32), state: 'completed'}]}), [active]);

const appState = {
  currentJobId: active.job_id,
  activeJob: true,
  result: null,
  displayedError: null
};
const completedSelection = reattach.applyHistorySelection(
  appState,
  'b'.repeat(32),
  { state: 'completed', result: { job_id: 'b'.repeat(32) } }
);
assert.equal(completedSelection.action, 'display');
assert.equal(appState.result.job_id, 'b'.repeat(32));
assert.equal(appState.currentJobId, active.job_id);
assert.equal(appState.activeJob, true);
assert.equal(reattach.cancelTarget(appState), active.job_id);

const failedSelection = reattach.applyHistorySelection(
  appState,
  'c'.repeat(32),
  { state: 'failed', message: 'historical failure' }
);
assert.equal(failedSelection.action, 'error');
assert.equal(appState.displayedError, 'historical failure');
assert.equal(appState.currentJobId, active.job_id);
assert.equal(appState.activeJob, true);
assert.equal(reattach.cancelTarget(appState), active.job_id);

(async () => {
  const calls = [];
  const payload = await reattach.fetchHistoryAndActive(async url => {
    calls.push(url);
    if (url.endsWith('/history')) return { jobs: [{ job_id: active.job_id }] };
    if (url.endsWith('/status')) return { active: true, current: active };
    throw new Error(`unexpected ${url}`);
  });
  assert.deepEqual(calls.sort(), [
    '/api/mof-screening/history',
    '/api/mof-screening/status'
  ]);
  assert.equal(payload.jobs.length, 1);
  assert.equal(payload.active.job_id, active.job_id);
  console.log('job reattach tests passed');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
