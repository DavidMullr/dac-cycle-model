(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.MofJobReattach = api;
}(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';

  const ACTIVE_STATES = new Set(['queued', 'running', 'cancelling']);

  function activeJob(statusDocument) {
    const jobs = activeJobs(statusDocument);
    if (jobs.length) return jobs[jobs.length - 1];
    const current = statusDocument?.current;
    if (!current || !ACTIVE_STATES.has(current.state)) return null;
    return current;
  }

  function activeJobs(statusDocument) {
    if (Array.isArray(statusDocument?.active_jobs)) {
      return statusDocument.active_jobs.filter(job => job && ACTIVE_STATES.has(job.state));
    }
    const current = statusDocument?.current;
    return current && ACTIVE_STATES.has(current.state) ? [current] : [];
  }

  function hasCapacity(concurrency, pendingStarts = 0) {
    const limit = concurrency?.max_concurrent_jobs ?? 1;
    return limit === 0 || (concurrency?.active_count ?? 0) + pendingStarts < limit;
  }

  function applyHistorySelection(state, jobId, inspection) {
    if (activeJob({ current: inspection })) {
      state.currentJobId = jobId;
      state.activeJob = true;
      return { action: 'reattach', job: inspection };
    }
    if (inspection?.state === 'completed' && inspection.result) {
      state.result = inspection.result;
      return { action: 'display', result: inspection.result };
    }
    state.displayedError = inspection?.message
      || inspection?.failure?.message
      || 'Saved job has no completed result.';
    return { action: 'error', message: state.displayedError };
  }

  function cancelTarget(state) {
    return state?.activeJob && state?.currentJobId
      ? state.currentJobId
      : null;
  }

  async function fetchHistoryAndActive(api) {
    const [history, status] = await Promise.all([
      api('/api/mof-screening/history'),
      api('/api/mof-screening/status')
    ]);
    return {
      jobs: history?.jobs || [],
      active: activeJob(status),
      activeJobs: activeJobs(status),
      concurrency: status?.concurrency
    };
  }

  return {
    ACTIVE_STATES,
    activeJob,
    activeJobs,
    hasCapacity,
    applyHistorySelection,
    cancelTarget,
    fetchHistoryAndActive
  };
}));
