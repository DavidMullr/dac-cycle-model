'use strict';

(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.MofProgressClock = api;
}(typeof globalThis === 'object' ? globalThis : this, function () {
  function createElapsedClock() {
    let jobId = null;
    let baseSeconds = 0;
    let anchoredAtMs = null;

    function reset(
      nextJobId,
      serverElapsedSeconds,
      monotonicNowMs,
      startedUtc = null,
      wallNowMs = Date.now()
    ) {
      jobId = nextJobId || null;
      baseSeconds = Number.isFinite(Number(serverElapsedSeconds))
        ? Math.max(0, Number(serverElapsedSeconds))
        : 0;
      const startedAtMs = Date.parse(startedUtc);
      if (Number.isFinite(startedAtMs) && Number.isFinite(Number(wallNowMs))) {
        baseSeconds = Math.max(
          baseSeconds,
          Math.max(0, Number(wallNowMs) - startedAtMs) / 1000
        );
      }
      anchoredAtMs = Number.isFinite(Number(monotonicNowMs))
        ? Number(monotonicNowMs)
        : null;
    }

    function observe(document, monotonicNowMs, wallNowMs = Date.now()) {
      const serverElapsed = Number(document?.elapsed_wall_s);
      if (!Number.isFinite(serverElapsed) || serverElapsed < 0) return null;
      if (!jobId || jobId !== document.job_id || anchoredAtMs === null) {
        reset(
          document.job_id,
          serverElapsed,
          monotonicNowMs,
          document.started_utc,
          wallNowMs
        );
      }
      const elapsed = Math.max(
        0,
        Number(monotonicNowMs) - anchoredAtMs
      ) / 1000;
      const projected = baseSeconds + elapsed;
      if (serverElapsed > projected) {
        reset(document.job_id, serverElapsed, monotonicNowMs);
        return serverElapsed;
      }
      return Math.max(serverElapsed, projected);
    }

    function elapsedSeconds(
      document,
      terminal,
      monotonicNowMs,
      wallNowMs = Date.now()
    ) {
      if (terminal) {
        const value = Number(document?.elapsed_wall_s);
        return Number.isFinite(value) && value >= 0 ? value : null;
      }
      return observe(document, monotonicNowMs, wallNowMs);
    }

    return { reset, observe, elapsedSeconds };
  }

  return { createElapsedClock };
}));
