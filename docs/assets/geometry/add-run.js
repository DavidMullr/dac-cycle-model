import {projectResult} from './native-result.mjs';

function openStore() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('dac-cycle-replays', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('runs', {keyPath:'id'});
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function storage(action, value) {
  const db = await openStore();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction('runs', action === 'getAll' ? 'readonly' : 'readwrite');
      const store = tx.objectStore('runs');
      const request = action === 'getAll' ? store.getAll() : store[action](value);
      tx.oncomplete = () => resolve(request.result);
      tx.onerror = tx.onabort = () => reject(tx.error || Error('Replay storage unavailable'));
    });
  } finally { db.close(); }
}

export function setupRunImport({listen, add, remove, selectedId}) {
  const el = id => document.getElementById(id);
  const jobs = el('px-local-jobs'), open = el('px-open-job'), file = el('px-run-file');
  const status = el('px-import-status'), historyStatus = el('px-history-status');
  let version = 0, historyVersion = 0;
  async function refresh() {
    const revision = ++historyVersion;
    open.disabled = true;
    historyStatus.textContent = 'Reading completed runs…';
    try {
      const res = await fetch('/api/mof-screening/history', {cache:'no-store'});
      if (!res.ok) throw Error();
      const history = await res.json();
      if (revision !== historyVersion) return;
      jobs.replaceChildren();
      for (const job of history.jobs) {
        if (job.state !== 'completed' || !/^[a-f0-9]{32}$/i.test(job.job_id)) continue;
        jobs.add(new Option([job.case_label, job.material, job.engine, job.job_id.slice(0,8)].filter(Boolean).join(' · '), job.job_id));
      }
      jobs.disabled = open.disabled = jobs.length === 0;
      historyStatus.textContent = jobs.length ? '' : 'No completed runs on this installation.';
    } catch {
      if (revision !== historyVersion) return;
      jobs.replaceChildren(); jobs.disabled = true;
      historyStatus.textContent = 'Local history is unavailable. You can still open an exported JSON file.';
    }
  }
  async function importBytes(readBytes) {
    const revision = ++version;
    status.textContent = 'Reading saved result…';
    try {
      const bytes = await readBytes();
      const run = projectResult(JSON.parse(new TextDecoder().decode(bytes)));
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
      run.id = Array.from(digest, b => b.toString(16).padStart(2,'0')).join('');
      if (revision !== version) return;
      add(run, true);
      let message = 'Added to Saved case in this browser.';
      try { await storage('put', run); }
      catch { message = 'Opened for this session. Browser storage is unavailable.'; }
      if (revision === version) status.textContent = message;
    } catch (error) {
      if (revision === version) status.textContent = 'Could not add run. ' +
        (error instanceof SyntaxError ? 'Choose a native result JSON file.' : error.message);
    }
  }
  listen(file, 'change', () => {
    const chosen = file.files[0];
    file.value = '';
    if (chosen) importBytes(() => chosen.arrayBuffer());
  });
  listen(open, 'click', () => {
    const id = jobs.value;
    if (!/^[a-f0-9]{32}$/i.test(id)) return;
    importBytes(async () => {
      const res = await fetch('/api/mof-screening/jobs/' + id + '/export?format=json', {cache:'no-store'});
      if (!res.ok) throw Error('The selected result could not be loaded.');
      return res.arrayBuffer();
    });
  });
  listen(el('px-refresh-jobs'), 'click', refresh);
  listen(el('px-add-run'), 'toggle', () => { if (el('px-add-run').open) refresh(); });
  listen(el('px-remove-run'), 'click', async () => {
    const id = selectedId();
    if (!id) return;
    ++version;
    try {
      await storage('delete', id);
      remove(id);
      status.textContent = 'Removed from this browser’s replay list.';
    } catch { status.textContent = 'Could not remove the browser copy.'; }
  });
  storage('getAll').then(runs => runs.forEach(run => add(run, false))).catch(() => {});
}
