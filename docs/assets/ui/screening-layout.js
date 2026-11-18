/* Presentation-only restructuring. Existing form nodes, IDs, handlers and warnings stay intact. */
(() => {
  'use strict';
  if (!['/screening', '/screening/'].includes(window.location.pathname)) return;

  const main = document.querySelector('main');
  const header = document.querySelector('header');
  const configuration = document.getElementById('configurationPanel');
  const historyTable = document.getElementById('historyTable');
  const form = document.getElementById('caseForm');
  if (!main || !configuration || !historyTable || !form) return;

  main.classList.add('ct-screening-main');
  const eyebrow = header?.querySelector('.eyebrow');
  const heading = header?.querySelector('h1');
  const description = header?.querySelector('p');
  if (eyebrow) eyebrow.textContent = 'Materials & simulations';
  if (heading) heading.textContent = 'Set up a material simulation';
  if (description) {
    description.textContent = 'Choose a MOF, COF, resin or custom material package, set the operating case and review the exact calculation boundary. Live runs and saved results are available from Monitor runs.';
  }

  const tabs = document.createElement('nav');
  tabs.className = 'ct-route-tabs';
  tabs.setAttribute('aria-label', 'Materials and simulations workspace');
  tabs.innerHTML = '<a href="#configurationPanel" aria-current="page">New simulation</a><a href="#historyHeading">Monitor runs</a><a href="/docs/workspace/results.html">Results &amp; comparison</a>';
  header?.append(tabs);

  const categoryNote = document.createElement('div');
  categoryNote.className = 'ct-inline-note';
  categoryNote.textContent = 'Material support includes MOFs, COFs, resin systems and format-checked custom packages. A package law, evidence basis and operating case remain distinct.';
  const materialHeading = document.getElementById('configureHeading');
  (materialHeading?.closest('.screening-section-heading') || materialHeading)
    ?.insertAdjacentElement('afterend', categoryNote);

  const presetLegend = [...form.querySelectorAll('legend')]
    .find(legend => legend.textContent.trim() === 'Comparison preset');
  const presetFieldset = presetLegend?.closest('fieldset');
  if (presetFieldset && !presetFieldset.closest('.ct-advanced-presets')) {
    const disclosure = document.createElement('details');
    disclosure.className = 'ct-advanced-presets';
    disclosure.innerHTML = '<summary>Comparison presets</summary><p>Each saved preset restores an exact recorded case. Its source, material and physical assumptions remain visible after loading.</p>';
    presetFieldset.before(disclosure);
    presetLegend.textContent = 'Saved comparison presets';
    disclosure.append(presetFieldset);
  }

  const addAnchor = id => {
    const target = document.getElementById(id);
    if (target) target.setAttribute('tabindex', '-1');
  };
  ['configurationPanel', 'previewHeading', 'parallelHeading', 'historyHeading'].forEach(addAnchor);

  const parallel = document.getElementById('parallelHeading')?.closest('section');
  const runPanel = document.getElementById('runPanel');
  const history = document.getElementById('historyHeading')?.closest('section');
  const workbench = configuration.closest('.screening-layout')?.querySelector('.screening-workbench');
  if (workbench && parallel && runPanel && history) {
    const disclosure = document.createElement('details');
    disclosure.id = 'ct-run-controls';
    disclosure.className = 'ct-run-controls';
    const summary = document.createElement('summary');
    summary.textContent = 'Run controls and history';
    const note = document.createElement('p');
    note.textContent = 'Inspect active calculations, adjust simultaneous-run limits and review saved runs.';
    const content = document.createElement('div');
    content.className = 'ct-run-control-content';
    parallel.before(disclosure);
    disclosure.append(summary, note, content);
    content.append(parallel, runPanel, history);
    const openControls = () => { disclosure.open = true; };
    document.querySelectorAll('a[href="#ct-run-controls"],a[href="#parallelHeading"],a[href="#historyHeading"]')
      .forEach(link => link.addEventListener('click', openControls));
    window.addEventListener('hashchange', () => {
      if (['#ct-run-controls', '#parallelHeading', '#historyHeading', '#runPanel'].includes(window.location.hash)) openControls();
    });
    if (['#ct-run-controls', '#parallelHeading', '#historyHeading', '#runPanel'].includes(window.location.hash)) openControls();
  }
})();
