/* Shared workspace navigation and appearance. Existing controllers stay untouched. */
(() => {
  'use strict';

  if (window.location.protocol === 'file:') return;

  const icons = {
    overview: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    material: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Zm0 0v9m8-4.5L12 12l-8-4.5M12 12v9"/>',
    results: '<path d="M4 20V9m5 11V4m5 16v-7m5 7V7"/><path d="M2 20h20"/>',
    layers: '<path d="m12 3 10 5-10 5L2 8l10-5Zm-10 9 10 5 10-5M2 16l10 5 10-5"/>',
    engineering: '<path d="m14 6 4 4M3 21l7-7m-4 4-3-3 9-9a6 6 0 0 1 8-3l-5 5 2 2 5-5a6 6 0 0 1-3 8l-9 9Z"/>',
    evidence: '<path d="M14 2H5v20h14V7l-5-5Zm0 0v5h5M8 12h8m-8 4h8"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 4 2c-1 .7-1.5 1-1.5 3m0 3h.01"/>',
    arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
    moon: '<path d="M21 13a9 9 0 0 1-10-10 9 9 0 1 0 10 10Z"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>'
  };

  const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.evidence}</svg>`;
  const primary = [
    ['/', 'Overview', 'overview'],
    ['/screening/', 'Materials &amp; simulations', 'material'],
    ['/docs/workspace/results.html', 'Results &amp; comparison', 'results'],
    ['/docs/workspace/bed-aging.html', 'Bed aging', 'layers'],
    ['/docs/geometry.html', '3D DAC setup', 'layers'],
    ['/docs/workspace/engineering.html', 'Engineering &amp; costs', 'engineering']
  ];
  const helpLinks = [
    ['/project-report/', 'Model and results', 'evidence'],
    ['/project-report/user_guide.html', 'Using the DAC model', 'help'],
    ['/project-report/evidence.html', 'Evidence and limits', 'evidence'],
    ['/docs/DATA_NEEDED.html', 'Data needed', 'evidence'],
    ['/docs/quickstart.html', 'Quick start', 'help']
  ];
  const aliases = new Map([
    ['/screening', '/screening/'], ['/screening/', '/screening/'],
    ['/results/', '/docs/workspace/results.html'], ['/results', '/docs/workspace/results.html'],
    ['/engineering/', '/docs/workspace/engineering.html'], ['/engineering', '/docs/workspace/engineering.html']
  ]);
  const routeTitles = new Map([...primary.map(([url, label]) => [url, label.replace('&amp;', '&')]), ...helpLinks.map(([url, label]) => [url, label])]);
  const path = window.location.pathname;
  const selectedPath = aliases.get(path) || path;
  const title = routeTitles.get(path) || primary.find(item => item[0] === selectedPath)?.[1].replace('&amp;', '&') || 'Workspace';
  const body = document.body;
  const main = body.querySelector('main');
  if (!main) return;
  const originalHeader = body.querySelector('header');
  const workspace = document.createElement('div');
  workspace.className = 'ct-workspace';
  const topbar = document.createElement('div');
  topbar.className = 'ct-topbar';
  topbar.innerHTML = `<button class="ct-icon-button ct-menu" id="ct-menu" type="button" aria-label="Open navigation" aria-controls="ct-sidebar" aria-expanded="false">${icon('menu')}</button><div class="ct-breadcrumb"><span>Carbon Team</span><span aria-hidden="true">/</span><strong>${title}</strong></div><div class="ct-topbar-actions"><a href="/docs/DATA_NEEDED.html" class="ct-report-link">Help &amp; evidence ${icon('arrow')}</a><button class="ct-theme-button" id="ct-theme" type="button"></button></div>`;
  workspace.append(topbar);
  if (originalHeader) {
    workspace.append(originalHeader);
  } else {
    const header = document.createElement('header');
    const h1 = main.querySelector('h1');
    if (h1) header.append(h1);
    workspace.append(header);
  }
  workspace.append(main);
  if (!main.id) main.id = 'ct-content';
  main.setAttribute('tabindex', '-1');
  if (path === '/docs/geometry.html') {
    const geometryHeading = originalHeader?.querySelector('h1');
    const geometryEyebrow = originalHeader?.querySelector('.eyebrow');
    if (geometryHeading) geometryHeading.textContent = '3D DAC setup';
    if (geometryEyebrow) geometryEyebrow.textContent = 'DAC process · saved cycle replay';
  }

  const skip = document.createElement('a');
  skip.className = 'ct-skip';
  skip.href = `#${main.id}`;
  skip.textContent = 'Skip to content';
  const sidebar = document.createElement('aside');
  sidebar.className = 'ct-sidebar';
  sidebar.id = 'ct-sidebar';
  sidebar.setAttribute('aria-label', 'Workspace navigation');
  const navLink = ([url, label, name], extraClass = '') => `<a href="${url}" class="ct-nav-link ${extraClass}"${selectedPath === url ? ' aria-current="page"' : ''}>${icon(name)}<span>${label}</span></a>`;
  sidebar.innerHTML = `<div class="ct-brand-row"><a class="ct-brand" href="/" aria-label="Carbon Team overview"><span class="ct-brand-mark">${icon('material')}</span><span>Carbon Team<small>DIRECT AIR CAPTURE</small></span></a><button class="ct-icon-button ct-menu-close" type="button" aria-label="Close navigation">${icon('close')}</button></div><nav class="ct-navigation" aria-label="Application"><div class="ct-nav-group ct-primary-nav"><p class="ct-nav-label">Workspace</p>${primary.map(item => navLink(item)).join('')}</div><div class="ct-nav-group"><p class="ct-nav-label">Help &amp; evidence</p>${helpLinks.map(item => navLink(item)).join('')}</div></nav><div class="ct-sidebar-footer"><span class="ct-footer-mark">C</span><div>Astérix project<small>DAC screening workspace</small></div></div>`;
  const backdrop = document.createElement('div');
  backdrop.className = 'ct-backdrop';
  backdrop.hidden = true;
  body.prepend(skip, sidebar, backdrop, workspace);
  body.classList.add('ct-ready');

  const themeButton = document.getElementById('ct-theme');
  const scheme = window.matchMedia('(prefers-color-scheme: dark)');
  const paintTheme = theme => {
    document.documentElement.dataset.ctTheme = theme;
    const dark = theme === 'dark';
    themeButton.innerHTML = `${icon(dark ? 'sun' : 'moon')}<span>${dark ? 'Light' : 'Dark'} mode</span>`;
    themeButton.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
  };
  const savedTheme = () => {
    try {
      return localStorage.getItem('carbon-team-theme');
    } catch (_) {
      return null;
    }
  };
  paintTheme(document.documentElement.dataset.ctTheme || (scheme.matches ? 'dark' : 'light'));
  themeButton.addEventListener('click', () => {
    const theme = document.documentElement.dataset.ctTheme === 'dark' ? 'light' : 'dark';
    paintTheme(theme);
    try {
      localStorage.setItem('carbon-team-theme', theme);
    } catch (_) {
      // The theme still works for this page.
    }
  });
  scheme.addEventListener('change', event => {
    if (!['light', 'dark'].includes(savedTheme())) paintTheme(event.matches ? 'dark' : 'light');
  });
  window.addEventListener('storage', event => {
    if (event.key === 'carbon-team-theme') {
      paintTheme(['light', 'dark'].includes(event.newValue)
        ? event.newValue
        : (scheme.matches ? 'dark' : 'light'));
    }
  });

  const menu = document.getElementById('ct-menu');
  const mobile = window.matchMedia('(max-width: 960px)');
  const setMenu = (open, returnFocus = true) => {
    open = open && mobile.matches;
    body.classList.toggle('ct-nav-open', open);
    menu.setAttribute('aria-expanded', String(open));
    backdrop.hidden = !open;
    workspace.inert = open;
    sidebar.inert = mobile.matches && !open;
    if (open) {
      sidebar.setAttribute('role', 'dialog');
      sidebar.setAttribute('aria-modal', 'true');
      sidebar.querySelector('.ct-menu-close').focus();
    } else {
      sidebar.removeAttribute('role');
      sidebar.removeAttribute('aria-modal');
      if (returnFocus) menu.focus();
    }
  };
  menu.addEventListener('click', () => setMenu(true));
  sidebar.querySelector('.ct-menu-close').addEventListener('click', () => setMenu(false));
  backdrop.addEventListener('click', () => setMenu(false));
  sidebar.addEventListener('keydown', event => {
    if (!body.classList.contains('ct-nav-open')) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      setMenu(false);
    }
    if (event.key === 'Tab') {
      const items = [...sidebar.querySelectorAll('a,button,summary')]
        .filter(element => element.getClientRects().length && !element.disabled);
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  mobile.addEventListener('change', () => setMenu(false, false));
  setMenu(false, false);

  const wrapTables = node => {
    const tables = node.matches?.('table') ? [node] : node.querySelectorAll?.('table') || [];
    tables.forEach(table => {
      if (table.parentElement.matches('.ct-table-scroll,.table-wrap,.tablewrap,.table-scroll,.scroll,.wrap')) return;
      const wrap = document.createElement('div');
      wrap.className = 'ct-table-scroll';
      table.before(wrap);
      wrap.append(table);
    });
  };
  wrapTables(main);
  new MutationObserver(records => {
    records.forEach(record => record.addedNodes.forEach(node => {
      if (node.nodeType === 1) wrapTables(node);
    }));
  }).observe(main, { childList: true, subtree: true });

  if (path === '/screening/' || path === '/screening') {
    const layoutScript = document.createElement('script');
    layoutScript.src = '/docs/assets/ui/screening-layout.js';
    layoutScript.defer = true;
    document.head.append(layoutScript);
  }
})();
