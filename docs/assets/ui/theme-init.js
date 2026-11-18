/* Apply the saved appearance before the page is painted. */
(() => {
  'use strict';
  let theme;
  try { theme = localStorage.getItem('carbon-team-theme'); } catch (_) { /* Storage can be unavailable. */ }
  if (theme !== 'light' && theme !== 'dark') {
    theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.dataset.ctTheme = theme;
})();
