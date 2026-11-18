/* A failed module or graphics context must leave a readable, usable page. */
(() => {
  'use strict';
  const fail = message => {
    document.getElementById('gx-loading').hidden = true;
    document.getElementById('gx-fallback').hidden = false;
    document.getElementById('gx-fallback-message').textContent = message;
    document.querySelectorAll('[data-gx-model],[data-gx-camera],.gx-controls input,.gx-zoom button,#px-motion,#px-label-mode,#px-focus-component,#px-show-bed').forEach(control => { control.disabled = true; });
    if (document.getElementById('px-replay').dataset?.ready !== 'true') {
      document.querySelectorAll('#px-intro select,#px-replay button,#px-replay input,#px-replay select,#px-field').forEach(control => { control.disabled = true; });
      document.getElementById('px-evidence-flag').textContent = 'Saved cycle unavailable';
      document.getElementById('px-current-phase').textContent = 'Unavailable';
    }
  };
  document.getElementById('gx-retry').addEventListener('click', () => window.location.reload());
  import('/docs/assets/geometry/viewer.js').then(module => module.start(fail)).catch(() => {
    fail('The local graphics module or saved geometry could not be loaded. Check that the complete project is present and WebGL 2 is available.');
  });
})();
