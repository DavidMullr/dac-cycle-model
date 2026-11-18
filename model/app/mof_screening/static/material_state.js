(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.MofMaterialState = api;
}(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';

  function createMaterialGate(initial = true) {
    let dirty = Boolean(initial);
    let revision = 0;
    let previewFingerprint = null;
    return {
      markDirty() {
        dirty = true;
        previewFingerprint = null;
        revision += 1;
        return revision;
      },
      clear() {
        dirty = false;
        previewFingerprint = null;
        revision += 1;
        return revision;
      },
      clearPreview() { previewFingerprint = null; },
      markPreviewValid(fingerprint) {
        if (!dirty) previewFingerprint = String(fingerprint);
      },
      canStart(fingerprint) {
        return !dirty && previewFingerprint === String(fingerprint);
      },
      revision() { return revision; },
      isCurrent(token) { return Number(token) === revision; },
      blocksPreview() { return dirty; },
      isDirty() { return dirty; }
    };
  }

  return { createMaterialGate };
}));
