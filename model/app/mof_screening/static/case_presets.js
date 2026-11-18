'use strict';

(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.MofCasePresets = api;
}(typeof globalThis === 'object' ? globalThis : this, function () {
  const DRY_AIR_48H_ID = 'dry_air_comparison_baseline_48h';
  const HUMID_CANDIDATE_COMMON_RH050_48H_ID =
    'humid_candidate_common_48h_rh050_exploratory';
  const YOUNG2021_HUMID_SHORT_CYCLE_ID = 'young2021_humid_short_cycle';
  const YOUNG2021_HOT116_RH0_48H_ID = 'young2021_hot116_48h_rh0';
  const YOUNG2021_HOT116_RH030_48H_ID = 'young2021_hot116_48h_rh030';
  const YOUNG2021_ADVISORY_RH0_48H_ID =
    'young2021_advisory_temperature_48h_rh0';
  const YOUNG2021_ADVISORY_RH030_48H_ID =
    'young2021_advisory_temperature_48h_rh030';
  const MMEN_INDEPENDENT_HUMID_RH0_48H_ID =
    'mmen_mcdonald2012_independent_humid_48h_rh0';
  const MMEN_INDEPENDENT_HUMID_RH030_48H_ID =
    'mmen_mcdonald2012_independent_humid_48h_rh030';

  function resolve(presets, presetId) {
    const list = Array.isArray(presets) ? presets : [];
    const preset = list.find(item => item?.id === presetId);
    if (!preset || !preset.case) {
      throw new Error(`Unknown case preset: ${presetId}`);
    }
    return preset;
  }

  function apply(presets, presetId, applyCase, invalidatePreview) {
    if (
      typeof applyCase !== 'function'
      || typeof invalidatePreview !== 'function'
    ) {
      throw new Error(
        'Case preset application requires case and preview callbacks.'
      );
    }
    const preset = resolve(presets, presetId);
    applyCase(preset.case);
    invalidatePreview();
    return preset;
  }

  function applySelectedMaterial(
    presets,
    presetId,
    applyCase,
    invalidatePreview
  ) {
    const preset = resolve(presets, presetId);
    if (preset.material_id) {
      throw new Error(
        `Preset ${presetId} selects a material and cannot preserve the current selection.`
      );
    }
    return apply(presets, presetId, applyCase, invalidatePreview);
  }

  return {
    DRY_AIR_48H_ID,
    HUMID_CANDIDATE_COMMON_RH050_48H_ID,
    YOUNG2021_HUMID_SHORT_CYCLE_ID,
    YOUNG2021_HOT116_RH0_48H_ID,
    YOUNG2021_HOT116_RH030_48H_ID,
    YOUNG2021_ADVISORY_RH0_48H_ID,
    YOUNG2021_ADVISORY_RH030_48H_ID,
    MMEN_INDEPENDENT_HUMID_RH0_48H_ID,
    MMEN_INDEPENDENT_HUMID_RH030_48H_ID,
    resolve,
    apply,
    applySelectedMaterial
  };
}));
