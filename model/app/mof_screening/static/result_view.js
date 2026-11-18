(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.MofResultView = api;
}(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';

  function residualSummary(value) {
    if (Array.isArray(value)) {
      const values = value.filter(item => Number.isFinite(Number(item))).map(Number);
      if (!values.length) return { known: false, value: null, basis: 'unknown' };
      return { known: true, value: Math.max(...values), basis: 'maximum relative' };
    }
    if (Number.isFinite(Number(value)) && value !== null && value !== '') {
      return { known: true, value: Number(value), basis: 'aggregate relative' };
    }
    return { known: false, value: null, basis: 'unknown' };
  }

  function closureStatus(value) {
    if (value === true) return 'PASS';
    if (value === false) return 'FAIL';
    return 'not recorded';
  }

  function clearExportLinks(elements) {
    for (const element of elements) {
      element.removeAttribute('href');
      element.setAttribute('aria-disabled', 'true');
    }
  }

  function setExportLink(element, href) {
    element.href = href;
    element.setAttribute('aria-disabled', 'false');
  }

  const ADVISORY_FAMILY = 'young2021_advisory_wadst_gab';

  function previewTemperatureFact(preview) {
    const policy = preview?.temperature_policy || {};
    if (preview?.material?.equilibrium_family === ADVISORY_FAMILY) {
      return {
        label: 'temperature policy',
        value: 'Record excursions',
        detail: 'measured support 25-70 C · former 120 C boundary unenforced'
      };
    }
    return {
      label: 'temperature guard',
      value: `${Number.isFinite(Number(policy.scenario_guard_C))
        ? Number(policy.scenario_guard_C).toFixed(1)
        : 'not recorded'} C`,
      detail: `${String(policy.registry_classification || 'unregistered').replaceAll('_', ' ')} · ${String(policy.mode || 'stop').replaceAll('_', ' ')}`
    };
  }

  function resultTemperatureFact(result) {
    const assessment = result?.temperature_guard_assessment;
    const policy = assessment?.policy || {};
    if (result?.material?.material_id === (
      'lewatit_vp_oc_1065_young2021_advisory_temperature_conditional'
    )) {
      return {
        label: 'Temperature policy',
        value: `Record excursions · peak ${Number.isFinite(Number(assessment?.peak_bed_temperature_C))
          ? `${Number(assessment.peak_bed_temperature_C).toFixed(2)} C`
          : 'not recorded'}`,
        detail: 'measured support 25-70 C · former 120 C boundary unenforced'
      };
    }
    return {
      label: 'Temperature guard',
      value: assessment
        ? `${String(assessment.status || 'not available').replaceAll('_', ' ')} · peak ${Number.isFinite(Number(assessment.peak_bed_temperature_C))
          ? `${Number(assessment.peak_bed_temperature_C).toFixed(2)} C`
          : 'not recorded'} · scenario guard ${Number.isFinite(Number(policy.scenario_guard_C))
          ? `${Number(policy.scenario_guard_C).toFixed(2)} C`
          : 'not recorded'} · excursion ${Number.isFinite(Number(assessment.scenario_guard_excursion_C))
          ? `${Number(assessment.scenario_guard_excursion_C).toFixed(2)} C`
          : 'not recorded'}`
        : 'Saved result · temperature-policy classification not recorded'
    };
  }

  return {
    residualSummary,
    closureStatus,
    clearExportLinks,
    setExportLink,
    previewTemperatureFact,
    resultTemperatureFact
  };
}));
