'use strict';

((root, factory) => {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.MofComparisonView = api;
})(typeof globalThis === 'undefined' ? this : globalThis, () => {
  const MAX_COMPARISON_JOBS = 32;

  function finiteMetric(value, digits = 3) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return 'unavailable';
    if (value !== 0 && Math.abs(value) < 10 ** -digits) {
      return value.toExponential(Math.min(digits, 3));
    }
    return value.toFixed(digits);
  }

  function percent(value, digits = 2) {
    return typeof value === 'number' && Number.isFinite(value)
      ? `${(value * 100).toFixed(digits)}%`
      : 'unavailable';
  }

  function evidenceLabel(value) {
    const status = String(value || '').toUpperCase();
    if (status.includes('REPRESENTATION_FAILED')) return 'representation failed';
    if (status.includes('USER_EDITED')) return 'user-edited assumption';
    if (status.includes('CONDITIONAL')) return 'conditional';
    if (status.includes('SYNTHETIC')) return 'synthetic';
    if (status.includes('HYPOTHETICAL')) return 'hypothetical';
    return status ? status.toLowerCase().replaceAll('_', ' ') : 'unavailable';
  }

  function statusLabel(value) {
    return String(value || 'unavailable').replaceAll('_', ' ');
  }

  function summaryRows(comparison) {
    const rows = comparison && Array.isArray(comparison.rows)
      ? comparison.rows
      : [];
    return rows.map(row => ({
      jobId: row.job_id,
      caseLabel: row.case_label,
      materialId: row.material_id,
      materialName: row.material_name || row.material_id || 'Unavailable material',
      collectedKgDay: finiteMetric(row.collected_CO2_kg_day),
      lastCycle: Number.isFinite(row.last_complete_cycle_CO2_kg_day)
        ? finiteMetric(row.last_complete_cycle_CO2_kg_day) : null,
      repeatability: row.cycle_repeatability_established,
      electricityKWhDay: finiteMetric(row.electricity_kWh_day),
      electricityKWhKg: finiteMetric(row.electricity_kWh_kg_CO2, 4),
      dryProductFraction: percent(row.dry_product_CO2_fraction),
      wetProductFraction: percent(row.wet_product_CO2_fraction),
      steamIntensity: Number.isFinite(row.steam_kg_per_kg_CO2)
        ? finiteMetric(row.steam_kg_per_kg_CO2, 2) : null,
      externalHeatIntensity: row.steam_generator_carrier === 'external_heat'
        && Number.isFinite(row.steam_generator_kWh_kg_CO2)
        ? finiteMetric(row.steam_generator_kWh_kg_CO2, 2) : null,
      evidence: evidenceLabel(row.material_evidence_status),
      numerical: statusLabel(row.numerical_status),
      conservation: statusLabel(row.conservation_status),
      hardLimits: statusLabel(row.hard_limits_status)
    }));
  }

  function selectionError(count) {
    if (!Number.isInteger(count) || count < 2) {
      return 'Select at least two saved jobs.';
    }
    if (count > MAX_COMPARISON_JOBS) {
      return `Select no more than ${MAX_COMPARISON_JOBS} saved jobs.`;
    }
    return null;
  }

  return {
    MAX_COMPARISON_JOBS,
    finiteMetric,
    percent,
    evidenceLabel,
    selectionError,
    summaryRows
  };
});
