'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const builderState = {
    baseMaterial: null,
    validMaterial: null,
    unsupportedFamily: null
  };

  function appApi() {
    return window.mofScreeningApp;
  }

  function value(id) {
    return $(id).value.trim();
  }

  function setValue(id, next) {
    $(id).value = next === null || next === undefined ? '' : String(next);
  }

  function listText(values) {
    return Array.isArray(values) ? values.join(', ') : '';
  }

  function curveText(rows) {
    return (rows || []).map(row => row.join(',')).join('\n');
  }

  function formText(next) {
    if (next === null || next === undefined) return '';
    if (typeof next === 'object') return JSON.stringify(next, null, 2);
    return String(next);
  }

  function provenanceText(entry) {
    if (!entry) return '';
    if (typeof entry === 'string') return entry;
    if (typeof entry !== 'object') return String(entry);
    if (entry.statement) return entry.statement;
    const classLabel = entry.class || 'declaration';
    const source = entry.source || entry.doi || entry.source_locator || '';
    return source ? `${classLabel}: ${source}` : classLabel;
  }

  function declarationMode(entry) {
    const className = String(entry?.class || entry?.guided_mode || '').toLowerCase();
    if (className.includes('synthetic')) return 'synthetic';
    if (
      className.includes('source') ||
      className.includes('measured') ||
      className.includes('published')
    ) return 'source';
    return 'assumption';
  }

  function overallEvidenceText() {
    const modes = [
      value('gmCo2EvidenceType'),
      value('gmWaterEvidenceType'),
      value('gmThermalEvidenceType'),
      value('gmKineticsEvidenceType')
    ].filter(Boolean);
    if (!modes.length) return 'Derived from the four property groups';
    const unique = [...new Set(modes)];
    return unique.length === 1
      ? `${unique[0]} for all declared property groups`
      : `mixed declarations: ${unique.join(', ')}`;
  }

  function clearGuidedFields() {
    const ids = [
      'gmMaterialId', 'gmName', 'gmEvidenceBasis',
      'gmEvidenceNotFor', 'gmShapingBasis', 'gmBinderSupport',
      'gmParticleGeometry', 'gmFamily', 'gmCo2Saturation',
      'gmCo2ReferenceLoading', 'gmCo2ReferencePressure', 'gmCo2Heat',
      'gmWaterSaturation', 'gmWaterHalfRh', 'gmWaterSipsExponent',
      'gmWaterHeat', 'gmActivityCo2Heat', 'gmActivityPenalty',
      'gmActivityCurveTemperature', 'gmActivityWaterSaturation',
      'gmActivityWaterHalfRh', 'gmActivityWaterSipsExponent',
      'gmActivityWaterHeat', 'gmActivityWaterSolverNormalizer',
      'gmSplitFastFraction', 'gmSplitSlowFraction',
      'gmCo2CurveCsv', 'gmSite1Capacity',
      'gmSite1Affinity', 'gmSite1Energy', 'gmSite2Capacity',
      'gmSite2Affinity', 'gmSite2Energy', 'gmSiteWaterMeasurementT',
      'gmSiteWaterReferenceT', 'gmSiteWaterHeat', 'gmWaterCurveCsv',
      'gmReferenceTemperature', 'gmMaximumTemperature', 'gmSolidCp',
      'gmNitrogenHenry', 'gmNitrogenHeat', 'gmOxygenHenry',
      'gmOxygenHeat', 'gmAdsorbedCp', 'gmAdsorptionRates',
      'gmDesorptionRates', 'gmCo2EvidenceType', 'gmWaterEvidenceType',
      'gmThermalEvidenceType', 'gmKineticsEvidenceType',
      'gmCo2Declaration', 'gmWaterDeclaration',
      'gmThermalDeclaration', 'gmKineticsDeclaration', 'gmCo2Validity',
      'gmWaterValidity', 'gmThermalValidity', 'gmKineticsValidity',
      'gmAssumptions', 'gmUncertaintyClassification',
      'gmUncertaintyStatement'
    ];
    for (const id of ids) setValue(id, '');
    $('guidedIssues').replaceChildren();
    $('guidedDownload').disabled = true;
    builderState.baseMaterial = null;
    builderState.validMaterial = null;
    builderState.unsupportedFamily = null;
    $('gmOverallEvidence').textContent = overallEvidenceText();
    updateFamilyVisibility();
  }

  function populateGuided(material) {
    if (!material) return;
    builderState.baseMaterial = JSON.parse(JSON.stringify(material));
    builderState.validMaterial = null;
    setValue('gmMaterialId', material.material_id);
    setValue('gmName', material.name);
    setValue('gmEvidenceBasis', material.evidence?.basis);
    setValue('gmEvidenceNotFor', material.evidence?.not_for);
    setValue('gmShapingBasis', material.shaped_material?.basis);
    setValue('gmBinderSupport', material.shaped_material?.binder_support);
    setValue('gmParticleGeometry', material.shaped_material?.particle_geometry);
    const supportedFamilies = new Set([
      'independent_langmuir_sips',
      'activity_shift_reference_isotherm',
      'split_rate_activity_shift_reference_isotherm',
      'site_resolved_concentration_langmuir'
    ]);
    builderState.unsupportedFamily = supportedFamilies.has(material.equilibrium_family)
      ? null
      : (material.equilibrium_family || null);
    setValue(
      'gmFamily',
      builderState.unsupportedFamily ? '' : material.equilibrium_family
    );
    setValue('gmReferenceTemperature', material.reference_temperature_K);
    setValue('gmMaximumTemperature', material.maximum_bed_temperature_C);
    setValue('gmSolidCp', material.solid_cp_J_kg_K);
    setValue('gmAdsorbedCp', listText(material.adsorbed_phase_cp_J_kg_K));
    setValue('gmAdsorptionRates', listText(material.kinetics?.adsorption_rate_s_inv));
    setValue('gmDesorptionRates', listText(material.kinetics?.desorption_rate_s_inv));

    const co2 = material.co2 || {};
    if (material.equilibrium_family === 'independent_langmuir_sips') {
      setValue('gmCo2Saturation', co2.saturation_mol_kg);
      setValue('gmCo2ReferenceLoading', co2.reference_loading_mol_kg);
      setValue('gmCo2ReferencePressure', co2.reference_partial_pressure_Pa);
      setValue('gmCo2Heat', co2.heat_J_mol);
    } else if (
      material.equilibrium_family === 'activity_shift_reference_isotherm'
      || material.equilibrium_family === 'split_rate_activity_shift_reference_isotherm'
    ) {
      setValue('gmActivityCo2Heat', co2.heat_J_mol);
      setValue('gmActivityPenalty', material.co2_water_penalty_per_mol_kg);
      setValue('gmActivityCurveTemperature', co2.reference_curve?.temperature_K);
      if (material.equilibrium_family === 'split_rate_activity_shift_reference_isotherm') {
        setValue('gmSplitFastFraction', co2.reservoir_fractions?.[0]);
        setValue('gmSplitSlowFraction', co2.reservoir_fractions?.[1]);
      }
      const curveRows = (co2.reference_curve?.pressure_Pa || []).map(
        (pressure, index) => [
          pressure,
          co2.reference_curve.loading_mol_kg[index]
        ]
      );
      setValue(
        'gmCo2CurveCsv',
        curveRows.length
          ? `pressure_Pa,loading_mol_kg\n${curveText(curveRows)}`
          : ''
      );
    } else if (material.equilibrium_family === 'site_resolved_concentration_langmuir') {
      for (const [index, suffix] of [[0, '1'], [1, '2']]) {
        setValue(`gmSite${suffix}Capacity`, co2.site_capacity_mol_kg?.[index]);
        setValue(`gmSite${suffix}Affinity`, co2.site_affinity_m3_mol?.[index]);
        setValue(`gmSite${suffix}Energy`, co2.site_internal_energy_J_mol?.[index]);
      }
      setValue('gmSiteWaterMeasurementT', material.water?.measurement_temperature_K);
      setValue('gmSiteWaterReferenceT', material.water?.reference_temperature_K);
      setValue('gmSiteWaterHeat', material.water?.sorption_heat_J_mol);
      setValue(
        'gmWaterCurveCsv',
        material.water?.points?.length
          ? `relative_humidity,loading_mol_kg\n${curveText(material.water.points)}`
          : ''
      );
    }

    const water = material.water || {};
    if (material.equilibrium_family === 'independent_langmuir_sips') {
      setValue('gmWaterSaturation', water.saturation_mol_kg);
      setValue('gmWaterHalfRh', water.half_loading_relative_humidity_at_reference);
      setValue('gmWaterSipsExponent', water.sips_exponent);
      setValue('gmWaterHeat', water.heat_J_mol);
    } else if (
      material.equilibrium_family === 'activity_shift_reference_isotherm'
      || material.equilibrium_family === 'split_rate_activity_shift_reference_isotherm'
    ) {
      setValue('gmActivityWaterSaturation', water.saturation_mol_kg);
      setValue('gmActivityWaterHalfRh', water.half_loading_relative_humidity_at_reference);
      setValue('gmActivityWaterSipsExponent', water.sips_exponent);
      setValue('gmActivityWaterHeat', water.heat_J_mol);
      setValue('gmActivityWaterSolverNormalizer', water.solver_normalizer_mol_kg);
    }
    setValue('gmNitrogenHenry', material.nitrogen?.henry_mol_kg_Pa_at_reference);
    setValue('gmNitrogenHeat', material.nitrogen?.heat_J_mol);
    setValue('gmOxygenHenry', material.oxygen?.henry_mol_kg_Pa_at_reference);
    setValue('gmOxygenHeat', material.oxygen?.heat_J_mol);

    const provenance = material.provenance || {};
    const co2Provenance = provenance.co2 || provenance.co2_sites || provenance.co2_anchor;
    const waterProvenance = provenance.water || provenance.water_curve;
    const thermalProvenance = provenance.thermal;
    const kineticsProvenance = provenance.kinetics;
    setValue('gmCo2EvidenceType', declarationMode(co2Provenance));
    setValue('gmWaterEvidenceType', declarationMode(waterProvenance));
    setValue('gmThermalEvidenceType', declarationMode(thermalProvenance));
    setValue('gmKineticsEvidenceType', declarationMode(kineticsProvenance));
    setValue('gmCo2Declaration', provenanceText(provenance.co2 || provenance.co2_sites || provenance.co2_anchor));
    setValue('gmWaterDeclaration', provenanceText(provenance.water || provenance.water_curve));
    setValue('gmThermalDeclaration', provenanceText(provenance.thermal));
    setValue('gmKineticsDeclaration', provenanceText(provenance.kinetics));
    const validity = material.validity || {};
    setValue('gmCo2Validity', formText(validity.co2 || validity.temperature_C));
    setValue('gmWaterValidity', formText(validity.water));
    setValue('gmThermalValidity', formText(validity.thermal || validity.material));
    setValue('gmKineticsValidity', formText(validity.kinetics));
    setValue('gmAssumptions', (material.assumptions || []).join('\n'));
    setValue('gmUncertaintyClassification', material.uncertainty?.classification);
    setValue('gmUncertaintyStatement', material.uncertainty?.statement);
    $('guidedIssues').replaceChildren();
    $('guidedDownload').disabled = true;
    $('gmOverallEvidence').textContent = overallEvidenceText();
    updateFamilyVisibility();
  }

  function updateFamilyVisibility() {
    const family = value('gmFamily');
    for (const panel of document.querySelectorAll('[data-guided-family]')) {
      panel.hidden = !String(panel.dataset.guidedFamily || '')
        .split(/\s+/)
        .filter(Boolean)
        .includes(family);
    }
    const site = family === 'site_resolved_concentration_langmuir';
    $('gmUseSiteLawCp').hidden = !site;
    const dryOnly = family === 'split_rate_activity_shift_reference_isotherm' ||
      (
        family === 'activity_shift_reference_isotherm' &&
        builderState.baseMaterial?.water?.model === 'dry_only'
      );
    for (const id of [
      'gmActivityWaterSaturation',
      'gmActivityWaterHalfRh',
      'gmActivityWaterSipsExponent',
      'gmActivityWaterHeat'
    ]) {
      $(id).closest('.screening-field').hidden = dryOnly;
      $(id).disabled = dryOnly;
    }
    $('gmActivityWaterSolverNormalizer').closest('.screening-field').hidden = !dryOnly;
    $('gmActivityWaterSolverNormalizer').disabled = !dryOnly;
    for (const id of ['gmSplitFastFraction', 'gmSplitSlowFraction']) {
      const field = $(id).closest('.screening-field');
      field.hidden = family !== 'split_rate_activity_shift_reference_isotherm';
      $(id).disabled = family !== 'split_rate_activity_shift_reference_isotherm';
    }
    for (const id of ['gmNitrogenHenry', 'gmNitrogenHeat', 'gmOxygenHenry', 'gmOxygenHeat']) {
      const field = $(id).closest('.screening-field');
      field.hidden = site;
      $(id).disabled = site;
    }
    $('gmStateOrder').textContent = site
      ? 'State and rate order: CO2 site 1, CO2 site 2, H2O. Site capacities are mol/kg and site energies are J/mol, zero or negative.'
      : family === 'split_rate_activity_shift_reference_isotherm'
        ? 'State and rate order: CO2 fast, CO2 slow, H2O, N2, O2. Both CO2 reservoirs share one thermodynamic curve and energy; fractions sum to 1.'
        : 'State and rate order: CO2, H2O, N2, O2. Cp entries are J/kg/K; LDF rates are s^-1. Zero rates are explicit no-transfer assumptions.';
  }

  function openBuilder({ reset }) {
    appApi()?.markMaterialDirty();
    if (reset) clearGuidedFields();
    $('guidedMaterialPanel').hidden = false;
    $('guidedMaterialPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function closeBuilder() {
    if (!builderState.validMaterial) appApi()?.markMaterialDirty();
    $('guidedMaterialPanel').hidden = true;
    $('guidedIssues').replaceChildren();
  }

  function renderIssues(issues) {
    const target = $('guidedIssues');
    target.replaceChildren();
    if (!issues?.length) {
      const node = document.createElement('div');
      node.className = 'screening-warning';
      node.textContent = 'Package validated. It is available in the material editor and for Preview.';
      target.append(node);
      return;
    }
    const heading = document.createElement('div');
    heading.className = 'screening-warning';
    heading.textContent = `${issues.length} missing or invalid field${issues.length === 1 ? '' : 's'}:`;
    target.append(heading);
    for (const issue of issues) {
      const node = document.createElement('div');
      node.className = 'screening-warning';
      node.textContent = `${issue.field}: ${issue.message}`;
      target.append(node);
    }
  }

  function collectDraft() {
    const family = value('gmFamily') || builderState.unsupportedFamily || '';
    const draft = {
      base_material: builderState.baseMaterial,
      material_id: value('gmMaterialId'),
      name: value('gmName'),
      equilibrium_family: family,
      evidence_basis: value('gmEvidenceBasis'),
      evidence_not_for: value('gmEvidenceNotFor'),
      shaping_basis: value('gmShapingBasis'),
      binder_support: value('gmBinderSupport'),
      particle_geometry: value('gmParticleGeometry'),
      reference_temperature_K: value('gmReferenceTemperature'),
      maximum_bed_temperature_C: value('gmMaximumTemperature'),
      solid_cp_J_kg_K: value('gmSolidCp'),
      co2_declaration: value('gmCo2Declaration'),
      water_declaration: value('gmWaterDeclaration'),
      thermal_declaration: value('gmThermalDeclaration'),
      kinetics_declaration: value('gmKineticsDeclaration'),
      co2_evidence_type: value('gmCo2EvidenceType'),
      water_evidence_type: value('gmWaterEvidenceType'),
      thermal_evidence_type: value('gmThermalEvidenceType'),
      kinetics_evidence_type: value('gmKineticsEvidenceType'),
      assumptions_text: value('gmAssumptions'),
      co2_validity: value('gmCo2Validity'),
      water_validity: value('gmWaterValidity'),
      thermal_validity: value('gmThermalValidity'),
      kinetics_validity: value('gmKineticsValidity'),
      uncertainty_classification: value('gmUncertaintyClassification'),
      uncertainty_statement: value('gmUncertaintyStatement'),
      adsorbed_phase_cp_J_kg_K: value('gmAdsorbedCp'),
      adsorption_rate_s_inv: value('gmAdsorptionRates'),
      desorption_rate_s_inv: value('gmDesorptionRates')
    };
    if (family === 'independent_langmuir_sips') {
      Object.assign(draft, {
        co2_saturation_mol_kg: value('gmCo2Saturation'),
        co2_reference_loading_mol_kg: value('gmCo2ReferenceLoading'),
        co2_reference_partial_pressure_Pa: value('gmCo2ReferencePressure'),
        co2_heat_J_mol: value('gmCo2Heat'),
        water_saturation_mol_kg: value('gmWaterSaturation'),
        water_half_loading_relative_humidity_at_reference: value('gmWaterHalfRh'),
        water_sips_exponent: value('gmWaterSipsExponent'),
        water_heat_J_mol: value('gmWaterHeat'),
        nitrogen_henry_mol_kg_Pa_at_reference: value('gmNitrogenHenry'),
        nitrogen_heat_J_mol: value('gmNitrogenHeat'),
        oxygen_henry_mol_kg_Pa_at_reference: value('gmOxygenHenry'),
        oxygen_heat_J_mol: value('gmOxygenHeat')
      });
    } else if (family === 'activity_shift_reference_isotherm') {
      Object.assign(draft, {
        co2_heat_J_mol: value('gmActivityCo2Heat'),
        co2_water_penalty_per_mol_kg: value('gmActivityPenalty'),
        co2_curve_temperature_K: value('gmActivityCurveTemperature'),
        co2_curve_csv: value('gmCo2CurveCsv'),
        water_saturation_mol_kg: value('gmActivityWaterSaturation'),
        water_half_loading_relative_humidity_at_reference: value('gmActivityWaterHalfRh'),
        water_sips_exponent: value('gmActivityWaterSipsExponent'),
        water_heat_J_mol: value('gmActivityWaterHeat'),
        water_solver_normalizer_mol_kg: value('gmActivityWaterSolverNormalizer'),
        nitrogen_henry_mol_kg_Pa_at_reference: value('gmNitrogenHenry'),
        nitrogen_heat_J_mol: value('gmNitrogenHeat'),
        oxygen_henry_mol_kg_Pa_at_reference: value('gmOxygenHenry'),
        oxygen_heat_J_mol: value('gmOxygenHeat')
      });
    } else if (family === 'split_rate_activity_shift_reference_isotherm') {
      Object.assign(draft, {
        co2_heat_J_mol: value('gmActivityCo2Heat'),
        co2_water_penalty_per_mol_kg: value('gmActivityPenalty'),
        co2_curve_temperature_K: value('gmActivityCurveTemperature'),
        co2_curve_csv: value('gmCo2CurveCsv'),
        split_fast_fraction: value('gmSplitFastFraction'),
        split_slow_fraction: value('gmSplitSlowFraction'),
        water_solver_normalizer_mol_kg: value('gmActivityWaterSolverNormalizer'),
        nitrogen_henry_mol_kg_Pa_at_reference: value('gmNitrogenHenry'),
        nitrogen_heat_J_mol: value('gmNitrogenHeat'),
        oxygen_henry_mol_kg_Pa_at_reference: value('gmOxygenHenry'),
        oxygen_heat_J_mol: value('gmOxygenHeat')
      });
    } else if (family === 'site_resolved_concentration_langmuir') {
      Object.assign(draft, {
        site1_capacity_mol_kg: value('gmSite1Capacity'),
        site1_affinity_m3_mol: value('gmSite1Affinity'),
        site1_internal_energy_J_mol: value('gmSite1Energy'),
        site2_capacity_mol_kg: value('gmSite2Capacity'),
        site2_affinity_m3_mol: value('gmSite2Affinity'),
        site2_internal_energy_J_mol: value('gmSite2Energy'),
        water_measurement_temperature_K: value('gmSiteWaterMeasurementT'),
        water_reference_temperature_K: value('gmSiteWaterReferenceT'),
        water_sorption_heat_J_mol: value('gmSiteWaterHeat'),
        water_curve_csv: value('gmWaterCurveCsv')
      });
    }
    return draft;
  }

  async function applyGuidedMaterial() {
    const api = appApi();
    api?.markMaterialDirty();
    const requestRevision = api?.materialRevision();
    const draft = collectDraft();
    const draftFingerprint = JSON.stringify(draft);
    builderState.validMaterial = null;
    $('guidedDownload').disabled = true;
    $('guidedApply').disabled = true;
    renderIssues([]);
    try {
      const response = await fetch('/api/mof-screening/material/guided', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ draft })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      if (
        !api?.isMaterialRevisionCurrent(requestRevision)
        || JSON.stringify(collectDraft()) !== draftFingerprint
      ) {
        api?.showError('Material inputs changed while validation was running; the stale response was ignored.');
        return;
      }
      renderIssues(payload.issues || []);
      if (!payload.valid || !payload.material) {
        api?.showError('The guided material package is incomplete; Preview and Start remain locked.');
        return;
      }
      builderState.validMaterial = payload.material;
      builderState.baseMaterial = JSON.parse(JSON.stringify(payload.material));
      api?.applyMaterial(payload.material);
      $('guidedDownload').disabled = false;
      api?.showError();
    } catch (error) {
      api?.showError(`Guided material: ${error.message}`);
    } finally {
      $('guidedApply').disabled = false;
    }
  }

  function downloadGuidedMaterial() {
    if (!builderState.validMaterial) return;
    const blob = new Blob(
      [JSON.stringify(builderState.validMaterial, null, 2) + '\n'],
      { type: 'application/json' }
    );
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${builderState.validMaterial.material_id}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function loadSyntheticExample() {
    appApi()?.markMaterialDirty();
    try {
      const response = await fetch('/api/mof-screening/materials/synthetic_independent_demo');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const material = await response.json();
      material.material_id = 'synthetic_guided_example';
      material.name = 'Synthetic guided independent example';
      material.synthetic = true;
      material.evidence = {
        ...material.evidence,
        status: 'SYNTHETIC',
        basis: 'Explicitly selected guided-form example; not a real material'
      };
      populateGuided(material);
      for (const id of [
        'gmCo2EvidenceType',
        'gmWaterEvidenceType',
        'gmThermalEvidenceType',
        'gmKineticsEvidenceType'
      ]) setValue(id, 'synthetic');
      $('gmOverallEvidence').textContent = overallEvidenceText();
      $('guidedMaterialPanel').hidden = false;
    } catch (error) {
      appApi()?.showError(`Synthetic example: ${error.message}`);
    }
  }

  async function loadCurveFile(inputId, textareaId) {
    appApi()?.markMaterialDirty();
    const file = $(inputId).files?.[0];
    if (!file) return;
    setValue(textareaId, await file.text());
    $(inputId).value = '';
    appApi()?.invalidatePreview();
  }

  function wire() {
    $('createMaterial').addEventListener('click', () => openBuilder({ reset: true }));
    $('guidedLoadCurrent').addEventListener('click', () => {
      const material = appApi()?.getCurrentMaterial();
      if (!material) {
        appApi()?.showError('Select or import a package before loading it into the guided form.');
        return;
      }
      populateGuided(material);
      appApi()?.clearMaterialDirty();
    });
    $('guidedApply').addEventListener('click', applyGuidedMaterial);
    $('guidedDownload').addEventListener('click', downloadGuidedMaterial);
    $('guidedClose').addEventListener('click', closeBuilder);
    $('guidedLoadSynthetic').addEventListener('click', loadSyntheticExample);
    $('gmUseSiteLawCp').addEventListener('click', () => {
      setValue(
        'gmAdsorbedCp',
        '663.1644845317488, 663.1644845317488, 1887.2867921009274'
      );
      builderState.validMaterial = null;
      $('guidedDownload').disabled = true;
      appApi()?.markMaterialDirty();
    });
    $('gmFamily').addEventListener('change', () => {
      builderState.unsupportedFamily = null;
      updateFamilyVisibility();
      appApi()?.markMaterialDirty();
    });
    $('guidedMaterialForm').addEventListener('input', () => {
      builderState.validMaterial = null;
      $('guidedDownload').disabled = true;
      appApi()?.markMaterialDirty();
      $('gmOverallEvidence').textContent = overallEvidenceText();
    });
    $('gmCo2CurveFile').addEventListener('change', () => loadCurveFile('gmCo2CurveFile', 'gmCo2CurveCsv'));
    $('gmWaterCurveFile').addEventListener('change', () => loadCurveFile('gmWaterCurveFile', 'gmWaterCurveCsv'));
    window.addEventListener('mof-screening-material', event => {
      if (
        !$('guidedMaterialPanel').hidden
        && event.detail?.material
        && event.detail.sourceLabel !== 'guided'
      ) {
        populateGuided(event.detail.material);
      }
    });
    updateFamilyVisibility();
  }

  window.addEventListener('DOMContentLoaded', wire);
})();
