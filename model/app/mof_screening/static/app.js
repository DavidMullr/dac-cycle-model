'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const materialName = name => String(name || 'Material').replace(/\s+v\d+$/, '');
  const progressClockApi = window.MofProgressClock || {
    createElapsedClock() {
      let jobId = null;
      let baseSeconds = 0;
      let anchoredAtMs = null;
      const reset = (
        nextJobId,
        serverElapsedSeconds,
        monotonicNowMs,
        startedUtc = null,
        wallNowMs = Date.now()
      ) => {
        jobId = nextJobId || null;
        baseSeconds = Number.isFinite(Number(serverElapsedSeconds))
          ? Math.max(0, Number(serverElapsedSeconds))
          : 0;
        const startedAtMs = Date.parse(startedUtc);
        if (Number.isFinite(startedAtMs) && Number.isFinite(Number(wallNowMs))) {
          baseSeconds = Math.max(
            baseSeconds,
            Math.max(0, Number(wallNowMs) - startedAtMs) / 1000
          );
        }
        anchoredAtMs = Number.isFinite(Number(monotonicNowMs))
          ? Number(monotonicNowMs)
          : null;
      };
      const observe = (
        stateDocument,
        monotonicNowMs,
        wallNowMs = Date.now()
      ) => {
        const serverElapsed = Number(stateDocument?.elapsed_wall_s);
        if (!Number.isFinite(serverElapsed) || serverElapsed < 0) return null;
        if (!jobId || jobId !== stateDocument.job_id || anchoredAtMs === null) {
          reset(
            stateDocument.job_id,
            serverElapsed,
            monotonicNowMs,
            stateDocument.started_utc,
            wallNowMs
          );
        }
        const projected = baseSeconds + Math.max(0, Number(monotonicNowMs) - anchoredAtMs) / 1000;
        if (serverElapsed > projected) {
          reset(stateDocument.job_id, serverElapsed, monotonicNowMs);
          return serverElapsed;
        }
        return Math.max(serverElapsed, projected);
      };
      const elapsedSeconds = (
        stateDocument,
        terminal,
        monotonicNowMs,
        wallNowMs = Date.now()
      ) => {
        if (terminal) {
          const value = Number(stateDocument?.elapsed_wall_s);
          return Number.isFinite(value) && value >= 0 ? value : null;
        }
        return observe(stateDocument, monotonicNowMs, wallNowMs);
      };
      return { reset, observe, elapsedSeconds };
    }
  };
  const state = {
    scope: null,
    material: null,
    preview: null,
    currentJobId: null,
    activeJob: false,
    concurrency: null,
    activeJobs: [],
    refreshingActive: false,
    startingPreviews: new Set(),
    result: null,
    comparison: null,
    pollTimer: null,
    liveProgressTimer: null,
    progressSnapshot: null,
    elapsedClock: progressClockApi.createElapsedClock(),
    previewRevision: 0,
    materialLoadRevision: 0
  };
  const materialGate = window.MofMaterialState.createMaterialGate(true);

  function errorMessage(value) {
    if (value && typeof value === 'object') return value.error || JSON.stringify(value);
    return String(value || 'Unknown error');
  }

  async function jsonRequest(url, options = {}) {
    const response = await fetch(url, { cache: 'no-store', ...options });
    let payload = null;
    try { payload = await response.json(); } catch (_) { payload = null; }
    if (!response.ok) {
      const field = payload?.field ? ` (${payload.field})` : '';
      throw new Error(`${errorMessage(payload)}${field}`);
    }
    return payload;
  }

  function showGlobalError(message = '') {
    const node = $('globalError');
    node.hidden = !message;
    node.textContent = message;
  }

  function tag(text, tone = '') {
    const node = document.createElement('span');
    node.className = `screening-tag ${tone}`.trim();
    node.textContent = text;
    return node;
  }

  function stateChip(text, tone = '') {
    const node = document.createElement('span');
    node.className = `screening-chip ${tone}`.trim();
    node.textContent = text;
    return node;
  }

  function number(value, digits = 3) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    const magnitude = Math.abs(Number(value));
    if (magnitude !== 0 && (magnitude < 0.001 || magnitude >= 100000)) return Number(value).toExponential(2);
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
  }

  function fraction(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    return `${(100 * Number(value)).toFixed(1)}%`;
  }

  function temperatureC(valueKelvin) {
    return valueKelvin === null || valueKelvin === undefined ? '—' : `${(Number(valueKelvin) - 273.15).toFixed(1)} °C`;
  }

  function download(name, payload, type = 'application/json') {
    const blob = new Blob([typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2) + '\n'], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = name;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function setPreviewInvalid() {
    state.previewRevision += 1;
    materialGate.clearPreview();
    state.preview = null;
    $('previewContent').hidden = true;
    $('previewEmpty').hidden = false;
    $('previewEmpty').textContent = 'Inputs changed. Preview the exact case again to unlock Start.';
    $('startButton').disabled = true;
    const chip = stateChip('Edits not previewed', 'warn');
    chip.id = 'previewState';
    $('previewState').replaceWith(chip);
  }

  function markMaterialDirty() {
    materialGate.markDirty();
    state.materialLoadRevision += 1;
    setPreviewInvalid();
  }

  function clearMaterialDirty() {
    materialGate.clear();
    state.materialLoadRevision += 1;
    setPreviewInvalid();
  }

  function currentPreviewFingerprint() {
    return JSON.stringify({ material: state.material, case: readCase() });
  }

  function renderMaterialTags() {
    const target = $('materialTags');
    target.replaceChildren();
    const material = state.material;
    if (!material) {
      target.append(tag('No material package loaded', 'bad'));
      return;
    }
    target.append(tag(`CURRENT · ${materialName(material.name)}`, 'good'));
    target.append(tag(material.material_id || 'material ID missing', ''));
    target.append(tag(material.equilibrium_family, 'good'));
    const evidenceStatus = material.evidence?.status || 'evidence unlabelled';
    target.append(tag(
      evidenceStatus,
      String(evidenceStatus).toUpperCase().includes('REPRESENTATION_FAILED')
        ? 'bad'
        : (material.synthetic ? 'warn' : '')
    ));
    target.append(tag(material.synthetic ? 'SYNTHETIC' : 'not synthetic', material.synthetic ? 'warn' : ''));
    if (material.provenance?.co2_temperature_shift?.gate_pass === false) {
      target.append(tag('family fit gate failed', 'bad'));
    }
    if (
      material.humidity_candidate?.schema
      === 'mof-screening-humidity-candidate/1'
    ) {
      target.append(tag('exploratory humid candidate · RH 0.5 preset', 'warn'));
    } else if (material.water?.model === 'dry_only') {
      target.append(tag('dry only · RH must be zero', 'warn'));
    }
    target.append(tag(material.shaped_material?.binder_support || 'binder/support not described', 'warn'));
    if ((material.uncertainty?.classification || '').toUpperCase().includes('UNQUANTIFIED')) {
      target.append(tag('uncertainty unquantified', 'warn'));
    }
  }

  function setMaterial(document, sourceLabel = '') {
    materialGate.clear();
    state.materialLoadRevision += 1;
    state.material = document;
    $('materialJson').value = JSON.stringify(document, null, 2);
    renderMaterialTags();
    if (sourceLabel) $('materialSelect').dataset.loadedLabel = sourceLabel;
    setPreviewInvalid();
    window.dispatchEvent(new CustomEvent('mof-screening-material', {
      detail: { material: document, sourceLabel }
    }));
  }

  async function loadMaterial(materialId) {
    markMaterialDirty();
    const loadRevision = state.materialLoadRevision;
    showGlobalError();
    try {
      const document = await jsonRequest(`/api/mof-screening/materials/${encodeURIComponent(materialId)}`);
      if (loadRevision !== state.materialLoadRevision) return;
      setMaterial(document, materialId);
    } catch (error) {
      showGlobalError(error.message);
    }
  }

  function applyMaterialFromEditor() {
    try {
      const document = JSON.parse($('materialJson').value);
      if (!document || typeof document !== 'object' || Array.isArray(document)) throw new Error('Material package must be a JSON object.');
      setMaterial(document, 'imported');
      showGlobalError();
    } catch (error) {
      showGlobalError(`Material JSON: ${error.message}`);
    }
  }

  function setValue(id, value) { $(id).value = value ?? ''; }

  function applyCase(caseDocument) {
    setValue('caseLabel', caseDocument.label);
    const engine = document.querySelector(`input[name="engine"][value="${caseDocument.engine}"]`);
    if (engine) engine.checked = true;
    const mode = document.querySelector(`input[name="weatherMode"][value="${caseDocument.weather.mode}"]`);
    if (mode) mode.checked = true;
    const runtimePolicy = caseDocument.runtime_policy || {
      mode: 'manual_stop',
      time_limit_s: null
    };
    const runtimeMode = document.querySelector(
      `input[name="runtimePolicy"][value="${runtimePolicy.mode}"]`
    );
    if (runtimeMode) runtimeMode.checked = true;
    const temperaturePolicy = caseDocument.temperature_policy || {
      mode: 'stop_at_scenario_guard'
    };
    const temperatureMode = document.querySelector(
      `input[name="temperaturePolicy"][value="${temperaturePolicy.mode}"]`
    );
    if (temperatureMode) temperatureMode.checked = true;
    setValue(
      'executionTimeLimitMinutes',
      runtimePolicy.time_limit_s === null
        || runtimePolicy.time_limit_s === undefined
        ? ''
        : runtimePolicy.time_limit_s / 60
    );
    setValue('weatherHours', caseDocument.weather.hours);
    setValue('weatherTemperature', caseDocument.weather.temperature_C);
    setValue('weatherRh', caseDocument.weather.relative_humidity);
    setValue('weatherPressure', caseDocument.weather.pressure_Pa);
    setValue('weatherCo2', caseDocument.weather.co2_ppm_dry);
    setValue('weatherStart', caseDocument.weather.start_utc);
    const equipment = caseDocument.equipment;
    setValue('eqMass', equipment.dry_shaped_mass_kg);
    setValue('eqThickness', equipment.bed_thickness_m);
    setValue('eqArea', equipment.bed_face_area_m2);
    setValue('eqParticle', equipment.particle_diameter_m);
    setValue('eqVoid', equipment.bed_void_fraction);
    setValue('eqAirflow', equipment.airflow_m3_s);
    setValue('eqHeater', equipment.heater_power_W);
    setValue('eqConductivity', equipment.effective_bed_conductivity_W_m_K);
    setValue('eqHeatTransfer', equipment.gas_solid_heat_transfer_W_m2_K);
    setValue('eqWallCapacity', equipment.wall_heat_capacity_J_K);
    setValue('eqWallLoss', equipment.wall_loss_UA_W_K);
    setValue('eqWallCooling', equipment.wall_cooling_UA_W_K);
    const cycle = caseDocument.cycle;
    setValue('cyAds', cycle.adsorption_s);
    setValue('cyEvac', cycle.evacuation_s);
    setValue('cyDes', cycle.desorption_s);
    setValue('cyCool', cycle.cooling_max_s);
    setValue('cyRepress', cycle.repressurization_max_s);
    setValue('cyTemperature', cycle.regeneration_temperature_C);
    setValue('cyPressure', cycle.regeneration_pressure_Pa);
    setValue('cyPurge', cycle.purge_s);
    setValue('collectionRoute', caseDocument.collection.product_route);
    setValue('initialBasis', caseDocument.initial_state.basis);
    $('initialCold').checked = Boolean(caseDocument.initial_state.cold_start);
    setValue('solverCells', caseDocument.solver.cells);
    setValue('solverRtol', caseDocument.solver.rtol);
    setValue('solverStep', caseDocument.solver.max_step_s);
    setValue('solverSamples', caseDocument.solver.samples_per_phase);
    updateWeatherVisibility();
    updateEngineApplicability();
    updateRuntimePolicyVisibility();
  }

  function selectedEngine() {
    return document.querySelector('input[name="engine"]:checked').value;
  }

  function selectedWeatherMode() {
    return document.querySelector('input[name="weatherMode"]:checked').value;
  }

  function selectedRuntimePolicyMode() {
    return document.querySelector('input[name="runtimePolicy"]:checked').value;
  }

  function readNumber(id) { return Number($(id).value); }

  function readCase() {
    const mode = selectedWeatherMode();
    const engine = selectedEngine();
    return {
      schema: 'mof-screening-case/1',
      label: $('caseLabel').value.trim(),
      engine,
      runtime_policy: {
        schema: 'mof-screening-runtime-policy/1',
        mode: selectedRuntimePolicyMode(),
        time_limit_s: selectedRuntimePolicyMode() === 'time_limit'
          ? readNumber('executionTimeLimitMinutes') * 60
          : null,
        classification: 'operational_runtime_control',
        physical_operating_condition: false
      },
      temperature_policy: {
        schema: 'mof-screening-temperature-policy/1',
        mode: document.querySelector('input[name="temperaturePolicy"]:checked').value
      },
      weather: {
        mode,
        hours: readNumber('weatherHours'),
        temperature_C: readNumber('weatherTemperature'),
        relative_humidity: readNumber('weatherRh'),
        pressure_Pa: readNumber('weatherPressure'),
        co2_ppm_dry: readNumber('weatherCo2'),
        start_utc: $('weatherStart').value.trim()
      },
      equipment: {
        dry_shaped_mass_kg: readNumber('eqMass'),
        bed_thickness_m: readNumber('eqThickness'),
        bed_face_area_m2: readNumber('eqArea'),
        particle_diameter_m: readNumber('eqParticle'),
        bed_void_fraction: readNumber('eqVoid'),
        airflow_m3_s: readNumber('eqAirflow'),
        heater_power_W: readNumber('eqHeater'),
        effective_bed_conductivity_W_m_K: readNumber('eqConductivity'),
        gas_solid_heat_transfer_W_m2_K: readNumber('eqHeatTransfer'),
        wall_heat_capacity_J_K: readNumber('eqWallCapacity'),
        wall_loss_UA_W_K: readNumber('eqWallLoss'),
        wall_cooling_UA_W_K: readNumber('eqWallCooling')
      },
      cycle: {
        adsorption_s: readNumber('cyAds'),
        evacuation_s: readNumber('cyEvac'),
        desorption_s: readNumber('cyDes'),
        cooling_max_s: readNumber('cyCool'),
        repressurization_max_s: readNumber('cyRepress'),
        regeneration_temperature_C: readNumber('cyTemperature'),
        regeneration_pressure_Pa: readNumber('cyPressure'),
        purge_s: readNumber('cyPurge')
      },
      solver: {
        cells: readNumber('solverCells'),
        rtol: readNumber('solverRtol'),
        max_step_s: readNumber('solverStep'),
        samples_per_phase: readNumber('solverSamples')
      },
      initial_state: {
        basis: $('initialBasis').value,
        cold_start: $('initialCold').checked
      },
      collection: {
        product_route: $('collectionRoute').value,
        knockout_model: 'ideal_pump_inlet_55C'
      }
    };
  }

  function updateWeatherVisibility() {
    const official = selectedWeatherMode() === 'official';
    $('weatherHours').max = official ? '24' : '96';
    $('fixedWeatherFields').hidden = official;
    $('officialWeatherFields').hidden = !official;
  }

  function updateEngineApplicability() {
    const spatial = selectedEngine() === 'spatial';
    for (const id of ['eqConductivity', 'eqHeatTransfer', 'eqWallCooling']) {
      $(id).disabled = !spatial;
      $(id).setAttribute('aria-disabled', String(!spatial));
    }
    $('transportApplicability').textContent = spatial
      ? 'Spatial only: through-thickness cells, bed conductivity and gas-solid heat transfer affect the local finite-transfer solution.'
      : 'Lumped companion: well-mixed transport and its reviewed cooling request are used. Bed conductivity, wall cooling UA, gas-solid heat transfer and through-thickness cells are retained in the case record but do not alter lumped execution.';
  }

  function updateRuntimePolicyVisibility() {
    const limited = selectedRuntimePolicyMode() === 'time_limit';
    $('executionTimeLimitMinutes').disabled = !limited;
    $('executionTimeLimitMinutes').setAttribute(
      'aria-disabled',
      String(!limited)
    );
  }

  function fact(label, value, detail = '') {
    const node = document.createElement('div');
    node.className = 'screening-fact';
    const eyebrow = document.createElement('span');
    eyebrow.textContent = label;
    const strong = document.createElement('strong');
    strong.textContent = value;
    node.append(eyebrow, strong);
    if (detail) {
      const small = document.createElement('small');
      small.textContent = detail;
      node.append(small);
    }
    return node;
  }

  function renderPreview(preview) {
    if (preview.material) {
      state.material = preview.material;
      $('materialJson').value = JSON.stringify(preview.material, null, 2);
      renderMaterialTags();
    }
    $('previewEmpty').hidden = true;
    $('previewContent').hidden = false;
    const temperatureFact = window.MofResultView.previewTemperatureFact(preview);
    $('previewFacts').replaceChildren(
      fact('material', preview.material_name, `${preview.material_id} · ${preview.engine}`),
      fact('horizon', `${preview.horizon_s / 3600} h`, `${preview.maximum_cycle_starts} bounded cycle starts`),
      preview.engine === 'spatial'
        ? fact('representation', `${preview.solver.cells} through-thickness cells`, 'spatial finite-transfer representation')
        : fact('representation', 'Lumped, well mixed', 'no through-thickness spatial cells'),
      fact(
        'execution control',
        preview.allocation?.runtime_policy?.mode === 'time_limit'
          ? `${number((preview.allocation.runtime_policy.time_limit_s || 0) / 60, 1)} min limit`
          : (preview.allocation?.runtime_policy?.mode === 'manual_stop'
            ? 'Manual stop'
            : 'Original execution policy not recorded'),
        preview.allocation?.runtime_policy
          ? 'operational control only · Stop remains available'
          : 'policy not recorded in this saved preview'
      ),
      fact(
        temperatureFact.label,
        temperatureFact.value,
        temperatureFact.detail
      )
    );
    const warningList = $('previewWarnings');
    warningList.replaceChildren();
    for (const warning of preview.warnings || []) {
      const node = document.createElement('div');
      node.className = 'screening-warning';
      node.textContent = warning;
      warningList.append(node);
    }
    const chip = stateChip('Preview valid · exact hashes bound', 'ok');
    chip.id = 'previewState';
    $('previewState').replaceWith(chip);
    const gate = state.scope?.gate || { start_enabled: false };
    updateStartAvailability();
    if (!gate.start_enabled) {
      const blocked = document.createElement('div');
      blocked.className = 'screening-warning';
      blocked.textContent = 'Execution is temporarily unavailable while kernel preflight is incomplete.';
      $('previewWarnings').append(blocked);
    }
  }

  async function previewCase() {
    showGlobalError();
    if (materialGate.blocksPreview()) {
      showGlobalError('Apply a complete material package or explicitly load another package before preview.');
      setPreviewInvalid();
      return;
    }
    if (!state.material) {
      showGlobalError('Load or apply a material package before preview.');
      return;
    }
    $('previewButton').disabled = true;
    $('previewButton').textContent = 'Validating…';
    const requestRevision = state.previewRevision;
    const materialRevision = materialGate.revision();
    const fingerprint = currentPreviewFingerprint();
    try {
      const preview = await jsonRequest('/api/mof-screening/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ material: state.material, case: readCase() })
      });
      if (
        requestRevision !== state.previewRevision
        || !materialGate.isCurrent(materialRevision)
        || fingerprint !== currentPreviewFingerprint()
      ) {
        setPreviewInvalid();
        showGlobalError('Inputs changed while preview was running; the stale response was ignored.');
        return;
      }
      state.preview = preview;
      materialGate.markPreviewValid(fingerprint);
      renderPreview(state.preview);
    } catch (error) {
      setPreviewInvalid();
      showGlobalError(error.message);
    } finally {
      $('previewButton').disabled = false;
      $('previewButton').textContent = 'Preview exact case';
    }
  }

  async function startJob() {
    if (!window.MofJobReattach.hasCapacity(state.concurrency, state.startingPreviews.size)) {
      showGlobalError('Maximum simultaneous runs reached. Raise the limit or wait for a run to finish.');
      return;
    }
    if (!state.preview || !state.preview.preview_id) return;
    const startingPreviewId = state.preview.preview_id;
    if (state.startingPreviews.has(startingPreviewId)) return;
    const fingerprint = currentPreviewFingerprint();
    if (materialGate.blocksPreview() || !materialGate.canStart(fingerprint)) {
      setPreviewInvalid();
      showGlobalError('Material inputs are incomplete or unapplied. Apply a complete package before Start.');
      return;
    }
    showGlobalError();
    state.result = null;
    $('resultsPanel').hidden = true;
    $('metricCards').replaceChildren();
    $('resultWarnings').replaceChildren();
    $('resultBasisCaption').textContent = '';
    $('resultDiagnostics').replaceChildren();
    $('startButton').disabled = true;
    state.startingPreviews.add(startingPreviewId);
    try {
      const started = await jsonRequest('/api/mof-screening/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preview_id: startingPreviewId })
      });
      state.currentJobId = started.job_id;
      state.elapsedClock.reset(
        null,
        0,
        performance.now()
      );
      state.activeJob = true;
      if (state.preview?.preview_id === startingPreviewId) state.preview = null;
      $('runPanel').hidden = false;
      const chip = stateChip('Queued', 'warn');
      chip.id = 'runState';
      $('runState').replaceWith(chip);
      $('runProgress').value = 0;
      $('runMessage').textContent = 'Starting';
      $('runProgressDetails').textContent = '';
      state.progressSnapshot = null;
      $('cancelButton').disabled = false;
      pollJob();
      refreshActiveJobs();
    } catch (error) {
      showGlobalError(error.message);
    } finally {
      state.startingPreviews.delete(startingPreviewId);
      updateStartAvailability();
    }
  }

  function pollJob() {
    clearTimeout(state.pollTimer);
    const monitoredJobId = state.currentJobId;
    if (!monitoredJobId) return;
    state.pollTimer = setTimeout(async () => {
      try {
        const stateDocument = await jsonRequest(`/api/mof-screening/jobs/${monitoredJobId}`);
        if (state.currentJobId !== monitoredJobId) return;
        state.progressSnapshot = {
          document: stateDocument,
          receivedAt: performance.now()
        };
        $('runProgress').value = Number(stateDocument.progress || 0);
        $('runMessage').textContent = stateDocument.message || '';
        $('runState').textContent = stateDocument.state;
        $('runState').className = `screening-chip ${stateDocument.state === 'completed' ? 'ok' : (['failed', 'timed_out'].includes(stateDocument.state) ? 'error' : 'warn')}`;
        $('runProgressDetails').textContent = progressDetail(stateDocument);
        if (['queued', 'running', 'cancelling'].includes(stateDocument.state)) {
          pollJob();
          return;
        }
        $('cancelButton').disabled = true;
        state.activeJob = false;
        state.progressSnapshot = {
          document: stateDocument,
          receivedAt: performance.now(),
          terminal: true
        };
        await refreshHistory();
        if (stateDocument.state === 'completed' && stateDocument.result) {
          state.result = stateDocument.result;
          renderResult(stateDocument.result);
        } else {
          const checkpoint = stateDocument.partial?.last_checkpoint;
          $('runProgressDetails').textContent = [
            progressDetail(stateDocument),
            checkpoint?.phase ? `Checkpoint phase: ${checkpoint.phase}` : null,
            checkpoint?.state_sha256 ? `State: ${checkpoint.state_sha256.slice(0, 12)}` : null,
            'Incomplete diagnostic retained; safe resume is not implemented.'
          ].filter(Boolean).join(' · ');
          showGlobalError(stateDocument.message || stateDocument.failure?.message || 'Screening ended before completion.');
        }
      } catch (error) {
        showGlobalError(error.message);
      }
    }, 900);
  }

  function durationLabel(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) return '—';
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const secs = Math.floor(value % 60);
    return [
      hours ? `${hours} h` : null,
      minutes ? `${minutes} min` : null,
      `${secs} s`
    ].filter(Boolean).join(' ');
  }

  function progressDetail(stateDocument) {
    const knownNumber = value => value !== null
      && value !== undefined
      && value !== ''
      && Number.isFinite(Number(value));
    let liveElapsed = null;
    if (
      state.currentJobId === stateDocument.job_id
      && knownNumber(stateDocument.elapsed_wall_s)
      && !state.progressSnapshot?.terminal
    ) {
      liveElapsed = state.elapsedClock.elapsedSeconds(
        stateDocument,
        false,
        performance.now(),
        Date.now()
      );
    } else if (knownNumber(stateDocument.elapsed_wall_s)) {
      liveElapsed = Number(stateDocument.elapsed_wall_s);
    }
    const parts = [];
    if (liveElapsed !== null) {
      parts.push(`${durationLabel(liveElapsed)} elapsed`);
    }
    if (stateDocument.current_cycle || stateDocument.current_phase) {
      parts.push(
        `cycle ${stateDocument.current_cycle ?? '—'} · ${stateDocument.current_phase ?? 'phase transition'}`
      );
    }
    if (knownNumber(stateDocument.simulated_s)) {
      parts.push(`${durationLabel(stateDocument.simulated_s)} simulated`);
    }
    if (
      knownNumber(stateDocument.phase_elapsed_s)
      && knownNumber(stateDocument.phase_duration_s)
    ) {
      parts.push(
        `phase ${durationLabel(stateDocument.phase_elapsed_s)} / ${durationLabel(stateDocument.phase_duration_s)}`
      );
    }
    if (stateDocument.runtime_notice?.message) {
      parts.push(stateDocument.runtime_notice.message);
    }
    return parts.join(' · ');
  }

  async function cancelJob() {
    const cancelJobId = window.MofJobReattach.cancelTarget(state);
    if (!cancelJobId) return;
    try {
      const payload = await jsonRequest('/api/mof-screening/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: cancelJobId })
      });
      $('runMessage').textContent = `Cancellation requested (${payload.state})`;
      $('cancelButton').disabled = true;
      pollJob();
    } catch (error) {
      showGlobalError(error.message);
    }
  }

  function metric(label, value, unit, primary = false) {
    const node = document.createElement('div');
    node.className = `screening-metric ${primary ? 'primary' : ''}`.trim();
    const title = document.createElement('span');
    title.textContent = label;
    const strong = document.createElement('strong');
    strong.textContent = value;
    const small = document.createElement('small');
    small.textContent = unit;
    node.append(title, strong, small);
    return node;
  }

  function evaluationFor(result) {
    return result.derived_evaluation || {
      target_attainment: result.target_attainment || {
        status: 'not_available',
        advisory: true,
        checks: [],
        statement: 'Target attainment was not recorded for this result.'
      },
      hard_limits: result.hard_limits || {
        status: 'not_assessed',
        checks: [],
        unassessed_limits: []
      },
      conservation: result.conservation || {
        status: (
          result.checks?.mass_closure_pass && result.checks?.energy_closure_pass
        ) ? 'pass' : 'fail'
      },
      numerical_reliability: result.numerical_reliability || result.numerical_qualification || {}
    };
  }

  function targetText(check) {
    if (!check || !check.known) return 'not available';
    const value = check.unit === 'K'
      ? temperatureC(check.achieved)
      : `${number(check.achieved, 0)} ${check.unit}`;
    const target = check.unit === 'K'
      ? temperatureC(check.target)
      : `${number(check.target, 0)} ${check.unit}`;
    const evidence = check.cycle !== null && check.cycle !== undefined
      ? ` · cycle ${check.cycle}, ${check.phase}, ${number(check.duration_s, 0)} s`
      : '';
    return `${value} achieved / ${target} target · ${check.passed ? 'met' : 'advisory miss'}${evidence}`;
  }

  function renderResult(result) {
    MofResultView.clearExportLinks([$('exportJson'), $('exportCsv')]);
    $('resultsPanel').hidden = false;
    const metrics = result.metrics || {};
    $('resultBasisCaption').textContent = [
      result.case_label,
      `${materialName(result.material?.name || result.material?.material_id)} · ${result.material?.material_id || '—'}`,
      `${result.engine} · ${result.horizon?.initial_state || 'initial basis unknown'}`,
      `${number(result.horizon?.simulated_s, 0)} of ${number(result.horizon?.requested_s, 0)} s requested`,
      result.horizon?.partial_final_cycle ? 'partial final cycle' : 'no partial final cycle'
    ].join(' · ');
    const evaluation = evaluationFor(result);
    const targets = evaluation.target_attainment || {};
    const hardLimits = evaluation.hard_limits || {};
    const conservation = evaluation.conservation || {};
    const numerical = evaluation.numerical_reliability || {};
    const targetMiss = String(targets.status || '').startsWith('advisory_miss');
    const hardFailure = hardLimits.status === 'fail';
    const closureFailure = conservation.status !== 'pass';
    const diagnostics = $('resultDiagnostics');
    diagnostics.className = `screening-diagnostic ${!hardFailure && !closureFailure ? 'ok' : 'warn'}`;
    diagnostics.textContent = [
      `Target attainment: ${targetMiss ? 'advisory miss' : String(targets.status || 'not available').replaceAll('_', ' ')}.`,
      targetMiss ? 'The target miss does not by itself reject or rank down this case.' : null,
      `Known hard limits: ${String(hardLimits.status || 'not assessed').replaceAll('_', ' ')}.`,
      `Conservation: ${closureFailure ? 'fail' : 'pass'}.`,
      `Numerical reliability: ${String(numerical.status || 'not established').replaceAll('_', ' ')}.`
    ].filter(Boolean).join(' ');
    $('metricCards').replaceChildren(
      metric('Collected CO₂/day', number(metrics.collected_CO2_kg_day, 4), 'kg/day · foreground metric', true),
      metric('Electricity/day', number(metrics.electricity_kWh_day, 3), 'kWh/day'),
      metric('Electricity intensity', number(metrics.electricity_kWh_kg_CO2, 3), 'kWh/kg CO₂'),
      metric('Completed cycles', `${result.horizon?.cycles_completed ?? '—'} / ${result.horizon?.cycles_started ?? '—'}`, 'completed / started'),
      metric('Dry CO₂ fraction', fraction(metrics.dry_product_CO2_fraction), 'product dry basis'),
      metric('Wet CO₂ fraction', fraction(metrics.wet_product_CO2_fraction), 'conditioned collected gas, wet basis'),
      metric('Peak bed temperature', temperatureC(metrics.peak_bed_temperature_K), 'achieved maximum'),
      metric('Water burden', `${number(metrics.water_loading_initial_mol_kg, 4)} → ${number(metrics.water_loading_final_mol_kg, 4)}`, 'mol/kg dry, initial → final'),
      metric('Peak local RH', fraction(metrics.local_relative_humidity_maximum), 'p_H₂O / p_sat(T_bed)'),
      metric('Targets', targetMiss ? 'ADVISORY MISS' : String(targets.status || 'N/A').toUpperCase(), 'requested T, P and cooling endpoints')
    );
    const evidence = stateChip(result.material?.synthetic ? 'SYNTHETIC PACKAGE' : (result.material?.evidence?.status || 'SCREENING'), result.material?.synthetic ? 'warn' : 'ok');
    evidence.id = 'resultEvidence';
    $('resultEvidence').replaceWith(evidence);

    const warnings = $('resultWarnings');
    warnings.replaceChildren();
    const warningText = [
      result.material?.evidence?.basis,
      ...(result.material?.assumptions || []),
      result.material?.validity?.water,
      targets.statement,
      numerical.statement,
      ...(hardLimits.unassessed_limits || []),
      result.horizon?.partial_final_cycle
        ? 'The requested horizon ends during a cycle; only completed phases and conserved ledger contributions are counted, and the tail is labelled separately.'
        : null,
    ].filter(Boolean);
    for (const text of warningText) {
      const node = document.createElement('div');
      node.className = 'screening-warning';
      node.textContent = text;
      warnings.append(node);
    }

    renderChart('pressurePlot', result.traces || [], [
      { key: 'pressure_Pa', label: 'maximum cell pressure', color: '#087c62' },
      { key: 'min_pressure_Pa', label: 'minimum cell pressure', color: '#557eac' }
    ], 'Pa');
    renderChart('temperaturePlot', result.traces || [], [
      { key: 'bed_temperature_mean_K', label: 'bed mean', color: '#c45832' },
      { key: 'bed_temperature_max_K', label: 'bed maximum', color: '#8a6d1d' },
      { key: 'wall_temperature_K', label: 'wall', color: '#266d72' }
    ], 'K');
    renderChart('loadingPlot', result.traces || [], [
      { key: 'co2_loading_mean_mol_kg', label: 'CO₂ loading', color: '#087c62' },
      { key: 'water_loading_mean_mol_kg', label: 'water loading', color: '#557eac' }
    ], 'mol/kg');
    renderChart('humidityPlot', result.traces || [], [
      { key: 'local_rh_mean', label: 'mean local RH', color: '#266d72' },
      { key: 'local_rh_max', label: 'maximum local RH', color: '#8a6d1d' }
    ], 'fraction');

    const details = $('resultDetails');
    details.replaceChildren();
    const grid = document.createElement('div');
    grid.className = 'screening-detail-grid';
    const massResidual = MofResultView.residualSummary(
      result.checks?.mass_relative_residual_by_species
    );
    const massClosure = MofResultView.closureStatus(
      result.checks?.mass_closure_pass
    );
    const energyClosure = MofResultView.closureStatus(
      result.checks?.energy_closure_pass
    );
    const targetDetails = (targets.checks || []).map(
      check => [check.label, targetText(check)]
    );
    const partialTargetDetails = (targets.partial_stage_observations || []).map(partial => {
      const achieved = partial.unit === 'K'
        ? temperatureC(partial.achieved)
        : (partial.achieved === null ? '—' : `${number(partial.achieved, 0)} ${partial.unit}`);
      const target = partial.target === null
        ? ''
        : (partial.unit === 'K'
          ? ` / ${temperatureC(partial.target)} target`
          : ` / ${number(partial.target, 0)} ${partial.unit} target`);
      return [
        `Partial cycle ${partial.cycle} · ${partial.phase}`,
        `${achieved}${target} observed over ${number(partial.duration_s, 0)} s · not counted as a completed-target miss`
      ];
    });
    const temperatureFact = window.MofResultView.resultTemperatureFact(result);
    const hardLimitDetails = (hardLimits.checks || []).map(
      check => [
        check.label,
        check.known ? (check.pass ? 'PASS' : 'FAIL') : 'not recorded'
      ]
    );
    const detailRows = [
      ['Engine', `${result.engine} · ${result.engine_identity}`],
      ['Material hash', result.input_identity?.material_sha256],
      ['Case hash', result.input_identity?.case_sha256],
      ['Horizon', `${number(result.horizon?.simulated_s, 0)} s simulated of ${number(result.horizon?.requested_s, 0)} s requested`],
      ['Cycle accounting', `${result.horizon?.cycles_completed ?? '—'} complete, ${result.horizon?.cycles_started ?? '—'} started${result.horizon?.partial_final_cycle ? ', final tail partial' : ''}`],
      ['Weather', `${result.weather?.mode} · ${result.weather?.interval_semantics}`],
      ['Execution control', result.execution_runtime_policy?.mode === 'time_limit'
        ? `${number(result.execution_runtime_policy.time_limit_s / 60, 1)} min selected limit · operational only`
        : (result.execution_runtime_policy?.mode === 'manual_stop'
          ? 'Manual stop · no elapsed-time deadline'
          : 'Saved result · original execution policy not recorded')],
      [temperatureFact.label, temperatureFact.value],
      ['Temperature policy detail', temperatureFact.detail],
      ...targetDetails,
      ...partialTargetDetails,
      ...hardLimitDetails,
      ['Hard-limit scope', String(hardLimits.status || 'not assessed').replaceAll('_', ' ')],
      ['Mass closure', `${massClosure} · ${massResidual.basis} ${massResidual.known ? number(massResidual.value, 8) : 'not recorded'}`],
      ['Energy closure', `${energyClosure} · gross relative ${number(result.checks?.energy_relative_residual, 8)}${result.checks?.energy_legacy_net_relative_residual === undefined ? '' : ` · net exchange ${number(result.checks.energy_legacy_net_relative_residual, 8)}`}`],
      ['Numerical reliability', `${String(numerical.status || 'not established').replaceAll('_', ' ')} · single runs do not qualify a numerical family`],
      ['Initial basis', result.horizon?.initial_state],
      ['CSS', result.horizon?.css_established ? 'Established' : 'Not established by this horizon'],
      ['Water handling', `${number(metrics.water_condensed_mol, 4)} mol condensed · ${number(metrics.water_in_product_mol, 4)} mol carried in product`],
      ['Water burden', `${number(metrics.water_loading_minimum_mol_kg, 4)}–${number(metrics.water_loading_maximum_mol_kg, 4)} mol/kg dry · ${number(metrics.local_relative_humidity_minimum, 4)}–${number(metrics.local_relative_humidity_maximum, 4)} local RH`],
      ['Humidity basis', result.water_burden?.local_relative_humidity_basis || 'not recorded'],
      ['Limits', 'No annual, hardware or measured-winner claim.']
    ];
    for (const [label, value] of detailRows) {
      const row = document.createElement('div');
      const strong = document.createElement('strong');
      strong.textContent = label;
      const span = document.createElement('span');
      span.textContent = value ?? '—';
      row.append(strong, span);
      grid.append(row);
    }
    details.append(grid);
    MofResultView.setExportLink(
      $('exportJson'),
      `/api/mof-screening/jobs/${result.job_id}/export?format=json`
    );
    MofResultView.setExportLink(
      $('exportCsv'),
      `/api/mof-screening/jobs/${result.job_id}/export?format=csv`
    );
    $('resultsPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function svgElement(name, attributes = {}) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    return node;
  }

  function renderChart(id, traces, series, unit) {
    const svg = $(id);
    svg.replaceChildren();
    if (!traces.length) return;
    const width = 600, height = 250, left = 52, right = 16, top = 18, bottom = 34;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const values = traces.flatMap(row => series.map(item => Number(row[item.key]))).filter(Number.isFinite);
    if (!values.length) return;
    let min = Math.min(...values), max = Math.max(...values);
    if (min === max) { min -= 1; max += 1; }
    const pad = (max - min) * 0.08;
    min -= pad;
    max += pad;
    const xMax = Math.max(...traces.map(row => Number(row.elapsed_s) || 0), 1);
    const x = value => left + plotWidth * Number(value || 0) / xMax;
    const y = value => top + plotHeight * (1 - (Number(value) - min) / (max - min));
    for (let i = 0; i <= 4; i++) {
      const yy = top + plotHeight * i / 4;
      svg.append(svgElement('line', {
        x1: left, y1: yy, x2: width - right, y2: yy,
        stroke: '#dce2e5', 'stroke-width': 1
      }));
      const label = svgElement('text', {
        x: left - 7, y: yy + 4, 'text-anchor': 'end',
        'font-size': 9, fill: '#617278'
      });
      label.textContent = number(max - (max - min) * i / 4, 1);
      svg.append(label);
    }
    const axis = svgElement('text', {
      x: left, y: height - 8, 'font-size': 9, fill: '#617278'
    });
    axis.textContent = `elapsed s · ${unit}`;
    svg.append(axis);
    series.forEach((item, index) => {
      const points = traces
        .filter(row => Number.isFinite(Number(row[item.key])))
        .map(row => `${x(row.elapsed_s)},${y(row[item.key])}`)
        .join(' ');
      svg.append(svgElement('polyline', {
        points, fill: 'none', stroke: item.color, 'stroke-width': 2,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round'
      }));
      const legendX = left + index * 170;
      svg.append(svgElement('line', {
        x1: legendX, y1: 8, x2: legendX + 15, y2: 8,
        stroke: item.color, 'stroke-width': 3
      }));
      const label = svgElement('text', {
        x: legendX + 20, y: 11, 'font-size': 9, fill: '#40545b'
      });
      label.textContent = item.label;
      svg.append(label);
    });
  }

  async function refreshHistory() {
    try {
      const payload = await window.MofJobReattach.fetchHistoryAndActive(jsonRequest);
      renderHistory(payload.jobs);
      applyActiveStatus(payload.activeJobs, payload.concurrency);
    } catch (error) {
      showGlobalError(error.message);
    }
  }

  function updateStartAvailability() {
    const valid = Boolean(state.preview?.preview_id)
      && !state.startingPreviews.has(state.preview?.preview_id)
      && state.scope?.gate?.start_enabled
      && !materialGate.blocksPreview()
      && materialGate.canStart(currentPreviewFingerprint());
    $('startButton').disabled = !valid || !window.MofJobReattach.hasCapacity(state.concurrency, state.startingPreviews.size);
  }

  function applyActiveStatus(jobs, concurrency) {
    state.activeJobs = jobs || [];
    state.concurrency = concurrency || state.concurrency || { max_concurrent_jobs: 1, active_count: state.activeJobs.length };
    const limit = state.concurrency.max_concurrent_jobs;
    $('concurrencyStatus').textContent = `${state.activeJobs.length} active · ${limit === 0 ? 'unlimited simultaneous runs' : `maximum ${limit} simultaneous runs`}`;
    const list = $('activeJobs');
    list.replaceChildren();
    for (const job of state.activeJobs) {
      const item = document.createElement('div');
      item.className = 'screening-warning';
      item.setAttribute('role', 'listitem');
      item.dataset.jobId = job.job_id;
      const title = document.createElement('strong');
      title.textContent = job.material_name || job.material_id || job.job_id.slice(0, 8);
      const detail = document.createElement('p');
      detail.textContent = `${job.job_id.slice(0, 8)} · ${job.state} · ${job.current_phase || job.message || 'starting'} · ${number((job.simulated_s || 0) / 3600, 2)} h simulated`;
      const open = document.createElement('button');
      open.className = 'secondary';
      open.textContent = `Open run ${job.job_id.slice(0, 8)}`;
      open.addEventListener('click', () => openJob(job.job_id));
      const stop = document.createElement('button');
      stop.className = 'danger';
      stop.textContent = `Cancel run ${job.job_id.slice(0, 8)}`;
      stop.disabled = job.state === 'cancelling';
      stop.addEventListener('click', async () => {
        stop.disabled = true;
        try {
          await jsonRequest('/api/mof-screening/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_id: job.job_id }) });
          await refreshActiveJobs();
        } catch (error) { showGlobalError(error.message); }
      });
      item.append(title, detail, open, stop);
      list.append(item);
    }
    const selected = state.activeJobs.find(job => job.job_id === state.currentJobId);
    if (!selected && state.activeJobs.length && (!state.activeJob || !state.currentJobId)) {
      reattachJob(state.activeJobs[state.activeJobs.length - 1]);
    }
    updateStartAvailability();
  }

  async function refreshActiveJobs() {
    if (state.refreshingActive) return;
    state.refreshingActive = true;
    try {
      const status = await jsonRequest('/api/mof-screening/status');
      applyActiveStatus(window.MofJobReattach.activeJobs(status), status.concurrency);
    } catch (error) { showGlobalError(error.message); }
    finally { state.refreshingActive = false; }
  }

  async function applyConcurrency() {
    const text = $('maxConcurrentJobs').value.trim();
    const value = Number(text);
    if (!text || !Number.isInteger(value) || value < 0) {
      showGlobalError('Enter an integer of 0 or more; 0 means unlimited.');
      return;
    }
    try {
      state.concurrency = await jsonRequest('/api/mof-screening/concurrency', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ max_concurrent_jobs: value }) });
      showGlobalError();
      await refreshActiveJobs();
    } catch (error) { showGlobalError(error.message); }
  }

  function reattachJob(activeJob) {
    state.currentJobId = activeJob.job_id;
    state.elapsedClock.reset(
      activeJob.job_id,
      Number(activeJob.elapsed_wall_s || 0),
      performance.now(),
      activeJob.started_utc,
      Date.now()
    );
    state.activeJob = true;
    state.progressSnapshot = {
      document: activeJob,
      receivedAt: performance.now()
    };
    $('runPanel').hidden = false;
    $('runProgress').value = Number(activeJob.progress || 0);
    $('runMessage').textContent = activeJob.message || 'Active screening job';
    $('runState').textContent = activeJob.state;
    $('runState').className = 'screening-chip warn';
    $('runProgressDetails').textContent = progressDetail(activeJob);
    $('cancelButton').disabled = activeJob.state === 'cancelling';
    updateStartAvailability();
    pollJob();
  }

  async function openJob(jobId) {
    try {
      const payload = await jsonRequest(`/api/mof-screening/jobs/${jobId}`);
      const selection = window.MofJobReattach.applyHistorySelection(
        state,
        jobId,
        payload
      );
      if (selection.action === 'reattach') {
        reattachJob(payload);
      } else if (selection.action === 'display') {
        renderResult(selection.result);
      } else if (selection.action === 'error') {
        showGlobalError(selection.message);
      }
    } catch (error) {
      showGlobalError(error.message);
    }
  }

  function renderHistory(jobs) {
    const table = $('historyTable');
    table.replaceChildren();
    const header = document.createElement('div');
    header.className = 'screening-history-row';
    for (const text of ['', 'Case', 'Material', 'CO₂ kg/day', 'kWh/kg', 'Inputs', 'State']) {
      const cell = document.createElement('span');
      cell.textContent = text;
      header.append(cell);
    }
    table.append(header);
    for (const job of jobs) {
      const row = document.createElement('div');
      row.className = 'screening-history-row';
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.value = job.job_id;
      check.disabled = !(job.valid && job.state === 'completed');
      check.setAttribute('aria-label', `Compare ${job.case_label || job.job_id}`);
      check.addEventListener('change', updateCompareButton);
      const caseCell = document.createElement('span');
      if (job.valid && job.state === 'completed') {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'secondary';
        button.textContent = job.case_label || 'Open result';
        button.addEventListener('click', () => openJob(job.job_id));
        caseCell.append(button);
      } else if (window.MofJobReattach.ACTIVE_STATES.has(job.state)) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'secondary';
        button.textContent = job.case_label || 'Open active run';
        button.addEventListener('click', () => openJob(job.job_id));
        caseCell.append(button);
      } else {
        caseCell.textContent = job.case_label || job.message || job.job_id;
      }
      const materialCell = document.createElement('span');
      materialCell.textContent = `${job.material || '—'}${job.synthetic ? ' · synthetic' : ''}`;
      const production = document.createElement('span');
      production.className = 'screening-num';
      production.textContent = number(job.metrics?.collected_CO2_kg_day, 4);
      const energy = document.createElement('span');
      energy.className = 'screening-num';
      energy.textContent = number(job.metrics?.electricity_kWh_kg_CO2, 3);
      const inputs = document.createElement('span');
      inputs.textContent = job.input_identity?.case_sha256?.slice(0, 10) || '—';
      const status = document.createElement('span');
      status.className = `screening-state ${job.state}`;
      status.textContent = job.target_attainment_status
        ? `${job.state} · target ${String(job.target_attainment_status).replaceAll('_', ' ')}`
        : job.state;
      row.append(check, caseCell, materialCell, production, energy, inputs, status);
      table.append(row);
    }
    if (jobs.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'screening-empty';
      empty.textContent = 'No saved screening jobs yet.';
      table.append(empty);
    }
    updateCompareButton();
  }

  function updateCompareButton() {
    const selected = [...document.querySelectorAll('#historyTable input[type="checkbox"]:checked')];
    const selectionError = window.MofComparisonView.selectionError(selected.length);
    $('compareButton').disabled = Boolean(selectionError);
    $('comparisonSelectionNote').textContent = selectionError
      ? selectionError
      : `${selected.length} selected · maximum ${window.MofComparisonView.MAX_COMPARISON_JOBS} jobs.`;
    $('downloadComparison').disabled = !state.comparison;
  }

  async function compareSelected() {
    const jobIds = [...document.querySelectorAll('#historyTable input[type="checkbox"]:checked')].map(node => node.value);
    const selectionError = window.MofComparisonView.selectionError(jobIds.length);
    if (selectionError) {
      showGlobalError(selectionError);
      return;
    }
    try {
      state.comparison = await jsonRequest('/api/mof-screening/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_ids: jobIds })
      });
      renderComparison(state.comparison);
    } catch (error) {
      showGlobalError(error.message);
    }
  }

  function renderComparison(comparison) {
    $('comparison').hidden = false;
    const basis = $('comparisonBasis');
    basis.replaceChildren();
    basis.append(tag(
      comparison.comparison_mode === 'exact_configuration'
        ? 'exact configuration'
        : (comparison.comparison_mode === 'controlled_comparison'
          ? 'controlled comparison'
          : 'incompatible basis'),
      comparison.compatible_comparison ? 'good' : 'bad'
    ));
    basis.append(tag(
      comparison.same_configuration ? 'same operating configuration' : 'operating/design inputs differ',
      comparison.same_configuration ? 'good' : 'warn'
    ));
    basis.append(tag(
      comparison.same_engine ? comparison.rows[0]?.engine : 'spatial/lumped mixed',
      comparison.same_engine ? 'good' : 'warn'
    ));
    basis.append(tag(
      comparison.same_source_identity ? 'source identity matches' : 'source identities differ',
      comparison.same_source_identity ? 'good' : 'bad'
    ));
    for (const field of comparison.basis_fields) {
      if (field.field === 'state_layout' && comparison.material_changed) {
        continue;
      }
      basis.append(tag(
        `${field.field}: ${field.matching ? 'match' : 'differ'}`,
        field.matching ? 'good' : 'warn'
      ));
    }
    if (comparison.family_layout_changed) {
      basis.append(tag('material-family state layout differs', 'good'));
    }
    for (const field of comparison.incompatible_fields || []) {
      basis.append(tag(`incompatible: ${field}`, 'bad'));
    }
    const targetMiss = comparison.rows.some(
      row => String(row.target_attainment_status || '').startsWith('advisory_miss')
    );
    const targetUnknown = comparison.rows.some(
      row => !['satisfied', 'advisory_miss'].includes(row.target_attainment_status)
    );
    const closureIssue = comparison.rows.some(row => !row.closure_pass);
    const hardLimitIssue = comparison.rows.some(
      row => row.hard_limits_status === 'fail'
    );
    const hardLimitUnknown = comparison.rows.some(
      row => !['pass', 'fail'].includes(row.hard_limits_status)
    );
    const numericalIssue = comparison.rows.some(
      row => row.numerical_status !== 'qualified'
    );
    const diagnostics = $('comparisonDiagnostics');
    const clean = comparison.compatible_comparison && !closureIssue && !hardLimitIssue && !targetUnknown && !hardLimitUnknown;
    diagnostics.className = `screening-diagnostic ${clean ? 'ok' : 'warn'}`;
    diagnostics.textContent = [
      comparison.comparison_mode === 'exact_configuration'
        ? 'The rows share an exact configuration and executable source identity.'
        : (comparison.comparison_mode === 'controlled_comparison'
          ? 'The rows form a controlled comparison on compatible source, weather, mass, engine and initial-state bases.'
          : 'Rows are shown for review but do not share a compatible source, weather, mass, engine or initial-state basis.'),
      targetMiss
        ? `One or more requested target misses are advisory and do not by themselves exclude or rank down a case.${targetUnknown ? ' Completed-stage target evidence is unavailable for at least one row; unknown is not treated as met.' : ''}`
        : (targetUnknown
          ? 'Completed-stage target attainment is unavailable for at least one row; unknown is not treated as met.'
          : 'All displayed completed-stage advisory operating targets are met.'),
      closureIssue ? 'At least one conservation check fails.' : 'Displayed conservation checks pass.',
      hardLimitIssue
        ? `At least one known hard limit fails.${hardLimitUnknown ? ' An additional hard-limit result is unknown and is not treated as pass.' : ''}`
        : (hardLimitUnknown
          ? 'One or more rows do not have an assessed hard-limit result; unknown is not treated as pass.'
          : 'All displayed known hard limits pass; unassessed limits remain listed separately.'),
      numericalIssue ? 'Numerical-family qualification is absent for at least one row.' : null
    ].filter(Boolean).join(' ');

    const summary = $('comparisonSummary');
    summary.replaceChildren();
    for (const item of window.MofComparisonView.summaryRows(comparison)) {
      const line = document.createElement('div');
      line.className = 'screening-comparison-summary-row';
      line.setAttribute('role', 'listitem');

      const identity = document.createElement('div');
      identity.className = 'screening-comparison-summary-material';
      const name = document.createElement('strong');
      name.textContent = item.materialName;
      const meta = document.createElement('small');
      meta.textContent = `${item.jobId.slice(0, 8)} · ${item.evidence}`;
      identity.append(name, meta);

      const objective = document.createElement('div');
      objective.className = 'screening-comparison-summary-objective';
      const objectiveLabel = document.createElement('small');
      objectiveLabel.textContent = 'Collected CO₂/day';
      const objectiveValue = document.createElement('strong');
      objectiveValue.textContent = `${item.collectedKgDay} kg/day`;
      objective.append(objectiveLabel, objectiveValue);

      const energy = document.createElement('div');
      energy.className = 'screening-comparison-summary-energy';
      const energyLabel = document.createElement('small');
      energyLabel.textContent = 'Electricity';
      const energyValue = document.createElement('span');
      energyValue.textContent = `${item.electricityKWhDay} kWh/day · ${item.electricityKWhKg} kWh/kg`;
      energy.append(energyLabel, energyValue);

      const condition = document.createElement('div');
      condition.className = 'screening-comparison-summary-status';
      const purity = document.createElement('span');
      purity.textContent = `Dry CO₂ ${item.dryProductFraction}`;
      const numerical = document.createElement('span');
      numerical.textContent = `Numerical ${item.numerical}`;
      const conservation = document.createElement('span');
      conservation.textContent = `Closure ${item.conservation}`;
      condition.append(purity, numerical, conservation);

      line.append(identity, objective, energy, condition);
      summary.append(line);
    }

    const table = $('comparisonTable');
    table.replaceChildren();
    const header = document.createElement('div');
    header.className = 'screening-history-row screening-comparison-row';
    const labels = [
      'Case', 'Material', 'Engine', 'CO₂ kg/day', 'kWh/day', 'kWh/kg',
      'Dry CO₂', 'Cycles', 'T achieved/target', 'P achieved/target',
      'Cooling achieved/target', 'Targets', 'Hard limits', 'Conservation',
      'Numerical', 'Source'
    ];
    for (const label of labels) {
      const cell = document.createElement('span');
      cell.textContent = label;
      header.append(cell);
    }
    table.append(header);
    for (const row of comparison.rows) {
      const line = document.createElement('div');
      line.className = 'screening-history-row screening-comparison-row';
      const targetChecks = Object.fromEntries(
        (row.target_checks || []).map(check => [check.id, check])
      );
      const targetValue = check => check?.known
        ? (check.unit === 'K'
          ? `${temperatureC(check.achieved)} / ${temperatureC(check.target)}`
          : `${number(check.achieved, 0)} / ${number(check.target, 0)} Pa`)
        : 'not available';
      const values = [
        row.case_label,
        `${row.material_name} · ${String(row.material_evidence_status || (row.synthetic ? 'synthetic' : 'evidence unavailable')).replaceAll('_', ' ')}`,
        row.engine,
        number(row.collected_CO2_kg_day, 4),
        number(row.electricity_kWh_day, 3),
        number(row.electricity_kWh_kg_CO2, 3),
        fraction(row.dry_product_CO2_fraction),
        `${row.cycles_completed ?? '—'} / ${row.cycles_started ?? '—'}${row.partial_final_cycle ? ' partial tail' : ''}`,
        targetValue(targetChecks.regeneration_temperature),
        targetValue(targetChecks.desorption_pressure),
        targetValue(targetChecks.cooling_temperature),
        String(row.target_attainment_status || 'not available').replaceAll('_', ' '),
        String(row.hard_limits_status || 'not assessed').replaceAll('_', ' '),
        row.closure_pass ? 'pass' : 'fail',
        String(row.numerical_status || 'unavailable').replaceAll('_', ' '),
        row.source_release_sha256?.slice(0, 10) || 'missing'
      ];
      for (const value of values) {
        const cell = document.createElement('span');
        cell.textContent = value;
        line.append(cell);
      }
      table.append(line);
    }
    $('comparisonNote').textContent = comparison.ranking_note;
    $('downloadComparison').disabled = false;
  }

  function wire() {
    $('loadHumidYoungDemo').addEventListener('click', async () => {
      try {
        const preset = window.MofCasePresets.resolve(
          state.scope?.defaults?.case_presets || [],
          window.MofCasePresets.YOUNG2021_HUMID_SHORT_CYCLE_ID
        );
        if (preset.material_id) {
          await loadMaterial(preset.material_id);
          $('materialSelect').value = preset.material_id;
        }
        applyCase(preset.case);
        setPreviewInvalid();
        $('casePresetNote').textContent = preset.evidence_context;
        showGlobalError();
      } catch (error) {
        showGlobalError(`Case preset: ${error.message}`);
      }
    });
    $('loadDryAirBaseline').addEventListener('click', () => {
      try {
        const preset = window.MofCasePresets.apply(
          state.scope?.defaults?.case_presets || [],
          window.MofCasePresets.DRY_AIR_48H_ID,
          applyCase,
          setPreviewInvalid
        );
        $('casePresetNote').textContent = preset.evidence_context;
        showGlobalError();
      } catch (error) {
        showGlobalError(`Case preset: ${error.message}`);
      }
    });
    $('loadHumidCandidate48h').addEventListener('click', () => {
      try {
        const preset = window.MofCasePresets.applySelectedMaterial(
          state.scope?.defaults?.case_presets || [],
          window.MofCasePresets.HUMID_CANDIDATE_COMMON_RH050_48H_ID,
          applyCase,
          setPreviewInvalid
        );
        $('casePresetNote').textContent = preset.evidence_context;
        showGlobalError();
      } catch (error) {
        showGlobalError(`Case preset: ${error.message}`);
      }
    });
    for (const [buttonId, presetId] of [
      ['loadReviewedLewatitHumid', 'reviewed_lewatit_humid_48h'],
      ['loadReviewedLewatitDry', 'reviewed_lewatit_dry_48h'],
      ['loadHotYoungDry48h', window.MofCasePresets.YOUNG2021_HOT116_RH0_48H_ID],
      ['loadHotYoungRh03048h', window.MofCasePresets.YOUNG2021_HOT116_RH030_48H_ID],
      ['loadYoungAdvisoryRh048h', window.MofCasePresets.YOUNG2021_ADVISORY_RH0_48H_ID],
      ['loadYoungAdvisoryRh03048h', window.MofCasePresets.YOUNG2021_ADVISORY_RH030_48H_ID],
      ['loadMmenIndependentRh048h', window.MofCasePresets.MMEN_INDEPENDENT_HUMID_RH0_48H_ID],
      ['loadMmenIndependentRh03048h', window.MofCasePresets.MMEN_INDEPENDENT_HUMID_RH030_48H_ID]
    ]) {
      $(buttonId).addEventListener('click', async () => {
        try {
          const preset = window.MofCasePresets.resolve(
            state.scope?.defaults?.case_presets || [],
            presetId
          );
          if (preset.material_id) {
            await loadMaterial(preset.material_id);
            $('materialSelect').value = preset.material_id;
          }
          applyCase(preset.case);
          setPreviewInvalid();
          $('casePresetNote').textContent = preset.evidence_context;
          showGlobalError();
        } catch (error) {
          showGlobalError(`Case preset: ${error.message}`);
        }
      });
    }
    $('materialSelect').addEventListener('change', event => loadMaterial(event.target.value));
    $('materialFile').addEventListener('change', async event => {
      const file = event.target.files?.[0];
      if (!file) return;
      markMaterialDirty();
      const loadRevision = state.materialLoadRevision;
      try {
        const text = await file.text();
        if (loadRevision !== state.materialLoadRevision) return;
        const parsed = JSON.parse(text);
        setMaterial(parsed, file.name);
        showGlobalError();
      } catch (error) {
        showGlobalError(`Material import: ${error.message}`);
      }
      event.target.value = '';
    });
    $('applyMaterial').addEventListener('click', applyMaterialFromEditor);
    $('downloadMaterial').addEventListener('click', () => {
      try {
        download(`${state.material?.material_id || 'material'}.json`, JSON.parse($('materialJson').value));
      } catch (error) {
        showGlobalError(`Material JSON: ${error.message}`);
      }
    });
    $('caseForm').addEventListener('input', setPreviewInvalid);
    $('materialJson').addEventListener('input', markMaterialDirty);
    document.querySelectorAll('input[name="weatherMode"]').forEach(node => {
      node.addEventListener('change', updateWeatherVisibility);
    });
    document.querySelectorAll('input[name="runtimePolicy"]').forEach(node => {
      node.addEventListener('change', () => {
        updateRuntimePolicyVisibility();
        setPreviewInvalid();
      });
    });
    document.querySelectorAll('input[name="temperaturePolicy"]').forEach(node => {
      node.addEventListener('change', setPreviewInvalid);
    });
    document.querySelectorAll('input[name="engine"]').forEach(node => {
      node.addEventListener('change', () => {
        setValue('solverCells', selectedEngine() === 'lumped' ? 1 : 4);
        updateEngineApplicability();
      });
    });
    $('weatherExample').addEventListener('change', event => {
      if (event.target.value) setValue('weatherStart', event.target.value);
    });
    $('previewButton').addEventListener('click', previewCase);
    $('startButton').addEventListener('click', startJob);
    $('cancelButton').addEventListener('click', cancelJob);
    $('applyConcurrency').addEventListener('click', applyConcurrency);
    $('refreshHistory').addEventListener('click', refreshHistory);
    $('compareButton').addEventListener('click', compareSelected);
    $('downloadComparison').addEventListener('click', () => {
      if (state.comparison) {
        download(`mof_screening_comparison_${Date.now()}.json`, state.comparison);
      }
    });
    $('editAndCopy').addEventListener('click', () => {
      if (!state.result) return;
      if (state.result.material_package) setMaterial(state.result.material_package, 'current');
      applyCase(state.result.case);
      setPreviewInvalid();
      $('configurationPanel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    state.liveProgressTimer = setInterval(() => {
      if (
        state.currentJobId
        && state.progressSnapshot
        && !state.progressSnapshot.terminal
      ) {
        $('runProgressDetails').textContent = progressDetail(
          state.progressSnapshot.document
        );
      }
    }, 1000);
  }

  window.mofScreeningApp = {
    getCurrentMaterial: () => state.material,
    applyMaterial: material => setMaterial(material, 'guided'),
    clearMaterialDirty,
    invalidatePreview: setPreviewInvalid,
    markMaterialDirty,
    materialRevision: () => materialGate.revision(),
    isMaterialRevisionCurrent: token => materialGate.isCurrent(token),
    showError: showGlobalError
  };

  async function init() {
    wire();
    showGlobalError();
    try {
      state.scope = await jsonRequest('/api/mof-screening/scope');
      state.concurrency = state.scope.concurrency;
      $('maxConcurrentJobs').value = String(state.concurrency?.max_concurrent_jobs ?? 0);
      const ready = state.scope.gate?.start_enabled === true;
      $('scopeStatus').textContent = ready
        ? `Screening diagnostic · ${state.scope.supported_families.length} material families`
        : 'Execution preflight in progress';
      $('scopeStatus').className = `screening-status ${ready ? 'ok' : 'warn'}`;
      const select = $('materialSelect');
      select.replaceChildren();
      for (const material of state.scope.defaults.built_in_materials) {
        const option = document.createElement('option');
        option.value = material.id;
        option.textContent = materialName(material.name);
        select.append(option);
      }
      const examples = $('weatherExample');
      for (const example of state.scope.defaults.official_weather_examples || []) {
        const option = document.createElement('option');
        option.value = example.start_utc;
        option.textContent = example.label;
        examples.append(option);
      }
      applyCase(state.scope.defaults.case);
      await loadMaterial(select.value);
      await refreshHistory();
      setInterval(refreshActiveJobs, 1500);
    } catch (error) {
      $('scopeStatus').textContent = 'Contract unavailable';
      $('scopeStatus').className = 'screening-status error';
      showGlobalError(error.message);
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
