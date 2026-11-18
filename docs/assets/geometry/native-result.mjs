// Project the native result into saved cycle samples. No integration or interpolation.
const molarMass = 0.0440095; // kg CO₂/mol
const phases = ['adsorption', 'evacuation', 'desorption', 'cooling', 'repressurization'];
const closeTime = (a, b) => Math.abs(a - b) < 1e-5;
function requireData(ok, message) {
  if (!ok) throw Error(message);
}
function finite(value, field) {
  requireData(Number.isFinite(value), 'Missing or invalid ' + field + '.');
  return value;
}
function curveSample(time, phase, loading, temperature, mass) {
  requireData(phases.includes(phase), 'Unrecognized cycle phase.');
  return {time_s: finite(time, 'sample time'), phase, loading: finite(loading, 'CO₂ loading'),
    solid_CO2_kg: loading * mass * molarMass, temperature_K: finite(temperature, 'bed temperature')};
}
function traceSample(row, origin, mass, factor) {
  const sample = curveSample(row.elapsed_s - origin, row.phase, finite(row.co2_loading_mean_mol_kg, 'CO₂ loading') * factor,
    row.bed_temperature_mean_K, mass);
  return {...sample, pressure_Pa: finite(row.pressure_Pa, 'pressure'),
    temperature_min_K: finite(row.bed_temperature_min_K, 'minimum temperature'),
    temperature_max_K: finite(row.bed_temperature_max_K, 'maximum temperature')};
}
function checkCycle(cycle) {
  const segments = cycle.phases, points = cycle.samples;
  requireData(segments.length && points.length && cycle.duration_s > 0, 'Cycle has no recorded duration or samples.');
  let end = 0;
  for (const p of segments) {
    requireData(phases.includes(p.phase) && closeTime(p.start_s, end) && p.end_s >= p.start_s, 'Inconsistent phase times.');
    end = p.end_s;
  }
  requireData(closeTime(end, cycle.duration_s) && closeTime(points[0].time_s, 0)
    && closeTime(points.at(-1).time_s, end), 'Saved samples do not cover the cycle.');
  for (let i = 1; i < points.length; i++)
    requireData(points[i].time_s >= points[i-1].time_s - 1e-7, 'Saved samples are out of order.');
}
function spatialCycles(result, mass) {
  const layout = result.state_layout, adapter = result.material.adapter;
  const count = layout.n_cells_half, width = layout.n_storage, [base, end] = layout.q;
  requireData(Number.isInteger(width) && width > 0 && Number.isInteger(count) && count > 0
    && base >= 0 && end - base === count * width && end <= layout.size, 'Invalid spatial state layout.');
  if (!adapter.species_for_storage)
    requireData(Number.isInteger(adapter.co2_states) && adapter.co2_states > 0 && adapter.co2_states <= width, 'Missing CO₂ site count.');
  const sites = adapter.species_for_storage
    ? adapter.species_for_storage.flatMap((species, i) => species === 0 ? [i] : [])
    : Array.from({length:adapter.co2_states}, (_, i) => i);
  requireData(Number.isInteger(count) && count > 0 && sites.length && sites.every(i => i < width)
    && end - base === count * width, 'Unrecognized spatial CO₂ state layout.');
  const byCycle = new Map();
  for (const row of result.traces) {
    if (!byCycle.has(row.cycle)) byCycle.set(row.cycle, []);
    byCycle.get(row.cycle).push(row);
  }
  let origin = 0;
  const candidates = new Set([1, sites.length]);
  // Older multi-site traces averaged across sites. Resolve their basis against
  // all available phase endpoints before applying a factor to any interior point.
  const cycles = result.cycles.map(c => {
    const rows = byCycle.get(c.cycle) || [];
    let index = 0;
    for (const p of c.phase_evidence) {
      const endTime = origin + p.end_s;
      while (index < rows.length && rows[index].elapsed_s < endTime - 1e-5) index++;
      requireData(index < rows.length && closeTime(rows[index].elapsed_s, endTime), 'Missing spatial phase endpoint.');
      let sum = 0;
      for (let cell = 0; cell < count; cell++)
        for (const site of sites) sum += finite(p.state_end[base + cell * width + site], 'CO₂ site inventory');
      const expected = sum / count, observed = finite(rows[index].co2_loading_mean_mol_kg, 'CO₂ loading');
      for (const factor of candidates)
        if (Math.abs(factor * observed - expected) > 1e-9 + Math.abs(expected) * 1e-7) candidates.delete(factor);
    }
    const cycle = {cycle:c.cycle, complete:c.complete, start_s:origin, duration_s:c.cycle_time_s,
      phases:c.phase_evidence.map(({phase, start_s, end_s}) => ({phase, start_s, end_s})), rows};
    origin += cycle.duration_s;
    return cycle;
  });
  requireData(candidates.size, 'The saved CO₂ trace disagrees with the phase-end bed inventories.');
  const factor = candidates.has(1) ? 1 : sites.length;
  return cycles.map(({rows, ...c}) => ({...c, samples:rows.map(row => traceSample(row, c.start_s, mass, factor))}));
}
function lumpedCycles(result, mass) {
  const raw = result.raw_model_evidence.summary, sites = result.state_layout.material_sites;
  requireData(Number.isInteger(sites) && sites > 0, 'Unrecognized lumped CO₂ state layout.');
  function stateSample(state, t, phase) {
    const loading = state.slice(0, sites).reduce((sum, v) => sum + finite(v, 'CO₂ site inventory'), 0);
    const temperature = state[sites + 3];
    return {...curveSample(t, phase, loading, temperature, mass), temperature_min_K:temperature, temperature_max_K:temperature};
  }
  const segments = [...result.cycles], horizon = result.horizon.simulated_s, cursor = raw.cursor;
  const lastTime = segments.at(-1)?.end_elapsed_s ?? 0;
  if (!closeTime(lastTime, horizon))
    segments.push({cycle:cursor.completed_cycles + 1, phase:cursor.phase, start_elapsed_s:cursor.phase_start_elapsed_s, end_elapsed_s:horizon});
  const cycles = [], traces = result.traces;
  requireData(traces.length === result.cycles.length, 'Missing lumped phase-end trace.');
  let previous;
  segments.forEach((p, index) => {
    let c = cycles.at(-1);
    if (!c || c.cycle !== p.cycle) {
      c = {cycle:p.cycle, start_s:p.start_elapsed_s, complete:p.cycle <= result.horizon.cycles_completed, phases:[], samples:[]};
      c.samples.push(previous ? {...previous, time_s:0, phase:p.phase} : stateSample(raw.initial_state, 0, p.phase));
      cycles.push(c);
    }
    const end = p.end_elapsed_s - c.start_s;
    c.phases.push({phase:p.phase, start_s:p.start_elapsed_s - c.start_s, end_s:end});
    if (index < traces.length) {
      const row = traces[index];
      requireData(row.phase === p.phase && row.cycle === p.cycle && closeTime(row.elapsed_s, p.end_elapsed_s), 'Lumped phase and trace times disagree.');
      previous = traceSample(row, c.start_s, mass, 1);
    } else previous = stateSample(raw.final_state, end, p.phase);
    c.samples.push(previous);
    c.duration_s = end;
  });
  return cycles;
}
export function projectResult(result) {
  requireData(result?.schema === 'mof-screening-result/1', 'Choose a native screening result JSON exported from Materials & simulations.');
  requireData(result.horizon?.complete === true, 'This run did not reach its requested horizon.');
  const engine = result.engine, mass = finite(result.case?.equipment?.dry_shaped_mass_kg, 'dry sorbent mass');
  requireData(mass > 0 && ['spatial','lumped'].includes(engine), 'Unsupported engine or sorbent mass.');
  requireData(Array.isArray(result.cycles) && Array.isArray(result.traces), 'Missing saved run data.');
  const cycles = engine === 'spatial' ? spatialCycles(result, mass) : lumpedCycles(result, mass);
  let horizon = 0, maxLoading = 0;
  cycles.forEach((c, i) => {
    requireData(c.cycle === i + 1 && closeTime(c.start_s, horizon), 'Cycle sequence is inconsistent.');
    checkCycle(c);
    horizon += c.duration_s;
    for (const p of c.samples) maxLoading = Math.max(maxLoading, p.loading);
  });
  requireData(cycles.length && closeTime(horizon, result.horizon.simulated_s), 'Cycle times disagree with the saved run horizon.');
  const label = [result.case_label, result.material.name || result.material.material_id, engine].filter(Boolean).join(' · ');
  const eq = result.case.equipment;
  return {isNative:true, engine, label, inventory_sampling:'trace', cycles, maxLoading, horizon,
    materialId:result.material.material_id, evidence:result.material.evidence?.status || 'Screening result',
    diagnostic:result.failure ? String(result.failure.message || result.failure) : '',
    metrics:{collected_CO2_kg:finite(result.metrics.collected_CO2_kg,'collected CO₂'), electricity_kWh:finite(result.metrics.electricity_kWh,'metered electricity')},
    case:{equipment:{dry_shaped_mass_kg:eq.dry_shaped_mass_kg}, cycle:{purge_s:result.case.cycle.purge_s}, collection:{product_route:result.case.collection.product_route}}};
}
