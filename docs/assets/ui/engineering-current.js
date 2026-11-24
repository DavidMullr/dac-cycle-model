(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const format = (value, digits = 2) => typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString(undefined, {maximumFractionDigits: digits}) : 'Not available';
  let report = null;
  let generation = 0;
  let busy = false;
  const status = text => { $('engineering-status').textContent = text; };
  const node = (tag, text) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; return el; };
  const setBusy = value => {
    busy = value;
    for (const id of ['calculate-costs', 'save-costs', 'export-json', 'export-csv']) $(id).disabled = value || !report;
  };
  async function request(url, body) {
    const response = await fetch(url, body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type':'application/json'}, body:JSON.stringify(body)
    });
    if (!response.ok) {
      let message = 'Request failed';
      try { message = (await response.json()).error || message; } catch (_) {}
      throw new Error(message);
    }
    return response;
  }
  function input(type, value, placeholder = '') {
    const el = node('input'); el.type = type; el.value = value ?? ''; el.placeholder = placeholder;
    if (type === 'number') { el.min = '0'; el.step = 'any'; }
    return el;
  }
  function costs() {
    const parse = field => field.value.trim() === '' ? null : Number(field.value);
    return {electricity_CHF_kWh:parse($('electricity-price')),
      ...(report?.regeneration?.generator_carrier ? {
        external_heat_CHF_kWh:parse($('external-heat-price')), feedwater_CHF_m3:parse($('feedwater-price')),
        makeup_fraction:parse($('makeup-fraction'))} : {}),
      components:[...$('cost-inputs').children].map(row => ({name:row.dataset.name,
        quantity:parse(row.querySelector('[data-field="quantity"]')),
        unit_cost_CHF:parse(row.querySelector('[data-field="price"]')),
        basis:row.querySelector('[data-field="basis"]').value}))};
  }
  function paint(value, replaceInputs = true) {
    report = value; $('engineering-content').hidden = false;
    $('run-basis').textContent = `${value.binding.label} · ${format(value.simulated_hours, 1)} simulated hours · worksheet revision ${value.worksheet_revision}`;
    $('run-metrics').replaceChildren();
    for (const [title, metric, unit] of [
      ['Collected CO₂', value.production.collected_CO2_kg_day, 'kg/day'],
      ['Electricity', value.electricity.kWh_kg_CO2, 'kWh/kg CO₂'],
      ['Pump-inlet condensate', value.water.condensate_kg_day, 'kg/day']
    ]) {
      const card = node('article'); const figure = node('p', `${format(metric, 3)} ${unit}`); figure.className = 'current-metric';
      card.append(node('h3', title), figure); $('run-metrics').append(card);
    }
    const rows = [
      ['Dry shaped material', `${format(value.bed.dry_shaped_mass_kg)} kg`, 'Selected case'],
      ['Bed volume', `${format(value.bed.volume_m3 * 1000)} L`, 'Face area × full thickness'],
      ['Full bed thickness', `${format(value.bed.thickness_m * 1000)} mm`, 'Selected case'],
      ['Airflow', `${format(value.fan.airflow_m3_s * 3600)} m³/h`, 'Selected case'],
      ['Bed pressure loss', `${format(value.fan.bed_pressure_drop_max_Pa)} Pa`, value.fan.scope],
      ['Fan electrical input', `${format(value.fan.model_fan_input_max_W)} W`, 'Model pressure loss and scenario efficiency'],
      ...(value.regeneration?.generator_carrier ? [
        ['Steam generator input', `${format(value.regeneration.generator_input_kWh_day)} kWh/day`, value.regeneration.generator_carrier],
        ...(value.regeneration.method === 'direct_steam' ? [
          ['Injected steam', `${format(value.regeneration.injected_kg_day)} kg/day`, 'At the bed inlet'],
          ['Bed drain', `${format(value.regeneration.bed_drain_kg_day)} kg/day`, 'Free liquid leaving the bed']
        ] : [])
      ] : []),
      ['Heater power request', `${format(value.thermal.heater_request_W)} W`, 'Capacity requested in the selected case'],
      ['Combined cooling limit', `${format(value.thermal.combined_cooling_limit_W)} W`, `Scenario cooling COP ${format(value.thermal.cooling_COP)}`],
      ['Mean heater electrical input', `${format(value.thermal.heater_average_electric_W)} W`, 'Saved heater energy / complete horizon'],
      ['Mean cooling electrical input', `${format(value.thermal.cooling_average_electric_W)} W`, 'Saved cooling energy / complete horizon'],
      ['Peak bed temperature', `${format(value.thermal.observed_peak_bed_C)} °C`, 'Maximum reported observation over the run'],
      ['Regeneration pressure request', `${format(value.vacuum.regeneration_request_Pa / 1000)} kPa absolute`, 'Requested, not guaranteed'],
      ['Lowest saved cell pressure', value.vacuum.lowest_saved_cell_pressure_Pa === null ? 'Not available' : `${format(value.vacuum.lowest_saved_cell_pressure_Pa / 1000)} kPa absolute`, 'Sampled trace observation'],
      ['Pump running input', `${format(value.vacuum.running_input_W)} W`, 'Scenario reference'],
      ['Water carried in product', `${format(value.water.water_in_product_kg_day, 3)} kg/day`, 'Saved product meter / horizon'],
      ['Dry / wet CO₂ fraction', `${format(value.production.dry_product_CO2_fraction === null ? null : value.production.dry_product_CO2_fraction * 100)}% / ${format(value.production.wet_product_CO2_fraction === null ? null : value.production.wet_product_CO2_fraction * 100)}%`, 'Model collection boundary']
    ];
    $('duties').replaceChildren();
    for (const values of rows) { const tr = node('tr'); tr.append(...values.map(v => node('td', v))); $('duties').append(tr); }
    if (replaceInputs) {
      $('electricity-price').value = value.costs.electricity_CHF_kWh ?? '';
      $('steam-prices').hidden = !value.regeneration?.generator_carrier;
      $('external-heat-label').hidden = value.regeneration?.generator_carrier !== 'external_heat';
      $('external-heat-price').value = value.costs.external_heat_CHF_kWh ?? '';
      $('feedwater-price').value = value.costs.feedwater_CHF_m3 ?? '';
      $('makeup-fraction').value = value.costs.makeup_fraction ?? '';
      $('cost-inputs').replaceChildren();
      for (const component of value.costs.components) {
        const tr = node('tr'); tr.dataset.name = component.name; tr.append(node('td', component.name));
        for (const [field, type, initial, hint] of [
          ['quantity','number',component.quantity,''], ['price','number',component.unit_cost_CHF,'Not priced'],
          ['basis','text',component.basis,'Supplier quote or assumption']
        ]) { const td = node('td'); const el = input(type, initial, hint); el.dataset.field = field; el.setAttribute('aria-label', component.name + ' ' + field); td.append(el); tr.append(td); }
        $('cost-inputs').append(tr);
      }
    }
    const price = value.costs.listed_components_total_CHF;
    const complete = price === null ? `Incomplete: ${value.costs.unpriced_components.length} unpriced components.` : `Listed component total: CHF ${format(price)}.`;
    const electric = value.costs.electricity_cost_CHF_day === null ? 'Enter an electricity price to calculate its running cost.' : `Modelled electricity cost: CHF ${format(value.costs.electricity_cost_CHF_day)}/day.`;
    $('cost-summary').textContent = `Priced subtotal: CHF ${format(value.costs.known_subtotal_CHF)}. ${complete} ${electric}`;
    if (value.regeneration?.generator_carrier) {
      const utility = value.costs.utilities_CHF_day;
      $('cost-summary').textContent += utility == null ? ' Enter applicable utility prices and the makeup fraction for a combined cost.'
        : ` Electricity, heat and makeup water total CHF ${format(utility)}/day.`;
    }
    $('source-identity').textContent = JSON.stringify({result:value.binding.selector, sha256:value.binding.result_sha256, input_identity:value.binding.input_identity}, null, 2);
    $('open-inputs').replaceChildren(...value.open_inputs.map(text => node('li', text)));
  }
  async function loadSelected() {
    const ticket = ++generation; report = null; $('engineering-content').hidden = true; $('run-basis').textContent = '';
    setBusy(true); status('Loading selected run…');
    try {
      const data = await (await request('/api/engineering?result=' + encodeURIComponent($('selected-run').value))).json();
      if (ticket !== generation) return;
      paint(data); status('Engineering requirements loaded from the selected result.');
      history.replaceState(null, '', '?result=' + encodeURIComponent($('selected-run').value));
    } catch (error) { if (ticket === generation) status(error.message); }
    finally { if (ticket === generation) setBusy(false); }
  }
  async function action(kind) {
    if (busy || !report) return;
    const ticket = generation;
    const identity = report.binding.result_sha256;
    const body = {result:report.binding.selector, costs:costs(), expected_revision:report.worksheet_revision};
    setBusy(true);
    try {
      const response = await request('/api/engineering/' + kind, body);
      if (kind.startsWith('export')) {
        const blob = await response.blob();
        if (ticket !== generation) return;
        const url = URL.createObjectURL(blob); const link = node('a');
        link.href = url; link.download = 'engineering_' + identity.slice(0, 12) + (kind.endsWith('csv') ? '.csv' : '.json');
        document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
        status('Exported the selected run and current worksheet.');
      } else {
        const updated = await response.json();
        if (ticket !== generation) return;
        paint(updated, false); status(kind === 'save' ? 'Worksheet saved for this exact result.' : 'Calculation updated; changes are not saved yet.');
      }
    } catch (error) { if (ticket === generation) status(error.message); }
    finally { if (ticket === generation) setBusy(false); }
  }
  $('selected-run').addEventListener('change', loadSelected);
  $('calculate-costs').addEventListener('click', () => action('calculate'));
  $('save-costs').addEventListener('click', () => action('save'));
  $('export-json').addEventListener('click', () => action('export?format=json'));
  $('export-csv').addEventListener('click', () => action('export?format=csv'));
  request('/api/results').then(r => r.json()).then(data => {
    for (const row of data.results) { const option = node('option', row.label); option.value = row.selector; $('selected-run').append(option); }
    const selected = new URLSearchParams(location.search).get('result') || 'reviewed:lewatit_humid';
    if (data.results.some(row => row.selector === selected)) $('selected-run').value = selected;
    $('selected-run').disabled = data.results.length === 0;
    if (data.results.length) return loadSelected();
    status('No complete results are available yet.');
  }).catch(error => status(error.message));
})();
