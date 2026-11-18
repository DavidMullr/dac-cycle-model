'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const resultView = require(
  path.join(__dirname, '..', 'app', 'mof_screening', 'static', 'result_view.js')
);

const actual = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'residual_summary.json'), 'utf8'));

const legacy = resultView.residualSummary(
  actual.checks.mass_relative_residual_by_species
);
assert.equal(legacy.known, true);
assert.equal(legacy.basis, 'aggregate relative');
assert.equal(legacy.value, 3.4196465009431665e-15);

const current = resultView.residualSummary([1e-12, 4e-12]);
assert.equal(current.known, true);
assert.equal(current.basis, 'maximum relative');
assert.equal(current.value, 4e-12);

assert.deepEqual(resultView.residualSummary(undefined), {
  known: false,
  value: null,
  basis: 'unknown'
});
assert.equal(resultView.closureStatus(undefined), 'not recorded');
assert.equal(resultView.closureStatus(true), 'PASS');
assert.equal(resultView.closureStatus(false), 'FAIL');

function fakeLink() {
  return {
    href: 'old-job-link',
    attributes: {},
    removeAttribute(name) { if (name === 'href') this.href = null; },
    setAttribute(name, value) { this.attributes[name] = value; }
  };
}
const jsonLink = fakeLink();
const csvLink = fakeLink();
resultView.clearExportLinks([jsonLink, csvLink]);
assert.equal(jsonLink.href, null);
assert.equal(csvLink.href, null);
assert.equal(jsonLink.attributes['aria-disabled'], 'true');

const advisoryFact = resultView.previewTemperatureFact({
  material: { equilibrium_family: 'young2021_advisory_wadst_gab' },
  temperature_policy: {
    mode: 'record_scenario_excursions',
    scenario_guard_C: 120,
    registry_classification: 'unqualified_scenario_guard'
  }
});
assert.deepEqual(advisoryFact, {
  label: 'temperature policy',
  value: 'Record excursions',
  detail: 'measured support 25-70 C · former 120 C boundary unenforced'
});

const strictFact = resultView.previewTemperatureFact({
  material: { equilibrium_family: 'young2021_wadst_gab' },
  temperature_policy: {
    mode: 'stop_at_scenario_guard',
    scenario_guard_C: 70,
    registry_classification: 'unqualified_scenario_guard'
  }
});
assert.equal(strictFact.label, 'temperature guard');
assert.equal(strictFact.value, '70.0 C');
assert.match(strictFact.detail, /stop at scenario guard/);

const advisoryResultFact = resultView.resultTemperatureFact({
  material: {
    material_id: 'lewatit_vp_oc_1065_young2021_advisory_temperature_conditional'
  },
  temperature_guard_assessment: {
    status: 'scenario_excursion_recorded',
    peak_bed_temperature_C: 119.25,
    policy: { scenario_guard_C: 120 }
  }
});
assert.equal(advisoryResultFact.label, 'Temperature policy');
assert.match(advisoryResultFact.value, /Record excursions/);
assert.match(advisoryResultFact.detail, /former 120 C boundary unenforced/);

const strictResultFact = resultView.resultTemperatureFact({
  material: { material_id: 'lewatit_vp_oc_1065_young2021_wadst_conditional' },
  temperature_guard_assessment: {
    status: 'within_scenario_guard',
    peak_bed_temperature_C: 65,
    scenario_guard_excursion_C: 0,
    policy: { scenario_guard_C: 70 }
  }
});
assert.equal(strictResultFact.label, 'Temperature guard');
assert.match(strictResultFact.value, /within scenario guard/);

const app = fs.readFileSync(
  path.join(__dirname, '..', 'app', 'mof_screening', 'static', 'app.js'),
  'utf8'
);
assert.match(app, /function renderResult[\s\S]*clearExportLinks/);
assert.match(app, /setExportLink/);
assert.match(app, /Manual stop/);
assert.match(app, /progressDetail/);
assert.match(app, /Incomplete diagnostic retained/);
assert.match(app, /reattachJob/);
assert.match(app, /fetchHistoryAndActive/);
assert.match(app, /runPanel[\s\S]*hidden = false/);
// Capacity during active-job reattachment is covered by the functional concurrency tests.
assert.match(app, /knownNumber/);
assert.match(app, /performance\.now\(\)/);
assert.doesNotMatch(app, /state_dimension_estimate/);
assert.match(app, /Lumped, well mixed/);
assert.match(app, /CURRENT ·/);
assert.match(app, /previewTemperatureFact/);
assert.match(app, /resultTemperatureFact/);

console.log('result view tests passed');
