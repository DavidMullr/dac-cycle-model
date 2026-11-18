'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const presets = require(path.join(
  __dirname, '..', 'app', 'mof_screening', 'static', 'case_presets.js'
));

const defaults = JSON.parse(fs.readFileSync(path.join(
  __dirname, '..', 'config', 'mof_screening', 'defaults.json'
), 'utf8'));
const reviewed = JSON.parse(fs.readFileSync(path.join(
  __dirname, 'fixtures', 'dry_air_case.json'
), 'utf8'));
const app = fs.readFileSync(path.join(
  __dirname, '..', 'app', 'mof_screening', 'static', 'app.js'
), 'utf8');
const page = fs.readFileSync(path.join(
  __dirname, '..', 'app', 'mof_screening', 'static', 'index.html'
), 'utf8');

let appliedCase = null;
let invalidations = 0;
const selectedMaterial = { material_id: 'selected-material' };
const activeJob = { job_id: 'a'.repeat(32), state: 'running' };
const preset = presets.apply(
  defaults.case_presets,
  presets.DRY_AIR_48H_ID,
  value => { appliedCase = value; },
  () => { invalidations += 1; }
);

assert.deepEqual(preset.case, reviewed);
assert.deepEqual(appliedCase, reviewed);
assert.equal(invalidations, 1);
assert.equal(preset.case.runtime_policy.mode, 'manual_stop');
assert.equal(preset.case.weather.relative_humidity, 0);
assert.equal(preset.case.weather.hours, 48);
assert.equal(preset.case.schema, 'mof-screening-case/1');
assert.equal(selectedMaterial.material_id, 'selected-material');
assert.equal(activeJob.state, 'running');
assert.match(page, /Dry-air comparison baseline \(48 h\)/);
assert.match(page, /Humid exploration baseline \(48 h\)/);
assert.match(page, /preserves the current candidate selection/);
assert.match(page, /Hot Young dry 48 h preview/);
assert.match(page, /Hot Young ambient 30% RH 48 h preview/);
assert.match(page, /Young advisory-extrapolation RH0 48 h/);
assert.match(page, /Young advisory-extrapolation 30% RH 48 h/);
assert.match(page, /mmen independent-component RH0 48 h/);
assert.match(page, /mmen independent-component 30% RH 48 h/);
assert.match(page, /\/screening\/case_presets\.js/);
assert.match(app, /loadDryAirBaseline[\s\S]*MofCasePresets\.apply/);
assert.match(
  app,
  /loadHumidCandidate48h[\s\S]*applySelectedMaterial[\s\S]*HUMID_CANDIDATE_COMMON_RH050_48H_ID/
);
assert.match(app, /YOUNG2021_HOT116_RH0_48H_ID/);
assert.match(app, /YOUNG2021_HOT116_RH030_48H_ID/);
assert.match(app, /YOUNG2021_ADVISORY_RH0_48H_ID/);
assert.match(app, /YOUNG2021_ADVISORY_RH030_48H_ID/);
assert.match(app, /MMEN_INDEPENDENT_HUMID_RH0_48H_ID/);
assert.match(app, /MMEN_INDEPENDENT_HUMID_RH030_48H_ID/);
assert.match(
  app,
  /MofCasePresets\.apply[\s\S]*applyCase[\s\S]*setPreviewInvalid/
);
assert.throws(
  () => presets.resolve(defaults.case_presets, 'missing'),
  /Unknown case preset/
);

let humidAppliedCase = null;
let humidInvalidations = 0;
const humidPreset = presets.applySelectedMaterial(
  defaults.case_presets,
  presets.HUMID_CANDIDATE_COMMON_RH050_48H_ID,
  value => { humidAppliedCase = value; },
  () => { humidInvalidations += 1; }
);
assert.equal(humidPreset.material_id, undefined);
assert.equal(humidPreset.case.weather.relative_humidity, 0.5);
assert.equal(humidPreset.case.weather.hours, 48);
assert.deepEqual(humidAppliedCase, humidPreset.case);
assert.equal(humidInvalidations, 1);
assert.equal(selectedMaterial.material_id, 'selected-material');
assert.throws(
  () => presets.applySelectedMaterial(
    defaults.case_presets,
    presets.YOUNG2021_HUMID_SHORT_CYCLE_ID,
    () => {},
    () => {}
  ),
  /cannot preserve the current selection/
);
const humidHandlerStart = app.indexOf("$('loadHumidCandidate48h')");
const humidHandlerEnd = app.indexOf(
  'for (const [buttonId, presetId] of',
  humidHandlerStart
);
const humidHandler = app.slice(humidHandlerStart, humidHandlerEnd);
assert.ok(humidHandler.length > 0);
assert.doesNotMatch(humidHandler, /loadMaterial\(/);

for (const [presetId, materialId] of [
  [
    presets.YOUNG2021_ADVISORY_RH0_48H_ID,
    'lewatit_vp_oc_1065_young2021_advisory_temperature_conditional'
  ],
  [
    presets.YOUNG2021_ADVISORY_RH030_48H_ID,
    'lewatit_vp_oc_1065_young2021_advisory_temperature_conditional'
  ],
  [
    presets.MMEN_INDEPENDENT_HUMID_RH0_48H_ID,
    'mmen_mg2_dobpdc_mcdonald2012_conditional'
  ],
  [
    presets.MMEN_INDEPENDENT_HUMID_RH030_48H_ID,
    'mmen_mg2_dobpdc_mcdonald2012_conditional'
  ]
]) {
  const selected = presets.resolve(defaults.case_presets, presetId);
  assert.equal(selected.material_id, materialId);
  assert.ok(selected.case);
}

console.log('case preset tests passed');
