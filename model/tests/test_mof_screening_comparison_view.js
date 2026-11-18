'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const comparisonView = require(
  path.join(
    __dirname,
    '..',
    'app',
    'mof_screening',
    'static',
    'comparison_view.js'
  )
);

assert.equal(comparisonView.MAX_COMPARISON_JOBS, 32);
assert.equal(comparisonView.selectionError(1), 'Select at least two saved jobs.');
assert.equal(comparisonView.selectionError(2), null);
assert.equal(comparisonView.selectionError(32), null);
assert.match(comparisonView.selectionError(33), /no more than 32/);

const rows = Array.from({ length: 32 }, (_, index) => ({
  job_id: `${index.toString(16).padStart(8, '0')}${'a'.repeat(24)}`,
  material_id: index === 0 ? 'hob_conditional' : `material_${index}`,
  material_name: index === 0 ? 'HOB conditional powder' : `Material ${index}`,
  collected_CO2_kg_day: index === 1 ? 0.0003696 : 12.3456,
  electricity_kWh_day: 7.891,
  electricity_kWh_kg_CO2: 0.65432,
  dry_product_CO2_fraction: 0.8123,
  material_evidence_status:
    index === 0
      ? 'CONDITIONAL_REFERENCE_CURVE_REPRESENTATION_FAILED_DRY_ONLY'
      : 'CONDITIONAL_SOURCE_BACKED',
  numerical_status: 'qualified',
  conservation_status: 'pass',
  hard_limits_status: 'pass'
}));
const summary = comparisonView.summaryRows({ rows });

assert.equal(summary.length, 32);
assert.equal(summary[0].collectedKgDay, '12.346');
assert.equal(summary[1].collectedKgDay, '3.696e-4');
assert.equal(summary[0].dryProductFraction, '81.23%');
assert.equal(summary[0].evidence, 'representation failed');
assert.equal(summary[0].numerical, 'qualified');

const html = fs.readFileSync(
  path.join(__dirname, '..', 'app', 'mof_screening', 'static', 'index.html'),
  'utf8'
);
const app = fs.readFileSync(
  path.join(__dirname, '..', 'app', 'mof_screening', 'static', 'app.js'),
  'utf8'
);
assert.ok(html.indexOf('id="comparisonSummary"') < html.indexOf('id="comparisonTable"'));
assert.match(html, /Select 2–32 saved jobs/);
assert.match(html, /id="solverCells"[^>]*max="32"/);
assert.match(app, /MofComparisonView\.summaryRows/);
assert.match(app, /MofComparisonView\.MAX_COMPARISON_JOBS/);

console.log('comparison view tests passed');
