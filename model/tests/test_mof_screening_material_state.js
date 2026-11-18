'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createMaterialGate } = require(
  path.join(__dirname, '..', 'app', 'mof_screening', 'static', 'material_state.js')
);

async function main() {
  const gate = createMaterialGate(true);
  assert.equal(gate.blocksPreview(), true);
  gate.clear();
  assert.equal(gate.blocksPreview(), false);
  const unchangedToken = gate.revision();
  await Promise.resolve();
  assert.equal(gate.isCurrent(unchangedToken), true);
  gate.markPreviewValid('same-inputs');
  assert.equal(gate.canStart('same-inputs'), true);

  // Field edit, failed apply, curve-file change/clear, and closing an
  // unapplied guided form all keep the old material blocked.
  for (const action of ['field edit', 'failed apply', 'curve file', 'close']) {
    gate.markDirty();
    assert.equal(gate.blocksPreview(), true, action);
  }
  gate.clear();
  assert.equal(gate.blocksPreview(), false);

  // A deferred response is applied only when its revision and fingerprint
  // remain current.
  const pendingToken = gate.revision();
  let resolveResponse;
  const deferred = new Promise(resolve => { resolveResponse = resolve; });
  gate.markDirty(); // An edit arrives before the old request resolves.
  resolveResponse({ valid: true });
  const staleResponse = await deferred;
  assert.equal(staleResponse.valid, true);
  assert.equal(gate.isCurrent(pendingToken), false);
  assert.equal(gate.canStart('old-response'), false);

  gate.clear();
  const currentToken = gate.revision();
  const currentResponse = await Promise.resolve({ valid: true });
  assert.equal(currentResponse.valid, true);
  assert.equal(gate.isCurrent(currentToken), true);
  gate.markPreviewValid('current-response');
  assert.equal(gate.canStart('current-response'), true);

const builder = fs.readFileSync(
  path.join(__dirname, '..', 'app', 'mof_screening', 'static', 'material_builder.js'),
  'utf8'
);
const app = fs.readFileSync(
  path.join(__dirname, '..', 'app', 'mof_screening', 'static', 'app.js'),
  'utf8'
);
assert.match(builder, /guidedMaterialForm[\s\S]*markMaterialDirty/);
assert.match(builder, /applyGuidedMaterial[\s\S]*markMaterialDirty/);
assert.match(builder, /applyGuidedMaterial[\s\S]*materialRevision/);
assert.match(builder, /draftFingerprint/);
assert.match(builder, /loadCurveFile[\s\S]*markMaterialDirty/);
assert.match(builder, /closeBuilder[\s\S]*markMaterialDirty/);
assert.match(app, /previewCase[\s\S]*blocksPreview\(\)/);
assert.match(app, /startJob[\s\S]*blocksPreview\(\)/);
assert.match(app, /requestRevision[\s\S]*previewRevision/);
assert.match(app, /materialLoadRevision/);
  assert.match(app, /canStart\(fingerprint\)/);

  console.log('material state tests passed');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
