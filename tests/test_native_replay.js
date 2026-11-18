const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const C = require('../docs/assets/geometry/process-contract.js');
const results = path.join(__dirname,'../docs/data/submission/results');
const read = file => JSON.parse(fs.readFileSync(path.join(results, file),'utf8'));
function near(a, b) { assert.ok(Math.abs(a-b) < 1e-9, a+' differs from '+b); }

async function main() {
  const {projectResult} = await import('../docs/assets/geometry/native-result.mjs');
  for (const file of fs.readdirSync(results)) {
    const source = read(file), run = projectResult(source);
    const factor = ['cof1000_humid.json','calf20_humid.json'].includes(file) ? 2 : 1;
    const cycle = run.cycles[0], row = source.traces[1];
    near(cycle.samples[1].solid_CO2_kg, row.co2_loading_mean_mol_kg * factor * source.case.equipment.dry_shaped_mass_kg * 0.0440095);
    assert.equal(run.cycles.at(-1).complete, false);
    assert.equal(run.horizon, source.horizon.simulated_s);
    assert.equal(run.metrics.collected_CO2_kg, source.metrics.collected_CO2_kg);
    if (factor === 2) {
      for (const r of source.traces) r.co2_loading_mean_mol_kg *= 2;
      near(projectResult(source).cycles[0].samples[1].solid_CO2_kg, cycle.samples[1].solid_CO2_kg);
      source.traces.at(-1).co2_loading_mean_mol_kg += 1;
      assert.throws(()=>projectResult(source), /disagrees/);
    }
  }
  const lumped = {schema:'mof-screening-result/1',engine:'lumped',material:{material_id:'two-site'},
    case:{equipment:{dry_shaped_mass_kg:3},cycle:{purge_s:0},collection:{product_route:'all_desorption_and_cooling'}},
    horizon:{complete:true,simulated_s:3600,cycles_completed:0},state_layout:{material_sites:2},cycles:[],traces:[],
    metrics:{collected_CO2_kg:0,electricity_kWh:0.2},raw_model_evidence:{summary:{
      initial_state:[0,0,0,0,0,298], final_state:[0.1,0.3,0,0,0,298.1],
      cursor:{completed_cycles:0,phase:'adsorption',phase_start_elapsed_s:0}}}};
  const partial = projectResult(lumped).cycles[0];
  assert.equal(partial.complete,false);
  assert.deepEqual(partial.samples.map(p=>p.time_s),[0,3600]);
  near(C.sampleAt(partial,3599).solid_CO2_kg,0);
  near(C.sampleAt(partial,3600).solid_CO2_kg,0.4 * 3 * 0.0440095);
  assert.throws(()=>projectResult({...lumped,horizon:{complete:false}}), /horizon/);
  assert.throws(()=>projectResult(lumped.case), /native screening result/);
  assert.equal(C.route('desorption',false).components.at(-1),'vent');
  assert.equal(C.route('desorption',true).components.at(-1),'collection');
  console.log('PASS: native run replay, stored site basis, partial cycles, sparse lumped inventory and purge routing.');
}
main().catch(error=>{ console.error(error); process.exitCode=1; });
