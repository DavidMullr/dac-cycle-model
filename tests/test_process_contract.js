const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {validate,inventoryRecords,route,snapshotAt,sampleAt,phaseAt,boundedTime,fieldValues,fieldScale,fieldHue,cycleSummary,edgeActive}=require('../docs/assets/geometry/process-contract.js');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../docs/data/process_saved_cycles_v1.json'),'utf8'));
const humid=JSON.parse(fs.readFileSync(path.join(__dirname,'../docs/data/lewatit_humid_cycle.json'),'utf8'));
const raw=fs.readFileSync(path.join(__dirname,'..',humid.identity.original_result));
const native=JSON.parse(raw);
assert.equal(require('node:crypto').createHash('sha256').update(raw).digest('hex'),humid.identity.result_sha256);
const cycle=native.cycles.filter(c=>c.complete).at(-1);
assert.equal(humid.cycle,cycle.cycle);
assert.equal(humid.case.weather.relative_humidity,0.65);
const trace=native.traces.filter(t=>t.cycle===cycle.cycle);
assert.equal(inventoryRecords(humid).length,trace.length);
for (const [i, row] of trace.entries()) {
  const sample=humid.samples[i];
  assert.equal(sample.time_s,row.elapsed_s-trace[0].elapsed_s);
  const grams=row.co2_loading_mean_mol_kg*native.case.equipment.dry_shaped_mass_kg*44.0095;
  assert.ok(Math.abs(sample.solid_CO2_kg*1000-grams)<1e-9);
}
const layout=native.state_layout;
const states=[cycle.phase_evidence[0].state_start,...cycle.phase_evidence.map(p=>p.state_end)];
for (const [i, state] of states.entries()) {
  const snapshot=humid.snapshots[i];
  const inventory=state.slice(...layout.q).filter((_,j)=>j%layout.n_storage===0);
  const kg=inventory.reduce((a,b)=>a+b,0)/layout.n_cells_half*humid.case.equipment.dry_shaped_mass_kg*0.0440095;
  assert.ok(Math.abs(sampleAt(humid,snapshot.time_s).solid_CO2_kg-kg)<1e-9);
  assert.deepEqual(snapshot.solid_CO2_mol_kg,inventory);
  assert.equal(snapshot.collected_CO2_kg,2*state[layout.product[0]]*0.0440095);
  assert.equal(snapshot.electricity_kWh,2*state[layout.E_electric_total]/3.6e6);
}
assert.ok(sampleAt(humid,3600).solid_CO2_kg>humid.samples[0].solid_CO2_kg);
assert.ok(sampleAt(humid,13000).solid_CO2_kg<sampleAt(humid,9000).solid_CO2_kg);
for(const dry of data.scenarios)assert.equal(inventoryRecords(dry),dry.snapshots);
data.scenarios.unshift(humid);
const original=JSON.stringify(data);validate(data);
for(const s of data.scenarios){
  for(const phase of s.phases){
    const r=route(phase.phase);
    if(phase.phase==='adsorption'){assert.equal(r.blower,true);assert.equal(r.pumped,false);assert.ok(r.components.includes('exhaust'));}
    if(phase.phase==='evacuation'){assert.equal(r.vent,true);assert.equal(r.collection,false);assert.ok(!r.components.includes('collection'));}
    if(['desorption','cooling'].includes(phase.phase)){assert.equal(r.collection,true);assert.equal(r.vent,false);assert.equal(r.pumped,true);}
    if(phase.phase==='repressurization'){assert.equal(r.blower,false);assert.equal(r.pumped,false);assert.equal(r.passiveFill,true);assert.ok(!r.components.includes('outlet'));}
  }
  for(let i=1;i<s.snapshots.length;i++){
    const now=s.snapshots[i],before=s.snapshots[i-1];
    assert.equal(snapshotAt(s,(before.time_s+now.time_s)/2),before,'No invented continuous full field');
    assert.equal(snapshotAt(s,now.time_s),now);
  }
}
assert.equal(JSON.stringify(data),original);
assert.throws(()=>route('invented'));
const fill=route('repressurization');
assert.equal(edgeActive({from:'inlet',to:'bed',kind:'air'},fill),true);
assert.equal(edgeActive({from:'blower',to:'inlet',kind:'air'},fill),false);
assert.equal(edgeActive({from:'bed',to:'outlet',kind:'air'},fill),false);
const bad=JSON.parse(original);bad.scenarios[0].identity.source_sha256='';assert.throws(()=>validate(bad));
for(const corrupt of [d=>{delete d.scenarios[0].samples[0].solid_CO2_kg;},d=>{delete d.scenarios[0].samples[0].temperature_min_K;},d=>{d.scenarios[0].snapshots[1].solid_CO2_mol_kg.pop();},d=>{d.scenarios[0].phases[1].start_s+=1;}]){
  const invalid=JSON.parse(original);corrupt(invalid);assert.throws(()=>validate(invalid),'Incomplete evidence must fail before plotting');
}
for(const s of data.scenarios){
  assert.equal(boundedTime(s,-1),0);
  assert.equal(boundedTime(s,s.duration_s+1),s.duration_s);
  assert.equal(boundedTime(s,NaN),0);
  for(const phase of s.phases){
    assert.equal(phaseAt(s,phase.start_s).phase,phase.phase,'A phase boundary selects the next gas route');
    assert.equal(sampleAt(s,phase.start_s).phase,phase.phase,'Duplicate boundary samples select the new phase');
    assert.equal(phaseAt(s,(phase.start_s+phase.end_s)/2).phase,phase.phase);
  }
  for(let i=1;i<s.samples.length;i++){
    const a=s.samples[i-1],b=s.samples[i];
    if(b.time_s-a.time_s>1e-5)assert.equal(sampleAt(s,(a.time_s+b.time_s)/2),a,'Never interpolate a saved scalar');
  }
  assert.equal(sampleAt(s,s.duration_s),s.samples.at(-1));
  assert.equal(snapshotAt(s,s.duration_s),s.snapshots.at(-1));
  const summary=cycleSummary(s);
  assert.equal(summary.collected_CO2_g,s.snapshots.at(-1).collected_CO2_kg*1000,'Full-device product is converted once');
  assert.equal(summary.electricity_kWh,s.snapshots.at(-1).electricity_kWh);
  assert.equal(summary.cold_start_kg_day,s.metrics.collected_CO2_kg_day);
  assert.equal(summary.cycle_equivalent_kg_day,s.latest_cycle_equivalent_kg_day);
  for(const snapshot of s.snapshots){
    assert.deepEqual(fieldValues(snapshot,'temperature'),snapshot.bed_temperature_K.map(k=>k-273.15));
    assert.deepEqual(fieldValues(snapshot,'loading'),snapshot.solid_CO2_mol_kg);
    assert.deepEqual(fieldValues(snapshot,'composition'),snapshot.gas_CO2_mole_fraction.map(v=>v*100));
    const scale=fieldScale(data,'loading');
    assert.ok(Math.min(...fieldValues(snapshot,'loading'))>=scale[0]);
    assert.ok(Math.max(...fieldValues(snapshot,'loading'))<=scale[1]);
  }
}
assert.deepEqual(fieldScale(data,'temperature'),[25,130]);
assert.deepEqual(fieldScale(data,'composition'),[0,100]);
assert.equal(fieldHue(-10,[0,100]),205);assert.equal(fieldHue(110,[0,100]),10);
assert.equal(JSON.stringify(data),original,'Playback helpers leave saved results immutable');
console.log('PASS: phase boundaries, uneven-time playback, saved fields, common colour scales, comparison units and immutable results.');
