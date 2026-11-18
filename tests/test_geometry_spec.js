'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const root=path.resolve(__dirname,'..');
const {fromSource}=require('../docs/assets/geometry/spec.js');
const source=JSON.parse(fs.readFileSync(path.join(root,'docs/data/architecture_cycle_pilot_scenario.json'),'utf8'));
const original=JSON.stringify(source);
const s=fromSource(source);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} differs from ${b}`);
near(s.mass,10);
near(s.pellet.width_m,10/(600*10*.024*.15));
near(s.pellet.width_m*s.pellet.bed_thickness_m*s.pellet.path_length_m*s.pellet.parallel_beds*s.pellet.bulk_density_kg_m3,s.mass);
near(s.pellet.mass_per_bed_kg,1);
near(s.channel.width_m,10/(2*900*.0005*100*.15));
near(s.channel.width_m*2*s.channel.coating_thickness_m*s.channel.parallel_units*s.channel.path_length_m*s.channel.coating_density_kg_m3,s.mass);
near(s.channel.pitch_m,.0032);
near(s.channel.stack_height_m,.32);
assert.equal(s.display.pellet_enlargement,4);
assert.equal(JSON.stringify(source),original,'The display must not mutate the saved scenario');
const clone=()=>JSON.parse(original);
for(const invalid of [null,0,-1,Infinity,'10',true]) {
  const bad=clone();bad.common.dry_composite_mass_kg=invalid;
  assert.throws(()=>fromSource(bad),/geometry/);
}
const missing=clone();missing.cases=[];assert.throws(()=>fromSource(missing),/unavailable/);
const tooMany=clone();tooMany.cases.find(row=>row.architecture==='coated_channel').geometry.parallel_units=201;
assert.throws(()=>fromSource(tooMany),/bounded viewer/);
const fractional=clone();fractional.cases.find(row=>row.architecture==='pellet').geometry.parallel_beds=1.5;
assert.throws(()=>fromSource(fractional),/bounded viewer/);
const oversized=clone();oversized.cases.find(row=>row.architecture==='pellet').geometry.pellet_diameter_m=.1;
assert.throws(()=>fromSource(oversized),/do not fit/);
const tooSmall=clone();tooSmall.cases.find(row=>row.architecture==='pellet').geometry.pellet_diameter_m=1e-9;
assert.throws(()=>fromSource(tooSmall),/display budget/);

(async()=>{
  const {airflowLayout,airflowPosition,airflowMarkerRadius}=await import('data:text/javascript;base64,'+fs.readFileSync(path.join(root,'docs/assets/geometry/airflow.js')).toString('base64'));
  for(const model of ['pellet','channel','bank'])for(const cutaway of [false,true])for(const explode of [false,true]) {
    const layout=airflowLayout(s,{model,cutaway,explode});
    assert.equal(layout.phase,'adsorption');assert.equal(layout.quantitative,false);
    assert.deepEqual(layout.direction,[0,0,-1]);
    for(const lane of layout.lanes) {
      near(lane.faceIn-lane.faceOut,1.5);
      assert.ok(lane.inlet>lane.faceIn && lane.outlet<lane.faceOut);
      assert.equal(airflowPosition(lane,.05).region,'inlet');
      assert.equal(airflowPosition(lane,.5).region,'bed');
      assert.equal(airflowPosition(lane,.95).region,'outlet');
      assert.ok(airflowPosition(lane,.2).z>airflowPosition(lane,.3).z);
      assert.ok(airflowMarkerRadius(lane,0)*2<lane.clearance,'Markers must fit within the open passage');
      near(airflowPosition(lane,-.1).z,airflowPosition(lane,.9).z);
      if(model==='channel') {
        const n=cutaway?24:s.channel.parallel_units;
        const pitch=s.channel.pitch_m*10*(s.channel.parallel_units/n)*(explode?1.7:1);
        const sheet=(2*s.channel.coating_thickness_m+s.channel.support_thickness_m)*10;
        const local=(lane.y-.06)%pitch;
        assert.ok(local>sheet && local<pitch,'Channel guide must stay in the open passage');
      } else {
        const base=model==='bank'?.10:0;
        const pitch=(s.pellet.bed_thickness_m+2*s.pellet.face_plate_thickness_m)*10+s.display.bank_spacing_m*10*(explode?2.8:1);
        const index=model==='bank'?Math.round((lane.y-base-(s.pellet.face_plate_thickness_m+s.pellet.bed_thickness_m/2)*10)/pitch):0;
        const bottom=base+index*pitch+s.pellet.face_plate_thickness_m*10;
        assert.ok(lane.y>bottom && lane.y<bottom+s.pellet.bed_thickness_m*10,'Bank guide must cross a bed, not the rack clearance');
      }
    }
    if(model==='bank')assert.equal(new Set(layout.lanes.map(l=>l.y)).size,s.pellet.parallel_beds);
  }
  assert.equal(JSON.stringify(source),original,'Flow guides must not mutate saved inputs');
  const nodes=new Map(),controls=[{disabled:false},{disabled:false}];
  const get=id=>{
    if(!nodes.has(id))nodes.set(id,{hidden:id==='gx-fallback',textContent:'',addEventListener(type,fn){this[type]=fn;}});
    return nodes.get(id);
  };
  let reloads=0;
  const boot=fs.readFileSync(path.join(root,'docs/assets/geometry/boot.js'),'utf8');
  vm.runInNewContext(boot,{document:{getElementById:get,querySelectorAll:()=>controls},window:{location:{reload(){reloads++;}}}},
    {importModuleDynamically:async()=>{throw new Error('Module unavailable');}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(get('gx-loading').hidden,true);
  assert.equal(get('gx-fallback').hidden,false);
  assert.match(get('gx-fallback-message').textContent,/could not be loaded/);
  assert.ok(controls.every(control=>control.disabled));
  get('gx-retry').click();assert.equal(reloads,1);
  console.log('PASS: source-derived geometry, flow direction and passage placement, immutable inputs, bounded invalid inputs and loading failure. No scientific execution.');
})().catch(error=>{console.error(error);process.exitCode=1;});
