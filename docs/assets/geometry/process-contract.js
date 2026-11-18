/* Pure saved-result mapping. No solver, input mutation, or inferred flow magnitude. */
(function(factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else window.CarbonProcess = factory();
})(function() {
  'use strict';
  const routes = {
    adsorption: ['intake','blower','inlet','bed','outlet','exhaust'],
    evacuation: ['bed','regeneration','conditioner','knockout','pump','diverter','vent'],
    desorption: ['bed','regeneration','conditioner','knockout','pump','diverter','collection'],
    cooling: ['bed','regeneration','conditioner','knockout','pump','diverter','collection'],
    repressurization: ['fill','inlet','bed']
  };
  function validate(data) {
    if (data.schema !== 'carbon-process-saved-cycle/1' || !data.scenarios?.length) throw Error('Unsupported saved-cycle schema');
    for (const s of data.scenarios) {
      if (s.identity.engine !== 'spatial' || s.coordinates.cells_half !== 16 || s.coordinates.cells_half !== s.snapshots[0].bed_temperature_K.length ||
          !s.samples.length || !s.phases.every(p=>routes[p.phase]) || s.geometry.representation !== 'transverse_symmetric_half_bed_parallel_channels') throw Error('Unsupported saved spatial layout');
      for (const key of ['case_sha256','source_sha256','material_sha256','result_sha256']) {
        if (!/^[a-f0-9]{64}$/.test(s.identity[key])) throw Error('Missing saved identity');
      }
      if (!s.complete || s.duration_s <= 0 || Math.abs(s.samples.at(-1).time_s-s.duration_s)>1e-5) throw Error('Incomplete cycle');
      if (!Number.isFinite(s.duration_s)) throw Error('Invalid cycle duration');
      for (const p of s.samples) for (const key of ['time_s','temperature_K','temperature_min_K','temperature_max_K','pressure_Pa']) {
        if (!Number.isFinite(p[key])) throw Error('Missing or nonfinite saved scalar');
      }
      if (s.inventory_sampling === 'trace' && !s.samples.every(p=>Number.isFinite(p.solid_CO2_kg))) {
        throw Error('Missing saved CO₂ inventory');
      }
      for (const p of s.snapshots) {
        for (const key of ['bed_temperature_K','solid_CO2_mol_kg','solid_H2O_mol_kg','gas_CO2_mole_fraction','gas_H2O_mole_fraction']) {
          if (p[key]?.length!==s.coordinates.cells_half || !p[key].every(Number.isFinite)) throw Error('Incomplete saved bed field');
        }
        for (const key of ['time_s','collected_CO2_kg','electricity_kWh','solid_CO2_kg']) if (!Number.isFinite(p[key])) throw Error('Invalid saved endpoint');
      }
      if (s.snapshots[0].time_s!==0 || Math.abs(s.snapshots.at(-1).time_s-s.duration_s)>1e-5) throw Error('Incomplete field endpoints');
      for(let i=0;i<s.phases.length;i++) {
        const p=s.phases[i],start=i?s.phases[i-1].end_s:0;
        if(!Number.isFinite(p.start_s)||!Number.isFinite(p.end_s)||p.end_s<=p.start_s||Math.abs(p.start_s-start)>1e-6)throw Error('Invalid phase timeline');
      }
      if(Math.abs(s.phases.at(-1).end_s-s.duration_s)>1e-5)throw Error('Incomplete phase timeline');
      for (let i=1;i<s.samples.length;i++) if (s.samples[i].time_s<s.samples[i-1].time_s) throw Error('Unordered trace');
      for (let i=1;i<s.snapshots.length;i++) if(s.snapshots[i].time_s<s.snapshots[i-1].time_s)throw Error('Unordered field snapshots');
      if (Math.abs(s.phases.reduce((a,p)=>a+p.collected_CO2_kg,0)-s.snapshots.at(-1).collected_CO2_kg)>1e-9) throw Error('Product mapping mismatch');
    }
    return data;
  }
  function snapshotAt(s, t) { return s.snapshots.filter(p=>p.time_s<=t+1e-6).at(-1) || s.snapshots[0]; }
  // The clock is elapsed physical time, independent of the solver's uneven saves.
  // At a shared boundary, the last trace is the start of the new phase.
  function sampleAt(s, t) {
    let lo=0, hi=s.samples.length;
    while(lo<hi){const mid=(lo+hi)>>>1;if(s.samples[mid].time_s<=t+1e-6)lo=mid+1;else hi=mid;}
    return s.samples[Math.max(0,lo-1)];
  }
  function inventoryRecords(s) {
    return s.inventory_sampling === 'trace' ? s.samples : s.snapshots;
  }
  function phaseAt(s,t) { return s.phases.find(p=>t<p.end_s-1e-6) || s.phases.at(-1); }
  function boundedTime(s,t) { return Math.max(0,Math.min(s.duration_s,Number.isFinite(t)?t:0)); }
  const fields = {
    temperature:{label:'Solid temperature',unit:'°C',key:'bed_temperature_K',factor:1,offset:-273.15,digits:2},
    loading:{label:'Solid CO₂ loading',unit:'mol/kg',key:'solid_CO2_mol_kg',factor:1,offset:0,digits:4},
    composition:{label:'Gas CO₂ fraction',unit:'%',key:'gas_CO2_mole_fraction',factor:100,offset:0,digits:3}
  };
  function fieldValues(snapshot,field) {
    const spec=fields[field];if(!spec)throw Error('Unknown bed field');
    return snapshot[spec.key].map(v=>v*spec.factor+spec.offset);
  }
  function fieldScale(data,field) {
    if(field==='temperature')return [25,130];
    if(field==='composition')return [0,100];
    if(!fields[field])throw Error('Unknown bed field');
    // One scale across all loaded materials and times; selecting a case cannot
    // make a small loading look like a large loading by rescaling the colours.
    let lo=0,hi=0;
    for(const s of data.scenarios)for(const snapshot of s.snapshots)for(const v of fieldValues(snapshot,field)){lo=Math.min(lo,v);hi=Math.max(hi,v);}
    return [Math.floor(lo*10)/10,Math.max(.1,Math.ceil(hi*10)/10)];
  }
  function fieldHue(value,scale) { return 205-195*Math.max(0,Math.min(1,(value-scale[0])/(scale[1]-scale[0]))); }
  function cycleSummary(s) {
    const end=s.snapshots.at(-1);
    return {duration_s:s.duration_s,collected_CO2_g:end.collected_CO2_kg*1000,electricity_kWh:end.electricity_kWh,
      cold_start_kg_day:s.metrics.collected_CO2_kg_day,cycle_equivalent_kg_day:s.latest_cycle_equivalent_kg_day};
  }
  function route(phase, collecting) {
    if (!routes[phase]) throw Error('Unknown phase');
    const diverted = collecting === false && ['desorption','cooling'].includes(phase);
    return {components: diverted ? [...routes[phase].slice(0,-1), 'vent'] : routes[phase], pumped:['evacuation','desorption','cooling'].includes(phase),
      blower:phase==='adsorption', collection:!diverted && ['desorption','cooling'].includes(phase),
      vent:diverted || phase==='evacuation', passiveFill:phase==='repressurization'};
  }
  function edgeActive(e,r) {
    if(e.kind==='air')return r.blower||(r.passiveFill&&e.from==='inlet'&&e.to==='bed');
    return e.kind==='pump'?r.pumped:e.kind==='vent'?r.vent:e.kind==='product'?r.collection:e.kind==='fill'?r.passiveFill:false;
  }
  return {validate, inventoryRecords, snapshotAt, sampleAt, phaseAt, boundedTime, fields, fieldValues, fieldScale, fieldHue, cycleSummary, route, edgeActive};
});
