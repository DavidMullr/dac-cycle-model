import {setupRunImport} from './add-run.js';
import {components} from './process-scene.js';

const $ = id => document.getElementById(id);
const pretty = phase => ({adsorption:'Adsorption',evacuation:'Evacuation',desorption:'Heating / desorption',cooling:'Cooling',repressurization:'Repressurization'}[phase] || phase);
const short = phase => ({desorption:'Heat & desorb',repressurization:'Refill'}[phase] || pretty(phase));
const colours = {adsorption:'#359c85',evacuation:'#638fba',desorption:'#ce914b',cooling:'#8e8ec2',repressurization:'#8d9d9d'};
const f = (n,d=3) => Number(n).toFixed(d);
const clock = t => {const whole=Math.floor(t+1e-6);return `${Math.floor(whole/3600)}:${String(Math.floor(whole%3600/60)).padStart(2,'0')}:${String(whole%60).padStart(2,'0')}`;};
const ns = 'http://www.w3.org/2000/svg';
function node(tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
function svg(tag,attrs={},text){const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;return el;}
const failed = s => s.evidence.status.includes('FAILED');
const evidenceLabel = s => {
  if (failed(s)) return 'Hot-temperature representation failed';
  return s.case.weather.relative_humidity > 0 ? 'Reviewed humid screening case' : 'Conditional dry model';
};
const replayURL = '/docs/data/lewatit_humid_cycle.json';
const dryReplayURL = '/docs/data/process_saved_cycles_v1.json';
async function loadSaved(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error('Saved replay unavailable: ' + url);
  return response.json();
}

export async function createPlayback({onChange,onFocus=()=>{},motionQuery}) {
  const [humid, dry] = await Promise.all([loadSaved(replayURL), loadSaved(dryReplayURL)]);
  const C = window.CarbonProcess;
  const data = C.validate({...dry, scenarios: [humid, ...dry.scenarios]});
  let inventoryDownload;
  let scenario=data.scenarios[0], comparison=null, time=0, selected='bed', field='temperature';
  let timer=null,lastTick=0,lastSnapshot=null,lastField=null,lastPhase=null;
  const reduced=motionQuery || window.matchMedia('(prefers-reduced-motion: reduce)');
  const events=new AbortController();
  const listen=(el,event,callback)=>el.addEventListener(event,callback,{signal:events.signal});
  const scales=Object.fromEntries(Object.keys(C.fields).map(key=>[key,C.fieldScale(data,key)]));
  const charts=new Map(), added=new Map();
  const originalTrendNote=$('px-trend-note').textContent;
  const atEnd=()=>scenario.isNative && !scenario.complete?'End of saved run':'Cycle complete';
  const W=360,H=150,L=39,R=10,T=17,B=27;
  const tabs=[...document.querySelectorAll('[data-px-tab]')];

  function showTab(name,focus=false){
    for(const tab of tabs){const active=tab.dataset.pxTab===name;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;$(tab.getAttribute('aria-controls')).hidden=!active;if(active&&focus)tab.focus();}
  }
  for(const tab of tabs){
    listen(tab,'click',()=>showTab(tab.dataset.pxTab));
    listen(tab,'keydown',e=>{const visible=tabs.filter(t=>!t.hidden);let i=visible.indexOf(tab);if(e.key==='ArrowRight')i=(i+1)%visible.length;else if(e.key==='ArrowLeft')i=(i+visible.length-1)%visible.length;else if(e.key==='Home')i=0;else if(e.key==='End')i=visible.length-1;else return;e.preventDefault();showTab(visible[i].dataset.pxTab,true);});
  }
  for(const s of data.scenarios){for(const id of ['px-scenario','px-compare']){const option=node('option',s.case.weather.relative_humidity ? s.label : s.label+' · dry');option.value=s.id;$(id).append(option);}}
  for(const [id,[title]] of Object.entries(components)){const option=node('option',title);option.value=id;$('px-component').append(option);}
  $('px-component').value=selected;

  function setPlayState(){
    $('px-play').textContent=timer?'Pause cycle':'Play cycle';$('px-play').setAttribute('aria-pressed',String(!!timer));
    $('px-cycle-label').textContent=`Saved cycle ${scenario.cycle} · ${time>=scenario.duration_s?(scenario.complete?'complete':'end of run'):timer?'playing':'paused'}`;
  }
  function stop(){if(timer!==null)clearInterval(timer);timer=null;setPlayState();}
  function seek(t){stop();time=C.boundedTime(scenario,t);update();}
  function play(){
    if(timer!==null){stop();return;}
    if(reduced.matches)return;
    if(time>=scenario.duration_s)time=0;
    lastTick=performance.now();
    timer=setInterval(()=>{const now=performance.now();time=C.boundedTime(scenario,time+(now-lastTick)/1000*Number($('px-speed').value));lastTick=now;update();if(time>=scenario.duration_s)stop();},100);
    setPlayState();update();
  }
  function selectComponent(id,focus=false){
    if(!components[id])return;
    selected=id;$('px-component').value=id;showTab('equipment');update();if(focus)onFocus(id);
  }
  function phaseButtons(){
    $('px-phases').replaceChildren();
    $('px-phase-track').querySelectorAll('.px-track-segment').forEach(el=>el.remove());
    for(const p of scenario.phases){
      const duration=p.end_s-p.start_s,b=node('button'),durationText=duration<60?`${f(duration,1)} s`:`${f(duration/60,1)} min`;
      b.type='button';b.dataset.phase=p.phase;b.style.setProperty('--phase-colour',colours[p.phase]);
      b.setAttribute('aria-label',`${pretty(p.phase)} · ${durationText}`);b.append(node('span',short(p.phase)),node('small',durationText));
      listen(b,'click',()=>seek(p.start_s));$('px-phases').append(b);
      const segment=node('div',undefined,'px-track-segment');segment.style.width=`${duration/scenario.duration_s*100}%`;segment.style.background=colours[p.phase];$('px-phase-track').append(segment);
    }
  }
  function series(s,type){
    if(type==='inventory')return C.inventoryRecords(s).map(p=>({time_s:p.time_s,value:p.solid_CO2_kg*1000}));
    if(type==='product'||type==='electricity')return s.snapshots.map(p=>({time_s:p.time_s,value:type==='product'?p.collected_CO2_kg*1000:p.electricity_kWh}));
    return s.samples.map(p=>({time_s:p.time_s,value:type==='temperature'?p.temperature_K-273.15:p.pressure_Pa/1000,
      min:type==='temperature'?p.temperature_min_K-273.15:undefined,max:type==='temperature'?p.temperature_max_K-273.15:undefined}));
  }
  function drawChart(id,type){
    const chart=$(id),step=['inventory','product','electricity'].includes(type),primary=series(scenario,type),other=comparison?series(comparison,type):null;
    const combined=[...primary,...(other||[])],values=combined.flatMap(p=>p.min===undefined?[p.value]:[p.value,p.min,p.max]);
    const maxTime=Math.max(scenario.duration_s,comparison?.duration_s||0);
    let lo=step?0:Math.min(...values),hi=Math.max(...values);const margin=Math.max((hi-lo)*.08,step?.001:.5);lo=step?0:lo-margin;hi+=margin;
    if(type==='pressure')lo=Math.min(0,Math.min(...values));
    const unit={temperature:'°C',pressure:'kPa',inventory:'g CO₂',product:'g CO₂',electricity:'kWh'}[type];
    const x=t=>L+t/maxTime*(W-L-R),y=v=>H-B-(v-lo)/(hi-lo)*(H-B-T);
    chart.replaceChildren();chart.setAttribute('viewBox',`0 0 ${W} ${H}`);
    chart.append(svg('title',{},`${unit} against elapsed cycle time. Solid: ${scenario.label}${comparison?`; dashed: ${comparison.label}`:''}. ${type==='inventory'?'Whole-bed adsorbed inventory at saved times.':step?'Held saved endpoints.':'Saved scalar samples.'}`));
    for(const p of scenario.phases)chart.append(svg('rect',{x:x(p.start_s),y:T,width:Math.max(0,x(p.end_s)-x(p.start_s)),height:H-B-T,fill:colours[p.phase],opacity:.055}));
    for(let i=0;i<3;i++){
      const v=lo+(hi-lo)*i/2,yy=y(v);chart.append(svg('line',{x1:L,x2:W-R,y1:yy,y2:yy,class:'px-chart-grid'}));
      chart.append(svg('text',{x:L-6,y:yy+4,'text-anchor':'end'},f(v,hi<5?2:0)));
    }
    chart.append(svg('text',{x:2,y:10},unit));
    for(let i=0;i<3;i++)chart.append(svg('text',{x:x(maxTime*i/2),y:H-7,'text-anchor':i===0?'start':i===2?'end':'middle'},`${f(maxTime*i/7200,1)} h`));
    const linePath=points=>points.map((p,i)=>`${i?(step?'H':'L'):'M'}${x(p.time_s)}${i&&step?`V${y(p.value)}`:` ${y(p.value)}`}`).join(' ');
    if(type==='temperature'){
      const upper=primary.map(p=>`${x(p.time_s)},${y(p.max)}`),lower=[...primary].reverse().map(p=>`${x(p.time_s)},${y(p.min)}`);
      chart.append(svg('polygon',{points:[...upper,...lower].join(' '),class:'px-chart-band'}));
    }
    if(other)chart.append(svg('path',{d:linePath(other),class:'px-chart-comparison'}));
    chart.append(svg('path',{d:linePath(primary),class:'px-chart-primary'}));
    for(const p of primary){const dot=svg('circle',{cx:x(p.time_s),cy:y(p.value),r:step?2.5:1.3,fill:'var(--ct-accent)',opacity:step?.8:.4});dot.append(svg('title',{},`${clock(p.time_s)} · ${f(p.value,3)} ${unit}`));chart.append(dot);}
    const cursor=svg('line',{x1:x(time),x2:x(time),y1:T-3,y2:H-B+2,class:'px-chart-cursor'}),point=svg('circle',{r:3.5,class:'px-chart-dot'});
    chart.append(cursor,point);charts.set(id,{chart,type,step,x,y,maxTime,cursor,point});
  }
  function paintPlots(){
    drawChart('px-inventory-plot','inventory');drawChart('px-temperature','temperature');if(scenario.engine!=='lumped')drawChart('px-pressure','pressure');else charts.delete('px-pressure');
    if(!scenario.isNative)drawChart('px-accumulation',$('px-meter-plot').value);else charts.delete('px-accumulation');
    const legend=$('px-chart-legend');legend.replaceChildren();
    for(const [s,cls] of [[scenario,''],[comparison,'px-comparator']])if(s){const label=node('span',undefined,cls);label.append(node('i'),document.createTextNode(s.label));legend.append(label);}
  }
  function moveCursors(sample,snapshot){
    for(const c of charts.values()){
      const p=c.type==='inventory'?(scenario.inventory_sampling==='trace'?sample:snapshot):(c.step?snapshot:sample);
      const v=c.type==='inventory'?p.solid_CO2_kg*1000:c.type==='temperature'?sample.temperature_K-273.15:c.type==='pressure'?sample.pressure_Pa/1000:c.type==='product'?snapshot.collected_CO2_kg*1000:snapshot.electricity_kWh;
      c.cursor.setAttribute('x1',c.x(time));c.cursor.setAttribute('x2',c.x(time));c.point.setAttribute('cx',c.x(p.time_s));c.point.setAttribute('cy',c.y(v));
    }
    let text=`Selected trace at ${clock(sample.time_s)} · ${f(sample.temperature_K-273.15,2)} °C${scenario.engine==='lumped'?'':' · '+f(sample.pressure_Pa/1000,2)+' kPa'}.`;
    if(comparison){const p=C.sampleAt(comparison,time);text+=` Comparison at ${clock(p.time_s)}: ${f(p.temperature_K-273.15,2)} °C · ${f(p.pressure_Pa/1000,2)} kPa${time>comparison.duration_s?' (cycle complete)':''}.`;}
    $('px-chart-readout').textContent=text;
  }
  for(const id of ['px-inventory-plot','px-temperature','px-pressure','px-accumulation']){
    listen($(id),'click',e=>{const c=charts.get(id);if(!c)return;const rect=c.chart.getBoundingClientRect();seek(((e.clientX-rect.left)/rect.width*W-L)/(W-L-R)*c.maxTime);});
    listen($(id),'keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();
      if(e.key==='Home')seek(0);else if(e.key==='End')seek(scenario.duration_s);else {
        const forward=e.key==='ArrowRight',points=id==='px-inventory-plot'?C.inventoryRecords(scenario):charts.get(id).step?scenario.snapshots:scenario.samples;
        const target=forward?points.find(p=>p.time_s>time+1e-6):points.filter(p=>p.time_s<time-1e-6).at(-1);
        seek(target?.time_s??(forward?scenario.duration_s:0));
      }
    });
  }
  function facts(rows,target){const dl=$(target);dl.replaceChildren();for(const [label,value] of rows){const group=node('div');group.append(node('dt',label),node('dd',value));dl.append(group);}}
  function inspector(phase,sample,snapshot){
    const r=C.route(phase),[title,description]=components[selected],active=r.components.includes(selected),eq=scenario.case.equipment;
    $('px-component-title').textContent=title;$('px-component-description').textContent=description;
    let state=active?'On active gas route':'Isolated from gas route';
    if(selected==='pump')state=r.pumped?'Motor on · withdrawal may be throttled':'Motor off';
    if(selected==='blower')state=r.blower?'Adsorption blower on':'Blower off';
    if(selected==='heater'||selected==='cooling')state='Thermal utility · phase policy below';
    if(selected==='drain')state='Drain interface · flow not sampled';
    $('px-component-state').textContent=state;$('px-component-state').dataset.active=String(active);
    const rows=[['Cycle phase',pretty(phase)]];
    if(selected==='bed')rows.push(['Mean solid temperature',`${f(sample.temperature_K-273.15,2)} °C · saved ${clock(sample.time_s)}`],['Mean absolute pressure',`${f(sample.pressure_Pa/1000,2)} kPa · saved ${clock(sample.time_s)}`],['Dry shaped inventory',`${eq.dry_shaped_mass_kg} kg`],['Bed thickness / face area',`${eq.bed_thickness_m*1000} mm / ${f(eq.bed_face_area_m2,6)} m²`],['Spatial field saved at',`${clock(snapshot.time_s)} · ${pretty(snapshot.phase)}`]);
    if(selected==='blower')rows.push(['Airflow request',`${eq.airflow_m3_s} actual m³/s during adsorption`]);
    if(selected==='pump')rows.push(['Running input','135 W (assumed)'],['Regeneration pressure request',`${scenario.case.cycle.regeneration_pressure_Pa/1000} kPa absolute; advisory`]);
    if(selected==='heater')rows.push(['Installed power',`${eq.heater_power_W/1000} kW (assumed)`],['Heater permission',scenario.phase_policy.phases[phase].heater],['Control request',scenario.phase_policy.phases[phase].request],['Instantaneous duty','Not available in the saved trace']);
    if(selected==='cooling'||selected==='conditioner')rows.push(['Combined cooling budget','2 kW thermal; COP 3 (assumed)'],['Allocation','Conditioner first; remainder to bed'],['Conditioner target','55 °C; actual inlet temperature not sampled']);
    if(selected==='intake'||selected==='fill')rows.push(['Feed humidity',`${f(100*scenario.case.weather.relative_humidity,0)}% RH`],['Feed CO₂',`${scenario.case.weather.co2_ppm_dry} ppm dry`]);
    if(selected==='collection')rows.push(['Cycle collected at saved endpoint',`${f(snapshot.collected_CO2_kg*1000,3)} g at ${clock(snapshot.time_s)}`],['48 h dry / wet CO₂ fraction',`${f(scenario.metrics.dry_product_CO2_fraction*100,2)}% / ${f(scenario.metrics.wet_product_CO2_fraction*100,2)}%`]);
    if(selected==='knockout'||selected==='drain')rows.push(['48 h modeled condensed water',`${f(scenario.metrics.water_condensed_mol,6)} mol`],['48 h water in product',`${scenario.metrics.water_in_product_mol.toExponential(2)} mol`]);
    facts(rows,'px-component-facts');
    if(phase!==lastPhase){
      const route=$('px-route-list');route.replaceChildren();r.components.forEach((id,i)=>{if(i)route.append(node('span','→'));const b=node('button',components[id][0]);b.type='button';listen(b,'click',()=>selectComponent(id));route.append(b);});
      $('px-route-note').textContent=r.passiveFill?'Passive ambient filling. Blower and vacuum pump are off.':r.vent?'Evacuation discharge is vented; it is excluded from collected CO₂.':r.collection?'Discharge reaches the collection interface. Pump withdrawal may be throttled to zero.':'Ambient air crosses the bed and leaves through the adsorption exhaust.';
      lastPhase=phase;
    }
  }
  function fields(snapshot){
    const spec=C.fields[field],values=C.fieldValues(snapshot,field),scale=scales[field],lo=Math.min(...values),hi=Math.max(...values),profile=$('px-profile');
    const timestamp=`Saved ${clock(snapshot.time_s)} · ${pretty(snapshot.phase)}. Held until the next phase boundary.`;
    $('px-field-title').textContent=spec.label;$('px-field-time').textContent=timestamp;$('px-cell-time').textContent=timestamp;
    $('px-field-key-title').textContent=spec.label;$('px-field-key-range').textContent=`${f(scale[0],field==='loading'?1:0)}–${f(scale[1],field==='loading'?1:0)} ${spec.unit}`;
    $('px-stage-field-time').textContent=`Bed field saved at ${clock(snapshot.time_s)} · ${pretty(snapshot.phase)}`;
    $('px-field-min').textContent=`${f(lo,spec.digits)} ${spec.unit}`;$('px-field-max').textContent=`${f(hi,spec.digits)} ${spec.unit}`;
    profile.replaceChildren();profile.setAttribute('viewBox','0 0 360 220');profile.setAttribute('aria-label',`${spec.label} across the transverse half-bed, saved at ${clock(snapshot.time_s)}`);
    const width=300,x0=42,plotLo=Math.min(scale[0],lo),plotHi=Math.max(scale[1],hi),y=v=>175-(v-plotLo)/(plotHi-plotLo)*95;
    values.forEach((value,i)=>{const cell=svg('rect',{x:x0+i*width/values.length,y:18,width:width/values.length-.6,height:34,fill:`hsl(${C.fieldHue(value,scale)} 60% 48%)`});cell.append(svg('title',{},`Cell ${i+1} · ${f(value,spec.digits)} ${spec.unit}`));profile.append(cell);});
    for(let i=0;i<3;i++){const v=plotLo+(plotHi-plotLo)*i/2;profile.append(svg('line',{x1:x0,x2:x0+width,y1:y(v),y2:y(v),class:'px-chart-grid'}),svg('text',{x:x0-6,y:y(v)+4,'text-anchor':'end'},f(v,plotHi<10?1:0)));}
    const x=i=>x0+(i+.5)*width/values.length;
    profile.append(svg('polyline',{points:values.map((v,i)=>`${x(i)},${y(v)}`).join(' '),class:'px-chart-primary'}));
    values.forEach((v,i)=>profile.append(svg('circle',{cx:x(i),cy:y(v),r:2.5,class:'px-chart-dot'})));
    profile.append(svg('text',{x:x0,y:70},spec.unit),svg('text',{x:x0,y:198},'Centre · 0 mm'),svg('text',{x:x0+width,y:198,'text-anchor':'end'},`Wall · ${scenario.case.equipment.bed_thickness_m*500} mm`));
    const body=$('px-cells');body.replaceChildren();snapshot.bed_temperature_K.forEach((t,i)=>{
      const tr=node('tr');for(const value of [i+1,f(scenario.coordinates.cell_centers_mm[i],3),f(t-273.15,3),f(snapshot.solid_CO2_mol_kg[i],5),snapshot.solid_H2O_mol_kg[i].toExponential(2),f(snapshot.gas_CO2_mole_fraction[i]*100,5),snapshot.gas_H2O_mole_fraction[i].toExponential(2)])tr.append(node('td',String(value)));body.append(tr);
    });
    $('px-solid').textContent=`${f(snapshot.solid_CO2_kg,6)} kg CO₂ held in the solid at this boundary.`;
  }
  function comparisons(){
    const scenarios=[scenario,...(comparison?[comparison]:[])],table=$('px-comparison'),head=node('tr');head.append(node('th','Quantity'));
    for(const s of scenarios)head.append(node('th',s.label));table.tHead.replaceChildren(head);
    const summaries=scenarios.map(C.cycleSummary),rows=[['Cycle duration','duration_s',v=>clock(v)],['Collected / cycle','collected_CO2_g',v=>`${f(v,3)} g`],['Electricity / cycle','electricity_kWh',v=>`${f(v,4)} kWh`],['48 h cold-start average','cold_start_kg_day',v=>`${f(v,4)} kg/day`],['Cycle equivalent','cycle_equivalent_kg_day',v=>`${f(v,4)} kg/day`]];
    const body=table.tBodies[0];body.replaceChildren();for(const [label,key,format] of rows){const tr=node('tr');tr.append(node('th',label));for(const s of summaries)tr.append(node('td',format(s[key])));body.append(tr);}
    $('px-comparison-note').textContent=comparison?'Complete saved cycles on the same full-device basis. The 3D scene follows the selected material.':'Choose a second material above to overlay its trends and compare the saved cycle totals.';
    const evidence=$('px-comparison-evidence');evidence.replaceChildren();for(const s of scenarios){const p=node('p',`${s.label}: ${evidenceLabel(s)}. ${failed(s)?'The hot-temperature fit fails its source-data holdouts; do not use this case to rank material performance.':'Cyclic steady state, mesh convergence and hardware performance remain unqualified.'}`);p.dataset.failed=String(failed(s));evidence.append(p);}
  }
  function meter(id,value,unit){const el=$(id);el.replaceChildren(document.createTextNode(value+' '),node('em',unit));}
  function update(){
    const state=current(), {sample,snapshot,phase}=state;
    $('px-time').value=String(time);$('px-time').setAttribute('aria-valuetext',`${clock(time)}, ${pretty(phase)}`);
    $('px-clock').textContent=clock(time);$('px-current-phase').textContent=time>=scenario.duration_s?atEnd():pretty(phase);
    $('px-phase-progress').style.left=`calc(${time/scenario.duration_s*100}% - 1px)`;
    const inventory=scenario.inventory_sampling==='trace'?sample:snapshot;
    meter('px-inventory',f(inventory.solid_CO2_kg*1000,2),'g');
    $('px-inventory-time').textContent=`${scenario.inventory_sampling==='trace'?'Sample':'Phase endpoint'} at ${clock(inventory.time_s)}`;
    meter('px-temp-value',f(sample.temperature_K-273.15,2),'°C');if(scenario.engine!=='lumped')meter('px-pressure-value',f(sample.pressure_Pa/1000,2),'kPa');
    const meters=scenario.isNative?scenario.metrics:snapshot;
    meter('px-product',f(meters.collected_CO2_kg*1000,3),'g');meter('px-electric',f(meters.electricity_kWh,4),'kWh');
    for(const id of ['px-scalar-time','px-pressure-time'])$(id).textContent=`Trace saved at ${clock(sample.time_s)}`;
    for(const id of ['px-product-time','px-electric-time'])$(id).textContent=scenario.isNative?'Run total at '+clock(scenario.horizon):`Endpoint saved at ${clock(snapshot.time_s)}`;
    $('px-phases').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.phase===phase)));
    $('px-previous').disabled=time<=0;$('px-next').disabled=time>=scenario.duration_s;
    if(scenario.isNative) {
      $('px-stage-field-time').textContent='Whole-bed mean saved at '+clock(sample.time_s);
      const route=C.route(phase,state.collecting);
      $('px-component-title').textContent=components[selected][0];
      $('px-component-description').textContent='Illustrative equipment arrangement.';
      $('px-component-state').textContent=route.components.includes(selected)?'On active gas route':'Outside active gas route';
      $('px-component-state').dataset.active=String(route.components.includes(selected));
      facts([['Cycle phase',pretty(phase)],['Dry sorbent',scenario.case.equipment.dry_shaped_mass_kg+' kg']], 'px-component-facts');
      $('px-route-list').textContent=route.components.map(id=>components[id][0]).join(' → ');
      $('px-route-note').textContent='Flow arrows show the route in the selected phase.';
    } else {
      inspector(phase,sample,snapshot);
      if(snapshot!==lastSnapshot||field!==lastField){fields(snapshot);lastSnapshot=snapshot;lastField=field;}
    }
    moveCursors(sample,snapshot);setPlayState();
    onChange(state);
  }
  function current(){
    const sample=C.sampleAt(scenario,time), phase=C.phaseAt(scenario,time);
    if(!scenario.isNative) return {scenario,sample,selected,time,phase:phase.phase,field,fieldScale:scales[field],snapshot:C.snapshotAt(scenario,time)};
    const collecting=(phase.phase==='desorption' && time-phase.start_s>=scenario.case.cycle.purge_s) ||
      (phase.phase==='cooling' && scenario.case.collection.product_route==='all_desorption_and_cooling');
    return {scenario,sample,selected,time,phase:phase.phase,field:'loading',fieldScale:[0,Math.max(scenario.maxLoading,1e-12)],
      meanLoading:sample.loading,snapshot:null,collecting};
  }
  function selectCycle(run,number){
    const cycle=run.cycles.find(c=>c.cycle===number);
    scenario={...run,...cycle};$('px-run-cycle').value=String(number);scenarioChanged();
  }
  function openCase(id){
    $('px-scenario').value=id;
    const run=added.get(id);
    if(!run){
      scenario=data.scenarios.find(s=>s.id===id);scenarioChanged();return;
    }
    comparison=null;$('px-compare').value='';
    const choices=$('px-run-cycle');choices.replaceChildren();
    for(const c of run.cycles)choices.add(new Option('Cycle '+c.cycle+(c.complete?'':' · partial'),String(c.cycle)));
    selectCycle(run,(run.cycles.findLast(c=>c.complete)||run.cycles.at(-1)).cycle);
  }
  function setMode(){
    const native=!!scenario.isNative, lumped=scenario.engine==='lumped';
    $('px-cycle-choice').hidden=$('px-remove-run').hidden=$('px-import-details').hidden=!native;
    for(const id of ['px-compare','px-field'])$(id).parentElement.hidden=native;
    for(const id of ['px-tab-bed','px-tab-compare'])$(id).hidden=native;
    for(const d of $('px-details').children)if(d.tagName==='DETAILS')d.hidden=native;
    $('px-pressure').closest('figure').hidden=$('px-pressure-value').parentElement.hidden=lumped;
    $('px-accumulation').closest('figure').hidden=native;
    $('px-pressure-label').textContent=native?'Maximum cell pressure':'Mean absolute pressure';
    $('px-pressure-stat').textContent=native?'Maximum across cells':'Mean';
    $('px-temp-stat').textContent=lumped?'Lumped bed':'Mean + cell range';
    $('px-product-label').textContent=native?'Collected over full run':'Collected in this cycle';
    $('px-electric-label').textContent=native?'Electricity over full run':'Electricity in this cycle';
    $('px-trend-note').textContent=native?(lumped?'The lumped model saves the initial state, phase endpoints and final state. ':'')+
      'The CO₂ readout holds the most recent saved mass. Select a point or use the arrow keys on the inventory graph to step through saved samples.':originalTrendNote;
    if(native)showTab('trends');
  }
  function makeDownload(){
    if (inventoryDownload) URL.revokeObjectURL(inventoryDownload);
    const csv = ['elapsed_cycle_s,phase,adsorbed_CO2_g', ...C.inventoryRecords(scenario).map(
      p => [p.time_s, p.phase, p.solid_CO2_kg*1000].join(',')
    )].join('\r\n');
    inventoryDownload = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
    $('px-inventory-csv').href = inventoryDownload;
    $('px-inventory-csv').download = scenario.id+'-cycle-'+scenario.cycle+'-bed-co2.csv';
    $('px-added-csv').href=inventoryDownload;
    $('px-added-csv').download=$('px-inventory-csv').download;
  }
  function scenarioChanged(){
    stop();time=0;lastSnapshot=null;lastPhase=null;
    if(comparison===scenario){comparison=null;$('px-compare').value='';}
    for(const option of $('px-compare').options)option.disabled=option.value===scenario.id;
    $('px-time').max=String(scenario.duration_s);$('px-clock-total').textContent=`of ${clock(scenario.duration_s)}`;
    setMode();
    makeDownload();
    if(scenario.isNative) {
      $('px-evidence-flag').textContent=[scenario.evidence,scenario.diagnostic].filter(Boolean).join(' · ');
      $('px-evidence-flag').dataset.failed=String(Boolean(scenario.diagnostic));
      $('px-scenario-context').textContent=scenario.engine+' · '+scenario.case.equipment.dry_shaped_mass_kg+' kg dry sorbent';
      $('px-inventory-basis').textContent=scenario.engine==='lumped'?'Whole bed · saved boundaries':'Whole bed · saved trace';
      $('px-field-key-title').textContent='Mean adsorbed CO₂';
      $('px-field-key-range').textContent='0–'+f(scenario.maxLoading,3)+' mol/kg';
      $('px-import-source').textContent=scenario.materialId+' · result SHA-256 '+scenario.id;
      phaseButtons();paintPlots();update();
      return;
    }
    $('px-evidence-flag').textContent=evidenceLabel(scenario);$('px-evidence-flag').dataset.failed=String(failed(scenario));
    $('px-average').textContent=`${f(scenario.metrics.collected_CO2_kg_day,6)} kg/day · 48 h cold-start average`;
    $('px-latest').textContent=`${f(scenario.latest_cycle_equivalent_kg_day,6)} kg/day · cycle equivalent; cyclic steady state unqualified`;
    $('px-total').textContent=`${f(scenario.metrics.electricity_kWh_day,3)} kWh/day · ${f(scenario.metrics.electricity_kWh_kg_CO2,2)} kWh/kg CO₂`;
    $('px-evidence').textContent=`${scenario.evidence.basis} Not established: ${scenario.evidence.not_for}`;
    const dense = scenario.inventory_sampling === 'trace';
    $('px-inventory-basis').textContent = dense ? 'Whole bed · saved trace' : 'Whole bed · phase endpoints';
    $('px-scenario-context').textContent = `${100*scenario.case.weather.relative_humidity}% RH · ${scenario.case.equipment.dry_shaped_mass_kg} kg dry sorbent`;
    $('px-replay-json').href = dense ? replayURL : dryReplayURL;
    $('px-native-json').href = dense ? '/' + scenario.identity.original_result : `/api/mof-screening/jobs/${encodeURIComponent(scenario.identity.job_id)}/export?format=json`;
    $('px-native-csv').hidden = dense;
    if (!dense) $('px-native-csv').href = `/api/mof-screening/jobs/${encodeURIComponent(scenario.identity.job_id)}/export?format=csv`;
    const list=$('px-identities');list.replaceChildren();for(const [key,value] of Object.entries(scenario.identity)){const p=node('p');p.append(node('strong',`${key}: `),document.createTextNode(String(value)));list.append(p);}
    const body=$('px-phase-ledger');body.replaceChildren();for(const p of scenario.phases){const tr=node('tr');for(const v of [pretty(p.phase),f((p.end_s-p.start_s)/60,2),f(p.collected_CO2_kg*1000,3),f(p.electricity_kWh.heater,5),f(p.electricity_kWh.pump,5),f(p.electricity_kWh.blower,5),f(p.electricity_kWh.cooling_and_auxiliaries,5),f(p.electricity_kWh.total,5)])tr.append(node('td',v));body.append(tr);}
    phaseButtons();paintPlots();comparisons();update();$('px-replay').dataset.ready='true';
  }
  listen($('px-scenario'),'change',()=>openCase($('px-scenario').value));
  listen($('px-run-cycle'),'change',()=>selectCycle(added.get(scenario.id),Number($('px-run-cycle').value)));
  setupRunImport({listen,selectedId:()=>scenario.isNative?scenario.id:null,
    add(run,open){
      if(!added.has(run.id)){
        added.set(run.id,run);
        $('px-scenario').add(new Option(run.label+' · '+run.id.slice(0,8),run.id));
      }
      if(open)openCase(run.id);
    },
    remove(id){
      added.delete(id);
      Array.from($('px-scenario').options).find(o=>o.value===id)?.remove();
      if(scenario.id===id)openCase(data.scenarios[0].id);
    }
  });
  listen($('px-compare'),'change',()=>{comparison=data.scenarios.find(s=>s.id===$('px-compare').value)||null;paintPlots();comparisons();update();showTab('trends');});
  listen($('px-component'),'change',()=>selectComponent($('px-component').value));
  listen($('px-time'),'input',()=>seek(Number($('px-time').value)));
  listen($('px-play'),'click',play);listen($('px-restart'),'click',()=>seek(0));
  listen($('px-next'),'click',()=>seek(scenario.phases.find(p=>p.start_s>time+1e-6)?.start_s??scenario.duration_s));
  listen($('px-previous'),'click',()=>seek(scenario.phases.filter(p=>p.start_s<time-1e-6).at(-1)?.start_s??0));
  listen($('px-field'),'change',()=>{field=$('px-field').value;update();showTab('bed');});
  listen($('px-meter-plot'),'change',()=>{drawChart('px-accumulation',$('px-meter-plot').value);update();});
  listen($('px-focus-component'),'click',()=>onFocus(selected));
  listen($('px-show-bed'),'click',()=>{selected='bed';$('px-component').value='bed';update();onFocus('bed');});
  listen(document.querySelector('a[href="#px-cell-details"]'),'click',()=>{$('px-cell-details').open=true;});
  listen($('px-focus'),'click',()=>{const enabled=document.body.classList.toggle('px-focus');$('px-focus').setAttribute('aria-pressed',String(enabled));$('px-focus').textContent=enabled?'Restore workspace':'Expand workspace';});
  listen(document,'visibilitychange',()=>{if(document.hidden)stop();});
  listen(document,'keydown',e=>{if(e.key!==' '||e.ctrlKey||e.metaKey||e.altKey||$('px-replay').hidden)return;if(e.target.closest('input,select,textarea,button,a,summary,[role="tab"]'))return;e.preventDefault();play();});
  const motionChanged=()=>{if(reduced.matches)stop();$('px-play').disabled=reduced.matches;$('px-play').title=reduced.matches?'Automatic playback is paused by your reduced-motion preference. Use the timeline and phase buttons.':'Play or pause saved cycle (Space)';};
  reduced.addEventListener('change',motionChanged);motionChanged();
  return {start:scenarioChanged,select:selectComponent,
    current,
    stop,dispose(){stop();if(inventoryDownload)URL.revokeObjectURL(inventoryDownload);events.abort();reduced.removeEventListener('change',motionChanged);}};
}
