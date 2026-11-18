import * as THREE from 'three';
import {OrbitControls} from '../vendor/three-r170/OrbitControls.js';
import {RoomEnvironment} from '../vendor/three-r170/RoomEnvironment.js';
import {RoundedBoxGeometry} from '../vendor/three-r170/RoundedBoxGeometry.js';
import {airflowLayout, airflowPosition, airflowMarkerRadius} from './airflow.js';
import {buildProcess} from './process-scene.js';
import {createPlayback} from './process-playback.js';

const $ = id => document.getElementById(id);
const U = 10; // One scene unit represents 100 mm.
const sourceURL = '/docs/data/architecture_cycle_pilot_scenario.json';
const descriptions = {
  pellet: {title: 'Pellet cassette', index: '01 / PACKED BED', subtitle: 'One of ten parallel beds',
    text: 'A shallow packed bed between two heated faces. The open corner reveals a representative adsorbent sample and the retaining frame.',
    badge: 'Pellets enlarged 4×',
    scope: 'Pellets are enlarged 4× and show a representative sample, not the particle count or packing structure. Particle shape, frame, seals and fasteners are illustrative. No cycle calculation runs in this viewer.'},
  channel: {title: 'Coated channels', index: '03 / STRUCTURED CONTACTOR', subtitle: '100 repeating channel units',
    text: 'Thin support sheets carry adsorbent on both faces. The open space between the coated faces forms the air passage.',
    badge: 'Coating shown at source thickness',
    scope: 'The 100 repeating units use the saved coating, support and gap dimensions. The outer frame and edge details are illustrative; this is not an engineered pressure boundary. Exploded spacing is for display only.'},
  bank: {title: 'Cassette bank', index: '02 / PARALLEL ASSEMBLY', subtitle: '10 kg dry composite · ten cassettes',
    text: 'Ten copies of the pellet cassette share the dry composite inventory. The rack shows one possible arrangement of parallel units.',
    badge: 'Rack arrangement illustrative',
    scope: 'Cassette inventory and bed dimensions come from the saved pilot preset. Rack arrangement, 12 mm clear spacing, mounting hardware and seals are illustrative. Manifolds, piping and equipment integration are unresolved.'}
};

export async function start(fail) {
  const response = await fetch(sourceURL);
  if (!response.ok) throw new Error('Geometry source unavailable');
  const specification = window.CarbonGeometry.fromSource(await response.json());
  const state = {model: 'process', cutaway: true, explode: false, grid: false, camera: 'iso', airflow: true, flowMotion: true};
  const canvas = $('gx-canvas'), stage = $('gx-stage');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let processView=null,labelsDirty=true;
  const playback=await createPlayback({motionQuery:reducedMotion,onFocus:id=>focusComponent(id),onChange:({scenario,phase,selected,snapshot,field,fieldScale,meanLoading,collecting})=>{
    if(state.model==='process'&&processView){
      processView.update(phase,selected,snapshot,field,fieldScale,meanLoading,collecting);
      labelsDirty=true;
      $('gx-stage-note').textContent=`${scenario.label} · cycle ${scenario.cycle} · ${scenario.case.equipment.dry_shaped_mass_kg} kg`;
      $('gx-flow-key').textContent=`${phase==='desorption'?'Heating / desorption':phase.charAt(0).toUpperCase()+phase.slice(1)} · flow direction only`;
      requestRender();
    }
  }});
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true, powerPreference: 'default'});
  } catch (_) {
    $('gx-layout').dataset.process='true';
    $('px-focus-component').disabled=$('px-show-bed').disabled=true;
    playback.start();
    fail('WebGL 2 is unavailable. The saved-cycle timeline, charts, bed values and comparison remain available.');
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false;
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-4, 4, 4, -4, .05, 100);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = .085;
  controls.minZoom = .55;
  controls.maxZoom = 5;
  controls.minPolarAngle = .025;
  controls.maxPolarAngle = Math.PI * .89;
  controls.autoRotateSpeed = .45;
  controls.screenSpacePanning = true;
  controls.enablePan = true;
  const pmrem = new THREE.PMREMGenerator(renderer);
  const environment = new RoomEnvironment();
  const environmentMap = pmrem.fromScene(environment, .035);
  scene.environment = environmentMap.texture;
  environment.dispose();
  pmrem.dispose();
  const key = new THREE.DirectionalLight(0xfff6e7, 2.5);
  key.position.set(-3, 8, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = key.shadow.camera.bottom = -11;
  key.shadow.camera.right = key.shadow.camera.top = 11;
  key.shadow.camera.far = 35;
  key.shadow.normalBias = .025;
  key.shadow.bias = -.0001;
  key.shadow.radius = 4;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xb6e8ff, 1.7);
  rim.position.set(5, 3, -5);
  scene.add(rim);
  scene.add(new THREE.HemisphereLight(0xe6f3ff, 0x5f747c, .65));
  const mats = {
    metal: new THREE.MeshStandardMaterial({color: 0xb2c0c9, metalness: .84, roughness: .29, envMapIntensity: 1.05}),
    edge: new THREE.MeshStandardMaterial({color: 0x637d8a, metalness: .76, roughness: .34}),
    screw: new THREE.MeshStandardMaterial({color: 0x788e9b, metalness: .95, roughness: .22}),
    seal: new THREE.MeshStandardMaterial({color: 0x263a43, metalness: .08, roughness: .85}),
    material: new THREE.MeshStandardMaterial({color: 0x16715b, metalness: .05, roughness: .69, envMapIntensity: .55}),
    coating: new THREE.MeshStandardMaterial({color: 0x125b4a, metalness: .12, roughness: .56, envMapIntensity: .55}),
    core: new THREE.MeshStandardMaterial({color: 0x0d4e40, metalness: .05, roughness: .88, envMapIntensity: .55}),
    flowIn: new THREE.LineBasicMaterial({color: 0x26b6e2, transparent: true, opacity: .55, toneMapped: false}),
    flowOut: new THREE.LineBasicMaterial({color: 0xd89b42, transparent: true, opacity: .55, toneMapped: false}),
    flowBed: new THREE.LineDashedMaterial({color: 0x70b0ad, transparent: true, opacity: .28, dashSize: .055, gapSize: .055, toneMapped: false}),
    flowMarker: new THREE.MeshBasicMaterial({vertexColors: false, toneMapped: false})
  };
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(40,40), new THREE.ShadowMaterial({opacity: .17, color: 0x081c26}));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = -.035;
  shadow.receiveShadow = true;
  scene.add(shadow);
  const groundGrid = new THREE.GridHelper(28,56,0x7897a7,0x7897a7);
  groundGrid.position.y = -.025;
  groundGrid.material.transparent = true;
  groundGrid.material.opacity = .11;
  groundGrid.visible = false;
  scene.add(groundGrid);
  const contactTexture = contactShadow();
  const contact = new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({map:contactTexture,transparent:true,depthWrite:false,opacity:.32}));
  contact.rotation.x = -Math.PI/2;
  contact.position.y = -.028;
  scene.add(contact);
  let model = new THREE.Group();
  scene.add(model);
  let frame = null, visible = true, contextLost = false;
  const clock = new THREE.Clock();
  state.airflow = $('gx-airflow').checked;
  state.flowMotion = $('gx-flow-motion').checked && !reducedMotion.matches;
  $('gx-flow-motion').checked = state.flowMotion;
  let flowGroup, flowLayout, flowHeads, flowTails, flowPhase = 0;
  const flowDummy = new THREE.Object3D(), flowColour = new THREE.Color();
  const flowColours = {inlet: 0x24abd5, outlet: 0xca8b2f, bed: 0x438e88};
  const flowAnchors = {inlet: new THREE.Vector3(), outlet: new THREE.Vector3()};
  const requestRender = () => {
    if (frame === null && visible && !document.hidden && !contextLost) frame = requestAnimationFrame(render);
  };
  function render() {
    frame = null;
    const delta = Math.min(clock.getDelta(), .05);
    const changed = controls.update(delta);
    const flowing = state.airflow && state.flowMotion && !reducedMotion.matches;
    if (flowing) flowPhase = (flowPhase + delta * .13) % 1;
    if(state.model==='process')processView?.animate(flowPhase,state.airflow);
    else if(flowing)updateAirflowMarkers();
    renderer.render(scene,camera);
    if(state.model==='process'){if(changed||labelsDirty){processView?.positionLabels(camera);labelsDirty=false;}}
    else positionAirflowLabels();
    canvas.dataset.ready = 'true';
    if (controls.autoRotate || changed || flowing) requestRender();
  }
  controls.addEventListener('change',()=>{labelsDirty=true;requestRender();});
  const clearCameraPreset = () => {
    document.querySelectorAll('[data-gx-camera]').forEach(button => button.setAttribute('aria-pressed','false'));
  };
  controls.addEventListener('start',clearCameraPreset);

  function mesh(geometry,material,parent,x=0,y=0,z=0) {
    const object = new THREE.Mesh(geometry,material);
    object.position.set(x,y,z);
    object.castShadow = true;
    object.receiveShadow = true;
    parent.add(object);
    return object;
  }
  function box(parent,w,h,d,x,y,z,material=mats.metal,radius=.009) {
    return mesh(new RoundedBoxGeometry(w,h,d,2,Math.min(radius,w*.2,h*.2,d*.2)),material,parent,x,y,z);
  }
  function bolt(parent,x,y,z) {
    mesh(new THREE.CylinderGeometry(.034,.034,.007,20),mats.screw,parent,x,y,z);
    mesh(new THREE.CylinderGeometry(.024,.024,.018,6),mats.screw,parent,x,y+.011,z);
    box(parent,.025,.002,.004,x,y+.021,z,mats.seal,.0005);
  }
  function lid(parent,w,d,thickness,y,cutaway) {
    const group = new THREE.Group();
    group.position.y = y;
    parent.add(group);
    if (!cutaway) box(group,w,thickness,d,0,thickness/2,0,mats.metal,.008);
    else {
      const shape = new THREE.Shape();
      const points=[[-w/2,-d/2],[w/2,-d/2],[w/2,d*.04],[-w*.02,d*.04],[-w*.02,d/2],[-w/2,d/2]];
      points.forEach(([x,z],i)=> i ? shape.lineTo(x,z) : shape.moveTo(x,z));
      shape.closePath();
      const geometry = new THREE.ExtrudeGeometry(shape,{depth:thickness,bevelEnabled:true,bevelSegments:2,steps:1,bevelSize:.003,bevelThickness:.002});
      geometry.rotateX(-Math.PI/2);
      mesh(geometry,mats.metal,group);
    }
    for (const x of [-w/2+.045,w/2-.045]) for (const z of [-d/2+.045,d/2-.045]) {
      if (cutaway && x>0 && z<0) continue;
      bolt(group,x,thickness+.003,z);
    }
    return group;
  }
  function cassette(parent,{particles=false,cutaway=false,explode=false}={}) {
    const p=specification.pellet, w=p.width_m*U, d=p.path_length_m*U, h=p.bed_thickness_m*U, t=p.face_plate_thickness_m*U;
    const fw=w+.12, fd=d+.12;
    box(parent,fw,t,fd,0,t/2,0,mats.metal,.006);
    const bottom=t;
    box(parent,.036,h,fd,-fw/2+.018,bottom+h/2,0,mats.edge);
    box(parent,.036,h,fd,fw/2-.018,bottom+h/2,0,mats.edge);
    for (const z of [-fd/2+.014,fd/2-.014]) {
      box(parent,fw,.022,.028,0,bottom+.011,z,mats.edge);
      box(parent,fw,.018,.028,0,bottom+h-.009,z,mats.edge);
      for (let i=1;i<=2;i++) box(parent,w,.007,.009,0,bottom+h*i/3,z,mats.screw,.001);
    }
    if (particles) pellets(parent,w,h,d,bottom);
    else box(parent,w,h*.95,d*.985,0,bottom+h/2,0,mats.core,.005);
    for (const x of [-fw/2+.022,fw/2-.022]) box(parent,.018,.012,fd-.03,x,bottom+h+.004,0,mats.seal,.003);
    const top=lid(parent,fw,fd,t,bottom+h+(explode?.62:0),cutaway);
    // Small edge ribs give the illustrative cover a manufactured appearance.
    for (let i=0;i<4;i++) {
      const x=-w*.39+i*w*.26;
      const ribLength=cutaway && x>0?fd*.43:fd*.89;
      const z=cutaway && x>0?fd*.24:0;
      box(top,.028,.024,ribLength,x,t+.012,z,mats.edge,.004);
    }
    return {w:fw,h:h+2*t+(explode?.62:0),d:fd};
  }
  function pellets(parent,w,h,d,bottom) {
    const radius=specification.pellet.pellet_diameter_m*U/2*specification.display.pellet_enlargement;
    const step=radius*2.22, nx=Math.floor((w-.07)/step), nz=Math.floor((d-.07)/step), ny=Math.max(1,Math.round(h/step));
    const count=nx*nz*ny;
    const geometry=new THREE.SphereGeometry(1,10,8);
    const cloud=new THREE.InstancedMesh(geometry,mats.material,count);
    cloud.castShadow=true; cloud.receiveShadow=true;
    const dummy=new THREE.Object3D(),colour=new THREE.Color();
    let seed=7412,index=0;
    const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
    for(let y=0;y<ny;y++)for(let z=0;z<nz;z++)for(let x=0;x<nx;x++) {
      const layerY=ny===1?h/2:radius+y*(h-2*radius)/(ny-1);
      dummy.position.set((x-(nx-1)/2)*step+((z+y)%2?.35:-.35)*radius+(random()-.5)*radius*.55,bottom+layerY+(random()-.5)*radius*.12,(z-(nz-1)/2)*step+(random()-.5)*radius*.55);
      const size=radius*(.88+random()*.15);
      dummy.scale.set(size*(.9+random()*.2),size*(.9+random()*.2),size);
      dummy.rotation.set(random()*3,random()*3,random()*3);
      dummy.updateMatrix();cloud.setMatrixAt(index,dummy.matrix);
      colour.setHSL(.435+(random()-.5)*.035,.42+random()*.12,.30+random()*.1);
      cloud.setColorAt(index,colour);index++;
    }
    cloud.instanceMatrix.needsUpdate=true;
    cloud.userData.representativeParticles=count;
    parent.add(cloud);
  }
  function channels(parent) {
    const c=specification.channel, n=state.cutaway?Math.min(24,c.parallel_units):c.parallel_units, w=c.width_m*U, d=c.path_length_m*U;
    const coat=c.coating_thickness_m*U, support=c.support_thickness_m*U;
    const pitch=c.pitch_m*U*(c.parallel_units/n)*(state.explode?1.7:1);
    const top=n*pitch;
    const supports=new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),mats.metal,n);
    const coatings=new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),mats.coating,n*2);
    supports.castShadow=coatings.castShadow=true;
    supports.receiveShadow=coatings.receiveShadow=true;
    const dummy=new THREE.Object3D();
    for(let i=0;i<n;i++) {
      const cut=state.cutaway && i>n*.62;
      const width=cut?w*.58:w;
      const x=cut?-(w-width)/2:0;
      const y=.06+i*pitch+coat+support/2;
      dummy.position.set(x,y,0);dummy.scale.set(width,support,d);dummy.updateMatrix();supports.setMatrixAt(i,dummy.matrix);
      for(let face=0;face<2;face++) {
        dummy.position.set(x,y+(face?1:-1)*(support+coat)/2,0);dummy.scale.set(width,coat,d);dummy.updateMatrix();coatings.setMatrixAt(i*2+face,dummy.matrix);
      }
    }
    supports.instanceMatrix.needsUpdate=coatings.instanceMatrix.needsUpdate=true;
    parent.add(supports,coatings);
    box(parent,w+.15,.06,d+.12,0,.015,0,mats.edge,.01);
    for(const x of [-w/2-.045,w/2+.045])for(const z of [-d/2-.025,d/2+.025]) {
      box(parent,.055,top+.11,.055,x,top/2+.02,z,mats.metal,.008);
      bolt(parent,x,top+.10,z);
    }
    for(const z of [-d/2-.025,d/2+.025])box(parent,w+.13,.045,.06,0,top+.07,z,mats.edge,.009);
    for(const x of [-w/2-.035,w/2+.035])box(parent,.055,.045,d+.1,x,top+.07,0,mats.edge,.009);
  }
  function bank(parent) {
    const p=specification.pellet;
    const thickness=(p.bed_thickness_m+2*p.face_plate_thickness_m)*U;
    const spacing=specification.display.bank_spacing_m*U*(state.explode?2.8:1);
    const total=p.parallel_beds*thickness+(p.parallel_beds-1)*spacing;
    const w=p.width_m*U+.12,d=p.path_length_m*U+.12;
    for(let i=0;i<p.parallel_beds;i++) {
      const tray=new THREE.Group();tray.position.y=.10+i*(thickness+spacing);
      if(state.explode)tray.position.x=(i%2?1:-1)*.10;
      cassette(tray,{cutaway:state.cutaway});parent.add(tray);
    }
    for(const x of [-w/2-.08,w/2+.08])for(const z of [-d/2-.06,d/2+.06]) {
      box(parent,.065,total+.3,.065,x,(total+.3)/2,z,mats.edge,.01);
      box(parent,.18,.045,.18,x,.025,z,mats.metal,.01);
      bolt(parent,x,total+.31,z);
    }
    for(const y of [.06,total+.24])for(const z of [-d/2-.06,d/2+.06])box(parent,w+.28,.06,.065,0,y,z,mats.metal,.01);
  }
  function buildAirflow() {
    flowLayout = airflowLayout(specification,state,U);
    flowGroup = new THREE.Group();
    flowGroup.visible = state.airflow;
    model.add(flowGroup);
    for (const lane of flowLayout.lanes) {
      for (const [from,to,material] of [[lane.inlet,lane.faceIn,mats.flowIn],[lane.faceIn,lane.faceOut,mats.flowBed],[lane.faceOut,lane.outlet,mats.flowOut]]) {
        const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(lane.x,lane.y,from),new THREE.Vector3(lane.x,lane.y,to)]);
        const line = new THREE.Line(geometry,material);
        line.computeLineDistances();flowGroup.add(line);
      }
    }
    const count = flowLayout.lanes.length * 4;
    flowHeads = new THREE.InstancedMesh(new THREE.ConeGeometry(.033,.12,10),mats.flowMarker,count);
    flowTails = new THREE.InstancedMesh(new THREE.CylinderGeometry(.008,.008,.16,6),mats.flowMarker,count);
    for (const object of [flowHeads,flowTails]) {
      object.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      object.frustumCulled = false;flowGroup.add(object);
    }
    const ys = flowLayout.lanes.map(lane => lane.y), first = flowLayout.lanes[0];
    const centreY = (Math.min(...ys)+Math.max(...ys))/2;
    flowAnchors.inlet.set(0,centreY+.20,first.inlet+.08);
    flowAnchors.outlet.set(0,centreY+.20,first.outlet-.08);
    updateAirflowMarkers();
  }
  function updateAirflowMarkers() {
    let index = 0;
    flowDummy.rotation.set(-Math.PI/2,0,0);
    for (const lane of flowLayout.lanes) for (let i=0;i<4;i++) {
      const point = airflowPosition(lane,flowPhase+i/4+.12);
      const radiusScale=airflowMarkerRadius(lane,point.z)/.033;
      flowDummy.scale.set(radiusScale,1,radiusScale);
      flowColour.setHex(flowColours[point.region]);
      flowDummy.position.set(point.x,point.y,point.z);flowDummy.updateMatrix();
      flowHeads.setMatrixAt(index,flowDummy.matrix);flowHeads.setColorAt(index,flowColour);
      flowDummy.position.z += .13;flowDummy.updateMatrix();
      flowTails.setMatrixAt(index,flowDummy.matrix);flowTails.setColorAt(index,flowColour);
      index++;
    }
    for (const object of [flowHeads,flowTails]) {
      object.instanceMatrix.needsUpdate = true;object.instanceColor.needsUpdate = true;
    }
  }
  function positionAirflowLabels() {
    if (!state.airflow) return;
    const width=stage.clientWidth,height=stage.clientHeight;
    const points=Object.entries(flowAnchors).map(([name,anchor])=>{
      const point=anchor.clone().applyMatrix4(model.matrixWorld).project(camera);
      return {name,x:THREE.MathUtils.clamp((point.x+1)*width/2,70,width-70),y:THREE.MathUtils.clamp((1-point.y)*height/2,155,height-145)};
    });
    if (Math.abs(points[0].x-points[1].x)<130 && Math.abs(points[0].y-points[1].y)<45) {
      points[0].y=Math.min(height-145,points[0].y+26);points[1].y=Math.max(155,points[1].y-26);
    }
    for (const point of points) {
      const label=$(`gx-flow-${point.name}`);
      label.style.left=`${point.x}px`;label.style.top=`${point.y}px`;
    }
  }
  function updateAirflowUI() {
    stage.dataset.airflow=String(state.airflow);
    if(state.model==='process'){
      $('gx-flow-inlet').hidden=$('gx-flow-outlet').hidden=true;
      $('gx-flow-key').hidden=!state.airflow;
      return;
    }
    $('gx-flow-motion').disabled=!state.airflow || reducedMotion.matches;
    $('gx-flow-motion-note').textContent=reducedMotion.matches?'Paused by reduced-motion preference':'Illustrative speed';
    $('gx-flow-explanation').textContent=state.model==='channel'
      ?'Air travels along the 150 mm passages between the coated faces.'
      :state.model==='bank'?'Air crosses each packed bed in parallel. Shared ducts and fan placement remain undefined.'
      :'Air crosses the 150 mm packed-bed length between the heated faces.';
    $('gx-flow-key').hidden=$('gx-flow-inlet').hidden=$('gx-flow-outlet').hidden=!state.airflow;
    canvas.setAttribute('aria-label',`${canvas.getAttribute('aria-label').split(' Airflow:')[0]} Airflow: ${state.airflow?'ambient air in, through the contactor, then air out. Adsorption direction only; no calculated velocity or concentration.':'hidden.'}`);
  }
  function clearModel() {
    processView?.dispose();processView=null;
    const geometries=new Set();
    model.traverse(object=>{if(object.geometry)geometries.add(object.geometry);if(object.isInstancedMesh)object.dispose();});
    geometries.forEach(geometry=>geometry.dispose());
    scene.remove(model);
    model=new THREE.Group();scene.add(model);
  }
  function rebuild() {
    clearModel();
    if(state.model==='process')processView=buildProcess({THREE,parent:model,box,mesh,mats,stage,onSelect:id=>playback.select(id)});
    else if(state.model==='pellet')cassette(model,{particles:true,cutaway:state.cutaway,explode:state.explode});
    else if(state.model==='channel')channels(model);
    else bank(model);
    // Fit the contactor before adding the direction guides, keeping its placement intact.
    const bounds=new THREE.Box3().setFromObject(model),size=bounds.getSize(new THREE.Vector3());
    const centre=bounds.getCenter(new THREE.Vector3());
    model.position.x=-centre.x;model.position.z=-centre.z;
    controls.target.set(0,Math.max(size.y/2,.18),0);
    contact.scale.set(size.x*1.5,size.z*2.1,1);
    if(state.model!=='process')buildAirflow();
    renderer.shadowMap.needsUpdate=true;
    camera.zoom=1;
    setCamera(state.camera);
    const process=state.model==='process';
    $('gx-layout').dataset.process=String(process);
    for(const id of ['px-intro','px-inspector','px-replay','px-readouts','px-scene-toolbar','px-details'])$(id).hidden=!process;
    $('gx-geometry-inspector').hidden=process;
    if(process){
      $('gx-stage-title').textContent='Complete DAC setup';$('gx-index').textContent='00 / CONNECTED PROCESS';
      $('gx-stage-note').textContent='Select equipment · saved cycle 10 · 10 kg dry shaped inventory';
      $('gx-render-note').textContent='Proposed layout · dimensions illustrative';
      $('gx-scope-note').textContent='Gas routes map to native phase boundary conditions. Valve hardware, piping, equipment placement and enlarged bed thickness are illustrative. Markers encode direction only; withdrawal can be throttled to zero. No live measurements or new calculation.';
      canvas.setAttribute('aria-label','Complete connected DAC setup: intake, blower, isolation valves, bed, exhaust, heater, shared cooling, pump-inlet cooler, knockout, vacuum pump, vent and CO2 collection. Use component selector for accessible inspection.');
      const current=playback.current();processView.update(current.phase,current.selected,current.snapshot,current.field,current.fieldScale,current.meanLoading,current.collecting);
      processView.setLabels($('px-label-mode').value);
      $('gx-stage-note').textContent=`${current.scenario.label} · cycle ${current.scenario.cycle} · ${current.scenario.case.equipment.dry_shaped_mass_kg} kg`;
      $('gx-flow-key').textContent=`${current.phase==='desorption'?'Heating / desorption':current.phase.charAt(0).toUpperCase()+current.phase.slice(1)} · flow direction only`;
    } else updateFacts(state.model);
    updateAirflowUI();
    stage.dataset.model=state.model;
    stage.dataset.exploded=String(state.explode);
    stage.dataset.cutaway=String(state.cutaway);
    $('gx-view-mode').textContent=process?'Schematic engineering arrangement':state.explode?'Exploded view · spacing increased':state.cutaway?'Cutaway view':'Assembled view';
    requestRender();
  }
  function setCamera(name) {
    state.camera=name;
    const target=controls.target.clone();
    const distance=12;
    const direction=name==='front'?new THREE.Vector3(0,0,1):name==='top'?new THREE.Vector3(0,1,.001):new THREE.Vector3(1.1,state.model==='pellet'?1.35:.85,1.6);
    camera.position.copy(direction.normalize().multiplyScalar(distance).add(target));
    camera.up.set(0,1,0);camera.lookAt(target);
    controls.update();resize();
    document.querySelectorAll('[data-gx-camera]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.gxCamera===name)));
    requestRender();
  }
  function focusComponent(id) {
    if(state.model!=='process'||!processView)return;
    const point=processView.componentPosition(id);
    if(!point)return;
    const offset=camera.position.clone().sub(controls.target);
    controls.target.copy(point);camera.position.copy(point).add(offset);camera.zoom=2.1;
    clearCameraPreset();camera.updateProjectionMatrix();controls.update();requestRender();
  }
  function resize() {
    labelsDirty=true;
    const width=stage.clientWidth,height=stage.clientHeight;
    if(!width||!height)return;
    renderer.setSize(width,height,false);
    const aspect=width/height;
    camera.updateMatrixWorld();
    const bounds=new THREE.Box3().setFromObject(model);
    let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z]) {
      const point=new THREE.Vector3(x,y,z).applyMatrix4(camera.matrixWorldInverse);
      minX=Math.min(minX,point.x);maxX=Math.max(maxX,point.x);minY=Math.min(minY,point.y);maxY=Math.max(maxY,point.y);
    }
    const availableX=Math.max(.5,(width-65)/width), availableY=Math.max(.42,(height-(state.model==='process'?195:235))/height);
    const half=Math.max((maxY-minY)/2/availableY,(maxX-minX)/2/(aspect*availableX),.5)*1.025;
    camera.left=-half*aspect;camera.right=half*aspect;
    camera.top=half;camera.bottom=-half;
    camera.updateProjectionMatrix();requestRender();
  }
  function updateFacts(name) {
    const data=descriptions[name],p=specification.pellet,c=specification.channel;
    $('gx-stage-title').textContent=$('gx-info-title').textContent=data.title;
    $('gx-index').textContent=data.index;$('gx-stage-note').textContent=data.subtitle;
    $('gx-description').textContent=data.text;$('gx-render-note').textContent=data.badge;
    $('gx-scope-note').textContent=data.scope;
    if(name==='channel' && state.cutaway) {
      $('gx-render-note').textContent='24 / 100 units · gaps expanded';
      $('gx-stage-note').textContent='Representative cutaway of a 100-unit module';
      $('gx-scope-note').textContent='The cutaway samples 24 of 100 repeating units, with expanded gaps so the layers remain visible. Readouts retain the source dimensions. Frame details are illustrative. Turn off Cutaway to show all 100 units at source spacing.';
    } else if(name==='channel') $('gx-render-note').textContent=state.explode?'100 units · gaps expanded':'100 units · source spacing';
    canvas.setAttribute('aria-label',`Rotatable 3D model: ${data.title}. ${$('gx-scope-note').textContent}`);
    const mm=(x,d=0)=>`${(x*1000).toFixed(d)} mm`;
    const rows=name==='channel'?[['Repeating units',String(c.parallel_units)],['Derived unit width',mm(c.width_m)],['Gas path',mm(c.path_length_m)],['Coating per face',mm(c.coating_thickness_m,2)],['Support thickness',mm(c.support_thickness_m,2)],['Clean air gap',mm(c.channel_gap_m,1)]]:name==='bank'?[['Parallel cassettes',String(p.parallel_beds)],['Dry composite total',`${specification.mass.toFixed(0)} kg`],['Dry composite per bed',`${p.mass_per_bed_kg.toFixed(1)} kg`],['Derived bed width',mm(p.width_m)],['Bed thickness',mm(p.bed_thickness_m)],['Gas path',mm(p.path_length_m)]]:[['Dry composite per bed',`${p.mass_per_bed_kg.toFixed(1)} kg`],['Derived bed width',mm(p.width_m)],['Bed thickness',mm(p.bed_thickness_m)],['Gas path',mm(p.path_length_m)],['Pellet diameter',mm(p.pellet_diameter_m,1)],['Source face plate',mm(p.face_plate_thickness_m,1)]];
    const fragment=document.createDocumentFragment();
    rows.forEach(([label,value])=>{const div=document.createElement('div');div.className='gx-fact';const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;div.append(dt,dd);fragment.append(div);});
    $('gx-facts').replaceChildren(fragment);
  }
  function theme() {
    const dark=document.documentElement.dataset.ctTheme==='dark';
    shadow.material.opacity=dark?.25:.16;
    contact.material.opacity=dark?.35:.22;
    groundGrid.material.opacity=dark?.12:.15;
    renderer.toneMappingExposure=dark?1:.95;
    flowColours.inlet=dark?0x65d8f4:0x1487b0;
    flowColours.outlet=dark?0xf1bf6d:0xb67b21;
    flowColours.bed=dark?0x7cd0c4:0x438e88;
    mats.flowIn.color.setHex(flowColours.inlet);mats.flowOut.color.setHex(flowColours.outlet);
    if(flowHeads&&state.model!=='process')updateAirflowMarkers();
    requestRender();
  }
  document.querySelectorAll('[data-gx-model]').forEach(button=>button.addEventListener('click',()=>{
    if(button.dataset.gxModel===state.model)return;
    state.model=button.dataset.gxModel;state.camera='iso';
    playback.stop();
    document.querySelectorAll('[data-gx-model]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
    rebuild();
  }));
  document.querySelectorAll('[data-gx-camera]').forEach(button=>button.addEventListener('click',()=>setCamera(button.dataset.gxCamera)));
  $('gx-cutaway').addEventListener('change',event=>{state.cutaway=event.target.checked;rebuild();});
  $('gx-explode').addEventListener('change',event=>{state.explode=event.target.checked;rebuild();});
  $('gx-airflow').addEventListener('change',event=>{
    state.airflow=event.target.checked;if(flowGroup)flowGroup.visible=state.airflow;
    updateAirflowUI();resize();requestRender();
  });
  $('gx-flow-motion').addEventListener('change',event=>{state.flowMotion=event.target.checked;requestRender();});
  $('px-motion').addEventListener('click',()=>{
    state.flowMotion=!state.flowMotion&&!reducedMotion.matches;
    $('px-motion').textContent=state.flowMotion?'Pause flow markers':'Animate flow markers';
    $('px-motion').setAttribute('aria-pressed',String(state.flowMotion));
    $('gx-flow-motion').checked=state.flowMotion;requestRender();
  });
  $('px-label-mode').addEventListener('change',()=>{processView?.setLabels($('px-label-mode').value);labelsDirty=true;requestRender();});
  $('gx-grid').addEventListener('change',event=>{state.grid=event.target.checked;groundGrid.visible=state.grid;requestRender();});
  $('gx-rotate').addEventListener('change',event=>{controls.autoRotate=event.target.checked;if(controls.autoRotate)clearCameraPreset();requestRender();});
  const zoom=factor=>{camera.zoom=THREE.MathUtils.clamp(camera.zoom*factor,controls.minZoom,controls.maxZoom);camera.updateProjectionMatrix();requestRender();};
  $('gx-zoom-in').addEventListener('click',()=>zoom(1.2));
  $('gx-zoom-out').addEventListener('click',()=>zoom(1/1.2));
  $('gx-reset').addEventListener('click',()=>{camera.zoom=1;controls.target.copy(new THREE.Box3().setFromObject(model).getCenter(new THREE.Vector3()));setCamera('iso');});
  canvas.addEventListener('keydown',event=>{
    if(event.ctrlKey || event.metaKey || event.altKey)return;
    const angle=.12;
    if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) {
      event.preventDefault();
      clearCameraPreset();
      const offset=camera.position.clone().sub(controls.target),spherical=new THREE.Spherical().setFromVector3(offset);
      if(event.key==='ArrowLeft')spherical.theta-=angle;
      if(event.key==='ArrowRight')spherical.theta+=angle;
      if(event.key==='ArrowUp')spherical.phi=Math.max(.03,spherical.phi-angle);
      if(event.key==='ArrowDown')spherical.phi=Math.min(Math.PI*.89,spherical.phi+angle);
      camera.position.copy(new THREE.Vector3().setFromSpherical(spherical).add(controls.target));controls.update();requestRender();
    } else if(['+','=','-','Home'].includes(event.key)) {
      event.preventDefault();if(event.key==='Home')$('gx-reset').click();else zoom(event.key==='-'?1/1.2:1.2);
    }
  });
  canvas.addEventListener('webglcontextlost',event=>{
    playback.stop();
    event.preventDefault();contextLost=true;
    if(frame!==null)cancelAnimationFrame(frame);frame=null;
    fail('The graphics context was lost. Reload the viewer to rebuild the local geometry.');
  });
  const resizeObserver=new ResizeObserver(resize);resizeObserver.observe(stage);
  const intersection=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)requestRender();});intersection.observe(stage);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)requestRender();});
  const themeObserver=new MutationObserver(theme);themeObserver.observe(document.documentElement,{attributes:true,attributeFilter:['data-ct-theme']});
  reducedMotion.addEventListener('change',()=>{
    $('px-motion').disabled=reducedMotion.matches;
    if(reducedMotion.matches){controls.autoRotate=false;$('gx-rotate').checked=false;state.flowMotion=false;$('gx-flow-motion').checked=false;}
    $('px-motion').textContent=state.flowMotion?'Pause flow markers':'Animate flow markers';
    $('px-motion').setAttribute('aria-pressed',String(state.flowMotion));
    updateAirflowUI();requestRender();
  });
  $('px-motion').disabled=reducedMotion.matches;
  $('px-motion').textContent=state.flowMotion?'Pause flow markers':'Animate flow markers';
  $('px-motion').setAttribute('aria-pressed',String(state.flowMotion));
  const raycaster=new THREE.Raycaster();let pressPoint=null;
  canvas.addEventListener('pointerdown',e=>{pressPoint=[e.clientX,e.clientY];});
  canvas.addEventListener('pointerup',e=>{
    if(state.model!=='process'||!pressPoint||Math.hypot(e.clientX-pressPoint[0],e.clientY-pressPoint[1])>5)return;
    const rect=canvas.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,1-(e.clientY-rect.top)/rect.height*2),camera);
    for(const hit of raycaster.intersectObject(model,true)){let o=hit.object;while(o&&!o.userData.component)o=o.parent;if(o){playback.select(o.userData.component);break;}}
  });
  rebuild();playback.start();theme();
  renderer.compile(scene,camera);
  renderer.render(scene,camera);
  canvas.dataset.ready='true';
  $('gx-loading').hidden=true;
  window.addEventListener('pageshow',event=>{if(event.persisted){visible=true;requestRender();}});
  window.addEventListener('pagehide',event=>{
    if(frame!==null)cancelAnimationFrame(frame);
    frame=null;
    if(event.persisted){visible=false;return;}
    contextLost=true;
    playback.dispose();
    resizeObserver.disconnect();intersection.disconnect();themeObserver.disconnect();controls.dispose();
    clearModel();Object.values(mats).forEach(material=>material.dispose());
    shadow.geometry.dispose();shadow.material.dispose();groundGrid.geometry.dispose();groundGrid.material.dispose();
    contact.geometry.dispose();contact.material.dispose();contactTexture.dispose();environmentMap.dispose();renderer.dispose();
  });
}

function contactShadow() {
  const canvas=document.createElement('canvas');canvas.width=canvas.height=128;
  const ctx=canvas.getContext('2d');
  const gradient=ctx.createRadialGradient(64,64,3,64,64,64);
  gradient.addColorStop(0,'rgba(5,16,23,.8)');gradient.addColorStop(.4,'rgba(5,16,23,.35)');gradient.addColorStop(1,'rgba(5,16,23,0)');
  ctx.fillStyle=gradient;ctx.fillRect(0,0,128,128);
  return new THREE.CanvasTexture(canvas);
}
