// Uses the existing viewer's meshes, materials, lights, orbit controls and renderer.
export const components = {
  intake: ['Air intake / filter','Proposed intake and filter; pressure loss and filter specification are unresolved.'],
  blower: ['Blower','Calculated adsorption airflow and modeled fan electricity. Fan/system intersection is not measured.'],
  inlet: ['Inlet isolation / distribution','Proposed valves map to native inlet boundary conditions. Manifold redistribution is unresolved.'],
  bed: ['Adsorption bed','One schematic contactor represents the complete saved dry shaped inventory. Inspect the actual transverse fields below.'],
  outlet: ['Adsorption outlet valve','Proposed isolation valve. Open during adsorption; closed during pumped phases and ambient filling.'],
  exhaust: ['Adsorption exhaust','Unadsorbed ambient CO₂ leaves here during adsorption; it is not collected product.'],
  regeneration: ['Regeneration isolation','Proposed isolation and throttle assembly. Native pressure control may throttle withdrawal to zero.'],
  conditioner: ['Pump-inlet cooler','Ideal 55 °C target; zero modeled pressure drop/holdup. Actual heat exchanger geometry is unspecified.'],
  knockout: ['Water knockout','Ideal modeled separation upstream of pump. Water drain is an interface; dissolved CO₂ loss is not modeled.'],
  pump: ['Vacuum pump','One assumed KNF N922, 135 W running input. Motor runs in pumped phases even when throttled. Wet-gas fitness is unqualified.'],
  diverter: ['Discharge routing valves','Proposed realization of the native evacuation vent versus desorption/cooling collection policy.'],
  vent: ['Evacuation vent','Evacuation discharge is excluded from collected product.'],
  collection: ['CO₂ collection interface','Model boundary outlet only. No qualified receiver, downstream dryer, compression or storage model.'],
  heater: ['Bed heater','2 kW installed heating assumption; enabled phase policy is separate from instantaneous duty.'],
  cooling: ['Shared cooling resource','One combined 2 kW thermal budget, COP 3. Conditioner receives priority; the bed receives the remainder.'],
  fill: ['Ambient fill connection','Proposed passive ambient bypass for repressurization. Blower off, adsorption outlet closed.'],
  drain: ['Condensate drain interface','Water handling interface only. No physical drain/seal design is qualified.']
};

export function buildProcess({THREE, parent, box, mesh, mats, stage, onSelect}) {
  const nodes = {
    intake:[-7,1,0], blower:[-5.2,1,0], inlet:[-3.3,1,0], bed:[-.4,1,0],
    outlet:[2.6,1,0], exhaust:[4.7,1,0], regeneration:[-.4,1,2.1],
    conditioner:[1.5,1,3.6], knockout:[3.5,1,3.6], pump:[5.5,1,3.6],
    diverter:[7.2,1,3.6], vent:[7.2,1,.5], collection:[7.2,1,6],
    heater:[-1.7,.45,-2.5], cooling:[-4.7,.55,3.7], fill:[-3.3,1,-2.5], drain:[3.5,.15,5.5]
  };
  const shortNames={intake:'Intake / filter',blower:'Blower',inlet:'Inlet valves',bed:'Adsorption bed',outlet:'Outlet valve',exhaust:'Air exhaust',
    regeneration:'Regen valve',conditioner:'Pump cooler',knockout:'Knockout',pump:'Vacuum pump',diverter:'Routing valves',vent:'Evac. vent',
    collection:'CO₂ collection',heater:'Heater',cooling:'Shared cooling',fill:'Ambient fill',drain:'Water drain'};
  const labels = document.createElement('div'); labels.className='px-labels'; stage.append(labels);
  const leaders=document.createElementNS('http://www.w3.org/2000/svg','svg');leaders.classList.add('px-label-leaders');leaders.setAttribute('aria-hidden','true');labels.append(leaders);
  const leaderLines={};
  const groups={}, markers=[], lines=[], ownedMaterials=[], fieldMaterials=[];
  const metal = mats.metal, green = mats.material;
  const accent = hex => { const m=new THREE.MeshStandardMaterial({color:hex,roughness:.5,metalness:.25});ownedMaterials.push(m);return m; };
  const teal=accent(0x248e80), orange=accent(0xd99848), blue=accent(0x448aaa);
  const cylinder = (g,r,h,x=0,y=0,z=0,mat=metal) => mesh(new THREE.CylinderGeometry(r,r,h,24),mat,g,x,y,z);
  for (const [id, xyz] of Object.entries(nodes)) {
    const g=new THREE.Group();g.position.set(...xyz);g.userData.component=id;parent.add(g);groups[id]=g;
    if (['inlet','outlet','regeneration','diverter'].includes(id)) {
      const valve=mesh(new THREE.OctahedronGeometry(.24),teal,g);
      cylinder(g,.035,.38,0,.30);cylinder(g,.19,.045,0,.49,0,teal);
      valve.rotation.z=Math.PI/4;
    } else if (id==='blower') {
      const body=cylinder(g,.43,.55,0,0,0,blue);body.rotation.z=Math.PI/2;
      box(g,.65,.17,.7,0,-.5,0,metal);
      for(let i=0;i<8;i++){const blade=box(g,.05,.34,.10,-.29,0,0,teal);blade.rotation.x=i*Math.PI/4;}
    } else if (id==='bed') {
      // Same rounded plates/material vocabulary as the retained cassette. Display
      // thickness is enlarged; coordinates and field values remain the saved 16 mm.
      box(g,2.5,.12,1.8,0,-.15,0,metal);
      for(let i=0;i<16;i++){
        const material=accent(0x248e80);fieldMaterials.push(material);
        box(g,2.4,.016,1.7,0,-.1+i*.017,0,material,.001);
      }
      box(g,2.5,.08,.83,0,.18,-.48,metal);
      for(const x of [-1.26,1.26]) box(g,.07,.4,1.9,x,.02,0,metal);
      for(const x of [-1,1])box(g,.10,1,.10,x,-.65,0,metal);
    } else if (id==='pump') {
      box(g,1.0,.6,.65,0,0,0,blue);
      for(const x of [-.33,.33])cylinder(g,.21,.35,x,.45,0,metal);
      box(g,1.15,.13,.85,0,-.42,0,metal);
    } else if (id==='conditioner') {
      box(g,.85,.8,.65,0,0,0,metal);
      for(let i=0;i<6;i++)box(g,.9,.03,.73,0,-.32+i*.13,0,blue);
    } else if (id==='knockout') {
      cylinder(g,.34,1.25,0,.08,0,metal);
      mesh(new THREE.SphereGeometry(.34,20,12),metal,g,0,.69,0).scale.y=.35;
      cylinder(g,.08,.45,0,-.75,0,blue);
    } else if (id==='cooling') {
      box(g,1.35,.8,.9,0,0,0,blue);
      const fan=cylinder(g,.28,.05,0,0,.48,metal);fan.rotation.x=Math.PI/2;
    } else if (id==='heater') {
      box(g,.95,.5,.65,0,0,0,orange);
      for(let i=0;i<4;i++)box(g,.7,.025,.02,0,-.12+i*.08,.34,metal);
    } else if (id==='intake') {
      box(g,.65,.85,.8,0,0,0,metal);
      for(let i=0;i<7;i++)box(g,.04,.65,.85,-.3+i*.1,0,0,blue);
    } else if (id==='collection') {
      // Open boundary bracket, deliberately not a tank or compressor.
      box(g,1,.06,.75,0,-.2,0,teal);
      for(const x of [-.5,.5])box(g,.06,.65,.75,x,.1,0,teal);
      cylinder(g,.18,.35,0,.1,0,teal);
    } else { cylinder(g,.16,.65,0,0,0,metal); }
    const label=document.createElement('button');label.type='button';label.className='px-label';label.textContent=shortNames[id];
    label.setAttribute('aria-label',components[id][0]);label.title=components[id][0];
    label.dataset.component=id;label.addEventListener('click',()=>onSelect(id));labels.append(label);
    const line=document.createElementNS('http://www.w3.org/2000/svg','line');leaders.append(line);leaderLines[id]=line;
  }
  const edges = [
    ['intake','blower','air'],['blower','inlet','air'],['inlet','bed','air'],['bed','outlet','air'],['outlet','exhaust','air'],
    ['bed','regeneration','pump'],['regeneration','conditioner','pump'],['conditioner','knockout','pump'],['knockout','pump','pump'],['pump','diverter','pump'],
    ['diverter','vent','vent'],['diverter','collection','product'],['fill','inlet','fill'],['knockout','drain','water'],
    ['heater','bed','heat'],['cooling','conditioner','utility'],['cooling','bed','utility']
  ];
  for (const [from,to,kind] of edges) {
    const a=new THREE.Vector3(...nodes[from]), b=new THREE.Vector3(...nodes[to]);
    // Orthogonal pipe elbows keep separate process branches legible.
    const points=[a,new THREE.Vector3(a.x,a.y,b.z),new THREE.Vector3(b.x,a.y,b.z),b]
      .filter((p,i,all)=>!i||p.distanceTo(all[i-1])>1e-4);
    const path=new THREE.CurvePath();for(let i=1;i<points.length;i++)path.add(new THREE.LineCurve3(points[i-1],points[i]));
    const material=accent(kind==='heat'?0xb97b39:kind==='utility'?0x3b667c:0x4e6773);
    const pipe=mesh(new THREE.TubeGeometry(path,24,kind==='utility'||kind==='heat'?.035:.065,8,false),material,parent);
    pipe.userData.component=from;
    lines.push({from,to,kind,material});
    if(['air','pump','vent','product','fill'].includes(kind)) {
      const head=mesh(new THREE.ConeGeometry(.12,.32,8),teal,parent);
      head.castShadow=false;markers.push({from,to,kind,head,path});
    }
  }
  const activeEdge=window.CarbonProcess.edgeActive;
  const haloMaterial=new THREE.MeshBasicMaterial({color:0xe4ae5a,transparent:true,opacity:.85,side:THREE.DoubleSide,depthWrite:false});ownedMaterials.push(haloMaterial);
  const halo=mesh(new THREE.RingGeometry(.62,.66,48),haloMaterial,parent);halo.rotation.x=-Math.PI/2;
  let routing=window.CarbonProcess.route('adsorption'), selected='bed',labelMode='all';
  function update(phase, selection=selected, snapshot, field='temperature', scale=[25,130], meanLoading, collecting) {
    selected=selection;routing=window.CarbonProcess.route(phase,collecting);
    for(const e of lines) {
      const on=activeEdge(e,routing);
      e.material.color.setHex(on?(e.kind==='product'?0xe7ae50:0x47bdb1):e.kind==='heat'?0xa46c39:e.kind==='utility'?0x436d89:0x4c5e67);
      e.material.emissive.setHex(on?0x123c36:0x000000);
    }
    labels.querySelectorAll('button').forEach(b=>{
      b.classList.toggle('px-active',routing.components.includes(b.dataset.component));
      b.setAttribute('aria-pressed',String(b.dataset.component===selected));
    });
    for(const e of markers)e.head.visible=activeEdge(e,routing);
    halo.position.set(nodes[selected][0],.03,nodes[selected][2]);halo.scale.setScalar(selected==='bed'?2.2:1);
    if (Number.isFinite(meanLoading)) {
      const hue=window.CarbonProcess.fieldHue(meanLoading,scale)/360;
      fieldMaterials.forEach(material=>material.color.setHSL(hue,.6,.48));
    } else if(snapshot){const values=window.CarbonProcess.fieldValues(snapshot,field);fieldMaterials.forEach((m,i)=>m.color.setHSL(window.CarbonProcess.fieldHue(values[i],scale)/360,.6,.48));}
  }
  return {
    update,
    setLabels(mode){labelMode=mode;},
    componentPosition(id){return groups[id]?.getWorldPosition(new THREE.Vector3());},
    animate(progress, visible=true) {
      for(const e of markers){e.head.visible=visible&&activeEdge(e,routing);if(!e.head.visible)continue;
        const t=(progress+.17)%1;e.head.position.copy(e.path.getPoint(t));
        e.head.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),e.path.getTangent(t).normalize());}
    },
    positionLabels(camera) {
      parent.updateMatrixWorld(true);
      const placed=[],width=stage.clientWidth,height=stage.clientHeight;
      const buttons=[...labels.querySelectorAll('button')].sort((a,b)=>{
        const rank=el=>el.dataset.component===selected?0:routing.components.includes(el.dataset.component)?1:2;return rank(a)-rank(b);
      });
      for(const b of buttons) {
        const id=b.dataset.component,line=leaderLines[id];
        const wanted=labelMode==='all'||id===selected||(labelMode==='active'&&routing.components.includes(id));
        const p=groups[b.dataset.component].getWorldPosition(new THREE.Vector3());p.y+=.68;p.project(camera);
        b.hidden=!wanted||p.z<-1||p.z>1||Math.abs(p.x)>1.2||Math.abs(p.y)>1.2;line.hidden=b.hidden;line.style.display=b.hidden?'none':'';
        if(b.hidden)continue;
        const anchorX=(p.x+1)*width/2,anchorY=(1-p.y)*height/2,bw=b.offsetWidth,bh=b.offsetHeight;
        let chosen=null;
        for(const dy of [0,-28,28,-56,56,-84,84,-112,112]){
          for(const dx of [0,-38,38,-76,76]){
            const x=Math.max(bw/2+9,Math.min(width-bw/2-9,anchorX+dx)),y=Math.max(112+bh/2,Math.min(height-130-bh/2,anchorY+dy));
            const box={left:x-bw/2-3,right:x+bw/2+3,top:y-bh/2-3,bottom:y+bh/2+3,x,y};
            if(!placed.some(r=>r.left<box.right&&r.right>box.left&&r.top<box.bottom&&r.bottom>box.top)){chosen=box;break;}
          }
          if(chosen)break;
        }
        if(!chosen)chosen={x:Math.max(bw/2+9,Math.min(width-bw/2-9,anchorX)),y:Math.max(160,Math.min(height-150,anchorY)),left:0,right:0,top:0,bottom:0};
        placed.push(chosen);b.style.left=`${chosen.x}px`;b.style.top=`${chosen.y}px`;
        line.setAttribute('x1',anchorX);line.setAttribute('y1',anchorY);line.setAttribute('x2',chosen.x);line.setAttribute('y2',chosen.y);
        line.style.opacity=Math.hypot(chosen.x-anchorX,chosen.y-anchorY)>12?'.5':'0';
      }
    },
    dispose(){labels.remove();ownedMaterials.forEach(m=>m.dispose());}
  };
}
