(() => {
  "use strict";

  const DEG = Math.PI / 180;
  const fingerMeta = {
    thumb:  { zh: "拇指", en: "THUMB",  color: "#f28c45" },
    index:  { zh: "食指", en: "INDEX",  color: "#4d83e6" },
    middle: { zh: "中指", en: "MIDDLE", color: "#20aa74" },
    ring:   { zh: "无名指", en: "RING", color: "#9b69d7" },
    pinky:  { zh: "小指", en: "PINKY",  color: "#e45f91" },
  };

  const common = (finger, joint, kind, code, label, english, hint, min, max, negative, positive) => ({
    finger, joint, kind, code, label, english, hint, min, max, negative, positive,
    id: `${finger}_${joint.toLowerCase()}_${kind}`,
    field: `${finger}_${joint.toLowerCase()}_${kind}_rad`,
    value: 0,
  });

  const dofs = [
    common("thumb", "CMC", "flex", "CF", "CMC 屈伸", "Carpometacarpal flexion", "拇指根部弯曲 / 伸直", -35, 75, "伸展", "屈曲"),
    common("thumb", "CMC", "abduction", "CA", "CMC 张合", "Carpometacarpal abduction", "拇指远离 / 靠近手掌", -40, 45, "内收", "外展"),
    common("thumb", "CMC", "opposition", "OP", "CMC 对掌", "Carpometacarpal opposition", "拇指旋向其他手指", -30, 70, "复位", "对掌"),
    common("thumb", "MCP", "flex", "MF", "MCP 屈伸", "Metacarpophalangeal flexion", "拇指掌指关节弯曲", -20, 80, "伸展", "屈曲"),
    common("thumb", "IP", "flex", "IF", "IP 屈伸", "Interphalangeal flexion", "拇指末端关节弯曲", -15, 90, "伸展", "屈曲"),
  ];

  ["index", "middle", "ring", "pinky"].forEach(finger => {
    dofs.push(
      common(finger, "MCP", "flex", "MF", "MCP 屈伸", "Metacarpophalangeal flexion", "指根弯曲 / 伸直", -25, 95, "伸展", "屈曲"),
      common(finger, "MCP", "abduction", "MA", "MCP 张合", "Metacarpophalangeal abduction", "手指左右展开 / 内收", -25, 25, "内收", "外展"),
      common(finger, "PIP", "flex", "PF", "PIP 屈伸", "Proximal interphalangeal flexion", "中间关节弯曲", 0, 110, "伸直", "屈曲"),
      common(finger, "DIP", "flex", "DF", "DIP 屈伸", "Distal interphalangeal flexion", "末端关节弯曲", 0, 90, "伸直", "屈曲"),
    );
  });

  const descriptions = {
    "thumb_CMC_flex": "控制拇指从掌面向内弯曲或向外伸展，是形成抓握包络的重要动作。",
    "thumb_CMC_abduction": "控制拇指离开或靠近掌面。张开虎口时，这个自由度会明显增大。",
    "thumb_CMC_opposition": "让拇指的指腹旋向其他四指，是捏取、抓笔和对掌动作的关键。",
    "thumb_MCP_flex": "控制拇指根部之后的掌指关节弯曲，继续收拢拇指。",
    "thumb_IP_flex": "控制拇指最末端关节弯曲，改变拇指指尖的朝向。",
    "MCP_flex": "控制手指从指根向掌心弯曲或伸直，是握拳动作的主要旋转。",
    "MCP_abduction": "控制相邻手指之间张开或并拢；只有指根 MCP 具有这个自由度。",
    "PIP_flex": "控制手指中间的第一道弯曲。握拳时通常具有最大的弯曲幅度。",
    "DIP_flex": "控制最靠近指尖的关节，使手指末节继续包裹物体。",
  };

  const state = { selected: 0, finger: "thumb", handedness: "Right", surface: "palm", yaw: -.22, pitch: .10, zoom: 1, dragging: false };
  const $ = id => document.getElementById(id);
  const canvas = $("handCanvas");
  const ctx = canvas.getContext("2d");
  const wrap = $("canvasWrap");
  let cssWidth = 0, cssHeight = 0, hitTargets = [], lastPointer = null, pointerStart = null;

  function dof(finger, joint, kind) { return dofs.find(d => d.finger === finger && d.joint === joint && d.kind === kind); }
  function val(finger, joint, kind) { return (dof(finger, joint, kind)?.value || 0) * DEG; }
  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function v(x=0,y=0,z=0) { return [x,y,z]; }
  function add(a,b) { return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]; }
  function scale(a,s) { return [a[0]*s,a[1]*s,a[2]*s]; }
  function norm(a) { const n=Math.hypot(...a)||1; return scale(a,1/n); }
  function cross(a,b) { return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]; }
  function mul(A,B) { return A.map((r,i)=>B[0].map((_,j)=>r[0]*B[0][j]+r[1]*B[1][j]+r[2]*B[2][j])); }
  function mv(A,p) { return A.map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]); }
  const I = () => [[1,0,0],[0,1,0],[0,0,1]];
  const rx = a => [[1,0,0],[0,Math.cos(a),-Math.sin(a)],[0,Math.sin(a),Math.cos(a)]];
  const ry = a => [[Math.cos(a),0,Math.sin(a)],[0,1,0],[-Math.sin(a),0,Math.cos(a)]];
  const rz = a => [[Math.cos(a),-Math.sin(a),0],[Math.sin(a),Math.cos(a),0],[0,0,1]];

  function poseSkeleton() {
    // EGO uses the wearer's egocentric handedness: looking down at your own
    // palm, the right thumb is on screen-right and the left thumb on screen-left.
    const mirror = state.handedness === "Right" ? -1 : 1;
    const palm = [
      v(-.62*mirror,-.55,0), v(.62*mirror,-.55,0), v(.78*mirror,.45,0),
      v(.48*mirror,1.22,0), v(-.50*mirror,1.25,0), v(-.78*mirror,.38,0),
    ];
    const roots = {
      index:v(-.39*mirror,1.08,0), middle:v(-.12*mirror,1.22,0),
      ring:v(.18*mirror,1.17,0), pinky:v(.46*mirror,1.02,0), thumb:v(-.66*mirror,.15,-.02),
    };
    // A relaxed, open hand at zero DOF. The fixed spread below is anatomy/layout,
    // not an extra joint angle. It keeps all five digits recognisable at startup.
    const baseAngles = { index:-.105*mirror, middle:-.025*mirror, ring:.065*mirror, pinky:.165*mirror };
    const lengths = { index:[.82,.58,.43], middle:[.91,.63,.45], ring:[.84,.59,.43], pinky:[.68,.46,.36] };
    const fingers = {};
    for (const f of ["index","middle","ring","pinky"]) {
      let R = mul(rz(baseAngles[f] + val(f,"MCP","abduction")*mirror), rx(val(f,"MCP","flex")));
      const points=[roots[f]];
      points.push(add(points.at(-1), mv(R,v(0,lengths[f][0],0))));
      R=mul(R,rx(val(f,"PIP","flex")));
      points.push(add(points.at(-1), mv(R,v(0,lengths[f][1],0))));
      R=mul(R,rx(val(f,"DIP","flex")));
      points.push(add(points.at(-1), mv(R,v(0,lengths[f][2],0))));
      fingers[f]={points, rotations:[null,null,null], finalR:R};
    }
    // Thumb points to the lateral side of its hand. `mirror` follows the EGO
    // first-person convention above, rather than a face-to-face observer view.
    let Rt=mul(rz(.98*mirror + val("thumb","CMC","abduction")*mirror), mul(ry(val("thumb","CMC","opposition")*mirror),rx(val("thumb","CMC","flex"))));
    const tp=[roots.thumb];
    tp.push(add(tp.at(-1),mv(Rt,v(0,.62,0))));
    Rt=mul(Rt,rx(val("thumb","MCP","flex"))); tp.push(add(tp.at(-1),mv(Rt,v(0,.45,0))));
    Rt=mul(Rt,rx(val("thumb","IP","flex"))); tp.push(add(tp.at(-1),mv(Rt,v(0,.35,0))));
    fingers.thumb={points:tp,finalR:Rt};
    return { palm, roots, fingers, mirror };
  }

  function viewPoint(p) {
    const R=mul(rx(state.pitch),ry(state.yaw));
    return mv(R,[p[0],p[1]-.72,p[2]]);
  }
  function project(p) {
    const q=viewPoint(p), perspective=1/(1+q[2]*.16), s=Math.min(cssWidth,cssHeight)*.205*state.zoom*perspective;
    return {x:cssWidth*.5+q[0]*s,y:cssHeight*.53-q[1]*s,z:q[2],s};
  }
  function line3(a,b,color,width=8,alpha=1) {
    const A=project(a), B=project(b);
    ctx.save(); ctx.globalAlpha=alpha; ctx.lineCap="round"; ctx.strokeStyle=color; ctx.lineWidth=width*Math.max(.72,(A.s+B.s)/2/(Math.min(cssWidth,cssHeight)*.205));
    ctx.beginPath(); ctx.moveTo(A.x,A.y); ctx.lineTo(B.x,B.y); ctx.stroke(); ctx.restore();
  }
  function circle3(p,r,fill,stroke="#fffef9",sw=3) {
    const P=project(p), radius=r*Math.max(.8,P.s/(Math.min(cssWidth,cssHeight)*.205));
    ctx.beginPath(); ctx.arc(P.x,P.y,radius,0,Math.PI*2); ctx.fillStyle=fill; ctx.fill(); if(sw){ctx.strokeStyle=stroke;ctx.lineWidth=sw;ctx.stroke();}
    return P;
  }
  function curve3(points,color,width,alpha=1) {
    const projected=points.map(project); if(projected.length<2)return;
    ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap="round";ctx.beginPath();ctx.moveTo(projected[0].x,projected[0].y);
    if(projected.length===2)ctx.lineTo(projected[1].x,projected[1].y);
    else ctx.quadraticCurveTo(projected[1].x,projected[1].y,projected[2].x,projected[2].y);
    ctx.stroke();ctx.restore();
  }
  function arrow3(origin,dir,color,label) {
    const end=add(origin,scale(norm(dir),.42)); line3(origin,end,color,3);
    const E=project(end); ctx.fillStyle=color; ctx.font="700 11px Inter, sans-serif"; ctx.fillText(label,E.x+5,E.y-5);
  }
  function selectedJointInfo(skel) {
    const d=dofs[state.selected], pts=skel.fingers[d.finger].points;
    const jointMap=d.finger==="thumb"?{CMC:0,MCP:1,IP:2}:{MCP:0,PIP:1,DIP:2};
    const idx=jointMap[d.joint];
    return {point:pts[idx],points:pts,index:idx,d};
  }

  function draw() {
    ctx.clearRect(0,0,cssWidth,cssHeight);
    const skel=poseSkeleton(), selected=selectedJointInfo(skel);
    const selectedValue=selected.d.value;
    selected.d.value=0;
    const neutral=poseSkeleton();
    selected.d.value=selectedValue;
    const shadow=skel.palm.map(p=>project([p[0],p[1],-.18]));
    ctx.beginPath(); shadow.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)); ctx.closePath(); ctx.fillStyle="rgba(18,45,36,.055)"; ctx.fill();
    const palm2=skel.palm.map(project);
    const isPalm=state.surface==="palm";
    const grad=ctx.createLinearGradient(0,cssHeight*.25,0,cssHeight*.75);grad.addColorStop(0,isPalm?"rgba(237,246,240,.96)":"rgba(231,237,234,.97)");grad.addColorStop(1,isPalm?"rgba(211,233,219,.82)":"rgba(205,216,211,.86)");
    ctx.beginPath();palm2.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fillStyle=grad;ctx.fill();ctx.strokeStyle="rgba(8,118,84,.28)";ctx.lineWidth=2;ctx.stroke();
    if(isPalm){
      // Three unmistakable palm creases. They are orientation cues, not landmarks.
      curve3([v(-.48*skel.mirror,.18,.02),v(0,.48,.02),v(.48*skel.mirror,.30,.02)],"rgba(37,112,83,.25)",3);
      curve3([v(-.42*skel.mirror,.72,.02),v(-.05*skel.mirror,.92,.02),v(.42*skel.mirror,.78,.02)],"rgba(37,112,83,.20)",2.5);
      curve3([v(-.46*skel.mirror,.02,.02),v(-.28*skel.mirror,-.28,.02),v(.08*skel.mirror,-.38,.02)],"rgba(37,112,83,.18)",2.5);
    } else {
      // Back-of-hand metacarpal hints make the reversed view visually explicit.
      ["index","middle","ring","pinky"].forEach(f=>curve3([v(0,-.32,-.015),skel.roots[f]],"rgba(73,94,86,.13)",2));
    }

    const neutralPoints=neutral.fingers[selected.d.finger].points;
    neutralPoints.slice(selected.index).slice(0,-1).forEach((point,index)=>{
      const next=neutralPoints[selected.index+index+1];
      line3(point,next,"#7f8a85",7,.18);
    });

    const bones=[];
    Object.entries(skel.fingers).forEach(([finger,data])=>data.points.slice(0,-1).forEach((p,i)=>bones.push({finger,a:p,b:data.points[i+1],z:(viewPoint(p)[2]+viewPoint(data.points[i+1])[2])/2})));
    bones.sort((a,b)=>a.z-b.z).forEach(b=>{
      line3(b.a,b.b,isPalm?"#dceee4":"#d7dfdb",b.finger==="thumb"?25:22,.98);
      line3(b.a,b.b,fingerMeta[b.finger].color,b.finger===selected.d.finger?10:7,b.finger===selected.d.finger?1:.48);
    });

    hitTargets=[];
    Object.entries(skel.fingers).forEach(([finger,data])=>data.points.forEach((point,index)=>{
      const isSelected=finger===selected.d.finger&&index===selected.index;
      const P=circle3(point,isSelected?10:6,isSelected?"#17201d":fingerMeta[finger].color,"#fffef9",isSelected?4:3);
      if(index<data.points.length-1) hitTargets.push({finger,index,x:P.x,y:P.y});
    }));
    if(!isPalm){
      Object.entries(skel.fingers).forEach(([finger,data])=>{
        const tip=project(data.points.at(-1)), before=project(data.points.at(-2));
        const angle=Math.atan2(tip.y-before.y,tip.x-before.x)+Math.PI/2;
        ctx.save();ctx.translate(tip.x,tip.y);ctx.rotate(angle);ctx.fillStyle="rgba(255,254,249,.92)";ctx.strokeStyle="rgba(57,70,65,.28)";ctx.lineWidth=1.5;ctx.beginPath();ctx.ellipse(0,7,finger==="thumb"?6:5,finger==="thumb"?9:8,0,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore();
      });
    }

    const p=selected.point;
    const baseAxis=selected.d.kind==="abduction"?v(0,0,1):selected.d.kind==="opposition"?v(0,1,0):v(1,0,0);
    arrow3(p,baseAxis,"#17201d","旋转轴");
    const P=project(p);ctx.font="700 12px Inter, sans-serif";ctx.fillStyle="#17201d";ctx.textAlign="center";ctx.fillText(`${fingerMeta[selected.d.finger].zh} · ${selected.d.label}`,P.x,P.y-23);ctx.textAlign="left";

    const origin=v(.78*skel.mirror,-.5,.03); arrow3(origin,v(.38*skel.mirror,0,0),"#e35e60","X"); arrow3(origin,v(0,.38,0),"#22a970","Y"); arrow3(origin,v(0,0,.38),"#4d83e6","Z");
  }

  function resize() {
    const rect=wrap.getBoundingClientRect(), ratio=Math.min(window.devicePixelRatio||1,2);cssWidth=rect.width;cssHeight=rect.height;
    canvas.width=Math.round(cssWidth*ratio);canvas.height=Math.round(cssHeight*ratio);canvas.style.width=`${cssWidth}px`;canvas.style.height=`${cssHeight}px`;ctx.setTransform(ratio,0,0,ratio,0,0);draw();
  }

  function renderTabs() {
    $("fingerTabs").innerHTML=Object.entries(fingerMeta).map(([key,m])=>`<button type="button" data-finger="${key}" class="${key===state.finger?"active":""}">${m.zh}<span>${m.en}</span></button>`).join("");
  }
  function renderList() {
    const items=dofs.filter(d=>d.finger===state.finger);
    $("dofList").innerHTML=items.map(d=>{const index=dofs.indexOf(d);return `<button type="button" class="dof-button ${index===state.selected?"active":""}" data-index="${index}"><span class="dof-code">${d.code}</span><span class="dof-copy"><strong>${d.label}</strong><small>${d.hint}</small></span><output>${Math.round(d.value)}°</output></button>`}).join("");
  }
  function updateSlider() {
    const d=dofs[state.selected], slider=$("angleSlider");slider.min=d.min;slider.max=d.max;slider.value=d.value;
    slider.style.setProperty("--range-fill",`${(d.value-d.min)/(d.max-d.min)*100}%`);
    $("rangeMin").textContent=`${d.min<0?"−":""}${Math.abs(d.min)}°`;$("rangeMax").textContent=`${d.max}°`;$("sliderTitle").textContent=d.label;
  }
  function updateLesson() {
    const d=dofs[state.selected], meta=fingerMeta[d.finger], key=descriptions[`${d.finger}_${d.joint}_${d.kind}`]?`${d.finger}_${d.joint}_${d.kind}`:`${d.joint}_${d.kind}`;
    $("selectedIndex").textContent=String(state.selected+1).padStart(2,"0");$("selectedFinger").textContent=meta.zh;$("selectedJoint").textContent=d.label;$("selectedHint").textContent=d.hint;$("selectedValue").textContent=`${Math.round(d.value)}°`;
    $("lessonCode").textContent=d.code;$("lessonName").textContent=`${meta.zh} ${d.label}`;$("lessonEnglish").textContent=d.english;$("lessonDescription").textContent=descriptions[key];
    $("negativeMeaning").textContent=d.negative;$("positiveMeaning").textContent=d.positive;$("fieldName").textContent=d.field;$("radValue").textContent=(d.value*DEG).toFixed(3);$("degValue").textContent=d.value.toFixed(1);
  }
  function renderUI() { renderTabs();renderList();updateSlider();updateLesson();draw(); }
  function select(index) { state.selected=clamp(index,0,dofs.length-1);state.finger=dofs[state.selected].finger;renderUI(); }
  function setAngle(value) { const d=dofs[state.selected];d.value=clamp(Number(value),d.min,d.max);document.querySelectorAll(".quick-presets button").forEach(b=>b.classList.remove("active"));renderList();updateSlider();updateLesson();draw(); }

  const presetValues={
    open:()=>0,
    fist:d=>d.kind==="flex"?(d.joint==="MCP"?75:d.joint==="PIP"?95:d.joint==="DIP"?70:d.joint==="IP"?65:45):d.kind==="opposition"?35:0,
    pinch:d=>d.finger==="thumb"?(d.kind==="opposition"?58:d.kind==="abduction"?18:d.kind==="flex"?35:0):d.finger==="index"?(d.joint==="MCP"&&d.kind==="flex"?35:d.joint==="PIP"?42:d.joint==="DIP"?25:0):d.kind==="flex"?(d.joint==="MCP"?60:d.joint==="PIP"?75:50):0,
    point:d=>d.finger==="index"?0:d.finger==="thumb"?(d.kind==="opposition"?25:d.kind==="flex"?25:0):d.kind==="flex"?(d.joint==="MCP"?72:d.joint==="PIP"?92:65):0,
  };
  function applyPreset(name) { dofs.forEach(d=>d.value=clamp(presetValues[name](d),d.min,d.max));document.querySelectorAll(".quick-presets button").forEach(b=>b.classList.toggle("active",b.dataset.preset===name));renderUI(); }

  $("fingerTabs").addEventListener("click",e=>{const b=e.target.closest("button[data-finger]");if(!b)return;state.finger=b.dataset.finger;state.selected=dofs.findIndex(d=>d.finger===state.finger);renderUI();});
  $("dofList").addEventListener("click",e=>{const b=e.target.closest("button[data-index]");if(b)select(Number(b.dataset.index));});
  $("angleSlider").addEventListener("input",e=>setAngle(e.target.value));
  $("minusAngle").addEventListener("click",()=>setAngle(dofs[state.selected].value-5));$("plusAngle").addEventListener("click",()=>setAngle(dofs[state.selected].value+5));$("zeroAngle").addEventListener("click",()=>setAngle(0));
  document.querySelectorAll(".quick-presets button").forEach(b=>b.addEventListener("click",()=>applyPreset(b.dataset.preset)));
  function setSurface(surface){
    state.surface=surface;state.yaw=surface==="palm"?-.22:Math.PI+.22;state.pitch=.10;state.zoom=1;
    document.querySelectorAll("[data-hand]").forEach(b=>b.classList.toggle("active",b.dataset.hand===state.handedness));
    document.querySelectorAll("[data-surface]").forEach(b=>b.classList.toggle("active",b.dataset.surface===surface));
    const hand=state.handedness==="Right"?"右手":"左手";
    $("surfaceTitle").textContent=`${hand} · ${surface==="palm"?"掌心面":"手背面"}`;
    const thumbSide=state.handedness==="Right"?"画面右侧":"画面左侧";
    $("surfaceHint").textContent=surface==="palm"?`可见掌纹，拇指在${thumbSide}`:"可见指甲，左右方向与掌心视图相反";draw();
  }
  document.querySelectorAll("[data-hand]").forEach(b=>b.addEventListener("click",()=>{state.handedness=b.dataset.hand;document.querySelectorAll("[data-hand]").forEach(x=>x.classList.toggle("active",x===b));setSurface(state.surface);}));
  document.querySelectorAll("[data-surface]").forEach(b=>b.addEventListener("click",()=>setSurface(b.dataset.surface)));
  $("resetPose").addEventListener("click",()=>applyPreset("open"));$("resetView").addEventListener("click",()=>setSurface(state.surface));

  wrap.addEventListener("pointerdown",e=>{state.dragging=true;lastPointer={x:e.clientX,y:e.clientY};pointerStart={x:e.clientX,y:e.clientY};wrap.classList.add("dragging");wrap.setPointerCapture(e.pointerId);});
  wrap.addEventListener("pointermove",e=>{if(!state.dragging)return;state.yaw+=(e.clientX-lastPointer.x)*.008;state.pitch=clamp(state.pitch+(e.clientY-lastPointer.y)*.007,-1.1,1.1);lastPointer={x:e.clientX,y:e.clientY};draw();});
  wrap.addEventListener("pointerup",e=>{const moved=pointerStart&&Math.hypot(e.clientX-pointerStart.x,e.clientY-pointerStart.y);state.dragging=false;wrap.classList.remove("dragging");if(!moved||moved<4){const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const hit=hitTargets.map(h=>({...h,dist:Math.hypot(h.x-x,h.y-y)})).sort((a,b)=>a.dist-b.dist)[0];if(hit&&hit.dist<22){const jointNames=hit.finger==="thumb"?["CMC","MCP","IP"]:["MCP","PIP","DIP"];const idx=dofs.findIndex(d=>d.finger===hit.finger&&d.joint===jointNames[hit.index]);if(idx>=0)select(idx);}}lastPointer=null;pointerStart=null;});
  wrap.addEventListener("wheel",e=>{e.preventDefault();state.zoom=clamp(state.zoom*(e.deltaY>0?.93:1.07),.68,1.55);draw();},{passive:false});
  window.addEventListener("resize",resize);

  // Query parameters are handy for documentation links and visual regression checks:
  // index.html?hand=Left&surface=back
  const params=new URLSearchParams(window.location.search);
  if(["Right","Left"].includes(params.get("hand")))state.handedness=params.get("hand");
  if(["palm","back"].includes(params.get("surface")))state.surface=params.get("surface");
  document.querySelectorAll("[data-hand]").forEach(b=>b.classList.toggle("active",b.dataset.hand===state.handedness));
  renderUI();resize();setSurface(state.surface);
})();
