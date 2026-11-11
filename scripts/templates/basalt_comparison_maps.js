(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('mapData').textContent);
  const cases = Object.keys(data.cases);
  const el = id => document.getElementById(id);
  let yaw = -.35, pitch = .85, zoom = 1, dragging = null;
  const palette = [[72,38,116],[54,91,141],[35,139,141],[79,185,112],[224,219,57]];
  function heightColor(t) {
    const q = Math.max(0,Math.min(.999,t))*(palette.length-1), i = Math.floor(q), f = q-i;
    return `rgb(${palette[i].map((v,j) => Math.round(v*(1-f)+palette[i+1][j]*f)).join(',')})`;
  }
  function draw() {
    const all = el('allPoints').checked;
    const visible = Object.fromEntries(cases.map(k => [k, data.cases[k].points.filter(p => all || Math.hypot(...p.map((v,j) => v-data.center[j])) <= 5)]));
    const combined = [...data.reference, ...cases.flatMap(k => visible[k])];
    const low = [0,1,2].map(j => Math.min(...combined.map(p=>p[j])));
    const high = [0,1,2].map(j => Math.max(...combined.map(p=>p[j])));
    const center = low.map((v,j)=>(v+high[j])/2);
    const radius = Math.max(.3,...high.map((v,j)=>v-low[j]))*.7;
    const rotate = p => {
      const d = p.map((v,j)=>v-center[j]);
      const a = Math.cos(yaw)*d[0]-Math.sin(yaw)*d[1];
      const b = Math.sin(yaw)*d[0]+Math.cos(yaw)*d[1];
      return [a, Math.sin(pitch)*b+Math.cos(pitch)*d[2], Math.cos(pitch)*b-Math.sin(pitch)*d[2]];
    };
    for (const key of cases) {
      const canvas = el('map_'+key), w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio||1;
      canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
      const ctx = canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
      const scale = Math.min(w,h)*.46/radius*zoom;
      const project = p => {const v=rotate(p); return [w/2+v[0]*scale,h/2-v[1]*scale,v[2]];};
      function line(a,b,color,width=1) {
        const p=project(a),q=project(b);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();
      }
      const step = radius>10?5:radius>3?1:.5;
      const gridLow = low.map(v=>Math.floor(v/step)*step), gridHigh=high.map(v=>Math.ceil(v/step)*step);
      const z=gridLow[2];
      for(let x=gridLow[0];x<=gridHigh[0]+.01;x+=step)line([x,gridLow[1],z],[x,gridHigh[1],z],'#dfe7ee');
      for(let y=gridLow[1];y<=gridHigh[1]+.01;y+=step)line([gridLow[0],y,z],[gridHigh[0],y,z],'#dfe7ee');
      const origin=[gridLow[0],gridLow[1],z];
      for(let j=0;j<3;j++) {
        const end=origin.map((v,k)=>v+(j===k?Math.min(1,radius*.6):0));
        line(origin,end,['#bb5454','#2f9868','#437fb5'][j],1.6);
        const p=project(end);ctx.fillStyle='#697f93';ctx.font='11px system-ui';ctx.fillText('XYZ'[j],p[0]+3,p[1]-3);
      }
      const points=visible[key].map(p=>({p,v:project(p)})).sort((a,b)=>a.v[2]-b.v[2]);
      for(const point of points) {
        const t=(point.p[2]-low[2])/Math.max(.01,high[2]-low[2]);
        ctx.fillStyle=heightColor(t);ctx.globalAlpha=.75;ctx.beginPath();ctx.arc(point.v[0],point.v[1],1.9,0,Math.PI*2);ctx.fill();
      }
      ctx.globalAlpha=1;ctx.strokeStyle='#192e43';ctx.lineWidth=1.6;ctx.beginPath();
      data.reference.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.stroke();
      ctx.fillStyle='#73879b';ctx.font='11px system-ui';ctx.fillText(`网格 ${step} 米 · ${all?'全部点':'距轨迹中心 5 米内'}`,9,h-12);
      el('count_'+key).textContent=`共 ${data.cases[key].points.length.toLocaleString()} 点 · 当前显示 ${visible[key].length.toLocaleString()} 点`;
      canvas.dataset.rendered='true';
    }
  }
  function setView(top) {
    yaw=top?0:-.35;pitch=top?Math.PI/2:.85;zoom=1;
    el('view3d').setAttribute('aria-pressed',String(!top));el('viewTop').setAttribute('aria-pressed',String(top));draw();
  }
  for(const key of cases) {
    const canvas=el('map_'+key);
    canvas.addEventListener('pointerdown',e=>{dragging={x:e.clientX,y:e.clientY,id:e.pointerId};canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',e=>{
      if(!dragging||dragging.id!==e.pointerId)return;
      yaw+=(e.clientX-dragging.x)*.008;pitch=Math.max(-1.5,Math.min(1.57,pitch+(e.clientY-dragging.y)*.008));
      dragging.x=e.clientX;dragging.y=e.clientY;
      el('view3d').setAttribute('aria-pressed','true');el('viewTop').setAttribute('aria-pressed','false');draw();
    });
    canvas.addEventListener('pointerup',()=>{dragging=null;});
    canvas.addEventListener('pointercancel',()=>{dragging=null;});
    canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.25,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
  }
  el('view3d').addEventListener('click',()=>setView(false));el('viewTop').addEventListener('click',()=>setView(true));
  el('resetMaps').addEventListener('click',()=>{el('allPoints').checked=false;setView(false);});
  el('allPoints').addEventListener('change',()=>{zoom=1;draw();});window.addEventListener('resize',draw);
  draw();
  const dialog=el('imageDialog');
  el('downloadVideo').href=document.querySelector('video source').getAttribute('src');
  for(const anchor of document.querySelectorAll('[data-zoom]'))anchor.addEventListener('click',e=>{
    e.preventDefault();const image=anchor.querySelector('img');el('largeImage').src=image.src;el('largeImage').alt=image.alt;dialog.showModal();
  });
  el('closeImage').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
})();
