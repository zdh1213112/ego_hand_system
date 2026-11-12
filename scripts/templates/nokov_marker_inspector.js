(function(root){
 'use strict';
 const S=root.NokovMarkerDiagnostics||(typeof require==='function'?require('./nokov_marker_diagnostics.js'):null);
 const {add,sub,mul,dot,cross,unit,finite}=S;
 function camera(){return {yaw:-.35,pitch:.7,span:.03,referenceSpan:.03,pan:[0,0],target:'pair'}}
 function rotate(v,c){const a=Math.cos(c.yaw),b=Math.sin(c.yaw),p=Math.cos(c.pitch),q=Math.sin(c.pitch);return [a*v[0]-b*v[1],b*q*v[0]+a*q*v[1]+p*v[2],-b*p*v[0]-a*p*v[1]+q*v[2]]}
 function fit(c,d){c.span=Math.max(.006,(d.distance||0)*2.6);c.referenceSpan=c.span;c.pan=[0,0];c.target='pair'}
 function zoomAt(c,amount,point,width,height){
  const factor=Math.exp(Math.max(-2,Math.min(2,amount))),next=c.span*factor,scale=Math.min(width,height)/c.span;
  if(!(next>0)||!Number.isFinite(next)||!Number.isFinite(Math.min(width,height)/next))return false;
  const offset=[(point[0]-width/2)/scale,-(point[1]-height/2)/scale];
  const pan=c.pan.map((v,k)=>v+offset[k]*(1-factor));
  if(!pan.every(Number.isFinite))return false;
  c.span=next;c.pan=pan;return true;
 }
 function panPixels(c,dx,dy,width,height){const scale=Math.min(width,height)/c.span;c.pan[0]-=dx/scale;c.pan[1]+=dy/scale}
 function target(d,c){if(c.target==='measured'&&finite(d.measured))return d.measured;if(c.target==='attached'||!finite(d.measured))return d.attached;return mul(add(d.measured,d.attached),.5)}
 function clipPolygon(points,bounds){
  let result=points;
  for(const [axis,bound,sign] of [[0,bounds[0],1],[0,bounds[1],-1],[1,bounds[2],1],[1,bounds[3],-1]]){
   const input=result;result=[];if(!input.length)break;
   let previous=input.at(-1),previousInside=(previous[axis]-bound)*sign>=0;
   for(const current of input){const inside=(current[axis]-bound)*sign>=0;
    if(inside!==previousInside){const t=(bound-previous[axis])/(current[axis]-previous[axis]);const intersection=add(previous,mul(sub(current,previous),t));intersection[axis]=bound;result.push(intersection)}
    if(inside)result.push(current);previous=current;previousInside=inside;
   }
  }
  return result;
 }
 function clipSegment(a,b,bounds){
  const delta=sub(b,a);let lo=0,hi=1;
  for(const [p,q] of [[-delta[0],a[0]-bounds[0]],[delta[0],bounds[1]-a[0]],[-delta[1],a[1]-bounds[2]],[delta[1],bounds[3]-a[1]]]){
   if(p===0){if(q<0)return null;continue}
   const t=q/p;if(p<0)lo=Math.max(lo,t);else hi=Math.min(hi,t);if(lo>hi)return null;
  }
  return [add(a,mul(delta,lo)),add(a,mul(delta,hi))];
 }
 function lengthText(m){
  if(!Number.isFinite(m))return '—';
  const a=Math.abs(m);if(a===0)return '0 mm';
  const [scale,suffix]=a>=1?[1,'m']:a>=.001?[1000,'mm']:a>=.000001?[1e6,'μm']:a>=1e-9?[1e9,'nm']:[1,'m'];
  return Number((m*scale).toPrecision(4)).toString()+' '+suffix;
 }
 function niceLength(m){const base=10**Math.floor(Math.log10(m));if(!base)return m;const ratio=m/base;return (ratio>=5?5:ratio>=2?2:1)*base}
 const css=`
 .marker-inspector-shell{position:fixed;left:5vw;top:5vh;width:90vw;height:88vh;max-width:1150px;max-height:95vh;margin:0;padding:0;resize:both;overflow:hidden;border:1px solid #8395a5;border-radius:8px;z-index:1000;background:#182029;color:#e8eef2}
 .marker-inspector{box-sizing:border-box;height:100%;display:flex;flex-direction:column;background:#182029;color:#e8eef2;font:14px 'Source Sans 3','Noto Sans CJK SC',sans-serif}
 .marker-inspector *{box-sizing:border-box}.marker-inspector header{display:flex;align-items:center;gap:12px;padding:12px 16px;margin:0;border-bottom:1px solid #43505d}.marker-inspector header strong{font-size:18px;flex:1}.marker-inspector-shell header{cursor:move}
 .marker-inspector .mi-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:8px 16px}.marker-inspector button,.marker-inspector select{font:inherit;background:#25323e;color:#e8eef2;border:1px solid #697a89;border-radius:4px;padding:6px 10px;cursor:pointer}.marker-inspector button:disabled{opacity:.4;cursor:default}
 .marker-inspector label{display:flex;gap:4px;align-items:center;font-size:13px}.marker-inspector input[type=range]{width:90px}.marker-inspector .mi-readout{padding:4px 16px 8px;line-height:1.65;font-variant-numeric:tabular-nums;min-height:48px}.marker-inspector .mi-status{color:#f2cb86}.marker-inspector .mi-axes{color:#b8cbd8;font-size:12px}
 .marker-inspector .mi-canvas-wrap{position:relative;flex:1;min-height:180px}.marker-inspector canvas{position:absolute;inset:0;display:block;width:100%;height:100%;touch-action:none;outline:none}.marker-inspector canvas:focus-visible{outline:2px solid #f2cb86;outline-offset:-2px}.marker-inspector footer{padding:8px 16px;border-top:1px solid #43505d;font-size:12px;color:#b6c4ce;line-height:1.6}.marker-inspector .mi-hint{color:#f2cb86}
 `;
 function create({getFrame,onNotice=()=>{}}){
  let host=null,popup=null,doc=null,canvas=null,observer=null,last=null,drag=null,ui={},c=camera(),fallback=false;
  const options={mesh:true,wire:true,normal:false,axes:false,depth:true,opacity:.4};
  function alive(){return host&&(fallback?host.isConnected&&host.open:popup&&!popup.closed)}
  function clean(){if(observer)observer.disconnect();observer=null;host=null;popup=null;canvas=null;doc=null;last=null;drag=null;ui={}}
  function close(){if(fallback&&host){host.close();host.remove();clean()}else if(popup&&!popup.closed){popup.close();clean()}else clean()}
  function node(tag,text,parent,attributes={}){const el=doc.createElement(tag);if(text!==null)el.textContent=text;for(const [k,v] of Object.entries(attributes))el.setAttribute(k,v);if(parent)parent.append(el);return el}
  function button(label,parent,fn){const el=node('button',label,parent,{type:'button'});el.onclick=fn;return el}
  function recenter(which){c.target=which;c.pan=[0,0];render()}
  function makeUI(){
   if(!doc.getElementById('marker-inspector-style')){const style=doc.createElement('style');style.id='marker-inspector-style';style.textContent=css;doc.head.append(style)}
   const pane=node('div',null,host,{class:'marker-inspector'}),header=node('header',null,pane);
   ui.title=node('strong','Marker 局部观察',header);button('关闭窗口',header,close);
   const tools=node('div',null,pane,{class:'mi-toolbar'});
   button('适配两点',tools,()=>{if(last){fit(c,last.d);render()}});
   button('定位附着点',tools,()=>recenter('attached'));
   ui.measured=button('定位实测点',tools,()=>recenter('measured'));
   button('放大 ＋',tools,()=>{zoomAt(c,-Math.log(2),[canvas.clientWidth/2,canvas.clientHeight/2],canvas.clientWidth,canvas.clientHeight);render()});
   button('缩小 −',tools,()=>{zoomAt(c,Math.log(2),[canvas.clientWidth/2,canvas.clientHeight/2],canvas.clientWidth,canvas.clientHeight);render()});
   button('重置视角',tools,()=>{c.yaw=-.35;c.pitch=.7;if(last)fit(c,last.d);render()});
   const optionsBar=node('div',null,pane,{class:'mi-toolbar'});
   for(const [key,label] of [['mesh','网格表面'],['wire','线框'],['normal','局部法线'],['axes','分轴方向'],['depth','遮挡提示']]){
    const wrap=node('label',null,optionsBar),input=node('input',null,wrap,{type:'checkbox'});input.checked=options[key];wrap.append(doc.createTextNode(label));input.onchange=()=>{options[key]=input.checked;render()};
   }
   const opacity=node('label','网格不透明度',optionsBar),slider=node('input',null,opacity,{type:'range',min:'0',max:'100',value:String(options.opacity*100),'aria-label':'局部网格不透明度'});slider.oninput=()=>{options.opacity=Number(slider.value)/100;render()};
   const readout=node('div',null,pane,{class:'mi-readout'});ui.status=node('div',null,readout,{class:'mi-status'});ui.axes=node('div',null,readout,{class:'mi-axes'});
   const wrap=node('div',null,pane,{class:'mi-canvas-wrap'});canvas=node('canvas',null,wrap,{tabindex:'0','aria-label':'可自由旋转和连续缩放的 marker 局部三维视图'});
   const footer=node('footer',null,pane);ui.hint=node('div',null,footer,{class:'mi-hint'});
   node('div','左键拖动旋转；Shift＋拖动／右键拖动平移；滚轮以鼠标位置连续缩放；双击适配两点。',footer);
   node('div','□ 实测 marker；◇ 网格附着点；箭头从附着点指向实测点。跟随主页面的选择与帧，视角独立。放大不增加原始数据精度。',footer);
   canvas.oncontextmenu=e=>e.preventDefault();
   canvas.onpointerdown=e=>{if(drag)return;drag={id:e.pointerId,x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===1||e.button===2};canvas.focus();canvas.setPointerCapture(e.pointerId);e.preventDefault()};
   canvas.onpointermove=e=>{if(!drag||e.pointerId!==drag.id)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.pan)panPixels(c,dx,dy,canvas.clientWidth,canvas.clientHeight);else{c.yaw=(c.yaw+dx*.008)%(Math.PI*2);c.pitch=(c.pitch+dy*.008)%(Math.PI*2)}render()};
   canvas.onpointerup=e=>{if(drag&&drag.id===e.pointerId){drag=null;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId)}};
   canvas.onpointercancel=canvas.onlostpointercapture=()=>{drag=null};
   canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),pixels=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?r.height:1);zoomAt(c,pixels*.002,[e.clientX-r.left,e.clientY-r.top],r.width,r.height);render()},{passive:false});
   canvas.ondblclick=()=>{if(last){fit(c,last.d);render()}};
   canvas.onkeydown=e=>{if(e.key==='+'||e.key==='='||e.key==='-'){e.preventDefault();zoomAt(c,(e.key==='-'?1:-1)*Math.log(2),[canvas.clientWidth/2,canvas.clientHeight/2],canvas.clientWidth,canvas.clientHeight);render()}else if(e.key.toLowerCase()==='f'&&last){fit(c,last.d);render()}};
   observer=new root.ResizeObserver(render);observer.observe(wrap);
   if(fallback){
    let moving=null;header.onpointerdown=e=>{if(e.target.closest('button'))return;const r=host.getBoundingClientRect();moving={id:e.pointerId,x:e.clientX,y:e.clientY,left:r.left,top:r.top};header.setPointerCapture(e.pointerId)};
    header.onpointermove=e=>{if(!moving||e.pointerId!==moving.id)return;host.style.left=Math.max(0,moving.left+e.clientX-moving.x)+'px';host.style.top=Math.max(0,moving.top+e.clientY-moving.y)+'px'};
    const openedHost=host;
    header.onpointerup=header.onpointercancel=header.onlostpointercapture=()=>{moving=null};host.onclose=()=>{openedHost.remove();if(host===openedHost)clean()};
   }
  }
  function open(){
   if(alive()){if(popup&&!fallback)popup.focus();update();return}
   clean();c=camera();popup=root.open('about:blank','_blank','popup,width=1100,height=800,resizable=yes,scrollbars=yes');fallback=!popup;
   if(fallback){doc=root.document;host=doc.createElement('dialog');host.className='marker-inspector-shell';doc.body.append(host);host.show();onNotice('浏览器未打开新窗口，已使用可拖动、可调整大小的页内浮窗。')}
   else{doc=popup.document;doc.title='Marker 局部观察';doc.documentElement.lang='zh-CN';doc.documentElement.style.cssText='height:100%;margin:0';doc.body.style.cssText='height:100%;margin:0';host=doc.body;const openedPopup=popup;popup.addEventListener('pagehide',()=>{if(popup===openedPopup&&!fallback)clean()},{once:true});onNotice('局部窗口已打开；选择 marker 或切换帧会同步更新。')}
   makeUI();update();render();
  }
  function update(){
   if(!alive()){if(host)clean();return}
   const next=getFrame();if(next===last)return;
   if(next&&(!last||next.marker!==last.marker))fit(c,next.d);
   last=next;render();
  }
  function render(){
   if(!alive()||!canvas)return;
   const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;
   const ratio=Math.min((doc.defaultView.devicePixelRatio||1),2),width=Math.round(w*ratio),height=Math.round(h*ratio);
   if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
   const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.fillStyle='#182029';ctx.fillRect(0,0,w,h);
   if(!last||!finite(last.d.attached)){ui.title.textContent='Marker 局部观察';ui.status.textContent='请在主页面选择一个 marker。';ui.axes.textContent='';ui.hint.textContent='';ui.measured.disabled=true;ctx.fillStyle='#c0cbd2';ctx.font='16px sans-serif';ctx.fillText('等待选择 marker',24,40);return}
   const {d,vertices,faces,faceId}=last,axes=d.axes||S.world,centre=target(d,c),scale=Math.min(w,h)/c.span;
   const view=p=>{if(!finite(p))return null;const delta=sub(p,centre),r=rotate(axes.map(a=>dot(delta,a)),c);return [r[0]-c.pan[0],r[1]-c.pan[1],r[2]]};
   const bounds=[-w/(2*scale),w/(2*scale),-h/(2*scale),h/(2*scale)];
   const pixel=p=>[w/2+p[0]*scale,h/2-p[1]*scale,p[2]];
   const inside=p=>p&&p[0]>=bounds[0]&&p[0]<=bounds[1]&&p[1]>=bounds[2]&&p[1]<=bounds[3];
   const vv=vertices.map(view),triangles=[];
   if(options.mesh||options.wire){
    faces.forEach((f,id)=>{const t=f.map(i=>vv[i]);if(!t.every(finite))return;const clipped=clipPolygon(t,bounds);if(clipped.length>=3)triangles.push({id,t,poly:clipped,depth:t.reduce((s,p)=>s+p[2],0)/3})});
    triangles.sort((a,b)=>a.depth-b.depth);
    for(const {id,t,poly} of triangles){ctx.beginPath();poly.map(pixel).forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();
     const n=unit(cross(sub(t[1],t[0]),sub(t[2],t[0]))),light=n?.5+.45*Math.abs(dot(n,[.3,.5,.81])):.7;
     if(options.mesh){ctx.globalAlpha=options.opacity;ctx.fillStyle=`rgb(${[195,177,156].map(v=>Math.round(v*light)).join(',')})`;ctx.fill()}
     if(options.wire||id===faceId){ctx.globalAlpha=id===faceId?1:.65;ctx.strokeStyle=id===faceId?'#f2cb86':'#83919c';ctx.lineWidth=id===faceId?1.5:.7;ctx.stroke()}
    }
    ctx.globalAlpha=1;
   }
   function line(a,b,colour,width=2,dashed=false){if(!finite(a)||!finite(b))return;const segment=clipSegment(view(a),view(b),bounds);if(segment)S.arrow(ctx,pixel(segment[0]),pixel(segment[1]),colour,width,dashed)}
   const av=view(d.attached),mv=view(d.measured),depthVertices=vv.map(v=>v&&[v[0]*1000,v[1]*1000,v[2]]),depthPoint=v=>v&&[v[0]*1000,v[1]*1000,v[2]];
   const hiddenM=options.depth&&options.mesh&&S.occluded(depthPoint(mv),depthVertices,faces),hiddenA=options.depth&&options.mesh&&S.occluded(depthPoint(av),depthVertices,faces);
   if(options.normal&&d.normal){
    const u=unit(cross(d.normal,Math.abs(d.normal[2])<.8?[0,0,1]:[1,0,0])),v=cross(d.normal,u),r=c.span*.16;
    const plane=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>view(add(d.attached,add(mul(u,a*r),mul(v,b*r))))),clipped=clipPolygon(plane,bounds);
    if(clipped.length){ctx.beginPath();clipped.map(pixel).forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle='rgba(142,184,237,.17)';ctx.fill()}
    const end=add(d.attached,mul(d.normal,c.span*.24));line(d.attached,end,'#8eb8ed',2);const p=view(end);if(inside(p)){const q=pixel(p);ctx.fillStyle='#a9cfff';ctx.font='13px sans-serif';ctx.fillText('n+',q[0]+6,q[1]-6)}
    if(d.foot)line(d.foot,d.measured,'#8eb8ed',2,true);
   }
   if(options.axes&&d.axes)d.axes.forEach((a,k)=>{const end=add(d.attached,mul(a,c.span*.22));line(d.attached,end,S.colours[k],1.5,true);const p=view(end);if(inside(p)){const q=pixel(p);ctx.fillStyle=S.colours[k];ctx.font='13px sans-serif';ctx.fillText('XYZ'[k]+'+',q[0]+5,q[1]-5)}});
   line(d.attached,d.measured,'#e6ebef',3,hiddenA||hiddenM);
   for(const [p,kind,hidden,label] of [[av,'attached',hiddenA,'网格附着点'],[mv,'measured',hiddenM,'实测 marker']])if(inside(p)){const q=pixel(p);S.glyph(ctx,q,kind,hidden,8);ctx.font='13px sans-serif';const width=ctx.measureText(label).width,x=Math.max(8,Math.min(w-width-8,q[0]+13)),y=Math.max(20,Math.min(h-45,q[1]-13));ctx.fillStyle='#182029';ctx.fillRect(x-3,y-15,width+6,20);ctx.fillStyle=kind==='measured'?'#8be0b9':'#f2cb86';ctx.fillText(label,x,y)}
   const bar=niceLength(100/scale),pixels=bar*scale;ctx.strokeStyle='#c0cbd2';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(20,h-25);ctx.lineTo(20+pixels,h-25);ctx.stroke();ctx.fillStyle='#c0cbd2';ctx.font='12px sans-serif';ctx.fillText(lengthText(bar),20,h-35);
   const mag=c.referenceSpan/c.span;ctx.fillText('视窗短边 '+lengthText(c.span)+' · '+(Number.isFinite(mag)?Number(mag.toPrecision(4)):'极高')+'×',20,22);
   ui.title.textContent=last.name+' · 局部观察';ui.status.textContent=`NOKOV #${last.frame} · ${last.time.toFixed(2)} s · `+(d.delta?'三维残差 '+(d.distance*1000).toFixed(2)+' mm':'本帧无实测点，仅显示附着点与网格');
   ui.axes.textContent=d.components?d.components.map((value,k)=>S.axisText(value,k,'hand')).join(' ｜ ')+' ｜ 法向 '+S.signed(d.normalDistance):'';
   ui.measured.disabled=!finite(d.measured);
   const outside=[];if(!inside(av))outside.push('附着点');if(mv&&!inside(mv))outside.push('实测点');
   ui.hint.textContent=outside.length?outside.join('、')+'在视窗外，可点击定位或“适配两点”。':(options.depth?'空心虚线点表示被网格遮挡。':'');
  }
  function state(){return {open:!!alive(),fallback,camera:{...c,pan:[...c.pan]},marker:last?.marker??null,frame:last?.frame??null}}
  return {open,close,update,state};
 }
 const api={camera,rotate,fit,zoomAt,panPixels,target,clipPolygon,clipSegment,lengthText,niceLength,create};
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.NokovMarkerInspector=api;
})(globalThis);
