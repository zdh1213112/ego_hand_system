(function(root){
 'use strict';
 const add=(a,b)=>a.map((v,i)=>v+b[i]), sub=(a,b)=>a.map((v,i)=>v-b[i]);
 const mul=(a,s)=>a.map(v=>v*s), dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
 const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 const finite=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
 const unit=p=>finite(p)&&Math.hypot(...p)>1e-10?mul(p,1/Math.hypot(...p)):null;
 const world=[[1,0,0],[0,1,0],[0,0,1]], colours=['#e28585','#83cbae','#8eb8ed'];
 function analyse(measured,attached,axes,normal){
  const valid=finite(measured)&&finite(attached),basis=axes&&axes.every(finite)?axes:null,n=unit(normal);
  const delta=valid?sub(measured,attached):null,normalDistance=delta&&n?dot(delta,n):null;
  return {measured,attached,axes:basis,normal:n,delta,
   components:delta&&basis?basis.map(a=>dot(delta,a)):null,
   distance:delta?Math.hypot(...delta):null,normalDistance,
   tangentDistance:normalDistance!==null?Math.hypot(...sub(delta,mul(n,normalDistance))):null,
   foot:normalDistance!==null?sub(measured,mul(n,normalDistance)):null};
 }
 function signed(value){
  if(!Number.isFinite(value))return '—';
  const mm=value*1000,rounded=Math.abs(mm)<.005?0:mm;
  return (rounded>0?'+':'')+rounded.toFixed(2)+' mm';
 }
 function axisText(value,axis,space){
  const label='XYZ'[axis]+' '+signed(value);
  if(!Number.isFinite(value)||Math.abs(value)<.000005)return label;
  const hand=[['小指侧','拇指侧'],['指尖方向','手腕方向'],['手背侧','掌心侧']];
  return label+' · '+(space==='hand'?hand[axis][value>0?0:1]:('世界 '+ 'XYZ'[axis]+(value>0?' 正向':' 负向')));
 }
 function occluded(point,vertices,faces,epsilon=.0003){
  if(!finite(point))return false;
  for(const face of faces){
   const [a,b,c]=face.map(i=>vertices[i]);
   if(!finite(a)||!finite(b)||!finite(c))continue;
   if(point[0]<Math.min(a[0],b[0],c[0])||point[0]>Math.max(a[0],b[0],c[0])||point[1]<Math.min(a[1],b[1],c[1])||point[1]>Math.max(a[1],b[1],c[1]))continue;
   const det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
   if(Math.abs(det)<1e-10)continue;
   const u=((b[1]-c[1])*(point[0]-c[0])+(c[0]-b[0])*(point[1]-c[1]))/det;
   const v=((c[1]-a[1])*(point[0]-c[0])+(a[0]-c[0])*(point[1]-c[1]))/det;
   if(u>=-1e-8&&v>=-1e-8&&u+v<=1+1e-8&&u*a[2]+v*b[2]+(1-u-v)*c[2]>point[2]+epsilon)return true;
  }
  return false;
 }
 function arrow(ctx,a,b,colour,width=2,dashed=false){
  if(!a||!b)return;
  ctx.save();ctx.strokeStyle=colour;ctx.fillStyle=colour;ctx.lineWidth=width;ctx.setLineDash(dashed?[5,4]:[]);
  ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
  const dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);
  if(length>6){const size=Math.min(7,length*.3),ux=dx/length,uy=dy/length;ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(b[0]-size*ux+size*.45*uy,b[1]-size*uy-size*.45*ux);ctx.lineTo(b[0]-size*ux-size*.45*uy,b[1]-size*uy+size*.45*ux);ctx.closePath();ctx.fill()}
  ctx.restore();
 }
 function glyph(ctx,p,kind,hidden=false,r=5){
  if(!p)return;
  ctx.save();ctx.setLineDash(hidden?[2,2]:[]);ctx.lineWidth=2;ctx.strokeStyle=kind==='measured'?'#8be0b9':'#f2cb86';ctx.fillStyle=ctx.strokeStyle;
  ctx.beginPath();
  if(kind==='measured')ctx.rect(p[0]-r,p[1]-r,r*2,r*2);
  else{ctx.moveTo(p[0],p[1]-r);ctx.lineTo(p[0]+r,p[1]);ctx.lineTo(p[0],p[1]+r);ctx.lineTo(p[0]-r,p[1]);ctx.closePath()}
  if(!hidden){ctx.save();ctx.globalAlpha=.3;ctx.fill();ctx.restore()}
  ctx.stroke();ctx.restore();
 }
 function tangentBasis(normal){
  const helper=Math.abs(normal[2])<.8?[0,0,1]:[1,0,0],a=unit(cross(normal,helper));
  return [a,cross(normal,a)];
 }
 function drawSpatial(ctx,d,project,options){
  const q=d.attached;if(!finite(q))return;
  ctx.save();ctx.font='12px monospace';
  if(options.normal&&d.normal){
   const [u,v]=tangentBasis(d.normal),radius=.009;
   const corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>project(add(q,add(mul(u,a*radius),mul(v,b*radius)))));
   ctx.beginPath();corners.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle='rgba(142,184,237,.16)';ctx.fill();ctx.strokeStyle='#8eb8ed';ctx.setLineDash([4,3]);ctx.stroke();ctx.setLineDash([]);
   const end=project(add(q,mul(d.normal,.018)));arrow(ctx,project(q),end,'#8eb8ed',2);ctx.fillStyle='#a9cfff';ctx.fillText('n+',end[0]+4,end[1]-4);
   if(d.foot){arrow(ctx,project(d.foot),project(d.measured),'#a9cfff',2,true);const p=project(d.foot);ctx.beginPath();ctx.arc(p[0],p[1],3,0,2*Math.PI);ctx.stroke()}
  }
  if(options.axes&&d.axes){
   d.axes.forEach((axis,k)=>{const end=project(add(q,mul(axis,.018)));arrow(ctx,project(q),end,colours[k],1,true);ctx.fillStyle=colours[k];ctx.fillText('XYZ'[k]+'+',end[0]+4,end[1]-4)});
   if(d.components){let from=q;d.components.forEach((value,k)=>{const to=add(from,mul(d.axes[k],value));arrow(ctx,project(from),project(to),colours[k],2.5);from=to})}
  }
  ctx.restore();
 }
 function drawOrthographic(canvas,d,vertices,faces,plane,{span=20,normal=false,visibility=false,space='hand'}={}){
  const w=canvas.clientWidth,h=canvas.clientHeight,pixelRatio=Math.min(root.devicePixelRatio||1,2);
  if(!w||!h)return;
  const width=Math.round(w*pixelRatio),height=Math.round(h*pixelRatio);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
  const ctx=canvas.getContext('2d');ctx.setTransform(pixelRatio,0,0,pixelRatio,0,0);ctx.fillStyle='#182029';ctx.fillRect(0,0,w,h);ctx.font='12px monospace';
  if(!finite(d.attached)||!d.axes){ctx.fillStyle='#c0cbd2';ctx.fillText('本帧坐标不可用',20,30);return}
  const [a,b]=plane,depth=unit(cross(d.axes[a],d.axes[b])),centre=[w/2,h/2],scale=Math.min(w-64,h-64)/(2*span);
  const project=p=>{if(!finite(p))return null;const delta=sub(p,d.attached);return [centre[0]+dot(delta,d.axes[a])*1000*scale,centre[1]-dot(delta,d.axes[b])*1000*scale,dot(delta,depth)]};
  const pv=vertices.map(project),step=span<=10?2:span<=20?5:10;
  ctx.save();ctx.beginPath();ctx.rect(centre[0]-span*scale,centre[1]-span*scale,2*span*scale,2*span*scale);ctx.clip();
  const visible=faces.filter(f=>{const t=f.map(i=>pv[i]);return t.every(finite)&&Math.max(...t.map(p=>p[0]))>=centre[0]-span*scale&&Math.min(...t.map(p=>p[0]))<=centre[0]+span*scale&&Math.max(...t.map(p=>p[1]))>=centre[1]-span*scale&&Math.min(...t.map(p=>p[1]))<=centre[1]+span*scale});
  visible.sort((f,g)=>f.reduce((s,i)=>s+pv[i][2],0)-g.reduce((s,i)=>s+pv[i][2],0));
  for(const f of visible){ctx.beginPath();f.forEach((id,k)=>k?ctx.lineTo(pv[id][0],pv[id][1]):ctx.moveTo(pv[id][0],pv[id][1]));ctx.closePath();ctx.fillStyle='#384047';ctx.fill();ctx.strokeStyle='#48515a';ctx.lineWidth=.45;ctx.stroke()}
  ctx.strokeStyle='#6d7d88';ctx.lineWidth=.6;
  for(let mm=-span;mm<=span;mm+=step){ctx.globalAlpha=mm===0?.75:.35;ctx.beginPath();ctx.moveTo(centre[0]+mm*scale,centre[1]-span*scale);ctx.lineTo(centre[0]+mm*scale,centre[1]+span*scale);ctx.moveTo(centre[0]-span*scale,centre[1]+mm*scale);ctx.lineTo(centre[0]+span*scale,centre[1]+mm*scale);ctx.stroke()}
  ctx.globalAlpha=1;
  if(normal)drawSpatial(ctx,d,project,{normal:true});
  const attached=project(d.attached),measured=project(d.measured);
  const hidden=visibility&&occluded(measured,pv,faces),attachedHidden=visibility&&occluded(attached,pv,faces);
  arrow(ctx,attached,measured,'#e2e7ed',2,hidden||attachedHidden);glyph(ctx,attached,'attached',attachedHidden,6);glyph(ctx,measured,'measured',hidden,6);
  ctx.restore();ctx.fillStyle='#d1dae0';ctx.font='12px monospace';
  ctx.fillText(`${'XYZ'[a]}${'XYZ'[b]} · ${space==='hand'?'手部':'世界'} · 每格 ${step} mm`,10,18);
  ctx.fillStyle=colours[a];ctx.fillText('XYZ'[a]+'+ →',w-58,h-12);ctx.fillStyle=colours[b];ctx.fillText('XYZ'[b]+'+ ↑',8,40);
  ctx.fillStyle='#d1dae0';ctx.fillText(`±${span} mm · ◇ 为原点`,10,h-12);
  if(measured&&(Math.abs(measured[0]-centre[0])>span*scale||Math.abs(measured[1]-centre[1])>span*scale)){ctx.fillStyle='#f2cb86';ctx.fillText('实测点超出范围，请扩大视窗',10,h-30)}
 }
 const api={add,sub,mul,dot,cross,finite,unit,world,colours,analyse,signed,axisText,occluded,arrow,glyph,drawSpatial,drawOrthographic};
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.NokovMarkerDiagnostics=api;
})(globalThis);
