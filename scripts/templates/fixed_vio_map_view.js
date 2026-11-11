'use strict';
const D=JSON.parse(document.getElementById('map-data').textContent),S=D.summary;
const $=id=>document.getElementById(id),names=Object.keys(D.maps);
const f=(v,n=2)=>v===null||v===undefined||!Number.isFinite(Number(v))?'—':Number(v).toFixed(n);
const cm=v=>v===null||v===undefined?null:v*100;
const colours={original_2cam:'#0067ff',monado_2cam:'#d500d9',monado_4cam:'#ff7300',reference:'#007f73',...D.comparison.colours};
const labels={original_2cam:'原版 Basalt · 双目',monado_2cam:'Monado · 双目',monado_4cam:'Monado · 四目',...D.comparison.labels};
const meanings={original_2cam:'原版 Basalt：双目图像 + IMU 的估计轨迹',monado_2cam:'Monado Basalt：双目图像 + IMU 的估计轨迹',monado_4cam:'Monado Basalt：四目图像 + IMU 的估计轨迹'};
let index=0,playing=false,last=null,target=0,animation=null,yaw=-.72,pitch=.48,zoom=1,drag=null;
const pan3d={x:0,y:0},pan2d={x:0,y:0};
function resetPanning(){pan3d.x=pan3d.y=pan2d.x=pan2d.y=0}
const current=()=>D.maps[$('case').value],overview=()=>$('mode').value==='overview';
const geometry=()=>$('coordinates').value==='imu0'?current().imu0:current();
const comparison=()=>D.comparison[$('coordinates').value];
const activeNames=()=>names.filter(n=>$('show_'+n).checked);
const colour=rgb=>`rgb(${rgb.join(',')})`;
const RECENT_TRAIL_SECONDS=3;
if(D.trajectory_only){
 document.title=`VIO / NOKOV · ${S.session} 算法轨迹对比`;
 document.querySelector('header h1').textContent=`VIO / NOKOV：${names.length} 种配置轨迹对比`;
 document.querySelector('section.card h2').textContent='有效参考区间位置 RMSE';
 document.querySelector('header .sub').textContent=`Session ${S.session} · 同一参考坐标 · 已保存的 VIO 轨迹`;
 document.querySelector('header .badge').textContent='固定标定 · 有效参考区间';
 $('summaryStats').innerHTML=names.map(name=>`<div class="stat"><span>${labels[name]}</span><strong id="rmse_${name}"></strong></div>`).join('')+'<div class="stat"><span>完整录制时长</span><strong id="duration"></strong></div>';
 $('sourceNote').textContent=`页面显示 ${f(D.time[0],3)}–${f(D.time.at(-1),3)} s 的 ${D.time.length} 个真值有效位姿（完整输出 ${S.input_poses} 帧）。首尾未覆盖区间保留在原轨迹中；未外推真值。当前展示一个重复运行，跨 session 报告另列三次运行范围。`;
 $('trajectoryDetails').href='#currentErrors';$('staticMaps').hidden=true;
 for(const id of ['showPoints','colourMode','radius','pointSize','showKeyframes'])$(id).closest('label').hidden=true;
 $('pointControls').querySelector('summary').textContent='姿态与误差选项';document.querySelector('.cloud-note').hidden=true;
 const selector=$('case');selector.innerHTML=names.map(name=>`<option value="${name}">${labels[name]}</option>`).join('');
 selector.value=names.includes('monado_4cam')?'monado_4cam':names[0];
 selector.parentElement.firstChild.textContent='姿态配置 ';
 $('accuracyDownload').href=D.exports.metrics;$('performanceDownload').href=D.exports.performance;$('summaryDownload').href=D.exports.summary;
 const group=$('metrics').closest('.metric-group');group.querySelector('h3').textContent='有效参考区间轨迹精度';
 const head=group.querySelector('thead tr');for(let i=0;i<3;i++)head.lastElementChild.remove();
 document.querySelector('.footer').textContent='已保存位姿与固定标定 · 真值覆盖区间内评价 · 轨迹单位 m';
 for(const id of ['map3d','map2d'])$(id).setAttribute('aria-label',`${names.length} 种配置与 NOKOV 真值的轨迹对比`);
}
const errorCache=Object.fromEntries(names.map(name=>{
 const series=D.errors.cases[name],position=series.position_error_m,orientation=series.orientation_error_deg;
 if(position.length!==D.time.length||orientation.length!==D.time.length)throw new Error('Per-frame error/time mismatch');
 const positionSquares=new Float64Array(position.length+1),orientationSquares=new Float64Array(orientation.length+1);
 let maxPosition=0,maxOrientation=0;
 for(let i=0;i<position.length;i++){
  if(!Number.isFinite(position[i])||!Number.isFinite(orientation[i]))throw new Error('Invalid per-frame error');
  positionSquares[i+1]=positionSquares[i]+position[i]*position[i];
  orientationSquares[i+1]=orientationSquares[i]+orientation[i]*orientation[i];
  maxPosition=Math.max(maxPosition,position[i]);maxOrientation=Math.max(maxOrientation,orientation[i]);
 }
 return [name,{position,orientation,positionSquares,orientationSquares,maxPosition,maxOrientation}];
}));

function visiblePoints(){
 const g=geometry(),origin=comparison().reference[0],radius=$('radius').value;
 return g.points.map((p,i)=>({p,i})).filter(({p})=>radius==='all'||Math.hypot(...p.map((v,k)=>v-origin[k]))<=Number(radius));
}
function viewExtent(){
 const g=geometry(),c=comparison(),pts=[];
 if($('showPoints').checked)pts.push(...visiblePoints().map(p=>p.p));
 for(const name of activeNames())pts.push(...c.online[name]);
 if($('showReference').checked)pts.push(...c.reference);
 if($('showKeyframes').checked)pts.push(...g.keyframes);
 if($('showAxes').checked)pts.push([0,0,0]);
 if(!pts.length)pts.push([0,0,0]);
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(const p of pts)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k])}
 return {center:lo.map((v,k)=>(v+hi[k])/2),radius:Math.max(.15,...hi.map((v,k)=>v-lo[k]))*.6,lo,hi};
}
function context(id){
 const c=$(id),w=c.clientWidth,h=c.clientHeight,d=devicePixelRatio||1;
 if(c.width!==Math.round(w*d)||c.height!==Math.round(h*d)){c.width=Math.round(w*d);c.height=Math.round(h*d)}
 const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);x.clearRect(0,0,w,h);return {c,x,w,h};
}
function path(x,points,project,color,start,end,alpha,width,dash=[],halo=false){
 if(end<=start)return 0;
 x.save();x.beginPath();
 for(let i=start;i<=end;i++){const p=project(points[i]);i===start?x.moveTo(...p):x.lineTo(...p)}
 x.lineCap='round';x.lineJoin='round';x.setLineDash(dash);
 if(halo){x.globalAlpha=.86;x.strokeStyle='#ffffff';x.lineWidth=width+3;x.stroke()}
 x.strokeStyle=color;x.globalAlpha=alpha;x.lineWidth=width;x.stroke();x.restore();return end-start;
}
function marker(x,p,color,size,kind='circle'){
 x.save();x.beginPath();
 if(kind==='square')x.rect(p[0]-size,p[1]-size,size*2,size*2);
 else if(kind==='triangle'){x.moveTo(p[0],p[1]-size*1.25);x.lineTo(p[0]+size,p[1]+size);x.lineTo(p[0]-size,p[1]+size);x.closePath()}
 else if(kind==='diamond'){x.moveTo(p[0],p[1]-size*1.3);x.lineTo(p[0]+size,p[1]);x.lineTo(p[0],p[1]+size*1.3);x.lineTo(p[0]-size,p[1]);x.closePath()}
 else x.arc(...p,size,0,Math.PI*2);
 x.fillStyle=color;x.fill();x.strokeStyle='#fff';x.lineWidth=1.8;x.stroke();x.restore();
}
function axes(x,project,origin,rotation,length,label,thin=false){
 const axisColours=['#df4f4f','#199a68','#397ee0'],ends=[];
 for(let k=0;k<3;k++){
  const end=origin.map((v,j)=>v+length*rotation[j*3+k]);ends.push(end);
  path(x,[origin,end],project,axisColours[k],0,1,1,thin?1.5:2.2);
  const a=project(origin),b=project(end),angle=Math.atan2(b[1]-a[1],b[0]-a[0]);
  x.beginPath();x.moveTo(...b);x.lineTo(b[0]-6*Math.cos(angle-.4),b[1]-6*Math.sin(angle-.4));x.lineTo(b[0]-6*Math.cos(angle+.4),b[1]-6*Math.sin(angle+.4));x.closePath();x.fillStyle=axisColours[k];x.fill();
  x.fillStyle=axisColours[k];x.font='11px system-ui';x.fillText('XYZ'[k]+label,b[0]+5,b[1]+[-6,12,-4][k]);
 }
 return ends;
}
function drawScene(x,project,e,viewId){
 const m=current(),g=geometry(),c=comparison(),active=activeNames(),end=overview()?D.time.length-1:index;
 let recent=index;while(recent>0&&D.time[recent]>D.time[index]-RECENT_TRAIL_SECONDS)recent--;
 let drawn=0;const eligible=visiblePoints();
 if($('showPoints').checked){
  x.save();x.globalAlpha=.42;
  for(const {p,i} of eligible){
   if(!overview()&&m.first_seen_s[i]>D.time[index]+1e-8)continue;
   let fill;
   if($('colourMode').value==='rgb'&&m.rgb)fill=colour(m.rgb[i]);
   else{const h=(p[2]-e.lo[2])/Math.max(.001,e.hi[2]-e.lo[2]);fill=`hsl(${175-145*h},57%,47%)`}
   x.beginPath();x.arc(...project(p),Number($('pointSize').value),0,Math.PI*2);x.fillStyle=fill;x.fill();drawn++;
  }
  x.restore();
 }
 const series=active.map(name=>({name,p:c.online[name],colour:colours[name],kind:['circle','square','triangle'][names.indexOf(name)],dash:[]}));
 if($('showReference').checked)series.push({name:'reference',p:c.reference,colour:colours.reference,kind:'diamond',dash:[7,4]});
 const stats={historySegments:0,recentSegments:0,futureSegments:0,maxDrawnIndex:series.length?index:-1,currentPoints:series.length};
 if(overview()&&index<end)for(const line of series){stats.futureSegments+=path(x,line.p,project,line.colour,index,end,.12,1.2,[3,5]);stats.maxDrawnIndex=end}
 if($('showKeyframes').checked){
  let k=g.keyframes.length-1;
  if(!overview()){k=0;while(k+1<g.keyframes.length&&m.keyframe_times_s[k+1]<=D.time[index])k++;if(m.keyframe_times_s[0]>D.time[index])k=-1}
  path(x,g.keyframes,project,'#64748b',0,k,.8,1.7,[2,5],true);
 }
 for(const line of series)stats.historySegments+=path(x,line.p,project,line.colour,0,recent,line.name==='reference'?.28:.22,line.name==='reference'?1.3:1.1,line.dash);
 for(const line of series)stats.recentSegments+=path(x,line.p,project,line.colour,recent,index,1,line.name==='reference'?3.6:3.1,line.dash,true);
 if($('showErrors').checked&&$('showReference').checked)for(const name of active)path(x,[c.online[name][index],c.reference[index]],project,colours[name],0,1,.9,1.3,[2,4]);
 for(const line of series)marker(x,project(line.p[index]),line.colour,line.name==='reference'?5.7:5.1,line.kind);
 $('mapStatus').textContent=`${f(D.time[index],3)} s · ${index+1} / ${D.time.length} 帧 · ${series.length} 条轨迹`+($('showPoints').checked?` · 点云 ${drawn} / ${g.points.length} 点`:'');
 const pan=viewId==='3d'?pan3d:pan2d;
 const state={index,overview:overview(),shownPoints:drawn,totalPoints:g.points.length,trajectoryEnd:end,coordinateFrame:$('coordinates').value,colourMode:$('colourMode').value,viewRadius:e.radius,includesMapInExtent:$('showPoints').checked,activeTrajectories:series.map(s=>s.name),panX:pan.x,panY:pan.y,recentWindowSeconds:RECENT_TRAIL_SECONDS,recentStartIndex:recent,recentStartTimeS:D.time[recent],...stats};
 window.mapRenderStates=window.mapRenderStates||{};window.mapRenderStates[viewId]=state;window.mapRenderState=state;
 window.mapViewInteractionState={pan3d:{...pan3d},pan2d:{...pan2d},yaw,pitch,zoom};
}
function draw3d(){
 const {x,w,h}=context('map3d'),e=viewExtent(),c=comparison(),scale=Math.min(w,h)*.39/e.radius*zoom;
 const co=Math.cos(yaw),si=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const project=p=>{const [a,b,z]=p.map((v,k)=>v-e.center[k]),u=co*a-si*b,v=si*a+co*b;return [w/2+pan3d.x+scale*u,h/2+pan3d.y-scale*(sp*v+cp*z)]};
 if($('showAxes').checked){
  const span=e.radius*.8,step=span/2;
  for(let j=-2;j<=2;j++){const v=j*step;path(x,[[-span,v,0],[span,v,0]],project,'#dce6ed',0,1,.8,1);path(x,[[v,-span,0],[v,span,0]],project,'#dce6ed',0,1,.8,1)}
  axes(x,project,[0,0,0],[1,0,0,0,1,0,0,0,1],e.radius*.4,$('coordinates').value==='imu0'?'₀':'ₙ');
 }
 if($('showBody').checked){const name=$('case').value,length=Math.max(.035,Math.min(.35,e.radius*.14));window.imuAxisEndpoints=axes(x,project,c.online[name][index],c.imu_rotations[name][index],length,'ᵢ',true)}
 drawScene(x,project,e,'3d');
 x.fillStyle='#617b8b';x.font='12px system-ui';x.fillText(($('coordinates').value==='imu0'?'共同首帧 IMU I₀':'NOKOV 世界 N')+' / m',15,23);
}
function draw2d(){
 const {x,w,h}=context('map2d'),e=viewExtent(),[a,b]=$('plane').value.split('').map(Number),scale=Math.min(w-90,h-75)/(e.radius*2)*zoom,cx=(w+35)/2,cy=(h-10)/2;
 const centerA=e.center[a]-pan2d.x/scale,centerB=e.center[b]+pan2d.y/scale;
 const project=p=>[cx+(p[a]-centerA)*scale,cy-(p[b]-centerB)*scale];
 x.font='11px system-ui';
 for(let k=-2;k<=2;k++){const v=k*e.radius/(2*zoom),px=cx+v*scale,py=cy-v*scale;x.strokeStyle='#dfe8ef';x.lineWidth=1;x.beginPath();x.moveTo(px,25);x.lineTo(px,h-35);x.moveTo(55,py);x.lineTo(w-15,py);x.stroke();x.fillStyle='#698295';x.fillText(f(centerA+v),px-12,h-15);x.fillText(f(centerB+v),8,py+4)}
 x.fillStyle='#4f6c80';x.fillText('XYZ'[a]+' / m',w-48,h-1);x.fillText('XYZ'[b]+' / m',15,16);
 x.save();x.beginPath();x.rect(55,25,w-70,h-60);x.clip();drawScene(x,project,e,'2d');x.restore();
}
function drawErrorChart(id,field,maxField,factor,unit){
 const {c,x,w,h}=context(id),L=64,R=20,T=24,B=30,active=activeNames();
 const duration=D.time[D.time.length-1],span=Math.max(duration,1e-9),plotWidth=w-L-R,plotHeight=h-T-B;
 const high=Math.max(.1,active.reduce((maximum,name)=>Math.max(maximum,errorCache[name][maxField]*factor),0))*1.12;
 const px=t=>L+t/span*plotWidth,py=v=>h-B-v/high*plotHeight;
 x.save();x.font='11px system-ui';
 if(index<D.time.length-1){
  const left=px(D.time[index]);x.fillStyle='#f3f6fa';x.fillRect(left,T,w-R-left,plotHeight);
  if(w-R-left>90){x.fillStyle='#91a3b1';x.textAlign='center';x.fillText('待播放',left+(w-R-left)/2,T+plotHeight/2);x.textAlign='left'}
 }
 for(let i=0;i<=3;i++){
  const value=high*i/3,y=py(value);x.beginPath();x.moveTo(L,y);x.lineTo(w-R,y);x.strokeStyle='#e0e8ef';x.lineWidth=1;x.stroke();
  x.fillStyle='#6b8295';x.textAlign='right';x.fillText(f(value),L-10,y+4);
 }
 for(let i=0;i<=6;i++){
  const time=duration*i/6;x.textAlign=i===0?'left':i===6?'right':'center';x.fillStyle='#6b8295';x.fillText(f(time,1)+' s',px(time),h-8);
 }
 x.textAlign='left';x.fillStyle='#526e83';x.font='12px system-ui';x.fillText(unit,L,15);
 x.save();x.beginPath();x.rect(L-1,T-1,plotWidth+2,plotHeight+2);x.clip();
 for(const name of active){
  const values=errorCache[name][field];x.beginPath();
  for(let i=0;i<=index;i++){const point=[px(D.time[i]),py(values[i]*factor)];i===0?x.moveTo(...point):x.lineTo(...point)}
  x.strokeStyle=colours[name];x.lineWidth=1.7;x.lineJoin='round';x.lineCap='round';x.stroke();
  marker(x,[px(D.time[index]),py(values[index]*factor)],colours[name],3.4);
 }
 x.restore();
 x.beginPath();x.moveTo(px(D.time[index]),T);x.lineTo(px(D.time[index]),h-B);x.strokeStyle='#344f65';x.lineWidth=1;x.stroke();
 if(!active.length){x.textAlign='center';x.fillStyle='#7b8e9d';x.fillText('未选择算法轨迹',L+plotWidth/2,T+plotHeight/2)}
 x.restore();
 Object.assign(c.dataset,{maxTime:String(duration),plotLeft:String(L),plotRight:String(R),currentIndex:String(index),maxDrawnIndex:String(active.length?index:-1),count:String(active.length?index+1:0)});
}
function renderCurrentErrors(){
 const count=index+1,rows={};
 for(const name of names){
  const series=errorCache[name];rows[name]={count,
   position_rmse_m:Math.sqrt(series.positionSquares[count]/count),
   orientation_rmse_deg:Math.sqrt(series.orientationSquares[count]/count),
   current_position_error_m:series.position[index],current_orientation_error_deg:series.orientation[index]};
 }
 $('currentErrorMetrics').innerHTML=names.map(name=>{const r=rows[name];return `<tr data-case="${name}"${$('show_'+name).checked?'':' hidden'}><td class="config" style="color:${colours[name]}">${labels[name]}</td><td>${r.count}</td><td>${f(cm(r.position_rmse_m))}</td><td>${f(r.orientation_rmse_deg)}</td><td>${f(cm(r.current_position_error_m))}</td><td>${f(r.current_orientation_error_deg)}</td></tr>`}).join('');
 $('errorRange').textContent=`累计区间 ${f(D.time[0],3)} → ${f(D.time[index],3)} s`;
 $('currentErrorTime').textContent=`当前采集时间戳：${D.errors.timestamps_ns[index]} ns`;
 drawErrorChart('errorPositionChart','position','maxPosition',100,'三维位置误差 / cm');
 drawErrorChart('errorOrientationChart','orientation','maxOrientation',1,'姿态旋转误差 / °');
 window.currentErrorState={index,rows,maxDrawnIndex:activeNames().length?index:-1,activeCases:activeNames()};
}
function nearestFrame(time){
 let lo=0,hi=D.time.length-1;
 while(lo<hi){const mid=(lo+hi)>>1;if(D.time[mid]<time)lo=mid+1;else hi=mid}
 if(lo>0&&time-D.time[lo-1]<=D.time[lo]-time)lo--;
 return lo;
}
function render(){
 const selected=$('case').value,m=current();draw3d();draw2d();
 renderCurrentErrors();
 $('time').textContent=f(D.time[index],3)+' s';$('frame').value=index;$('frameInfo').textContent=`${index+1} / ${D.time.length}`;
 $('duration').textContent=f(S.duration_s,2)+' s';
 for(const name of names){$('rmse_'+name).textContent=f(cm(D.maps[name].stats.all.position_rmse_m))+' cm';$('rmse_'+name).style.color=colours[name];$('legend_'+name).classList.toggle('off',!$('show_'+name).checked)}
 $('legend_reference').classList.toggle('off',!$('showReference').checked);
 const frame=$('coordinates').value==='imu0'?'imu0':'nokov_world';
 if(D.trajectory_only){const exports=D.exports.cases[selected];$('downloads').innerHTML=`<a href="${exports.errors}">所选配置逐帧位姿与误差</a>`+(exports.performance?`<a href="${exports.performance}">所选配置逐帧后端耗时</a>`:'')}
 else $('downloads').innerHTML=(m.rgb?`<a href="maps/${selected}/map_rgb_${frame}.ply">所选点云 RGB PLY</a>`:`<a href="maps/${selected}/map_nokov_world.ply">所选点云 PLY</a>`)+`<a href="evaluation/${selected}_errors.csv">所选配置逐帧误差</a><a href="performance/${selected}_frame_times.csv">所选配置逐帧后端耗时</a>`;
 $('coordinateNote').textContent=$('coordinates').value==='imu0'?'共同首帧 IMU：原点与轴向取 NOKOV 首帧 IMU 位姿，单位 m。':'NOKOV 场地世界坐标，单位 m。';
 $('metrics').innerHTML=names.map(name=>{const r=D.maps[name].stats,a=r.all,q=r.accuracy||{},re=r.rpe_1s_all;return `<tr><td class="config" style="color:${colours[name]}">${labels[name]}</td><td>${a.poses}</td><td>${f(cm(a.position_rmse_m))}</td><td>${f(cm(q.position_p95_m))}</td><td>${f(cm(q.position_max_m??a.position_max_m))}</td><td>${f(a.orientation_rmse_deg)}</td><td>${f(cm(re?.translation_m?.rmse))} / ${f(re?.rotation_deg?.rmse)}</td><td>${f(cm(q.final_position_error_m))}</td><td>${f(q.pose_output_coverage_pct,1)}</td>`+(D.trajectory_only?'':`<td>${r.mapping.landmarks}</td><td>${f(r.mapping.reprojection_rmse_px)}</td><td>${f(r.mapping.wall_s)}</td>`)+`</tr>`}).join('');
 $('performanceMetrics').innerHTML=names.map(name=>{const r=D.maps[name].stats,p=r.performance||{},b=p.backend_frame_ms||{};return `<tr><td class="config" style="color:${colours[name]}">${labels[name]}</td><td>${f(b.mean)}</td><td>${f(b.median)}</td><td>${f(b.p95)}</td><td>${f(b.max)}</td><td>${f(p.wall_ms_per_frame)}</td><td>${f(p.throughput_fps,1)}</td><td>${f(p.realtime_speedup,2)}${p.realtime_speedup===null||p.realtime_speedup===undefined?'':'×'}</td><td>${f(p.cpu_percent,0)}</td><td>${f(p.peak_rss_mib,1)}</td><td>${f(p.process_wall_s??r.wall_s)}</td></tr>`}).join('');
}
function pause(){playing=false;last=null;if(animation!==null)cancelAnimationFrame(animation);animation=null;$('play').textContent='播放'}
function tick(now){if(!playing)return;if(last!==null)target+=(now-last)/1000*Number($('speed').value);last=now;while(index+1<D.time.length&&D.time[index+1]<=target)index++;render();if(index===D.time.length-1)pause();else animation=requestAnimationFrame(tick)}
$('trajectoryLegend').innerHTML=names.map((name,i)=>`<label id="legend_${name}" class="trajectory-item"><input id="show_${name}" type="checkbox" checked><span class="line-key" style="--line:${colours[name]}">${['●','■','▲'][i]}</span><span><b>${labels[name]}</b><small>${meanings[name]}</small></span></label>`).join('')+`<label id="legend_reference" class="trajectory-item"><input id="showReference" type="checkbox" checked><span class="line-key truth" style="--line:${colours.reference}">◆</span><span><b>${D.comparison.reference_label||'NOKOV 真值（IMU 中心）'}</b><small>头环刚体换算至 IMU 中心的参考轨迹</small></span></label>`;
$('play').onclick=()=>{if(playing){pause();return}if(index===D.time.length-1)index=0;target=D.time[index];playing=true;last=null;$('play').textContent='暂停';animation=requestAnimationFrame(tick)};
$('reset').onclick=()=>{pause();index=0;$('mode').value='progressive';render()};
$('frame').max=D.time.length-1;$('frame').oninput=()=>{pause();index=Number($('frame').value);render()};
for(const id of ['errorPositionChart','errorOrientationChart'])$(id).onclick=e=>{
 const chart=$(id),bounds=chart.getBoundingClientRect(),left=Number(chart.dataset.plotLeft),right=Number(chart.dataset.plotRight);
 const fraction=Math.max(0,Math.min(1,(e.clientX-bounds.left-left)/(bounds.width-left-right)));
 pause();index=nearestFrame(fraction*Number(chart.dataset.maxTime));target=D.time[index];render();
};
for(const id of ['mode','radius','plane','showPoints','showReference','showKeyframes','showAxes','showBody','showErrors','coordinates',...names.map(n=>'show_'+n)])$(id).onchange=()=>{zoom=1;resetPanning();render()};
for(const id of ['colourMode','pointSize'])$(id).onchange=render;
$('case').onchange=()=>{zoom=1;resetPanning();render()};
$('home').onclick=()=>{yaw=-.72;pitch=.48;zoom=1;resetPanning();render()};
$('home2d').onclick=()=>{pan2d.x=pan2d.y=0;draw2d()};
function idleCursor(id){return id==='map2d'||$('interaction3d').value==='pan'?'move':'grab'}
function finishDrag(e){
 if(!drag||drag.id!==e.currentTarget.id||drag.pointerId!==e.pointerId)return;
 drag=null;e.currentTarget.style.cursor=idleCursor(e.currentTarget.id);
}
for(const id of ['map3d','map2d']){
 const canvas=$(id);canvas.style.cursor=idleCursor(id);
 canvas.oncontextmenu=e=>e.preventDefault();
 canvas.onpointerdown=e=>{
  if(drag||![0,1,2].includes(e.button))return;
  e.preventDefault();
  const pan=id==='map2d'||$('interaction3d').value==='pan'||e.shiftKey||e.button!==0;
  drag={id,pointerId:e.pointerId,x:e.clientX,y:e.clientY,pan};
  canvas.setPointerCapture(e.pointerId);canvas.style.cursor=pan?'move':'grabbing';
 };
 canvas.onpointermove=e=>{
  if(!drag||drag.id!==id||drag.pointerId!==e.pointerId)return;
  e.preventDefault();const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
  if(drag.pan){const offset=id==='map3d'?pan3d:pan2d;offset.x+=dx;offset.y+=dy}
  else{yaw+=dx*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.008))}
  drag.x=e.clientX;drag.y=e.clientY;
  if(id==='map3d')draw3d();else draw2d();
 };
 canvas.onpointerup=e=>{finishDrag(e);if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId)};
 canvas.onpointercancel=finishDrag;canvas.onlostpointercapture=finishDrag;
}
$('interaction3d').onchange=()=>{$('map3d').style.cursor=idleCursor('map3d')};
for(const id of ['map3d','map2d'])$(id).onwheel=e=>{e.preventDefault();zoom=Math.max(.1,Math.min(15,zoom*Math.exp(-e.deltaY*.001)));draw3d();draw2d()};
const hasRGB=names.every(n=>D.maps[n].rgb);
$('colourMode').querySelector('[value=rgb]').disabled=!hasRGB;$('colourMode').value=hasRGB?'rgb':'height';
if(hasRGB&&D.rgb_static_plot_available)$('staticMaps').href='rgb_maps_imu0.png';
addEventListener('resize',render);
addEventListener('pageshow',()=>{pause();index=0;$('mode').value='progressive';$('coordinates').value='imu0';render()});render();
