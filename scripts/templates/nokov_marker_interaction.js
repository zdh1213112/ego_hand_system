(function(root){
 'use strict';
 const finite=p=>Array.isArray(p)&&p.length>=2&&Number.isFinite(p[0])&&Number.isFinite(p[1]);
 function segmentDistance(point,a,b){
  if(!finite(point)||!finite(a)||!finite(b))return Infinity;
  const dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;
  const t=length?Math.max(0,Math.min(1,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length)):0;
  return Math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy);
 }
 function pickMarker(point,measured,attached,{markers=true,residuals=true,selected=-1,radius=12}={}){
  let best=-1,score=Infinity;
  for(let k=0;k<measured.length;k++){
   const active=k===selected,m=measured[k],a=attached[k],pair=finite(m)&&finite(a);
   const endpoints=[];
   if((markers||residuals||active)&&finite(m))endpoints.push(m);
   if(((residuals&&pair)||active)&&finite(a))endpoints.push(a);
   for(const q of endpoints){
    const distance=Math.hypot(point[0]-q[0],point[1]-q[1]);
    if(distance<=radius&&(distance<score||(distance===score&&active))){score=distance;best=k}
   }
  }
  if(best>=0)return best;
  for(let k=0;k<measured.length;k++){
   if(!residuals&&k!==selected)continue;
   const distance=segmentDistance(point,measured[k],attached[k]);
   if(distance<=6&&distance<score){score=distance;best=k}
  }
  return best;
 }
 function startGesture(x,y,id){return {startX:x,startY:y,lastX:x,lastY:y,pointerId:id,moved:false}}
 function moveGesture(gesture,x,y){
  const moved=gesture.moved||Math.hypot(x-gesture.startX,y-gesture.startY)>5;
  const delta=moved?[x-(gesture.moved?gesture.lastX:gesture.startX),y-(gesture.moved?gesture.lastY:gesture.startY)]:[0,0];
  return {gesture:{...gesture,lastX:x,lastY:y,moved},delta};
 }
 const api={segmentDistance,pickMarker,startGesture,moveGesture};
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.NokovMarkerInteraction=api;
})(globalThis);
