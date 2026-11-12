const {test} = require('node:test');
const assert = require('node:assert/strict');
const I = require('../scripts/templates/nokov_marker_inspector.js');
const S = require('../scripts/templates/nokov_marker_diagnostics.js');
const near = (a,b,tol=1e-12) => assert.ok(Math.abs(a-b)<tol, `${a} != ${b}`);

test('zoom keeps the point below the cursor fixed and has no former 3x limit', () => {
  const c=I.camera(),w=1000,h=600,p=[731,173],before=structuredClone(c);
  const position=()=>[(p[0]-w/2)*c.span/Math.min(w,h)+c.pan[0],-(p[1]-h/2)*c.span/Math.min(w,h)+c.pan[1]];
  const world=position();
  I.zoomAt(c,-Math.log(2),p,w,h);
  position().forEach((v,k)=>near(v,world[k]));
  I.zoomAt(c,Math.log(2),p,w,h);
  near(c.span,before.span);c.pan.forEach((v,k)=>near(v,before.pan[k]));
  for(let k=0;k<40;k++)I.zoomAt(c,-Math.log(2),[w/2,h/2],w,h);
  assert.ok(c.referenceSpan/c.span>1e12);
  assert.ok(c.span>0 && Number.isFinite(c.span));
});

test('panning is measured in screen pixels at every zoom and camera rotation preserves distances', () => {
  for(const span of [.2,.02,1e-10]){
    const c={...I.camera(),span},scale=500/span;
    I.panPixels(c,90,-75,800,500);
    near(-c.pan[0]*scale,90);near(c.pan[1]*scale,-75);
    const v=[.003,-.007,.009];near(Math.hypot(...I.rotate(v,c)),Math.hypot(...v));
  }
});

test('fit and endpoint targets handle coincident points and missing measurements without inventing a residual', () => {
  const d=S.analyse(null,[.01,.02,.03],S.world,[0,0,1]),c=I.camera();
  I.fit(c,d);assert.ok(c.span>0);assert.equal(d.distance,null);
  c.target='measured';assert.deepEqual(I.target(d,c),d.attached);
  const same=S.analyse(d.attached,d.attached,S.world,[0,0,1]);I.fit(c,same);
  assert.ok(c.span>0);assert.equal(same.distance,0);
  const pair=S.analyse([.02,.04,.06],d.attached,S.world,[0,0,1]);I.fit(c,pair);
  assert.deepEqual(I.target(pair,c),[.015,.03,.045]);
  c.target='attached';assert.deepEqual(I.target(pair,c),pair.attached);
});

test('high zoom clips an enclosing triangle even when all its vertices are offscreen', () => {
  const r=1e-9,bounds=[-r,r,-r,r],triangle=[[-1,-1,.2],[1,-1,.2],[0,1,.2]];
  const result=I.clipPolygon(triangle,bounds);
  assert.ok(result.length>=4);
  result.forEach(p=>{assert.ok(p.every(Number.isFinite));assert.ok(Math.abs(p[0])<=r*1.001);assert.ok(Math.abs(p[1])<=r*1.001);near(p[2],.2)});
  let area=0;result.forEach((p,k)=>{const q=result[(k+1)%result.length];area+=p[0]*q[1]-q[0]*p[1]});
  near(Math.abs(area)/2/(4*r*r),1,1e-6);
  assert.deepEqual(I.clipPolygon([[1,1,0],[2,1,0],[1,2,0]],bounds),[]);
});

test('clipped residual segments remain finite and outside or degenerate segments are handled', () => {
  const bounds=[-1e-8,1e-8,-1e-8,1e-8];
  const line=I.clipSegment([-1,0,0],[1,0,.2],bounds);
  assert.equal(line.length,2);line.forEach(p=>assert.ok(p.every(Number.isFinite)));
  near(line[0][0],-1e-8);near(line[1][0],1e-8);
  assert.equal(I.clipSegment([2,2,0],[2,3,0],bounds),null);
  assert.deepEqual(I.clipSegment([0,0,0],[0,0,0],bounds),[[0,0,0],[0,0,0]]);
});

test('scale bar remains readable below a millimetre', () => {
  assert.equal(I.lengthText(.002),'2 mm');
  assert.equal(I.lengthText(.000002),'2 μm');
  assert.equal(I.lengthText(2e-9),'2 nm');
  assert.equal(I.lengthText(0),'0 mm');
  near(I.niceLength(.0000036),.000002);
});
