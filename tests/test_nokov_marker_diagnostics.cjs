const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const spatial = require('../scripts/templates/nokov_marker_diagnostics.js');
const near = (a,b) => assert.ok(Math.abs(a-b)<1e-10, `${a} != ${b}`);

test('signed local components and normal/tangent lengths reconstruct the same residual', () => {
  const q = [.1,.2,.3], m = [.102,.197,.304];
  const d = spatial.analyse(m,q,spatial.world,[0,0,1]);
  [d.components[0],d.components[1],d.components[2]].forEach((v,k)=>near(v,[.002,-.003,.004][k]));
  near(d.normalDistance,.004); near(d.tangentDistance,Math.hypot(.002,.003));
  near(Math.hypot(d.normalDistance,d.tangentDistance),d.distance);
  near(spatial.dot(spatial.sub(d.foot,q),d.normal),0);
  const inward = spatial.analyse([.1,.2,.297],q,spatial.world,[0,0,1]);
  near(inward.normalDistance,-.003);
  const bent = spatial.analyse(m,q,spatial.world,[1,0,0]);
  near(bent.normalDistance,.002);
});

test('hand coordinates are invariant under rigid rotation and translation', () => {
  const rotate = p => [-p[1],p[2],-p[0]], translate=p=>spatial.add(rotate(p),[3,2,1]);
  const q=[.1,.2,.3],m=[.102,.197,.304];
  const first=spatial.analyse(m,q,spatial.world,[0,0,1]);
  const moved=spatial.analyse(translate(m),translate(q),spatial.world.map(rotate),rotate([0,0,1]));
  first.components.forEach((v,k)=>near(v,moved.components[k]));
  near(first.normalDistance,moved.normalDistance);
});

test('missing measurements and degenerate axes/normals remain unavailable, real zero stays zero', () => {
  const missing=spatial.analyse(null,[0,0,0],spatial.world,[0,0,1]);
  assert.equal(missing.components,null);assert.equal(missing.normalDistance,null);assert.equal(missing.foot,null);
  const noNormal=spatial.analyse([1,0,0],[0,0,0],[null,null,null],[0,0,0]);
  assert.equal(noNormal.axes,null);assert.equal(noNormal.tangentDistance,null);
  assert.equal(spatial.signed(null),'—');assert.equal(spatial.signed(NaN),'—');
  assert.equal(spatial.signed(0),'0.00 mm');assert.equal(spatial.signed(-.00000001),'0.00 mm');
  assert.match(spatial.axisText(-.002,0,'hand'),/拇指侧/);
  assert.match(spatial.axisText(.002,0,'hand'),/小指侧/);
  assert.match(spatial.axisText(-.002,2,'hand'),/掌心侧/);
});

test('occlusion uses interpolated triangle depth and does not mark surface or foreground points hidden', () => {
  const vertices=[[0,0,.002],[10,0,.004],[0,10,.002]],faces=[[0,1,2]];
  assert.equal(spatial.occluded([2,2,0],vertices,faces),true);
  assert.equal(spatial.occluded([2,2,.0024],vertices,faces),false);
  assert.equal(spatial.occluded([2,2,.006],vertices,faces),false);
  assert.equal(spatial.occluded([9,9,0],vertices,faces),false);
  assert.equal(spatial.occluded(null,vertices,faces),false);
  assert.equal(spatial.occluded([2,2,0],[[0,0,1],[1,1,1],[2,2,1]],faces),false);
});

test('all diagnostics are opt-in; collapse suspends enabled options and world mode preserves raw delta', () => {
  const template=fs.readFileSync(path.join(__dirname,'../scripts/templates/nokov_mano.html'),'utf8');
  assert.doesNotMatch(template.match(/<details id="direction-options"[^>]*>/)[0],/\bopen\b/);
  for(const key of ['axes','normal','views','visibility']) assert.doesNotMatch(template.match(new RegExp('<input id="direction-'+key+'"[^>]*>'))[0],/\bchecked\b/);
  const nodes={'direction-options':{open:false},'direction-space':{value:'hand'},'direction-span':{value:'20'}};
  for(const key of ['axes','normal','views','visibility'])nodes['direction-'+key]={checked:true};
  const arrays={measured:new Float32Array(72),attached:new Float32Array(72),normals:new Float32Array(72),axes:new Float32Array([0,1,0,-1,0,0,0,0,1])};
  arrays.measured.set([.002,.003,.004]);arrays.normals.set([0,0,1]);
  const sandbox=vm.createContext({$:id=>nodes[id],spatial,index:0,selectedMarker:0,directionArrays:arrays});
  vm.runInContext(template.slice(template.indexOf('function directionOptions()'),template.indexOf('function draw(){')),sandbox);
  let settings=vm.runInContext('directionOptions()',sandbox);
  assert.equal(settings.axes,false);assert.equal(settings.views,false);assert.equal(settings.normal,false);assert.equal(settings.visibility,false);
  nodes['direction-options'].open=true;settings=vm.runInContext('directionOptions()',sandbox);assert.equal(settings.axes,true);
  const local=vm.runInContext('currentDirection(directionOptions())',sandbox);
  near(local.components[0],arrays.measured[1]);near(local.components[1],-arrays.measured[0]);
  nodes['direction-space'].value='world';
  const world=vm.runInContext('currentDirection(directionOptions())',sandbox);
  world.components.forEach((v,k)=>near(v,arrays.measured[k]));near(world.distance,local.distance);
});
