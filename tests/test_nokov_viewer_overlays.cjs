const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const template = fs.readFileSync(path.join(__dirname, '../scripts/templates/nokov_mano.html'), 'utf8');
const source = template.slice(template.indexOf('const overlays=D.overlays,'), template.lastIndexOf('</script>'));

function fixture() {
  class Element {
    constructor() { this.children = []; this.value = ''; this.textContent = ''; this.readyState = 0; this.paused = true; this.currentTime = 0; this.duration = 30; }
    append(node) { this.children.push(node); }
    replaceChildren() { this.children = []; }
    removeAttribute(key) { delete this[key]; }
    load() { this.readyState = 0; }
    pause() { this.paused = true; }
    play() { this.paused = false; return Promise.resolve(); }
  }
  const nodes = {};
  const $ = id => nodes[id] ||= new Element();
  const variant = folder => ({label: folder, poster: '', links: {}, summary: {frames: 900, limitations: []}, alignment: {}, videos: {
    '4views': {label: '四路拼接', file: `${folder}/four.mp4`, duration_s: 30, width: 1920, height: 1200},
    'cam0_left': {label: 'cam0', file: `${folder}/cam0.mp4`, duration_s: 30, width: 1920, height: 1200},
  }});
  const context = vm.createContext({$, document: {createElement: () => new Element()},
    D: {overlays: {default: 'original', variants: {original: variant('original'), aligned: variant('aligned')}}}});
  vm.runInContext(source, context);
  $('overlay-video').readyState = 4; $('overlay-video').oncanplay();
  return {$, context, run: code => vm.runInContext(code, context)};
}

test('switches version and camera at the same playback time and preserves playing state', async () => {
  const {$} = fixture(), video = $('overlay-video');
  video.readyState = 4; video.currentTime = 12.5; video.paused = false;
  $('overlay-version').value = 'aligned'; $('overlay-version').onchange();
  assert.equal(video.src, 'aligned/four.mp4');
  video.readyState = 1; video.onloadedmetadata();
  assert.equal(video.paused, true, 'metadata alone cannot resume an unbuffered seek');
  video.readyState = 4; video.oncanplay();
  assert.equal(video.currentTime, 12.5); assert.equal(video.paused, false);
  $('overlay-camera').value = 'cam0_left'; $('overlay-camera').onchange();
  assert.equal(video.src, 'aligned/cam0.mp4');
  video.readyState = 1; video.onloadedmetadata();
  video.readyState = 4; video.oncanplay();
  assert.equal(video.currentTime, 12.5); assert.equal(video.paused, false);
  assert.equal($('overlay-download').href, 'aligned/cam0.mp4');
});

test('rapid switches retain requested time and ignore stale loadedmetadata callback', () => {
  const {$} = fixture(), video = $('overlay-video');
  video.readyState = 4; video.currentTime = 9.25; video.paused = false;
  $('overlay-version').value = 'aligned'; $('overlay-version').onchange();
  const staleCallback = video.onloadedmetadata;
  $('overlay-camera').value = 'cam0_left'; $('overlay-camera').onchange();
  video.currentTime = 0; staleCallback();
  assert.equal(video.currentTime, 0); assert.equal(video.paused, true);
  video.readyState = 1; video.onloadedmetadata();
  video.readyState = 4; video.oncanplay();
  assert.equal(video.currentTime, 9.25); assert.equal(video.paused, false);
});

test('failed loads clear download link and display failure instead of success', () => {
  const {$} = fixture();
  $('overlay-video').onerror();
  assert.match($('overlay-error').textContent, /加载失败/);
  assert.equal($('overlay-download').href, undefined);
});

test('switching during a buffered seek preserves the pending play intent', () => {
  const {$} = fixture(), video = $('overlay-video');
  video.readyState = 4; video.currentTime = 9; video.paused = false;
  $('overlay-version').value = 'aligned'; $('overlay-version').onchange();
  video.readyState = 1; video.onloadedmetadata();
  $('overlay-camera').value = 'cam0_left'; $('overlay-camera').onchange();
  video.readyState = 1; video.onloadedmetadata();
  assert.equal(video.currentTime, 9);
  video.readyState = 4; video.seeking = true; video.oncanplay();
  assert.equal(video.paused, true);
  video.seeking = false; video.onseeked();
  assert.equal(video.paused, false);
});

test('an aborted old play request cannot report a failure on the new video', async () => {
  const {$} = fixture(), video = $('overlay-video');
  let rejectPlay;
  video.play = () => new Promise((resolve, reject) => { rejectPlay = reject; });
  video.readyState = 4; video.currentTime = 6; video.paused = false;
  $('overlay-version').value = 'aligned'; $('overlay-version').onchange();
  video.readyState = 1; video.onloadedmetadata();
  video.readyState = 4; video.oncanplay();
  $('overlay-camera').value = 'cam0_left'; $('overlay-camera').onchange();
  rejectPlay(new Error('play aborted by new source'));
  await Promise.resolve();
  assert.equal($('overlay-error').textContent, '');
});

test('missing pixel metrics display an em dash instead of zero', () => {
  const {run} = fixture();
  assert.equal(run('overlayMetrics({count:2,median:null,p95:null})'), '2 个点；中位 — px / P95 — px');
  assert.equal(run('overlayMetrics({count:0,median:null,p95:null})'), '无验证观测');
});

test('partial results show candidate status, separate evaluated and failed frames', () => {
  const {$, run} = fixture();
  run(`D.overlays.variants.aligned.alignment = {
    status: 'partial', approved_for_default: false, by_camera: {
      '0': {status: 'failed', validation_frames: [46,120], validated_frames: [], failed_frames: [46,120],frame_roles:{46:'development',120:'validation'}}
    }
  }`);
  $('overlay-version').value = 'aligned'; $('overlay-version').onchange();
  assert.equal($('overlay-state').textContent, '候选校正：部分改善，失败与未验证范围见下方');
  const text = $('overlay-validation').children[0].children.map(node => node.textContent);
  assert.ok(text.includes('验证未通过'));
  assert.ok(text.includes('评估视频帧：46、120'));
  assert.ok(text.includes('通过视频帧：无'));
  assert.ok(text.includes('未通过视频帧：46、120'));
  assert.ok(text.includes('样本角色：46 · development；120 · validation'));
});
