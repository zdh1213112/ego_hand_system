const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const template = fs.readFileSync(path.join(__dirname, '../scripts/templates/nokov_mano.html'), 'utf8');
const formatSource = template.match(/function residualText\(value\)\{[^\n]+\}/)[0];
const residualText = vm.runInNewContext(`(${formatSource})`);

test('readout converts metres to mm and preserves real zero', () => {
  assert.equal(residualText(.00123456), '1.23 mm');
  assert.equal(residualText(0), '0.00 mm');
});

test('readout displays an em dash for every missing or non-finite value', () => {
  for (const value of [null, undefined, NaN, Infinity, -Infinity]) {
    assert.equal(residualText(value), '—');
  }
});

test('residual curves leave a gap at missing frames instead of drawing zero', () => {
  const source = template.slice(template.indexOf('function drawResidual(){'), template.indexOf('\nfunction seek('));
  const paths = [];
  let current;
  const canvas = {
    clearRect() {}, fillText() {},
    beginPath() { current = []; },
    moveTo(x, y) { current.push(['move', x, y]); },
    lineTo(x, y) { current.push(['line', x, y]); },
    stroke() { paths.push(current); },
  };
  vm.runInNewContext(`${source};drawResidual()`, {
    context: () => ({x: canvas, w: 300, h: 110}), index: 0,
    D: {p95: [.001, null, .003], median: [.001, null, .003], time: [0, 1, 2]},
  });
  for (const curve of paths.slice(1, 3)) {
    assert.deepEqual(curve, [['move', 46, 78.5], ['move', 285, 65.5]]);
  }
});
