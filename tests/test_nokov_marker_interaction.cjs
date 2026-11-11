const test = require('node:test');
const assert = require('node:assert/strict');
const {pickMarker, segmentDistance, startGesture, moveGesture} = require('../scripts/templates/nokov_marker_interaction.js');

test('measured point, attached point and long residual segment select the same marker', () => {
  const measured = [[20, 20], [90, 90]], attached = [[70, 20], [110, 90]];
  assert.equal(pickMarker([21, 20], measured, attached), 0);
  assert.equal(pickMarker([70, 21], measured, attached), 0);
  assert.equal(pickMarker([45, 24], measured, attached), 0);
  assert.equal(pickMarker([90, 89], measured, attached), 1);
  assert.equal(pickMarker([160, 160], measured, attached), -1);
});

test('a missing measurement cannot create an imaginary residual', () => {
  assert.equal(segmentDistance([0, 0], null, [5, 5]), Infinity);
  assert.equal(pickMarker([5, 5], [null], [[5, 5]]), -1);
  assert.equal(pickMarker([5, 5], [null], [[5, 5]], {selected: 0}), 0);
});

test('hidden items are not clickable; the selected pair remains available', () => {
  const measured = [[20, 20]], attached = [[60, 20]];
  assert.equal(pickMarker([20, 20], measured, attached, {markers: false, residuals: false}), -1);
  assert.equal(pickMarker([60, 20], measured, attached, {markers: false, residuals: false, selected: 0}), 0);
  assert.equal(segmentDistance([22, 20], [20, 20], [20, 20]), 2);
});

test('tap jitter does not rotate, while dragging back to the start remains a drag', () => {
  let state = startGesture(10, 10, 3);
  let result = moveGesture(state, 13, 13);
  assert.equal(result.gesture.moved, false);
  assert.deepEqual(result.delta, [0, 0]);
  result = moveGesture(result.gesture, 20, 15);
  assert.equal(result.gesture.moved, true);
  assert.deepEqual(result.delta, [10, 5]);
  result = moveGesture(result.gesture, 10, 10);
  assert.equal(result.gesture.moved, true);
  assert.deepEqual(result.delta, [-10, -5]);
  assert.equal(result.gesture.pointerId, 3);
});
