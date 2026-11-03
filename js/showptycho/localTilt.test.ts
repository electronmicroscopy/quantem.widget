import { expect, test } from 'vitest';
import { makeGrid } from './localTilt';

test('equal overlapping scan windows cover the chosen transition', () => {
  const windows=makeGrid([48,400,112,368],128,.75);
  expect(windows).toHaveLength(40);
  expect(windows[0].bounds).toEqual([48,176,112,240]);
  expect(windows[5].bounds).toEqual([80,208,112,240]);
  expect(windows[39].bounds).toEqual([272,400,240,368]);
});
