// @ts-expect-error Vitest raw source import is not part of the widget bundle.
import source from "./index.tsx?raw";
import { expect, test } from "vitest";

// A compare pass awaits its range readback after adopting its buffers. If a
// newer pass starts meanwhile, it adopts its own buffers and the engine retires
// these; recording them as the next drag's delta base would write destroyed
// buffers. Only the latest pass records them; a superseded pass still sets its
// ranges and paints, because the newer pass may be waiting on a volume load.
test("a compare pass superseded during its range readback paints but does not record the delta base", () => {
  const body = source.slice(source.indexOf("const updateRoiCompareGpuSlots"), source.indexOf("if (await updateRoiCompareGpuSlots())"));
  const readback = body.indexOf("await engine.computeRangeBatch(batchSlots)");
  const ranges = body.indexOf("compareGpuRangesRef.current.set(frame, range)", readback);
  const guard = body.indexOf("if (generation === compareViGen) {", readback);
  const record = body.indexOf("compareIncrementalRef.current = {", readback);
  expect(readback).toBeGreaterThan(0);
  expect(ranges).toBeGreaterThan(readback);
  expect(guard).toBeGreaterThan(ranges);
  expect(record).toBeGreaterThan(guard);
  expect(body.slice(readback, guard)).not.toContain("return false");
});
