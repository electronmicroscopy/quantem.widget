import { describe, expect, it } from "vitest";
import { residentDisplayValues, residentLogTable, residentRawValues, residentScalarType } from "./batchValues";

describe("resident exact-count batches", () => {
  it("normalizes a bounded view without mutating or truncating raw counts", () => {
    const storage = new Uint16Array([111, 0, 1, 255, 50615, 65535, 222]);
    const bytes = new DataView(storage.buffer, 2, 10);
    const before = storage.slice();
    const display = residentDisplayValues(bytes, { dtype: "<u2", mask_area: 23017 });
    expect(Array.from(display)).toEqual([0, 1, 255, 50615, 65535].map(v => Math.fround(v / 23017)));
    expect(storage).toEqual(before);
  });
  it("uses the exact normalization table carried after the count section", () => {
    const storage = new ArrayBuffer(4 + 8 + 65536 * 4);
    const counts = new Uint16Array(storage, 4, 4);
    counts.set([0, 1, 50615, 65535]);
    const table = new Float32Array(storage, 12, 65536);
    for (let i = 0; i < table.length; i++) table[i] = i / 14472;
    const display = residentDisplayValues(new DataView(storage, 4),
      {dtype: "<u2", mask_area: 14472, normalization_offset: 8});
    expect(Array.from(display)).toEqual(Array.from(counts, count => table[count]));
    expect(display.length).toBe(4);
  });
  it("retains float32 fallback values and their bounded view", () => {
    const storage = new Float32Array([111, -2.5, 0, 1.25, 222]);
    const display = residentDisplayValues(new DataView(storage.buffer, 4, 12));
    expect(Array.from(display)).toEqual([-2.5, 0, 1.25]);
    expect(display.buffer).toBe(storage.buffer);
  });
  it("rejects invalid compact-batch normalization instead of showing corrupted values", () => {
    const bytes = new DataView(new Uint16Array([1]).buffer);
    for (const mask_area of [undefined, -1, NaN, Infinity]) {
      expect(() => residentDisplayValues(bytes, {dtype: "<u2", mask_area})).toThrow("nonnegative detector mask area");
    }
  });
  it("retains tiny positive log values and matches the CPU display across the full count domain", () => {
    const storage=new ArrayBuffer(8+65536*4);
    const values=new Float32Array(storage,8,65536);
    for(let i=0;i<values.length;i++)values[i]=i/36864;
    const original=values.slice();
    const actual=residentLogTable(new DataView(storage,4),4);
    const expected=Float32Array.from(values,value=>Math.log1p(value));
    expect(actual).toEqual(expected);
    expect(actual[1]).toBeGreaterThan(0);
    expect(values).toEqual(original);
  });
});


describe("resident uint32 scientific transport", () => {
  it("retains every raw bit above float32 precision and exposes the exact bounded export view", () => {
    const storage = new Uint32Array([123, 0, 65535, 65536, 16777217, 4294967295, 456]);
    const bytes = new DataView(storage.buffer, 4, 20);
    const info = {dtype:"<u4",shape:[1,1,5],bytes:20,mask_area:3};
    const raw = residentRawValues(bytes, info);
    expect(raw).toBeInstanceOf(Uint32Array);
    expect(raw.buffer).toBe(storage.buffer);
    expect(raw.byteOffset).toBe(4);
    expect(Array.from(raw)).toEqual([0,65535,65536,16777217,4294967295]);
    const display = residentDisplayValues(bytes, info);
    expect(display.buffer).not.toBe(raw.buffer);
    expect(Array.from(display)).toEqual(Array.from(raw, count => Math.fround(count/3)));
    const exported = new Uint8Array(raw.buffer,raw.byteOffset,raw.byteLength).slice();
    expect(new Uint32Array(exported.buffer)).toEqual(new Uint32Array([0,65535,65536,16777217,4294967295]));
    expect(storage[0]).toBe(123);expect(storage[6]).toBe(456);
  });
  it.each(["<u2","<u4"])("empty masks derive zero display values without mutating %s counts", dtype => {
    const raw = dtype === "<u2" ? new Uint16Array([0,65535]) : new Uint32Array([16777217,4294967295]);
    const before = raw.slice();
    expect(residentDisplayValues(new DataView(raw.buffer),{dtype,mask_area:0})).toEqual(new Float32Array(2));
    expect(raw).toEqual(before);
  });
  it.each(["<f8",">u4","u4","|u4","<i4",""])("rejects explicit unknown scalar code %s", dtype => {
    expect(() => residentScalarType({dtype})).toThrow("Unsupported resident scalar");
  });
  it("rejects a null scalar code instead of treating it as legacy float32", () => {
    expect(() => residentScalarType({dtype:null} as unknown as {dtype:string})).toThrow("Unsupported resident scalar");
  });
  it("rejects mismatched shapes, truncated buffers, invalid alignment, and foreign tables", () => {
    const bytes = new DataView(new ArrayBuffer(20));
    for (const info of [
      {dtype:"<u4",shape:[1,2,3]}, {dtype:"<u4",shape:[-1,1,5]},
      {dtype:"<u4",bytes:24}, {dtype:"<u4",normalization_offset:16},
      {dtype:"<u4",normalization_offset:24}, {dtype:"<u2",normalization_offset:4},
    ]) expect(() => residentRawValues(bytes,info)).toThrow();
    expect(() => residentRawValues(new DataView(bytes.buffer,1,16),{dtype:"<u4"})).toThrow("alignment");
    expect(() => residentRawValues(new DataView(bytes.buffer,0,19),{dtype:"<u4"})).toThrow("bounds");
  });
});
