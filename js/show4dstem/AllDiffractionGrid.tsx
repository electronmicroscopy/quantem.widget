import * as React from "react";
import { GPUColormapEngine } from "../colormaps";
import { extractBytes } from "../format";
import { getGPUDevice, isSoftwareGPUAdapter } from "../.generated/engine/device/webgpu";

type DiffractionSlots = {
  engine: GPUColormapEngine;
  slots: Map<number, number>;
  ranges: Map<number, {min: number; max: number}>;
};

/** Upload native patterns once; the common grid owns every interaction. */
export function AllDiffractionGrid({bytes, indices, rows, cols, selectionLabel, renderGrid}: {
  bytes: DataView | undefined; indices: number[]; rows: number; cols: number;
  selectionLabel: string; renderGrid: (data: DiffractionSlots) => React.ReactNode;
}) {
  const [engine, setEngine] = React.useState<GPUColormapEngine | null>(null);
  const [data, setData] = React.useState<DiffractionSlots | null>(null);
  const [error, setError] = React.useState("");
  React.useEffect(() => {
    let disposed = false;
    let renderer: GPUColormapEngine | null = null;
    void (async () => {
      try {
        const device = await getGPUDevice();
        if (!device || isSoftwareGPUAdapter()) throw new Error("Hardware WebGPU is required for the all-pattern display.");
        renderer = new GPUColormapEngine(device);
        if (disposed) renderer.destroy(); else setEngine(renderer);
      } catch (cause) { if (!disposed) setError(String(cause)); }
    })();
    return () => { disposed = true; renderer?.destroy(); };
  }, []);
  React.useEffect(() => {
    if (!engine || !bytes || !indices.length) return;
    const payload = extractBytes(bytes);
    const length = rows * cols;
    if (payload.byteLength !== indices.length * length * 4) return;
    let cancelled = false;
    const values = new Float32Array(payload.buffer, payload.byteOffset, payload.byteLength / 4);
    const slots = new Map(indices.map((frame, index) => [frame, index]));
    indices.forEach((_, index) => engine.uploadData(index, values.subarray(index * length, (index + 1) * length), cols, rows));
    void (async () => {
      try {
        const ranges = await engine.computeRangeBatch([...slots.values()]);
        if (cancelled) return;
        const range = {min: Math.min(...ranges.map(r => r.min)), max: Math.max(...ranges.map(r => r.max))};
        setData({engine, slots, ranges: new Map(indices.map(frame => [frame, range]))});
      } catch (cause) { if (!cancelled) setError(String(cause)); }
    })();
    return () => { cancelled = true; };
  }, [engine, bytes, indices, rows, cols]);
  return <section aria-label="All diffraction patterns" style={{width:"100%"}}>
    <div style={{fontSize:11, marginBottom:4}}>Diffraction patterns · {selectionLabel} · shared contrast</div>
    {error && <div role="alert">{error}</div>}
    {!indices.length && <div>No scan positions selected.</div>}
    {indices.length > 0 && data && renderGrid(data)}
  </section>;
}
