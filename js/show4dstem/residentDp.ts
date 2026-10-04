import { GPUColormapEngine, COLORMAPS } from "../colormaps";
import { captureGpuCanvas } from "./captureGpuCanvas";

/** Optional resident-pattern capability; CPU-only sources retain frameAt(). */
export type ResidentPatternSource = {
  getDevice(): GPUDevice;
  frameAtBuffer(scan: number): { buffer: GPUBuffer; n: number };
};
export type DpDisplayOptions = {
  width: number; height: number; colormap: string; log: boolean;
  minPct: number; maxPct: number; min: number | null; max: number | null;
  zoom: number; panX: number; panY: number;
};

/** Own display resources, borrow scientific buffers, and hydrate only settled data. */
export class ResidentDpDisplay {
  private engine: GPUColormapEngine;
  private context: GPUCanvasContext;
  private slot = -1;
  private version = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private closed = false;
  private options: DpDisplayOptions;
  private pending = false;
  constructor(
    private canvas: HTMLCanvasElement,
    readonly device: GPUDevice,
    options: DpDisplayOptions,
    private onPending: (pending: boolean) => void,
    private onError: (error: unknown) => void,
  ) {
    this.engine = new GPUColormapEngine(device);
    this.options = options;
    const context = this.engine.configureCanvas(canvas, options.width, options.height);
    if (!context) { this.engine.destroy(); throw Error("Cannot create the DP canvas. Reload the viewer."); }
    this.context = context;
    context.configure({device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC});
  }
  private setPending(value: boolean) {
    if (value === this.pending) return;
    this.pending = value;
    this.onPending(value);
  }
  updateOptions(options: DpDisplayOptions) {
    this.options = options;
    if (this.slot >= 0 && !this.closed) this.draw();
  }
  show(sources: ResidentPatternSource[], scan: number, onSettled: (data: Float32Array) => void) {
    if (this.closed) throw Error("DP display is closed; reload the viewer.");
    if (!sources.length || sources.some(source => source.getDevice() !== this.device))
      throw Error("DP averaging requires loaded sources on the same GPU.");
    const {width, height} = this.options;
    const slots = sources.map((source, index) => {
      const frame = source.frameAtBuffer(scan);
      if (frame.n !== width * height) throw Error("DP dimensions do not match the resident source.");
      const slot = index + 1;
      this.engine.borrowBuffer(slot, frame.buffer, width, height);
      return slot;
    });
    this.slot = slots[0];
    if (slots.length > 1) {
      if (!this.engine.averageResidentSlotsInto(0, slots)) throw Error("Cannot average the selected DP buffers.");
      this.slot = 0;
    }
    const version = ++this.version;
    this.setPending(true);
    this.draw();
    // Metadata describes submitted content, not measured screen presentation.
    this.canvas.dataset.residentDp = JSON.stringify({scan, count: sources.length, version, submittedAt: performance.now()});
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = undefined;
      // A snapshot copy is submitted before awaiting mapping; later drags cannot
      // overwrite it. An older completion never publishes stats for a newer DP.
      void this.engine.readDataSlots([this.slot]).then(([data]) => {
        if (this.closed || version !== this.version) return;
        if (!data || data.some(value => !Number.isFinite(value) || value < 0)) throw Error("DP decoding failed; reload the data folder.");
        onSettled(data);
        this.setPending(false);
      }).catch(error => { if (!this.closed && version === this.version) this.onError(error); });
    }, 80);
  }
  private draw() {
    const o = this.options;
    this.engine.uploadLUT(o.colormap, COLORMAPS[o.colormap] || COLORMAPS.viridis);
    const transform = {zoom: o.zoom, panX: o.panX, panY: o.panY};
    if (o.min != null && o.max != null) {
      return this.engine.renderPanelSlotsDirectToCanvas([this.slot], {
        vmin: o.log ? Math.log1p(Math.max(0, o.min)) : o.min,
        vmax: o.log ? Math.log1p(Math.max(0, o.max)) : o.max,
      }, o.log, this.context, {width: o.width, height: o.height, panelCount: 1,
        cols: 1, rows: 1, gap: 0, bgRgb: 0, transforms: [transform], smooth: false});
    }
    return this.engine.renderSlotDirectWithGpuRangeToCanvas(this.slot, o.minPct, o.maxPct,
      o.log, this.context, {width: o.width, height: o.height, bgRgb: 0, transform, smooth: false});
  }
  capture() { return captureGpuCanvas(this.device, this.context, () => this.draw()); }
  invalidate() {
    this.version++;
    clearTimeout(this.timer);
    this.timer = undefined;
    this.setPending(false);
  }
  destroy() {
    if (this.closed) return;
    this.closed = true;
    this.invalidate();
    this.context.unconfigure();
    this.engine.destroy();
  }
}
