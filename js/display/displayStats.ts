// Per-frame display statistics (the percentile histogram behind auto contrast and the
// histogram widget's bins) off the input thread. The worker runs the same functions as
// the input-thread reference, so every window and bin is bit-identical.
import { computeHistogramFromBytes, percentileHistogram, type PercentileHistogram } from "./stats";

export type FrameStats = { histogram: PercentileHistogram; displayBins: number[] };

/** Both display statistics of one frame: percentileWindow reads any contrast window from `histogram`. */
export function frameStats(data: ArrayLike<number>): FrameStats {
  const histogram = percentileHistogram(data);
  // Both displays use the same finite range; avoid scanning the full frame twice.
  return { histogram, displayBins: computeHistogramFromBytes(data, 256, histogram.min, histogram.max) };
}

/** Worker source defining the reference statistics functions verbatim (they call nothing else). */
export function frameStatsWorkerFunctions(): string {
  return [percentileHistogram, computeHistogramFromBytes].map(fn => fn.toString()).join("\n");
}

/** Worker expression for frameStats(`array`), through the functions frameStatsWorkerFunctions defines. */
export function frameStatsWorkerCall(array: string): string {
  return `(() => { const histogram = ${percentileHistogram.name}(${array}); return {histogram, displayBins: ${computeHistogramFromBytes.name}(${array}, 256, histogram.min, histogram.max)}; })()`;
}

export function displayStatsWorkerSource(): string {
  return `${frameStatsWorkerFunctions()}
self.onmessage = ({data}) => {
  const stats = ${frameStatsWorkerCall("data.data")};
  self.postMessage({id: data.id, stats}, stats.histogram.bins ? [stats.histogram.bins.buffer] : []);
};`;
}

/** One worker thread for the frame statistics of a widget; the input thread only copies the frame. */
export class DisplayStatsWorker {
  private worker: Worker | null = null;
  private serial = 0;
  private pending = new Map<number, { resolve: (stats: FrameStats) => void; reject: (error: Error) => void }>();
  private disposed = false;

  constructor() {
    if (typeof Worker === "undefined") return;
    const url = URL.createObjectURL(new Blob([displayStatsWorkerSource()], { type: "text/javascript" }));
    try {
      this.worker = new Worker(url);
      this.worker.onmessage = ({ data }) => {
        const job = this.pending.get(data.id);
        this.pending.delete(data.id);
        job?.resolve(data.stats);
      };
      this.worker.onerror = event => this.fail(new Error(`display statistics worker failed: ${event.message}`));
    } catch {
      // Hosts that disallow blob workers compute the same statistics on the input thread.
      this.worker = null;
    } finally { URL.revokeObjectURL(url); }
  }

  /** The statistics of `data`, which stays the caller's: the worker receives a copy. */
  stats(data: Float32Array): Promise<FrameStats> {
    if (this.disposed) return Promise.reject(new Error("display statistics are closed"));
    if (!this.worker) return Promise.resolve(frameStats(data));
    const id = ++this.serial;
    const copy = data.slice();
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage({ id, data: copy }, [copy.buffer]);
    });
  }

  private fail(error: Error) {
    for (const job of this.pending.values()) job.reject(error);
    this.pending.clear();
    this.worker?.terminate();
    this.worker = null;
  }

  dispose() {
    this.disposed = true;
    this.fail(new Error("display statistics are closed"));
  }
}
