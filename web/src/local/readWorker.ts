// Web Worker: read + parse one Arina .h5 file off the main thread. Reading File bytes
// (File.arrayBuffer / handle.getFile().arrayBuffer) is single-threaded-bound on the main
// thread to ~2.3 GB/s, but a pool of these workers reads in parallel at ~10 GB/s (the file
// delivery + ArrayBuffer allocation parallelize across threads). Each worker also runs the
// fast jsfive B-tree parse, then transfers the file buffer + blockMeta back (zero-copy).
import { readH5Volume } from "../../../js/.generated/engine/io/backends/webgpu/h5reader";

interface ReadRequest { id: number; name: string; file?: File; handle?: FileSystemFileHandle; }

self.onmessage = async (e: MessageEvent<ReadRequest>) => {
  const { id, name, file, handle } = e.data;
  try {
    const f = file ?? (handle ? await handle.getFile() : null);
    if (!f) { (self as unknown as Worker).postMessage({ id, error: "no file source" }); return; }
    const buffer = await f.arrayBuffer();                 // the parallel-read win happens here
    const vol = readH5Volume(buffer, name);
    const spec = vol.chunks[0];
    (self as unknown as Worker).postMessage(
      { id, name, nFrames: spec.nFrames, nBlocksPerFrame: spec.nBlocksPerFrame, blockElems: spec.blockElems,
        detSize: spec.detSize, srcDtype: vol.srcDtype, blockMeta: spec.blockMeta, buffer },
      [buffer, spec.blockMeta.buffer],
    );
  } catch (e) {
    (self as unknown as Worker).postMessage({
      id,
      error: e instanceof Error ? e.message : String(e),
    });
  }
};
