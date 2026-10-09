// Copy the WebGPU science kernels that Show4DSTEM and ShowPtycho bundle
// (detector reductions, HDF5/bslz4 browser IO, SSB, QEM tables) from
// quantem.gpu into js/.generated/engine. Display kernels (colormaps, display
// FFT, statistics, display geometry) are widget source under js/display and
// are not synced. The science kernels share the widget's WebGPU device: the
// generated device/webgpu.ts re-exports js/display/device.ts so detector
// buffers borrowed by the colormap engine live on the same device.

import { spawnSync } from "child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

// quantem.gpu domains the widget bundles take as-is. Everything else listed in
// quantem.gpu's webgpu/sources.json (display, device, geometry, parity) is
// either widget-owned or unused by the widget.
export const SCIENCE_DOMAINS = ["detector/", "dpc/", "formats/", "io/", "ssb/"];

export function syncGpuWebgpuSources({ targetDir = "js/.generated/engine" } = {}) {
  const outputDir = path.isAbsolute(targetDir) ? targetDir : path.join(repoRoot, targetDir);
  const python = process.env.PYTHON || "python";
  const code = `
import json
import os
from pathlib import Path

source_root = os.environ.get("QUANTEM_GPU_SRC")
if source_root:
    root = Path(source_root) / "quantem" / "gpu"
else:
    from importlib.resources import files

    root = files("quantem.gpu")
names = json.loads(root.joinpath("webgpu", "sources.json").read_text(encoding="utf-8"))
print(json.dumps({
    "root": str(root),
    "sources": {
        name: root.joinpath(*name.split("/")).read_text(encoding="utf-8")
        for name in names
        if name.startswith(tuple(${JSON.stringify(SCIENCE_DOMAINS)}))
    },
}))
`;
  const runExport = (env = process.env) => spawnSync(python, ["-c", code], {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
    env,
  });

  let result = runExport();
  if (result.status !== 0 && !process.env.QUANTEM_GPU_SRC) {
    const home = process.env.HOME || "";
    const srcDir = [
      path.resolve(repoRoot, "../quantem.gpu/src"),
      path.resolve(repoRoot, "../../quantem.gpu/src"),
      home ? path.resolve(home, "repos/quantem.gpu/src") : "",
    ].find((dir) => dir && existsSync(dir));
    if (srcDir) result = runExport({ ...process.env, QUANTEM_GPU_SRC: srcDir });
  }
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim().split("\n").pop();
    throw new Error(
      "show4dstem and showptycho bundle WebGPU science kernels from quantem.gpu, " +
      "and no quantem.gpu source was found. Set QUANTEM_GPU_SRC to the quantem.gpu " +
      "src directory (QUANTEM_GPU_SRC=/path/to/quantem.gpu/src npm run build), or " +
      "install quantem.gpu in the Python named by PYTHON. Build only the other " +
      `bundles with: npm run build -- show2d show3d ... (${detail})`
    );
  }

  const { root, sources } = JSON.parse(result.stdout);
  // Recreate the tree so renamed or deleted quantem.gpu files cannot remain importable.
  rmSync(outputDir, { recursive: true, force: true });
  for (const [name, text] of Object.entries(sources)) {
    const dest = path.join(outputDir, name);
    mkdirSync(path.dirname(dest), { recursive: true });
    writeFileSync(dest, text, "utf8");
  }
  mkdirSync(path.join(outputDir, "device"), { recursive: true });
  writeFileSync(
    path.join(outputDir, "device", "webgpu.ts"),
    "// Generated: science kernels use the widget's one WebGPU device.\n" +
    'export * from "../../../display/device";\n',
    "utf8",
  );
  console.log(`synced ${Object.keys(sources).length} quantem.gpu science kernel files from ${root} -> ${targetDir}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  syncGpuWebgpuSources();
}
