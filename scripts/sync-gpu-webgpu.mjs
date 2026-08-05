// Sync canonical WebGPU browser-compute sources from quantem.gpu into the
// widget frontend tree before bundling. Browsers need TypeScript/WGSL bundled
// into the anywidget JS artifact, but quantem.gpu owns the reusable kernel
// source.

import { spawnSync } from "child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

export function syncGpuWebgpuSources({ targetDir = "js/.generated/engine" } = {}) {
  const outputDir = path.isAbsolute(targetDir) ? targetDir : path.join(repoRoot, targetDir);
  const python = process.env.PYTHON || "python";
  const code = `
import json
import os
from pathlib import Path

names = (
    "device/webgpu.ts",
    "io/backends/webgpu/bslz4.ts",
    "io/backends/webgpu/h5reader.ts",
    "io/backends/webgpu/local-h5.ts",
    "detector/compute/webgpu/backend.ts",
    "dpc/compute/webgpu/fft.ts",
    "dpc/compute/webgpu/kernels.ts",
    "ssb/compute/webgpu/backend.ts",
    "ssb/compute/webgpu/optimizer.ts",
    "ssb/compute/webgpu/protocol.ts",
    "ssb/compute/webgpu/kernels/common.ts",
    "ssb/compute/webgpu/kernels/fft128.ts",
    "ssb/compute/webgpu/kernels/fft256.ts",
    "ssb/compute/webgpu/kernels/fft512.ts",
    "ssb/compute/webgpu/kernels/fft1024.ts",
    "ssb/compute/webgpu/kernels/index.ts",
)
source_root = os.environ.get("QUANTEM_GPU_SRC")
if source_root:
    root = Path(source_root) / "quantem" / "gpu"
else:
    from importlib.resources import files

    root = files("quantem.gpu")
print(json.dumps({
    name: root.joinpath(*name.split("/")).read_text(encoding="utf-8")
    for name in names
}))
`;
  const runExport = (env = process.env) => spawnSync(python, ["-c", code], {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
    env,
  });

  let result = runExport();
  if (result.status !== 0) {
    const home = process.env.HOME || "";
    const srcDirs = [
      process.env.QUANTEM_GPU_SRC,
      path.resolve(repoRoot, "../quantem.gpu/src"),
      path.resolve(repoRoot, "../../quantem.gpu/src"),
      home ? path.resolve(home, "repos/quantem.gpu/src") : "",
      home ? path.resolve(home, "quantem.gpu/src") : "",
    ].filter((srcDir) => srcDir && existsSync(srcDir));
    if (srcDirs.length) {
      const pythonPath = [
        ...srcDirs,
        process.env.PYTHONPATH || "",
      ].filter(Boolean).join(path.delimiter);
      result = runExport({
        ...process.env,
        PYTHONPATH: pythonPath,
        QUANTEM_GPU_SRC: srcDirs[0],
      });
    }
  }
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    throw new Error(
      "Unable to sync WebGPU sources from quantem.gpu. Install quantem.gpu in " +
      "the active Python environment, set PYTHON explicitly, or set " +
      `QUANTEM_GPU_SRC to the quantem.gpu/src directory. ${detail}`
    );
  }

  const sources = JSON.parse(result.stdout);
  // This tree is generated exclusively from the explicit GPU manifest above.
  // Recreate it so renamed or deleted domain files cannot remain importable.
  rmSync(outputDir, { recursive: true, force: true });
  mkdirSync(outputDir, { recursive: true });
  let changed = 0;
  let unchanged = 0;
  for (const [name, text] of Object.entries(sources)) {
    const dest = path.join(outputDir, name);
    mkdirSync(path.dirname(dest), { recursive: true });
    const current = existsSync(dest) ? readFileSync(dest, "utf8") : null;
    if (current === text) {
      unchanged += 1;
      continue;
    }
    writeFileSync(dest, text, "utf8");
    changed += 1;
  }
  console.log(
    `synced quantem.gpu WebGPU domains -> ${targetDir} (${changed} updated, ${unchanged} unchanged)`
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  syncGpuWebgpuSources();
}
