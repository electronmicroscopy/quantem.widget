// Copy the WebGPU science kernels that Show4DSTEM and ShowPtycho bundle
// (detector reductions, HDF5/bslz4 browser IO, SSB, QEM tables) from
// quantem.gpu into js/.generated/engine. Display kernels (colormaps, display
// FFT, statistics, display geometry) are widget source under js/display and
// are not synced. The science kernels share the widget's WebGPU device: the
// generated device/webgpu.ts re-exports js/display/device.ts so detector
// buffers borrowed by the colormap engine live on the same device, and
// js/display/borrowedImage.ts re-exports quantem.gpu's borrowed-image contract
// (detector/webgpu/borrowed-image.ts), so every bundle that draws through
// js/display needs this sync.
//
// gpu-ref.txt names the one quantem.gpu ref every CI, docs and release build
// checks out. The sync writes MANIFEST.json beside the kernels (the ref and
// commit they came from, a sha256 per file); the build ships it in the wheel
// as static/engine-manifest.json and quantem.widget.profile() prints it, so a
// bundle always says which quantem.gpu kernels it carries.

import { spawnSync } from "child_process";
import { createHash } from "crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

// quantem.gpu domains the widget bundles take as-is. Everything else listed in
// quantem.gpu's webgpu/sources.json (display, device, geometry, parity) is
// either widget-owned or unused by the widget.
export const SCIENCE_DOMAINS = ["detector/", "dpc/", "formats/", "io/", "ssb/"];
export const PINNED_GPU_REF = readFileSync(path.join(repoRoot, "gpu-ref.txt"), "utf8").trim();

/** quantem.gpu's src directory beside this checkout (``../quantem.gpu/src``), or null. */
export function siblingGpuSource(root = repoRoot) {
  const candidate = path.resolve(root, "..", "quantem.gpu", "src");
  return existsSync(path.join(candidate, "quantem", "gpu", "webgpu", "sources.json")) ? candidate : null;
}

export function syncGpuWebgpuSources({ targetDir = "js/.generated/engine" } = {}) {
  const outputDir = path.isAbsolute(targetDir) ? targetDir : path.join(repoRoot, targetDir);
  const python = process.env.PYTHON || "python";
  // Where the kernels come from: QUANTEM_GPU_SRC, else the quantem.gpu installed in PYTHON when PYTHON
  // is set, else a quantem.gpu checkout beside this one, else the quantem.gpu installed in "python".
  const env = { ...process.env };
  if (!env.QUANTEM_GPU_SRC && !env.PYTHON) {
    const sibling = siblingGpuSource();
    if (sibling) {
      env.QUANTEM_GPU_SRC = sibling;
      console.log(`using the quantem.gpu checkout beside this one, ${sibling}; set QUANTEM_GPU_SRC or PYTHON to choose another`);
    }
  }
  const code = `
import json
import os
from pathlib import Path

source_root = os.environ.get("QUANTEM_GPU_SRC")
installed = None
if source_root:
    root = Path(source_root) / "quantem" / "gpu"
else:
    from importlib.metadata import version
    from importlib.resources import files

    root = files("quantem.gpu")
    installed = version("quantem.gpu")
names = json.loads(root.joinpath("webgpu", "sources.json").read_text(encoding="utf-8"))
print(json.dumps({
    "root": str(root),
    "installed": installed,
    "sources": {
        name: root.joinpath(*name.split("/")).read_text(encoding="utf-8")
        for name in names
        if name.startswith(tuple(${JSON.stringify(SCIENCE_DOMAINS)}))
    },
}))
`;
  const result = spawnSync(python, ["-c", code], { encoding: "utf8", maxBuffer: 20 * 1024 * 1024, env });
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim().split("\n").pop();
    throw new Error(
      "The widget bundles take quantem.gpu's borrowed-image contract and WebGPU science " +
      "kernels, and no quantem.gpu source was found. Set QUANTEM_GPU_SRC to the quantem.gpu " +
      "src directory (QUANTEM_GPU_SRC=/path/to/quantem.gpu/src npm run build) of a checkout " +
      `at ${PINNED_GPU_REF} (gpu-ref.txt), clone quantem.gpu beside this repository (../quantem.gpu), ` +
      `or name a Python with quantem.gpu installed (PYTHON=/path/to/python npm run build). (${detail})`
    );
  }

  const { root, installed, sources } = JSON.parse(result.stdout);
  const generated = {
    "device/webgpu.ts":
      "// Generated: science kernels use the widget's one WebGPU device.\n" +
      'export * from "../../../display/device";\n',
  };
  // Recreate the tree so renamed or deleted quantem.gpu files cannot remain importable.
  rmSync(outputDir, { recursive: true, force: true });
  for (const [name, text] of Object.entries({ ...sources, ...generated })) {
    const dest = path.join(outputDir, name);
    mkdirSync(path.dirname(dest), { recursive: true });
    writeFileSync(dest, text, "utf8");
  }
  const git = gitCheckout(root);
  const manifest = {
    quantem_gpu: { ref: git.describe, git_sha: git.sha, version: installed, pinned_ref: PINNED_GPU_REF },
    files: sha256s(outputDir, Object.keys(sources)),
    generated: sha256s(outputDir, Object.keys(generated)),
  };
  writeFileSync(path.join(outputDir, "MANIFEST.json"), JSON.stringify(manifest, null, 1) + "\n", "utf8");
  const commit = git.sha ? ` at ${git.describe} (${git.sha.slice(0, 12)})` : installed ? ` (quantem.gpu ${installed})` : " (not a git checkout)";
  const pin = (git.describe ?? installed) === PINNED_GPU_REF ? "" : `; gpu-ref.txt pins ${PINNED_GPU_REF}`;
  console.log(`synced ${Object.keys(sources).length} quantem.gpu science kernel files from ${root}${commit}${pin} -> ${targetDir}`);
}

/** The tag description and commit of the quantem.gpu checkout holding ``root``, or nulls for an installed package. */
function gitCheckout(root) {
  const git = (...args) => spawnSync("git", ["-C", root, ...args], { encoding: "utf8" });
  // sources.json tracked by that repository: a quantem.gpu checkout, not an env that happens to sit in one
  if (git("ls-files", "--error-unmatch", "webgpu/sources.json").status !== 0) return { describe: null, sha: null };
  return {
    describe: git("describe", "--tags", "--always", "--dirty").stdout.trim(),
    sha: git("rev-parse", "HEAD").stdout.trim(),
  };
}

/** sha256 of each named file under ``dir``, by name in sorted order, so a reader can check the bundle's sources. */
function sha256s(dir, names) {
  return Object.fromEntries([...names].sort().map((name) => [
    name, createHash("sha256").update(readFileSync(path.join(dir, name))).digest("hex"),
  ]));
}

// Run directly (node scripts/sync-gpu-webgpu.mjs), not imported by build.mjs, quantem.live or
// `node -e` (no script path). Node resolves symlinks and URL-encodes the module URL, so compare
// like with like: a raw `file://${path}` misses paths with spaces and the sync silently did nothing.
if (process.argv[1] && import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href) {
  syncGpuWebgpuSources();
}
