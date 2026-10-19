// Bundle each widget as a self-contained ESM file.
// anywidget loads bundles via Blob URL; relative imports break in that context.
// esbuild flattens everything into one file per widget.
//
//   npm run build                     all bundles
//   npm run build -- show2d show3d    only the named bundles
//
// Every build first syncs js/.generated/engine from quantem.gpu (QUANTEM_GPU_SRC
// or an installed quantem.gpu; CI checks out the ref in gpu-ref.txt): show4dstem
// and showptycho bundle its WebGPU science kernels, and every bundle that draws
// through js/display validates borrowed count images with its contract.

import { build, context } from "esbuild";
import { copyFileSync, mkdirSync, rmSync } from "fs";
import { syncGpuWebgpuSources } from "./sync-gpu-webgpu.mjs";

const watch = process.argv.includes("--watch");
const widgets = [
  "plot2d",
  "show1d",
  "show2d",
  "show3d",
  "show3dslices",
  "show4dstem",
  "showdiffraction",
  "showpdf",
  "showptycho",
  "chooselattice",
  "showcif",
];
const requested = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
const unknown = requested.filter((name) => !widgets.includes(name));
if (unknown.length) throw new Error(`unknown bundle(s): ${unknown.join(", ")}`);
const selected = requested.length ? widgets.filter((name) => requested.includes(name)) : widgets;

try {
  syncGpuWebgpuSources();
} catch (error) {
  console.error(`\nNOT built: ${selected.join(", ")}\n${error.message}`);
  process.exit(1);
}

// A full build replaces the whole static tree so retired bundles cannot linger.
if (!requested.length) rmSync("src/quantem/widget/static", { recursive: true, force: true });
mkdirSync("src/quantem/widget/static", { recursive: true });
// The wheel names the quantem.gpu sources its bundles carry; quantem.widget.profile() prints it.
copyFileSync("js/.generated/engine/MANIFEST.json", "src/quantem/widget/static/engine-manifest.json");

const baseOpts = {
  bundle: true,
  format: "esm",
  jsx: "automatic",
  target: "es2022",
  define: { "process.env.NODE_ENV": '"production"' },
  minify: true,
  sourcemap: false,
  legalComments: "none",
};

for (const name of selected) {
  const opts = {
    ...baseOpts,
    entryPoints: [`js/${name}/index.tsx`],
    outfile: `src/quantem/widget/static/${name}.js`,
  };
  if (watch) {
    const watcher = await context(opts);
    await watcher.watch();
    console.log(`watching ${name}...`);
  } else {
    const start = Date.now();
    await build(opts);
    console.log(`built ${name}.js (${Date.now() - start}ms)`);
  }
}

if (!watch) console.log("done.");
