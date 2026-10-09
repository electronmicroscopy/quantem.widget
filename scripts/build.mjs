// Bundle each widget as a self-contained ESM file.
// anywidget loads bundles via Blob URL; relative imports break in that context.
// esbuild flattens everything into one file per widget.
//
//   npm run build                     all bundles
//   npm run build -- show2d show3d    only the named bundles
//
// show4dstem and showptycho bundle WebGPU science kernels synced from
// quantem.gpu (QUANTEM_GPU_SRC or an installed quantem.gpu). Every other bundle
// is widget source only and builds without quantem.gpu.

import { build, context } from "esbuild";
import { rmSync, copyFileSync, mkdirSync, existsSync } from "fs";
import { syncGpuWebgpuSources } from "./sync-gpu-webgpu.mjs";

const watch = process.argv.includes("--watch");
const widgets = [
  { name: "plot2d" },
  { name: "show1d" },
  { name: "show2d" },
  { name: "show3d" },
  { name: "show3dslices" },
  { name: "show4dstem", science: true },
  { name: "showdiffraction" },
  { name: "showptycho", science: true },
  { name: "chooselattice" },
  { name: "showcif" },
];
const requested = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
const unknown = requested.filter((name) => !widgets.some((widget) => widget.name === name));
if (unknown.length) throw new Error(`unknown bundle(s): ${unknown.join(", ")}`);
const selected = requested.length ? widgets.filter((widget) => requested.includes(widget.name)) : widgets;

let syncError = null;
if (selected.some((widget) => widget.science)) {
  try {
    syncGpuWebgpuSources();
  } catch (error) {
    syncError = error;
  }
}

// A full build replaces the whole static tree so retired bundles cannot linger.
if (!requested.length) rmSync("src/quantem/widget/static", { recursive: true, force: true });
mkdirSync("src/quantem/widget/static", { recursive: true });

const baseOpts = {
  bundle: true,
  format: "esm",
  jsx: "automatic",
  target: "es2022",
  define: { "process.env.NODE_ENV": '"production"' },
  loader: { ".css": "text" },
  minify: true,
  sourcemap: false,
  legalComments: "none",
};

for (const widget of selected) {
  if (widget.science && syncError) continue;
  const opts = {
    ...baseOpts,
    entryPoints: [`js/${widget.name}/index.tsx`],
    outfile: `src/quantem/widget/static/${widget.name}.js`,
  };
  if (watch) {
    const watcher = await context(opts);
    await watcher.watch();
    console.log(`watching ${widget.name}...`);
  } else {
    const start = Date.now();
    await build(opts);
    console.log(`built ${widget.name}.js (${Date.now() - start}ms)`);
  }
  // Copy CSS sibling if present (anywidget _css trait reads from static/).
  for (const cssName of [`${widget.name}.css`, "styles.css"]) {
    const cssSrc = `js/${widget.name}/${cssName}`;
    if (existsSync(cssSrc)) {
      copyFileSync(cssSrc, `src/quantem/widget/static/${widget.name}.css`);
      break;
    }
  }
}

if (syncError) {
  const missing = selected.filter((widget) => widget.science).map((widget) => widget.name);
  console.error(`\nNOT built: ${missing.join(", ")}\n${syncError.message}`);
  process.exit(1);
}
if (!watch) console.log("done.");
