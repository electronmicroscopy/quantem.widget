import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// Standalone WebGPU 4D-STEM browser. No backend: the GUI is the quantem.live
// Browse page, the data layer reads a locally-picked folder of Arina .h5 files
// and decodes them on the GPU via the shared js/engine WGSL engine.
const here = dirname(fileURLToPath(import.meta.url));
const widgetRoot = resolve(here, "..");

// `vite build --mode offline` inlines EVERYTHING into one self-contained index.html so the app
// opens by double-click over file:// (no server) - the real "ship a folder of one HTML + .h5
// files, click the HTML, pick the folder" path. Data still comes from the local folder picker
// at runtime; only the app code is inlined. Normal `vite build` keeps the fast multi-file dev/
// served output. (jsfive is dynamically imported -> inline it too.)
const offline = process.env.OFFLINE_HTML === "1";

export default defineConfig({
  plugins: [react(), ...(offline ? [viteSingleFile()] : [])],
  define: { __QWIDGET_OFFLINE_HTML__: JSON.stringify(offline) },
  server: {
    // allow importing the engine symlink target (widget/js/engine) + jsfive from
    // the parent node_modules, both outside the web/ root.
    fs: { allow: [here, widgetRoot] },
  },
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
});
