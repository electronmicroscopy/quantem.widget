import { useEffect } from "react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { Routes, Route, Navigate } from "react-router-dom";
import { ShortcutRegistryProvider } from "./components/ShortcutRegistry";
import Browse from "./pages/browse/Browse";
import { colors, fontSizes } from "./theme";
import { scanFolder, type LocalFile } from "./local/store";

// Reuse the quantem.live dashboard theme verbatim (same MUI palette / type / custom
// nav700 breakpoint) so the Browse GUI renders exactly as it does in the live app.
const theme = createTheme({
  palette: { mode: "light", background: { default: colors.text.white, paper: colors.text.white } },
  typography: {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    fontSize: fontSizes.lg,
  },
  breakpoints: { values: { xs: 0, sm: 600, md: 900, lg: 1200, xl: 1536, nav700: 700 } },
});
declare module "@mui/material/styles" { interface BreakpointOverrides { nav700: true } }

// Notify the mounted Browse page that scanFolder ran (picker OR a dev hook) so it re-reads the
// dataset tree. Browse is always rendered now - no separate gate page.
function announceFolder() { window.dispatchEvent(new Event("quantem-folder-loaded")); }

export default function App() {
  // Dev / CDP verification hooks (the OS folder picker can't be driven headlessly).
  useEffect(() => {
    // Served-folder hook: __loadServed("http://host/", ["sub/a_master.h5", "sub/a_data_000001.h5", ...]).
    (window as unknown as { __loadServed: (base: string, paths: string[]) => Promise<void> }).__loadServed =
      async (base: string, paths: string[]) => {
        const files: LocalFile[] = paths.map((p) => ({
          name: p.split("/").pop()!, relPath: p, bytes: async () => (await fetch(base + p)).arrayBuffer(),
        }));
        await scanFolder(files);
        announceFolder();
      };
    // Real disk-path hook: reads File objects, exactly what the picker yields - no http.
    (window as unknown as { __loadFileList: (list: FileList) => Promise<void> }).__loadFileList =
      async (list: FileList) => {
        const files: LocalFile[] = Array.from(list).filter((f) => /\.h5$/i.test(f.name)).map((f) => ({
          name: f.name, relPath: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name, bytes: () => f.arrayBuffer(), source: f,
        }));
        await scanFolder(files);
        announceFolder();
      };
    // Full-pipeline timing hook: open the first scanned dataset and return its __perf record.
    (window as unknown as { __openFirst: () => Promise<unknown> }).__openFirst =
      async () => {
        const { getSessions, bfGeometry, datasetMeanDp } = await import("./local/store");
        const sess = getSessions();
        const s = sess.find((x) => x.files.length > 0);
        if (!s) return { error: "no sessions" };
        const f = s.files[0];
        const wall0 = performance.now();
        await bfGeometry(s.source, s.date, f.name);
        const wallMs = Math.round(performance.now() - wall0);
        const dp = await datasetMeanDp(s.source, s.date, f.name);
        let dpSum = 0; for (let i = 0; i < dp.length; i++) dpSum += dp[i];
        const perf = (window as unknown as { __perf?: unknown[] }).__perf || [];
        return { dataset: f.name, nFiles: s.files.length, wallMs, dpSum, dpLen: dp.length, perf: perf[perf.length - 1] };
      };
    // Per-dataset decode-timing probe: force-load one dataset, return wall + __perf record.
    (window as unknown as { __timeDataset: (s: string, d: string, n: string, detBin?: 1 | 2 | 4, dtype?: "uint8" | "uint16") => Promise<unknown> }).__timeDataset =
      async (source: string, date: string, name: string, detBin: 1 | 2 | 4 = 1, dtype: "uint8" | "uint16" = "uint8") => {
        const { bfGeometry, datasetMeanDp } = await import("./local/store");
        const t0 = performance.now();
        try { await bfGeometry(source, date, name, detBin, dtype); } catch (e) { return { error: String(e).slice(0, 120) }; }
        const wall = Math.round(performance.now() - t0);
        const dp = await datasetMeanDp(source, date, name, detBin, dtype);
        let dpSum = 0; for (let i = 0; i < dp.length; i++) dpSum += dp[i];
        const perf = (window as unknown as { __perf?: { key: string; loadDecodeMs: number }[] }).__perf || [];
        return { wall, detBin, dtype, dpLen: dp.length, dpSum, perf: perf[perf.length - 1] };
      };
    // CoM/DPC parity probe.
    (window as unknown as { __comStats: () => Promise<unknown> }).__comStats =
      async () => {
        const { getSessions, datasetComStats } = await import("./local/store");
        const s = getSessions().find((x) => x.files.length > 0);
        if (!s) return { error: "no sessions" };
        return await datasetComStats(s.source, s.date, s.files[0].name);
      };
    // WGSL compute parity hook: run maskedSum (BF/DF) + maskedCoM on a DETERMINISTIC
    // index-function fixture (value = (s*31 + d*17) % 251) so a numpy reference reproduces
    // it bit-for-bit. tests/test_wgsl_parity.py drives this over CDP on a real GPU and asserts
    // the WGSL output matches numpy - the automated gate that the widget's WebGPU BF/DF/CoM
    // compute stays correct (the Python torch path already has test_dpc_virtual_parity.py).
    (window as unknown as { __wgslParity: (sc: number, dr: number, dc: number) => Promise<unknown> }).__wgslParity =
      async (scanCount: number, detRows: number, detCols: number) => {
        const { DetectorCompute } = await import("../../js/.generated/engine/detector/compute/webgpu/backend");
        const detSize = detRows * detCols;
        const stack = new Uint8Array(scanCount * detSize);
        for (let s = 0; s < scanCount; s++) for (let d = 0; d < detSize; d++) stack[s * detSize + d] = (s * 31 + d * 17) % 251;
        const cy = (detRows - 1) / 2, cx = (detCols - 1) / 2, radius = Math.min(detRows, detCols) * 0.25;
        const mask = new Uint32Array(detSize);
        for (let row = 0; row < detRows; row++) for (let col = 0; col < detCols; col++) {
          const dy = row - cy, dx = col - cx; mask[row * detCols + col] = dy * dy + dx * dx <= radius * radius ? 1 : 0;
        }
        const compute = await DetectorCompute.create(stack, scanCount, detSize);
        if (!compute) return { error: "no WebGPU device" };
        const vi = await compute.maskedSum(mask);
        const com = await compute.maskedCoM(mask, detCols);
        const dpcY = await compute.maskedDpc(mask, detCols, "row");
        const dpcX = await compute.maskedDpc(mask, detCols, "col");
        return {
          virtual: Array.from(vi),
          comY: Array.from(com.comY),
          comX: Array.from(com.comX),
          dpcY: Array.from(dpcY),
          dpcX: Array.from(dpcX),
          scanCount,
          detRows,
          detCols,
        };
      };
    // bslz4 Strategy-D parity + kernel-time verify hook.
    (window as unknown as { __verifyD: (url: string) => Promise<unknown> }).__verifyD =
      async (url: string) => {
        const { readH5Volume } = await import("../../js/.generated/engine/io/backends/webgpu/h5reader");
        const { verifyFusedD } = await import("../../js/.generated/engine/io/backends/webgpu/bslz4");
        const buf = await (await fetch(url)).arrayBuffer();
        const vol = readH5Volume(buf, url.split("/").pop()!);
        if (vol.srcDtype === "uint8") return { error: "uint8 source has no fused-D path" };
        const r = await verifyFusedD(vol.chunks[0], vol.srcDtype);
        return { srcDtype: vol.srcDtype, nFrames: vol.nFrames, detSize: vol.detSize, ...r };
      };
  }, []);
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ShortcutRegistryProvider>
        <Routes>
          <Route path="/browse/*" element={<Browse />} />
          <Route path="*" element={<Navigate to="/browse" replace />} />
        </Routes>
      </ShortcutRegistryProvider>
    </ThemeProvider>
  );
}
