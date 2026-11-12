import * as React from "react";
import { getGPUInfo, getRenderPath, onGPULost, webGPULostReason, type RenderPath } from "../display/device";

/** Which path draws the widget's pixels, once the shared WebGPU device request settles. */
function useRenderPath(): RenderPath | null {
  const [path, setPath] = React.useState<RenderPath | null>(null);
  React.useEffect(() => {
    let disposed = false;
    const update = () => { void getRenderPath().then(next => { if (!disposed) setPath(next); }); };
    update();
    const stopWatching = onGPULost(update);  // a lost device moves the widgets to JavaScript
    return () => { disposed = true; stopWatching(); };
  }, []);
  return path;
}

/**
 * Status chip naming the render path: "WebGPU" on a hardware adapter, "CPU"
 * when the Canvas2D/JS reference draws the same pixels. Styled like the other
 * muted title-row status chips. A widget that knows which path drew the frame
 * on screen passes it as `drawnBy`, so the chip never claims WebGPU for pixels
 * the CPU drew.
 */
export function RenderPathBadge({ colors, drawnBy = null }: {
  colors: { controlBg: string; textMuted: string; border: string };
  /** The path that drew the frame on screen, when the widget knows it; it overrides the device's path. */
  drawnBy?: RenderPath | null;
}) {
  const devicePath = useRenderPath();
  if (!devicePath) return null;
  const path = drawnBy ?? devicePath;
  const lost = webGPULostReason();
  const title = path === "WebGPU"
    ? `Display math runs on WebGPU (${getGPUInfo()}).`
    : devicePath === "WebGPU"
      ? "This frame was drawn in JavaScript on the CPU although the browser has a WebGPU adapter; the status line says why."
      : lost !== null
        ? `The WebGPU device was lost (${lost}): display math runs in JavaScript with the same pixel values until the page reloads.`
        : "No hardware WebGPU in this browser: display math runs in JavaScript with the same pixel values.";
  return (
    <span
      data-render-path={path}
      title={title}
      style={{
        marginLeft: 4, padding: "0 4px", fontSize: 9, fontWeight: 500, borderRadius: 3,
        backgroundColor: colors.controlBg, color: colors.textMuted, border: `1px solid ${colors.border}`,
        whiteSpace: "nowrap", verticalAlign: "middle",
      }}
    >
      {path}
    </span>
  );
}
