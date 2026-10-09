import * as React from "react";
import { createRoot } from "react-dom/client";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import {
  COLORMAPS,
  createGPUColormapEngine,
  GPUColormapEngine,
  renderToOffscreen,
} from "../display/colormaps";
import { extractBytes, downloadBlob } from "../format";
import { detectTheme, getThemeColors, useTheme } from "../theme";
import { useHideStaticFallback } from "../staticFallback";

type Model = {
  get(key: string): any;
  set(key: string, value: unknown): void;
  save_changes(): void;
  on(event: string, callback: () => void): void;
  off(event: string, callback: () => void): void;
};

function render({ model, el }: { model: Model; el: HTMLElement }) {
  let themeColors = getThemeColors(detectTheme().theme);
  const host = document.createElement("div");
  host.dataset.quantemPlot2d = "true";
  host.style.cssText =
    "width:100%;min-width:260px;font:12px system-ui;";
  const preview = document.createElement("img");
  preview.alt = "Plot2D saved preview; rerun the cell for interaction";
  preview.style.cssText = "display:none;width:100%;height:auto";
  const canvas = document.createElement("canvas");
  canvas.dataset.quantemScientificOutput = "plot2d-map";
  canvas.style.cssText =
    "width:100%;display:block;touch-action:none;cursor:crosshair";
  const controls = document.createElement("div");
  controls.style.cssText =
    "display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:4px 8px";
  const reset = document.createElement("button");
  reset.textContent = "Reset View";
  const zoomIn = document.createElement("button");
  zoomIn.textContent = "Zoom In";
  const zoomOut = document.createElement("button");
  zoomOut.textContent = "Zoom Out";
  const zoomLabel = document.createElement("span");
  zoomLabel.setAttribute("aria-live", "polite");
  const save = document.createElement("button");
  save.textContent = "Save PNG";
  const status = document.createElement("span");
  status.textContent = "Preparing display…";
  const readout = document.createElement("div");
  readout.style.cssText =
    "height:22px;padding:3px 8px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
  const colorControl = document.createElement("span");
  colorControl.style.cssText = "display:inline-flex;gap:6px;align-items:center;flex-shrink:0";
  const colorRoot = createRoot(colorControl);
  function ColorControl() {
    const { colors } = useTheme();
    useHideStaticFallback(model, { current: host }, Boolean(bitmap));
    React.useLayoutEffect(() => {
      themeColors = colors;
      host.style.color = colors.text;
      host.style.background = colors.bg;
      status.style.color = zoomLabel.style.color = colors.textMuted;
      for (const button of [reset, zoomIn, zoomOut, save]) {
        Object.assign(button.style, {
          color: colors.text, background: colors.controlBg,
          border: `1px solid ${colors.border}`, borderRadius: "3px",
          font: "10px system-ui", padding: "3px 8px", cursor: "pointer",
        });
      }
      schedule();
    }, [colors]);
    return <>
      <span>Color</span>
      <Select size="small" value={model.get("cmap")}
        inputProps={{ "aria-label": "Color map" }}
        sx={{ fontSize: 10, color: colors.text, bgcolor: colors.controlBg,
          "& .MuiSelect-select": { py: 0.5 },
          "& .MuiSelect-icon": { color: colors.text },
          "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
          "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent } }}
        MenuProps={{ PaperProps: { sx: { maxHeight: 320, bgcolor: colors.controlBg,
          color: colors.text, border: `1px solid ${colors.border}` } } }}
        onChange={(event) => {
          model.set("cmap", event.target.value);
          model.save_changes();
        }}>
        {Object.keys(COLORMAPS).map(name => <MenuItem key={name} value={name}>{name}</MenuItem>)}
      </Select>
    </>;
  }
  function updateColorControl() {
    colorRoot.render(<ColorControl />);
  }
  controls.append(colorControl, zoomIn, zoomOut, save, reset, zoomLabel, status);
  host.append(preview, canvas, controls, readout);
  el.append(host);
  let engine: GPUColormapEngine | null = null;
  let bitmap: CanvasImageSource | null = null;
  let source = new Float64Array();
  let displayCmap = model.get("cmap");
  let displayMin = model.get("vmin");
  let displayMax = model.get("vmax");
  let pointer: { clientX: number; clientY: number } | null = null;
  let disposed = false,
    generation = 0,
    frame = 0;
  let queue = Promise.resolve();
  let bounds = (
    model.get("view_bounds").length
      ? model.get("view_bounds")
      : model.get("grid").bounds
  ).slice();
  let drag: { x: number; y: number; bounds: number[] } | null = null;
  let wheelTimer = 0;
  let readoutFrame = 0;
  function showReadout() {
    if (!readoutFrame) readoutFrame = requestAnimationFrame(() => {
      readoutFrame = 0;
      updateReadout();
    });
  }
  const fullBounds = () => model.get("grid").bounds as number[];
  function applyMaxWidth() {
    host.style.maxWidth = `${model.get("max_width") ?? 600}px`;
    schedule();
  }
  const geometry = () => ({
    width: Math.max(260, host.clientWidth),
    height: model.get("plot_height_px"),
    left: 65,
    top: 30,
    right: 18,
    bottom: 105,
  });
  let paintedBounds = bounds.slice();
  let paintedGeometry = geometry();
  const formatValue = (value: number) => Number(value.toPrecision(4)).toString();
  const formatTick = (value: number, span: number) =>
    formatValue(Math.abs(value) < span * 1e-12 ? 0 : value);
  function commit() {
    model.set("view_bounds", bounds.slice());
    model.save_changes();
  }
  function clampBounds(next: number[]) {
    const original = fullBounds();
    return [0, 2].flatMap((index) => {
      const span = Math.min(
        original[index + 1] - original[index],
        Math.max(
          (original[index + 1] - original[index]) / 100,
          next[index + 1] - next[index],
        ),
      );
      const low = Math.max(
        original[index],
        Math.min(original[index + 1] - span, next[index]),
      );
      return [low, low + span];
    });
  }
  function paint() {
    frame = 0;
    if (disposed || canvas.style.display === "none") return;
    const layout = geometry(),
      width = layout.width - layout.left - layout.right,
      height = layout.height - layout.top - layout.bottom;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(layout.width * ratio);
    canvas.height = Math.round(layout.height * ratio);
    canvas.style.height = `${layout.height}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(ratio, ratio);
    ctx.fillStyle = themeColors.bg;
    ctx.fillRect(0, 0, layout.width, layout.height);
    const grid = model.get("grid"),
      original = fullBounds();
    const zoom = (original[1] - original[0]) / (bounds[1] - bounds[0]);
    zoomLabel.textContent = `${formatValue(zoom)}× · ${zoom > 1.001 ? "Drag to pan" : "Zoom in to pan"}`;
    canvas.style.cursor = drag
      ? "grabbing"
      : zoom > 1.001
        ? "grab"
        : "crosshair";
    if (bitmap) {
      ctx.save();
      ctx.translate(layout.left, layout.top + height);
      ctx.scale(1, -1);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(
        bitmap,
        ((bounds[0] - original[0]) / (original[1] - original[0])) * grid.cols,
        ((bounds[2] - original[2]) / (original[3] - original[2])) * grid.rows,
        ((bounds[1] - bounds[0]) / (original[1] - original[0])) * grid.cols,
        ((bounds[3] - bounds[2]) / (original[3] - original[2])) * grid.rows,
        0,
        0,
        width,
        height,
      );
      ctx.restore();
    }
    ctx.strokeStyle = themeColors.border;
    ctx.strokeRect(layout.left, layout.top, width, height);
    ctx.font = "12px system-ui";
    ctx.fillStyle = themeColors.text;
    ctx.textAlign = "center";
    ctx.fillText(model.get("title"), layout.left + width / 2, 17);
    for (let tick = 0; tick <= 4; tick++) {
      const fraction = tick / 4;
      ctx.textAlign = "center";
      ctx.fillText(
        formatTick(
          bounds[0] + fraction * (bounds[1] - bounds[0]),
          bounds[1] - bounds[0],
        ),
        layout.left + fraction * width,
        layout.top + height + 17,
      );
      ctx.textAlign = "right";
      ctx.fillText(
        formatTick(
          bounds[2] + fraction * (bounds[3] - bounds[2]),
          bounds[3] - bounds[2],
        ),
        layout.left - 7,
        layout.top + height * (1 - fraction) + 4,
      );
    }
    ctx.textAlign = "center";
    ctx.fillText(model.get("x_label"), layout.left + width / 2, layout.top + height + 37);
    ctx.save();
    ctx.translate(15, layout.top + height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(model.get("y_label"), 0, 0);
    ctx.restore();
    const lut = COLORMAPS[displayCmap];
    if (lut)
      for (let i = 0; i < 256; i++) {
        ctx.fillStyle = `rgb(${lut[i * 3]},${lut[i * 3 + 1]},${lut[i * 3 + 2]})`;
        ctx.fillRect(
          layout.left + (i * width) / 256,
          layout.height - 52,
          width / 256 + 0.5,
          10,
        );
      }
    ctx.fillStyle = themeColors.text;
    ctx.textAlign = "left";
    ctx.fillText(formatValue(displayMin), layout.left, layout.height - 27);
    ctx.textAlign = "right";
    ctx.fillText(formatValue(displayMax), layout.left + width, layout.height - 27);
    ctx.textAlign = "center";
    ctx.fillText(model.get("colorbar_label"), layout.left + width / 2, layout.height - 8);
    const line = model.get("horizontal_line");
    if (line != null && line >= bounds[2] && line <= bounds[3]) {
      const lineY =
        layout.top + height * (1 - (line - bounds[2]) / (bounds[3] - bounds[2]));
      ctx.strokeStyle = "#e649a0";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(layout.left, lineY);
      ctx.lineTo(layout.left + width, lineY);
      ctx.stroke();
    }
    paintedBounds = bounds.slice();
    paintedGeometry = layout;
    updateReadout();
    canvas.setAttribute(
      "aria-label",
      `${model.get("title")}; ${model.get("x_label")}; ${model.get("y_label")}; ${model.get("colorbar_label")}`,
    );
    canvas.dataset.paintGeneration = String(
      Number(canvas.dataset.paintGeneration || 0) + 1,
    );
  }
  function schedule() {
    if (!frame) frame = requestAnimationFrame(paint);
  }
  const ready = createGPUColormapEngine()
    .then((value) => {
      engine = value;
      return value;
    })
    .catch(() => null);
  function prepare() {
    const current = ++generation;
    const bytes = extractBytes(model.get("data_bytes"));
    if (!bytes.length) {
      const saved = model.get("_static_fallback_jpeg");
      if (saved) preview.src = `data:${model.get("_static_fallback_mime") || "image/png"};base64,${saved}`;
      else preview.removeAttribute("src");
      preview.style.display = saved ? "block" : "none";
      canvas.style.display = controls.style.display = "none";
      readout.textContent = "Saved preview · rerun the cell for interaction";
      return;
    }
    queue = queue
      .then(async () => {
        await ready;
        if (disposed || current !== generation) return;
        const grid = model.get("grid"),
          bytes = extractBytes(model.get("data_bytes"));
        const nextSource = new Float64Array(
          bytes.slice(0, grid.rows * grid.cols * 8).buffer,
        );
        const cmap = model.get("cmap"),
          vmin = model.get("vmin"), vmax = model.get("vmax");
        const display = Float32Array.from(nextSource), lut = COLORMAPS[cmap];
        if (!lut) {
          status.textContent = "Unsupported colormap";
          return;
        }
        let next: CanvasImageSource | null = null;
        if (engine) {
          engine.uploadData(0, display, grid.cols, grid.rows);
          const rendered = await engine.renderSlotsToImageBitmapAsync(
            [0],
            [{ vmin, vmax }],
            false,
            cmap,
            lut,
          );
          next = rendered?.[0] ?? null;
        }
        const usedGPU = Boolean(next);
        if (!next) {
          next = renderToOffscreen(
            display,
            grid.cols,
            grid.rows,
            lut,
            vmin,
            vmax,
          );
        }
        if (disposed || current !== generation) {
          if (next instanceof ImageBitmap) next.close();
          return;
        }
        if (bitmap instanceof ImageBitmap) bitmap.close();
        // Publish source, color metadata and pixels in the same synchronous paint.
        bitmap = next;
        preview.style.display = "none";
        canvas.style.display = "block";
        controls.style.display = "flex";
        source = nextSource;
        displayCmap = cmap;
        displayMin = vmin;
        displayMax = vmax;
        status.textContent = `${usedGPU ? "WebGPU display" : "CPU display"} · wheel to zoom`;
        cancelAnimationFrame(frame);
        paint();
        updateColorControl();
      })
      .catch((error) => {
        if (!disposed && current === generation)
          status.textContent = `Display error: ${String(error)}`;
      });
  }
  function plotPosition(event: MouseEvent) {
    const layout = geometry(),
      rect = canvas.getBoundingClientRect();
    const col =
      (event.clientX - rect.left - layout.left) / (layout.width - layout.left - layout.right);
    const row =
      1 - (event.clientY - rect.top - layout.top) / (layout.height - layout.top - layout.bottom);
    return col >= 0 && col <= 1 && row >= 0 && row <= 1 ? [col, row] : null;
  }
  canvas.onpointerdown = (event) => {
    if (event.button !== 0 || !plotPosition(event)) return;
    event.preventDefault();
    event.stopPropagation();
    drag = { x: event.clientX, y: event.clientY, bounds: bounds.slice() };
    canvas.setPointerCapture(event.pointerId);
    schedule();
  };
  canvas.onpointermove = (event) => {
    const layout = geometry();
    const width = layout.width - layout.left - layout.right,
      height = layout.height - layout.top - layout.bottom;
    if (drag) {
      const dx =
        ((event.clientX - drag.x) / width) * (drag.bounds[1] - drag.bounds[0]);
      const dy =
        ((event.clientY - drag.y) / height) * (drag.bounds[3] - drag.bounds[2]);
      bounds = clampBounds([
        drag.bounds[0] - dx,
        drag.bounds[1] - dx,
        drag.bounds[2] + dy,
        drag.bounds[3] + dy,
      ]);
      schedule();
    }
    pointer = { clientX: event.clientX, clientY: event.clientY };
    showReadout();
  };
  function updateReadout() {
    if (!pointer || !bitmap) { readout.title = readout.textContent = ""; return; }
    const layout = paintedGeometry, rect = canvas.getBoundingClientRect();
    const width = layout.width - layout.left - layout.right,
      height = layout.height - layout.top - layout.bottom;
    const colFraction = (pointer.clientX - rect.left - layout.left) / width;
    const rowFraction = 1 - (pointer.clientY - rect.top - layout.top) / height;
    if (
      colFraction < 0 ||
      colFraction >= 1 ||
      rowFraction < 0 ||
      rowFraction >= 1
    ) {
      readout.title = readout.textContent = "";
      return;
    }
    const x = paintedBounds[0] + colFraction * (paintedBounds[1] - paintedBounds[0]),
      y = paintedBounds[2] + rowFraction * (paintedBounds[3] - paintedBounds[2]);
    const grid = model.get("grid"),
      original = fullBounds();
    const col = Math.floor(
      ((x - original[0]) / (original[1] - original[0])) * grid.cols,
    );
    const row = Math.floor(
      ((y - original[2]) / (original[3] - original[2])) * grid.rows,
    );
    const binX =
      original[0] + ((col + 0.5) * (original[1] - original[0])) / grid.cols;
    const binY =
      original[2] + ((row + 0.5) * (original[3] - original[2])) / grid.rows;
    readout.title = readout.textContent = `Bin (${row}, ${col}) · x ${formatValue(binX)} · y ${formatValue(binY)} · value ${source[row * grid.cols + col]?.toPrecision(6)}`;
  }
  canvas.onpointerup =
    canvas.onpointercancel =
    canvas.onlostpointercapture =
      () => {
        if (!drag) return;
        drag = null;
        commit();
        schedule();
      };
  canvas.onpointerleave = () => {
    pointer = null;
    showReadout();
  };
  function zoomBy(factor: number, position = [0.5, 0.5]) {
    bounds = clampBounds(
      [0, 2].flatMap((lowIndex, axis) => {
        const span = bounds[lowIndex + 1] - bounds[lowIndex];
        const anchor = bounds[lowIndex] + position[axis] * span;
        return [
          anchor - position[axis] * span * factor,
          anchor + (1 - position[axis]) * span * factor,
        ];
      }),
    );
    schedule();
  }
  canvas.addEventListener(
    "wheel",
    (event) => {
      const position = plotPosition(event);
      if (!position) return;
      event.preventDefault();
      event.stopPropagation();
      const delta =
        event.deltaY *
        (event.deltaMode === 1
          ? 16
          : event.deltaMode === 2
            ? geometry().height
            : 1);
      zoomBy(Math.exp(Math.max(-1, Math.min(1, delta * 0.002))), position);
      clearTimeout(wheelTimer);
      wheelTimer = window.setTimeout(commit, 150);
    },
    { passive: false },
  );
  reset.onclick = () => {
    clearTimeout(wheelTimer);
    drag = null;
    bounds = fullBounds().slice();
    commit();
    schedule();
  };
  canvas.ondblclick = reset.onclick as () => void;
  zoomIn.onclick = () => {
    zoomBy(1 / 1.5);
    commit();
  };
  zoomOut.onclick = () => {
    zoomBy(1.5);
    commit();
  };
  save.onclick = () =>
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, "plot2d.png");
    });
  const observers: [string, () => void][] = [];
  for (const key of ["data_bytes", "cmap", "vmin", "vmax"])
    observers.push([`change:${key}`, prepare]);
  for (const key of [
    "horizontal_line",
    "title",
    "x_label",
    "y_label",
    "colorbar_label",
    "plot_height_px",
  ])
    observers.push([`change:${key}`, schedule]);
  observers.push(["change:max_width", applyMaxWidth]);
  observers.push(["change:cmap", updateColorControl]);
  observers.push([
    "change:view_bounds",
    () => {
      bounds = (
        model.get("view_bounds").length ? model.get("view_bounds") : fullBounds()
      ).slice();
      schedule();
    },
  ]);
  observers.forEach(([event, handler]) => model.on(event, handler));
  const resizeObserver = new ResizeObserver(schedule);
  resizeObserver.observe(host);
  applyMaxWidth();
  updateColorControl();
  prepare();
  return () => {
    disposed = true;
    generation++;
    cancelAnimationFrame(frame);
    clearTimeout(wheelTimer);
    cancelAnimationFrame(readoutFrame);
    colorRoot.unmount();
    resizeObserver.disconnect();
    observers.forEach(([event, handler]) => model.off(event, handler));
    if (bitmap instanceof ImageBitmap) bitmap.close();
    void Promise.all([queue, ready]).then(() => engine?.destroy());
    host.remove();
  };
}

export default { render };
