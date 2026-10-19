import { formatScaleLabel, formatZoomLabel, roundToNiceValue } from "../figure";

const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

// Scale bar and zoom of the navigation image, drawn on its HiDPI overlay canvas: white bar and text
// with a solid black outline, so they read on any image (a bright virtual-BF one too), unlike the
// faint drop shadow of the shared drawScaleBarHiDPI.
export function drawNavScaleBar(
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: string,
  imageWidth: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const effectiveZoom = zoom * scaleX;

  const targetBarPx = 60;
  const barThickness = 5;
  const border = 2;
  const fontSize = 16;
  const margin = 12;

  // A bar of a round physical length, about targetBarPx long on screen.
  const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
  const nicePhysical = roundToNiceValue(targetPhysical);
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;
  const barX = cssWidth - barPx - margin;
  const barY = cssHeight - margin;

  // Bar: black border rectangle, then white fill on top.
  ctx.fillStyle = "black";
  ctx.fillRect(barX - border, barY - border, barPx + 2 * border, barThickness + 2 * border);
  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  // Labels: white text over a black outline; the length centred above the bar, the zoom at the
  // bottom left.
  ctx.font = `${fontSize}px ${FONT}`;
  ctx.lineJoin = "round";
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = "black";
  ctx.fillStyle = "white";
  ctx.textBaseline = "bottom";
  ctx.textAlign = "center";
  drawOutlinedText(ctx, formatScaleLabel(nicePhysical, unit), barX + barPx / 2, barY - 4);
  ctx.textAlign = "left";
  drawOutlinedText(ctx, formatZoomLabel(zoom), margin, cssHeight - margin + barThickness);

  ctx.restore();
}

function drawOutlinedText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number) {
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}
