import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

/** The fields of a Find drift result the chart reads (see drift.ts DriftResult). */
export type DriftReadout = {
  frames: number[];
  shift_px: [number, number][];
  velocity_px_per_frame: [number, number];
  times_s?: number[] | null;
  velocity_px_per_s?: [number, number] | null;
  fit_r2: [number, number];
  shift: [number, number][] | null;
  velocity: [number, number] | null;
  pixel_unit: string;
  frame_interval_s: number | null;
  velocity_unit: string;
  backend: string;
};

const ROW = "#4fc3f7";   // drift along rows
const COL = "#ffb74d";   // drift along columns
const HEIGHT = 112;
const PAD = { left: 34, right: 6, top: 6, bottom: 16 };

/** A short number: 3 significant figures, plain notation. */
function short(value: number): string {
  if (!Number.isFinite(value)) return "–";
  if (value === 0) return "0";
  const digits = Math.max(0, 2 - Math.floor(Math.log10(Math.abs(value))));
  return value.toFixed(Math.min(4, digits));
}

/**
 * Drift against time, rows and columns, with the fitted lines, and the velocity in words above it.
 * Against each frame's own time when the frames have times (frame_times_s or a time frame axis); in
 * physical units when the viewer also knows its pixel size, else pixels; without times, per frame.
 */
export function DriftChart(props: {
  drift: DriftReadout | null;
  status: string;
  width: number;
  colors: { text: string; textMuted: string; border: string };
}) {
  const { drift, status, width, colors } = props;
  if (!drift) {
    return (
      <Box sx={{ flex: "0 0 auto", borderTop: `1px solid ${colors.border}`, px: "6px", py: "4px" }}>
        <Typography sx={{ fontSize: 10, color: colors.textMuted }}>{status}</Typography>
      </Box>
    );
  }
  const timed = Array.isArray(drift.times_s) && drift.times_s.length === drift.frames.length && !!drift.velocity_px_per_s;
  const physical = timed && !!drift.shift && !!drift.velocity;
  const unit = physical ? drift.pixel_unit : "px";
  const times = timed ? drift.times_s! : drift.frames.map((frame) => frame - drift.frames[0]);
  const values = physical ? drift.shift! : drift.shift_px;
  const slope: [number, number] = physical ? drift.velocity! : timed ? drift.velocity_px_per_s! : drift.velocity_px_per_frame;
  const speed = Math.hypot(slope[0], slope[1]);
  const all = values.flat().concat([0]);
  const low = Math.min(...all), high = Math.max(...all);
  const span = high - low || 1;
  const tMax = Math.max(...times) || 1;
  const plotW = width - PAD.left - PAD.right, plotH = HEIGHT - PAD.top - PAD.bottom;
  const x = (t: number) => PAD.left + (t / tMax) * plotW;
  const y = (v: number) => PAD.top + (1 - (v - low) / span) * plotH;
  // fitted lines through the measured mean at the mean time, as the least-squares fit is
  const meanT = times.reduce((a, b) => a + b, 0) / times.length;
  const fitted = (axis: 0 | 1) => {
    const mean = values.reduce((sum, value) => sum + value[axis], 0) / values.length;
    const at = (t: number) => mean + slope[axis] * (t - meanT);
    return `M${x(0)},${y(at(0))} L${x(tMax)},${y(at(tMax))}`;
  };
  const per = timed ? `${unit}/s` : "px/frame";
  return (
    <Box data-testid="show3d-drift-chart" sx={{ flex: "0 0 auto", borderTop: `1px solid ${colors.border}`, px: "1px", pt: "3px" }}>
      <Typography sx={{ fontSize: 11, fontWeight: 600, color: colors.text, px: "5px" }}>
        Drift {short(speed)} {per}
      </Typography>
      <Typography sx={{ fontSize: 10, color: colors.textMuted, px: "5px", lineHeight: 1.35 }}>
        <span style={{ color: ROW }}>row {short(slope[0])}</span> · <span style={{ color: COL }}>col {short(slope[1])}</span> {per}
        {" · "}R² {drift.fit_r2[0].toFixed(2)}/{drift.fit_r2[1].toFixed(2)} · {drift.frames.length} frames · {drift.backend === "webgpu" ? "WebGPU" : "CPU"}
      </Typography>
      <svg width={width} height={HEIGHT} role="img" aria-label={`Drift against time: ${short(speed)} ${per}`}>
        <line x1={PAD.left} x2={width - PAD.right} y1={y(0)} y2={y(0)} stroke={colors.border} strokeWidth={1} />
        <text x={PAD.left - 3} y={PAD.top + 8} fontSize={9} fill={colors.textMuted} textAnchor="end">{short(high)}</text>
        <text x={PAD.left - 3} y={PAD.top + plotH} fontSize={9} fill={colors.textMuted} textAnchor="end">{short(low)}</text>
        <text x={PAD.left - 3} y={PAD.top + plotH / 2 + 3} fontSize={9} fill={colors.textMuted} textAnchor="end">{unit}</text>
        <text x={width - PAD.right} y={HEIGHT - 3} fontSize={9} fill={colors.textMuted} textAnchor="end">
          {timed ? `${short(tMax)} s` : `${short(tMax)} frames`}
        </text>
        {([0, 1] as const).map((axis) => (
          <g key={axis}>
            <path d={fitted(axis)} stroke={axis === 0 ? ROW : COL} strokeWidth={1} strokeDasharray="3 2" fill="none" />
            {values.map((value, i) => (
              <circle key={i} cx={x(times[i])} cy={y(value[axis])} r={2.2} fill={axis === 0 ? ROW : COL} />
            ))}
          </g>
        ))}
      </svg>
    </Box>
  );
}
