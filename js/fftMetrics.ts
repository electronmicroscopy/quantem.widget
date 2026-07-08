export type FftQualityMetrics = {
  sharp: number | null;
  peaks: number;
  snr: number | null;
  ring: number | null;
};

type FftMetricRegion = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type FftMetricOptions = {
  sampling?: number | null;
  unit?: string | null;
  region?: FftMetricRegion | null;
  maxSamplesPerDim?: number;
};

const DEFAULT_LATTICE_PERIOD_RANGE_A: [number, number] = [1.0, 6.0];

function samplingToAngstrom(sampling: number | null | undefined, unit: string | null | undefined): number | null {
  if (sampling == null || !Number.isFinite(sampling) || sampling <= 0) return null;
  const normalized = String(unit || "").trim().toLowerCase();
  if (!normalized || normalized === "px" || normalized === "pixel" || normalized === "pixels") return null;
  if (normalized === "a" || normalized === "å" || normalized === "angstrom" || normalized === "angstroms") return sampling;
  if (normalized === "nm" || normalized === "nanometer" || normalized === "nanometers") return sampling * 10;
  if (normalized === "pm" || normalized === "picometer" || normalized === "picometers") return sampling * 0.01;
  if (normalized === "um" || normalized === "µm" || normalized === "micrometer" || normalized === "micrometers") return sampling * 10000;
  if (normalized === "m") return sampling * 1e10;
  return null;
}

function bandBounds(width: number, height: number, samplingA: number | null): { lo: number; hi: number } | null {
  const rMax = Math.min(width, height) / 2;
  if (rMax < 16) return null;
  if (samplingA && samplingA > 0) {
    const n = Math.min(width, height);
    const [minPeriodA, maxPeriodA] = DEFAULT_LATTICE_PERIOD_RANGE_A;
    const lo = Math.max(2, n * samplingA / maxPeriodA);
    const hi = Math.min(rMax - 1, n * samplingA / minPeriodA);
    if (hi > lo + 2) return { lo, hi };
  }
  return { lo: 0.15 * rMax, hi: 0.6 * rMax };
}

function metricFormat(value: number | null, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "--";
  if (Math.abs(value) >= 100) return value.toFixed(0);
  return value.toFixed(digits);
}

export function formatFftQualityLabel(metrics: FftQualityMetrics | null): string {
  if (!metrics) return "";
  return `FFT sharp ${metricFormat(metrics.sharp)} · peaks ${metrics.peaks} · SNR ${metricFormat(metrics.snr)}`;
}

export function computeFftQualityMetrics(
  mag: Float32Array,
  width: number,
  height: number,
  options: FftMetricOptions = {},
): FftQualityMetrics | null {
  if (mag.length !== width * height || width < 32 || height < 32) return null;

  const region = options.region;
  const x0 = Math.max(0, Math.floor(region?.x ?? 0));
  const y0 = Math.max(0, Math.floor(region?.y ?? 0));
  const regionW = Math.max(1, Math.min(width - x0, Math.floor(region?.width ?? width)));
  const regionH = Math.max(1, Math.min(height - y0, Math.floor(region?.height ?? height)));
  if (regionW < 32 || regionH < 32) return null;

  const samplingA = samplingToAngstrom(options.sampling, options.unit);
  const bounds = bandBounds(regionW, regionH, samplingA);
  if (!bounds) return null;

  const cx = x0 + regionW / 2;
  const cy = y0 + regionH / 2;
  const rMax = Math.min(regionW, regionH) / 2;
  const dcRadius = Math.max(2, rMax * 0.015);
  const stride = Math.max(1, Math.ceil(Math.max(regionW, regionH) / Math.max(64, options.maxSamplesPerDim ?? 768)));
  const rLoSq = bounds.lo * bounds.lo;
  const rHiSq = bounds.hi * bounds.hi;
  const dcSq = dcRadius * dcRadius;
  const rMaxSq = (rMax - 1) * (rMax - 1);

  let totalPower = 0;
  let bandPower = 0;
  let bandCount = 0;
  let bandSum = 0;
  let bandSumSq = 0;
  let strongest = 0;

  const angleBins = 16;
  const angleSum = new Float64Array(angleBins);
  const angleCount = new Uint32Array(angleBins);

  for (let y = y0; y < y0 + regionH; y += stride) {
    const dy = y - cy;
    const row = y * width;
    for (let x = x0; x < x0 + regionW; x += stride) {
      const dx = x - cx;
      const rSq = dx * dx + dy * dy;
      if (rSq <= dcSq || rSq >= rMaxSq) continue;
      const v = Math.max(0, mag[row + x]);
      const power = v * v;
      totalPower += power;
      if (rSq < rLoSq || rSq > rHiSq) continue;
      bandPower += power;
      bandCount++;
      bandSum += v;
      bandSumSq += v * v;
      if (v > strongest) strongest = v;
      const theta = Math.atan2(dy, dx);
      const bin = Math.max(0, Math.min(angleBins - 1, Math.floor(((theta + Math.PI) / (2 * Math.PI)) * angleBins)));
      angleSum[bin] += v;
      angleCount[bin]++;
    }
  }

  if (totalPower <= 1e-20 || bandCount < 8) return null;

  const mean = bandSum / bandCount;
  const variance = Math.max(0, bandSumSq / bandCount - mean * mean);
  const std = Math.sqrt(variance);
  const threshold = mean + Math.max(3, std > 0 ? 5 * std : mean * 2);
  const neighborStep = stride;
  let peaks = 0;

  for (let y = y0 + neighborStep; y < y0 + regionH - neighborStep; y += stride) {
    const dy = y - cy;
    const row = y * width;
    for (let x = x0 + neighborStep; x < x0 + regionW - neighborStep; x += stride) {
      const dx = x - cx;
      const rSq = dx * dx + dy * dy;
      if (rSq < rLoSq || rSq > rHiSq) continue;
      const center = mag[row + x];
      if (center < threshold) continue;
      let isPeak = true;
      for (let oy = -neighborStep; oy <= neighborStep && isPeak; oy += neighborStep) {
        for (let ox = -neighborStep; ox <= neighborStep; ox += neighborStep) {
          if (ox === 0 && oy === 0) continue;
          if (mag[(y + oy) * width + x + ox] >= center) {
            isPeak = false;
            break;
          }
        }
      }
      if (isPeak) peaks++;
    }
  }

  let ring = null;
  let validAngles = 0;
  let angleMeanSum = 0;
  const angleMeans = new Float64Array(angleBins);
  for (let i = 0; i < angleBins; i++) {
    if (angleCount[i] === 0) continue;
    angleMeans[i] = angleSum[i] / angleCount[i];
    angleMeanSum += angleMeans[i];
    validAngles++;
  }
  if (validAngles > 1) {
    const angleMean = angleMeanSum / validAngles;
    if (angleMean > 1e-20) {
      let angleSq = 0;
      for (let i = 0; i < angleBins; i++) {
        if (angleCount[i] === 0) continue;
        const d = angleMeans[i] - angleMean;
        angleSq += d * d;
      }
      ring = Math.max(0, 1 - Math.sqrt(angleSq / validAngles) / angleMean);
    }
  }

  return {
    sharp: 100 * bandPower / totalPower,
    peaks,
    snr: std > 1e-20 ? Math.max(0, (strongest - mean) / std) : (mean > 1e-20 ? strongest / mean : null),
    ring,
  };
}

export function summarizeFftQualityMetrics(metrics: Array<FftQualityMetrics | null>): FftQualityMetrics | null {
  const valid = metrics.filter((m): m is FftQualityMetrics => m != null);
  if (valid.length === 0) return null;
  let sharpSum = 0, sharpCount = 0, peaks = 0, snr = 0, snrCount = 0, ringSum = 0, ringCount = 0;
  for (const metric of valid) {
    if (metric.sharp != null && Number.isFinite(metric.sharp)) {
      sharpSum += metric.sharp;
      sharpCount++;
    }
    peaks += metric.peaks;
    if (metric.snr != null && Number.isFinite(metric.snr)) {
      snr = Math.max(snr, metric.snr);
      snrCount++;
    }
    if (metric.ring != null && Number.isFinite(metric.ring)) {
      ringSum += metric.ring;
      ringCount++;
    }
  }
  return {
    sharp: sharpCount > 0 ? sharpSum / sharpCount : null,
    peaks,
    snr: snrCount > 0 ? snr : null,
    ring: ringCount > 0 ? ringSum / ringCount : null,
  };
}
