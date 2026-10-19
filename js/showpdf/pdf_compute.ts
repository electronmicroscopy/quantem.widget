// G(r) and g(r) in the browser, for the standalone HTML export (no Python kernel). From a 1D I(k)
// and its k axis, it follows quantem.diffraction.PairDistributionFunction:
//   calculate_Gr: background fit, F(k), frequency filter, S(k), k window, sine transform -> G(r)
//   damp_origin_oscillations: low-r correction of G(r) for a supplied density
//   calculate_gr: g(r)
// As in the library, the number density is an input, never estimated. Math.fround reproduces the
// library's float32 arithmetic where it changes a result. pdf_compute.numpy.test.ts checks parity.

export interface GrParams {
  kMinFit: number;     // 0 = data minimum (k_min_fit=None)
  kMaxFit: number;     // 0 = data maximum (k_max_fit=None)
  kMinWindow: number;  // <= 0 = kMinFit (k_min_window=None)
  kMaxWindow: number;  // <= 0 = kMaxFit (k_max_window=None)
  kLowpass: number;    // 0 = off
  kHighpass: number;   // 0 = off
  rMax: number;
  rStep: number;
}

export interface GrResult {
  bg: Float64Array;     // background B(k) over the full k axis
  fk: Float64Array;     // F(k) times 2*pi, before the window
  fkWin: Float64Array;  // windowed F(k): the input of the sine transform
  sk: Float64Array;     // S(k) = 1 + F(k)/k (before 2*pi)
  r: Float64Array;
  gr: Float64Array;     // reduced PDF G(r)
  // Bounds actually used (0 sentinels resolved), needed by the damping.
  kMinFit: number;
  kMaxFit: number;
  kMinWindow: number;
  kMaxWindow: number;
}

export interface DampingParams {
  density: number;     // number density rho0 (atoms/Å³), > 0
  rCut: number;        // upper end (Å) of the corrected interval [0, rCut]
  maxIter?: number;    // default 6, as in the library
  tolerance?: number;  // default 1e-3 (S(k) units), as in the library
}

export interface DampingResult {
  fkWin: Float64Array;  // windowed corrected F(k) used for the transform
  gr: Float64Array;     // G(r) with the origin oscillations corrected
}

// G(r) as calculate_Gr computes it, without the damping. `qq` must be the library's exact float64
// k axis: the background fit selects k bins by comparing it with the fit range, as fit_bg_batched does.
export function computeGr(qq: Float64Array, ik: Float64Array, p: GrParams): GrResult {
  const n = qq.length;
  const dk = qq[1] - qq[0];
  const { kMinFit, kMaxFit, kMinWindow, kMaxWindow } = resolveWindowBounds(qq, p);
  const { bg, f } = fitBackground(qq, ik, kMinFit, kMaxFit);

  // F(k) = (I - B) * k / f, with k and f floored as in calculate_Gr
  const fFloor = 1e-10 * maxValue(f);
  const fkUnfiltered = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const kSafe = Math.max(qq[i], 1e-10);
    const fSafe = Math.max(f[i], fFloor);
    fkUnfiltered[i] = (ik[i] - bg[i]) * kSafe / fSafe;
  }
  // The filter uses calculate_Gr's dk, from its float32 k: with the float64 dk the kernel radius
  // ceil(3 * k_lowpass / dk) can differ by a sample.
  const dk32 = Math.fround(Math.fround(qq[1]) - Math.fround(qq[0]));
  const fk = frequencyFiltering(fkUnfiltered, p.kLowpass, p.kHighpass, dk32);

  // S(k) = 1 + F(k)/k is taken before F(k) gets its 2*pi factor, as in calculate_Gr.
  const sk = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const kSafe = Math.max(qq[i], 1e-10);
    sk[i] = qq[i] > 0 ? 1 + fk[i] / kSafe : 1;
  }
  for (let i = 0; i < n; i++) fk[i] *= 2 * Math.PI;

  const wk = kWindow(qq, kMinWindow, kMaxWindow);
  const fkWin = fk.map((v, i) => v * wk[i]);
  const r = rGrid(p.rMax, p.rStep);
  const gr = sineTransform(qq, r, fkWin, dk);
  return { bg, fk, fkWin, sk, r, gr, kMinFit, kMaxFit, kMinWindow, kMaxWindow };
}

type KBounds = Pick<GrParams, "kMinFit" | "kMaxFit" | "kMinWindow" | "kMaxWindow">;

// The fit and window bounds calculate_Gr uses. An unset (0) fit end is the end of the library's
// float32 k axis, so the last float64 bin can fall just outside the fit, as in the library; an
// unset window end is the fit end.
export function resolveWindowBounds(qq: Float64Array, p: KBounds): KBounds {
  const kMinFit = p.kMinFit > 0 ? p.kMinFit : Math.fround(qq[0]);
  const kMaxFit = p.kMaxFit > 0 ? p.kMaxFit : Math.fround(qq[qq.length - 1]);
  return {
    kMinFit,
    kMaxFit,
    kMinWindow: p.kMinWindow > 0 ? p.kMinWindow : kMinFit,
    kMaxWindow: p.kMaxWindow > 0 ? p.kMaxWindow : kMaxFit,
  };
}

// The k window of _k_window: zero outside [kmin, kmax]; inside, a sin^2 ramp over the first 10% of
// the band times the Lorch factor sin(pi*k/kmax) / (pi*k/kmax). Evaluated in float32 like the
// library, which decides the bins in band and gives w = 0 at kmin and, as sin(float32(pi)) < 0,
// a tiny negative w at kmax.
export function kWindow(k: Float64Array, kmin: number, kmax: number): Float64Array {
  const f32 = Math.fround;
  // Only the ramp width e and the ramp end kmin + e are float64 sums in the library.
  const e = 0.1 * (kmax - kmin);
  const e32 = f32(e);
  const rampEnd32 = f32(kmin + e);
  const kmin32 = f32(kmin);
  const kmax32 = f32(kmax);
  const halfPi32 = f32(0.5 * Math.PI);
  const pi32 = f32(Math.PI);
  const wk = new Float64Array(k.length);
  for (let i = 0; i < k.length; i++) {
    const k32 = f32(k[i]);
    if (k32 < kmin32 || k32 > kmax32) continue;   // out of band: 0
    let ramp = 1;
    if (k32 < rampEnd32) {
      const t = f32(f32(k32 - kmin32) / e32);
      const s = f32(Math.sin(f32(halfPi32 * t)));
      ramp = f32(s * s);
    }
    const x = f32(k32 / kmax32);
    const px = f32(pi32 * x);
    const sinc = x === 0 ? 1 : f32(f32(Math.sin(px)) / px);
    wk[i] = f32(ramp * sinc);
  }
  return wk;
}

// The low-r correction of damp_origin_oscillations (Kaplow et al., in the Yoshimoto-Omote form)
// for a supplied density: the deviation of G(r) from -4*pi*rho0*r on [0, rCut] is moved back into
// S(k), until that correction stops changing.
export function dampOriginOscillations(qq: Float64Array, res: GrResult, p: DampingParams): DampingResult {
  const n = qq.length;
  const dk = qq[1] - qq[0];
  const r = res.r;
  const rho0 = p.density;
  const maxIter = p.maxIter ?? 6;
  const tolerance = p.tolerance ?? 1e-3;

  // Corrected k columns: [kMinFit, kMaxWindow], compared in float32 as the library does.
  const kLo32 = Math.fround(res.kMinFit);
  const kHi32 = Math.fround(res.kMaxWindow);
  const kIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    const k32 = Math.fround(qq[i]);
    if (k32 >= kLo32 && k32 <= kHi32) kIdx.push(i);
  }
  const qScaled = kIdx.map((i) => qq[i] * 2 * Math.PI);

  // r points of the correction integrals: [0, rCut], compared in float32 as the library does.
  const rCut32 = Math.fround(p.rCut);
  const rIdx: number[] = [];
  for (let i = 0; i < r.length; i++) {
    if (r[i] >= 0 && Math.fround(r[i]) <= rCut32) rIdx.push(i);
  }
  const rShort = rIdx.map((i) => r[i]);

  const wk = kWindow(qq, res.kMinWindow, res.kMaxWindow);
  const skCor = Float64Array.from(res.sk);
  // The windowed F(k) of the corrected S(k), and its G(r).
  const transform = () => {
    const fkWin = new Float64Array(n);
    for (let i = 0; i < n; i++) fkWin[i] = qq[i] * (skCor[i] - 1.0) * wk[i] * 2 * Math.PI;
    return { fkWin, gr: sineTransform(qq, r, fkWin, dk) };
  };

  let corrected = transform();
  let betaPrev: Float64Array | null = null;
  for (let it = 0; it < maxIter; it++) {
    const gShort = rIdx.map((i) => corrected.gr[i]);
    const { alpha, beta } = alphaBeta(qScaled, rShort, gShort);
    if (betaPrev !== null) {
      // The library's torch.max over an empty beta fails here.
      if (beta.length === 0) {
        throw new Error("the damping k range [k min fit, k max window] contains no data");
      }
      let change = 0;
      for (let c = 0; c < beta.length; c++) change = Math.max(change, Math.abs(beta[c] - betaPrev[c]));
      if (change < tolerance) break;
    }
    betaPrev = beta;
    for (let c = 0; c < kIdx.length; c++) {
      const i = kIdx[c];
      skCor[i] = skCor[i] - beta[c] + rho0 * alpha[c];
    }
    corrected = transform();
  }
  return corrected;
}

// g(r) = 1 + G(r) / (4*pi*r*rho0), 0 at r = 0 and clamped to >= 0, as
// calculate_gr(set_pdf_positive=True).
export function computeGofR(r: Float64Array, gr: Float64Array, rho0: number): Float64Array {
  const out = new Float64Array(r.length);
  for (let i = 0; i < r.length; i++) {
    out[i] = r[i] > 0 ? Math.max(0, 1 + gr[i] / (4 * Math.PI * r[i] * rho0)) : 0;
  }
  return out;
}

// --- Background fit: fit_bg_batched for a single curve --------------------

// Fits B(k) = c + i0 exp(-k^2 / (2 s0^2)) + i1 exp(-k^4 / (2 s1^4)) to I(k) on [kmin, kmax] and
// returns B(k) and f(k) = B(k) - c over the full k axis.
function fitBackground(qq: Float64Array, ik: Float64Array, kmin: number, kmax: number) {
  const n = qq.length;
  const k2 = qq.map((k) => k * k);
  const dk = qq[1] - qq[0];

  const fitIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    if (qq[i] >= kmin && qq[i] <= kmax) fitIdx.push(i);
  }
  const nf = fitIdx.length;
  if (nf === 0) throw new Error("the k fit range contains no data");
  const kFit = fitIdx.map((i) => qq[i]);
  const k2Fit = fitIdx.map((i) => k2[i]);
  const ikFit = fitIdx.map((i) => Math.max(ik[i], 1e-10));

  // Residual weights: a sin^2 taper over the first 0.25 1/Å above kmin (0 where it is <= 1e-4)
  // suppresses the central-beam tail; dividing by a line falling with k favours high k.
  const lastK = kFit[nf - 1];
  const weights = kFit.map((k) => {
    const s = Math.sin(0.5 * Math.PI * Math.min(Math.max((k - kmin) / 0.25, 0), 1));
    const taper = s * s;
    return (taper > 1e-4 ? taper : 0) / (lastK - 0.9 * k + dk);
  });

  // Initial guess: c = min I, amplitudes = median I - min I, widths = mean k.
  const sorted = [...ikFit].sort((a, b) => a - b);
  const c0 = Math.max(sorted[0], 1e-10);
  const median = sorted[Math.floor((nf - 1) / 2)];   // the lower middle value, as torch.median
  const amp = Math.max(median - c0, 1e-10);
  let kSum = 0;
  for (const k of qq) kSum += k;
  const sig = Math.max(kSum / n, 1e-3);
  // The fit works on the parameters' logs, which keeps them positive.
  const theta0 = [c0, amp, sig, amp, sig].map((v) => Math.log(v));

  // Weighted residuals (B - I) * w and their derivatives in log c, log i0, log s0, log i1, log s1.
  const residuals = (theta: number[]) => {
    const [c, i0, s0, i1, s1] = theta.map((v) => Math.exp(v));
    const r = new Float64Array(nf);
    const jac: number[][] = [];
    for (let j = 0; j < nf; j++) {
      const e0 = clampedExp(-k2Fit[j] / (2 * s0 * s0));
      const e1 = clampedExp(-(k2Fit[j] * k2Fit[j]) / (2 * s1 * s1 * s1 * s1));
      const B = c + i0 * e0 + i1 * e1;
      const w = weights[j];
      r[j] = (B - ikFit[j]) * w;
      const t0 = i0 * e0;
      const t1 = i1 * e1;
      jac.push([
        c * w,
        t0 * w,
        t0 * (k2Fit[j] / (s0 * s0)) * w,
        t1 * w,
        t1 * (2.0 * k2Fit[j] * k2Fit[j] / (s1 * s1 * s1 * s1)) * w,
      ]);
    }
    return { r, jac };
  };

  return backgroundCurves(levenbergMarquardt(theta0, residuals), k2);
}

// B(k) and f(k) = B(k) - c (floored at 1e-10 max B) for the fitted log parameters, rounded to
// float32 as fit_bg_batched returns them: F(k) divides by f(k), which amplifies the rounding of
// B(k) where f(k) is small.
function backgroundCurves(theta: number[], k2: Float64Array) {
  const [c, i0, s0, i1, s1] = theta.map((v) => Math.exp(v));
  const bg = k2.map((kSq) => {
    const e0 = clampedExp(-kSq / (2 * s0 * s0));
    const e1 = clampedExp(-(kSq * kSq) / (2 * s1 * s1 * s1 * s1));
    return c + i0 * e0 + i1 * e1;
  });
  const fFloor = 1e-10 * maxValue(bg);
  const f = bg.map((b) => Math.max(b - c, fFloor));
  return { bg: bg.map((v) => Math.fround(v)), f: f.map((v) => Math.fround(v)) };
}

// Levenberg-Marquardt over the log parameters, as fit_bg_batched runs it for one curve: up to 200
// Jacobian updates with up to 30 damping trials each, no log parameter moving more than 1 per step.
function levenbergMarquardt(
  theta0: number[],
  residuals: (theta: number[]) => { r: Float64Array; jac: number[][] },
) {
  const maxLogStep = 1.0;
  const loss = (theta: number[]) => {
    const { r } = residuals(theta);
    let sum = 0;
    for (let j = 0; j < r.length; j++) sum += r[j] * r[j];
    return sum;
  };

  let theta = theta0;
  let currentLoss = loss(theta);
  let lam = 1.0;
  for (let outer = 0; outer < 200; outer++) {
    const { r, jac } = residuals(theta);
    // Normal equations: J^T J and J^T r.
    const JtJ = Array.from({ length: 5 }, () => [0, 0, 0, 0, 0]);
    const Jtr = [0, 0, 0, 0, 0];
    for (let j = 0; j < r.length; j++) {
      for (let a = 0; a < 5; a++) {
        Jtr[a] += jac[j][a] * r[j];
        for (let b = 0; b < 5; b++) JtJ[a][b] += jac[j][a] * jac[j][b];
      }
    }
    const diag = JtJ.map((row, i) => Math.max(row[i], 1e-12));
    const prevLoss = currentLoss;
    let accepted = false;
    for (let inner = 0; inner < 30; inner++) {
      // Step: (J^T J + lam * diag + 1e-12) delta = J^T r, shortened to maxLogStep.
      const A = JtJ.map((row, i) => row.map((v, j) => v + (i === j ? lam * diag[i] + 1e-12 : 0)));
      const delta = solve5(A, Jtr);
      let largest = 1e-30;
      for (const d of delta) largest = Math.max(largest, Math.abs(d));
      const stepScale = Math.min(maxLogStep / largest, 1.0);
      const trial = theta.map((v, i) => v - delta[i] * stepScale);
      const trialLoss = loss(trial);
      if (isFinite(trialLoss) && trialLoss < currentLoss) {
        theta = trial;
        currentLoss = trialLoss;
        accepted = true;
        lam = Math.max(lam * 0.3, 1e-12);   // accepted: toward Gauss-Newton
        break;
      }
      lam = Math.min(lam * 3.0, 1e12);   // rejected: toward gradient descent
    }
    if (!accepted) break;
    if ((prevLoss - currentLoss) / (prevLoss + 1e-30) < 1e-9) break;   // converged
  }
  return theta;
}

// Solves the 5 x 5 system A x = b by Gauss-Jordan elimination with partial pivoting.
function solve5(A: number[][], b: number[]) {
  const n = 5;
  const M = A.map((row, i) => [...row, b[i]]);   // augmented matrix [A | b]
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) {
      if (Math.abs(M[r][c]) > Math.abs(M[pivot][c])) pivot = r;
    }
    [M[c], M[pivot]] = [M[pivot], M[c]];
    const d = M[c][c] || 1e-300;   // a zero (or NaN) pivot is replaced by 1e-300
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const factor = M[r][c] / d;
      for (let k = c; k <= n; k++) M[r][k] -= factor * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-300));
}

// exp(x) with x clamped to [-100, 0], as in the background model of fit_bg_batched.
function clampedExp(x: number) {
  return Math.exp(Math.min(Math.max(x, -100), 0));
}

// The largest value, skipping NaN (-Infinity if there is none).
function maxValue(a: Float64Array) {
  let max = -Infinity;
  for (const v of a) {
    if (v > max) max = v;
  }
  return max;
}

// --- Frequency filter: _frequency_filtering and core.utils.filter ---------

// Gaussian low-pass, high-pass (F(k) minus its smoothed copy) or band-pass (difference of the two
// smoothings) of F(k), with sigmas in 1/Å (0 = off).
function frequencyFiltering(fk: Float64Array, kLowpass: number, kHighpass: number, dk: number) {
  const smooth = (sigma: number) => gaussianFilter1d(fk, gaussianKernel1d(sigma / dk));
  if (kLowpass > 0 && kHighpass > 0) {
    // _frequency_filtering raises a ValueError here too.
    if (kHighpass > kLowpass) {
      throw new Error("k highpass must not exceed k lowpass for band-pass filtering");
    }
    const low = smooth(kLowpass);
    const high = smooth(kHighpass);
    return high.map((v, i) => v - low[i]);
  }
  if (kLowpass > 0) return smooth(kLowpass);
  if (kHighpass > 0) {
    const high = smooth(kHighpass);
    return fk.map((v, i) => v - high[i]);
  }
  return fk;
}

// Normalised Gaussian of radius ceil(3 sigma) samples, as gaussian_kernel_1d.
function gaussianKernel1d(sigma: number) {
  const radius = Math.ceil(3.0 * sigma);
  const n = 2 * radius + 1;
  const kernel = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - radius;
    const v = Math.exp(-0.5 * (x * x) / (sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));
    kernel[i] = v;
    sum += v;
  }
  for (let i = 0; i < n; i++) kernel[i] /= sum;
  return kernel;
}

// Convolution with replicate ("edge") padding, as gaussian_filter_1d.
function gaussianFilter1d(arr: Float64Array, kernel: Float64Array) {
  const n = arr.length;
  const pad = Math.floor(kernel.length / 2);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let j = 0; j < kernel.length; j++) {
      let idx = i + j - pad;
      if (idx < 0) idx = 0;
      else if (idx >= n) idx = n - 1;
      acc += arr[idx] * kernel[j];
    }
    out[i] = acc;
  }
  return out;
}

// --- Transforms -----------------------------------------------------------

// The r grid of calculate_Gr: torch.arange(0, r_max, r_step) built in float64 and cast to float32.
function rGrid(rMax: number, rStep: number) {
  const length = Math.max(0, Math.ceil(rMax / rStep));
  if (length === 0) throw new Error("the r grid is empty: r max must be positive");
  const r = new Float64Array(length);
  for (let i = 0; i < length; i++) r[i] = Math.fround(i * rStep);
  return r;
}

// Discrete sine transform G(r) = (2/pi)*dk*2*pi * sum_k sin(2*pi*k*r) * fkWin(k), with G[0] = 0
// (calculate_Gr and the damping zero the first r point).
function sineTransform(qq: Float64Array, r: Float64Array, fkWin: Float64Array, dk: number) {
  const out = new Float64Array(r.length);
  const scale = (2 / Math.PI) * dk * 2 * Math.PI;
  for (let ri = 0; ri < r.length; ri++) {
    let acc = 0;
    const twoPiR = 2 * Math.PI * r[ri];
    for (let ki = 0; ki < qq.length; ki++) acc += Math.sin(twoPiR * qq[ki]) * fkWin[ki];
    out[ri] = scale * acc;
  }
  if (r.length > 0) out[0] = 0;
  return out;
}

// The integrals of _compute_alpha_beta over r in [0, rCut], one per corrected k column (Q = 2*pi*k):
// alpha of the ideal -4*pi*r (per unit density) and beta of the current G(r).
function alphaBeta(qScaled: number[], rShort: number[], gShort: number[]) {
  const alpha = new Float64Array(qScaled.length);
  const beta = new Float64Array(qScaled.length);
  for (let c = 0; c < qScaled.length; c++) {
    const Q = qScaled[c] === 0 ? 1e-12 : qScaled[c];
    const alphaIntegrand = new Float64Array(rShort.length);
    const betaIntegrand = new Float64Array(rShort.length);
    for (let i = 0; i < rShort.length; i++) {
      const s = Math.sin(Q * rShort[i]) / Q;
      alphaIntegrand[i] = -4 * Math.PI * rShort[i] * s;
      betaIntegrand[i] = gShort[i] * s;
    }
    alpha[c] = trapz(alphaIntegrand, rShort);
    beta[c] = trapz(betaIntegrand, rShort);
  }
  return { alpha, beta };
}

// Trapezoid-rule integral of y over x.
function trapz(y: ArrayLike<number>, x: ArrayLike<number>) {
  let s = 0;
  for (let i = 1; i < y.length; i++) s += 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1]);
  return s;
}
