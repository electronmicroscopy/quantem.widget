/**
 * Small numerical kit for the diffraction simulator: quaternions (scalar
 * first, the quantem convention: v_lab = R(q) v_crystal), base64 float32
 * decoding, and a complex Hermitian eigensolver (cyclic Jacobi) for the
 * Bloch wave calculation in the browser.
 */

export type Quat = [number, number, number, number];
export type Vec3 = [number, number, number];

export function qmult(a: Quat, b: Quat): Quat {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

export function qnormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  const s = q[0] < 0 ? -1 / n : 1 / n;
  return [q[0] * s, q[1] * s, q[2] * s, q[3] * s];
}

export function quatFromAxisAngle(axis: Vec3, angle: number): Quat {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const s = Math.sin(angle / 2) / n;
  return [Math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s];
}

/** Row-major 3x3 rotation matrix R with v_lab = R v_crystal. */
export function quatToMatrix(q: Quat): number[] {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

export function matVec(R: number[], v: Vec3): Vec3 {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];
}

/** R^T v: lab to crystal frame. */
export function matTVec(R: number[], v: Vec3): Vec3 {
  return [
    R[0] * v[0] + R[3] * v[1] + R[6] * v[2],
    R[1] * v[0] + R[4] * v[1] + R[7] * v[2],
    R[2] * v[0] + R[5] * v[1] + R[8] * v[2],
  ];
}

/** Quaternion putting the crystal-frame unit direction d along +z (lab). */
export function quatFromZoneAxis(d: Vec3, inPlaneDeg = 0): Quat {
  const n = Math.hypot(d[0], d[1], d[2]) || 1;
  const v: Vec3 = [d[0] / n, d[1] / n, d[2] / n];
  const axis: Vec3 = [v[1], -v[0], 0]; // v x z
  const sinT = Math.hypot(axis[0], axis[1]);
  const angle = Math.atan2(sinT, v[2]);
  const qTilt = sinT < 1e-12 ? quatFromAxisAngle([1, 0, 0], v[2] > 0 ? 0 : Math.PI) : quatFromAxisAngle(axis, angle);
  const qSpin = quatFromAxisAngle([0, 0, 1], (inPlaneDeg * Math.PI) / 180);
  return qnormalize(qmult(qSpin, qTilt));
}

export function decodeF32(b64: string): Float32Array {
  if (!b64) return new Float32Array(0);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Smallest integer direction indices [uvw] with u a + v b + w c along d (crystal Cartesian), or null. */
export function directionIndices(cell: number[][], d: Vec3, maxMult = 8): [number, number, number] | null {
  // fractional coordinates of d: solve d = cell^T uvw  ->  uvw = inv(cell^T) d
  const m = invert3(transpose3(cell));
  if (!m) return null;
  const f = matVec(m, d);
  const fm = Math.max(Math.abs(f[0]), Math.abs(f[1]), Math.abs(f[2])) || 1;
  const v = f.map((x) => x / fm);
  for (let mult = 1; mult <= maxMult; mult++) {
    const w = v.map((x) => x * mult);
    if (w.every((x) => Math.abs(x - Math.round(x)) < 0.02)) {
      const ints = w.map((x) => Math.round(x)) as [number, number, number];
      // accept only if the integer direction is within 0.3 degrees of d
      const c = cell;
      const v: Vec3 = [
        ints[0] * c[0][0] + ints[1] * c[1][0] + ints[2] * c[2][0],
        ints[0] * c[0][1] + ints[1] * c[1][1] + ints[2] * c[2][1],
        ints[0] * c[0][2] + ints[1] * c[1][2] + ints[2] * c[2][2],
      ];
      const cosang = (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]) / ((Math.hypot(...v) * Math.hypot(...d)) || 1);
      if (Math.abs(cosang) < Math.cos((0.3 * Math.PI) / 180)) return null;
      const g = gcd3(ints);
      return ints.map((x) => x / g) as [number, number, number];
    }
  }
  return null;
}

/** Miller-Bravais direction [u v t w] -> three-index [U V W] = [2u+v, u+2v, w]. */
export function fourToThree(v: number[]): [number, number, number] {
  const [u, vv, , w] = v;
  return [2 * u + vv, u + 2 * vv, w];
}

/** Three-index direction [U V W] of a hexagonal cell -> smallest integer [u v t w]. */
export function threeToFour(d: [number, number, number]): [number, number, number, number] {
  const [U, V, W] = d;
  // u = (2U - V)/3, v = (2V - U)/3, t = -(u + v): scale by 3 and reduce
  const ints = [2 * U - V, 2 * V - U, -(U + V), 3 * W];
  const g = Math.max(1, ints.reduce((a, b) => gcd(a, b), 0));
  return ints.map((x) => x / g) as [number, number, number, number];
}

/** Parse "1 1 0", "110", "1,-1,0", "[1-10]" or a 4-index "0001" / "1 0 -1 0"; 4 indices are Miller-Bravais. */
export function parseDirection(text: string): [number, number, number] | null {
  const t = text.trim().replace(/[\[\]()]/g, "");
  let parts: string[];
  if (/[\s,]/.test(t)) parts = t.split(/[\s,]+/).filter(Boolean);
  else parts = t.match(/-?\d/g) || [];
  if (parts.length !== 3 && parts.length !== 4) return null;
  const v = parts.map(Number);
  if (v.some((x) => !isFinite(x)) || v.every((x) => x === 0)) return null;
  if (v.length === 4) {
    if (Math.abs(v[0] + v[1] + v[2]) > 1e-9) return null; // u + v + t must vanish
    return fourToThree(v);
  }
  return v as [number, number, number];
}

function gcd(a: number, b: number): number {
  a = Math.abs(a); b = Math.abs(b);
  while (b) [a, b] = [b, a % b];
  return a;
}
function gcd3(v: [number, number, number]): number {
  return Math.max(1, gcd(gcd(v[0], v[1]), v[2]));
}

export function transpose3(a: number[][]): number[] {
  return [a[0][0], a[1][0], a[2][0], a[0][1], a[1][1], a[2][1], a[0][2], a[1][2], a[2][2]];
}

export function invert3(m: number[]): number[] | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-14) return null;
  const inv = [
    A, -(b * i - c * h), b * f - c * e,
    B, a * i - c * g, -(a * f - c * d),
    C, -(a * h - b * g), a * e - b * d,
  ];
  return inv.map((x) => x / det);
}

/**
 * Eigendecomposition of a complex Hermitian matrix by cyclic Jacobi
 * rotations. re/im are row-major n x n; returns eigenvalues and the
 * eigenvectors as columns (vecRe[i*n + j] = component i of eigenvector j).
 * O(n^3) per sweep, a handful of sweeps: milliseconds for n ~ 100.
 */
export function eighComplex(re: Float64Array, im: Float64Array, n: number) {
  const a = Float64Array.from(re);
  const b = Float64Array.from(im);
  const vr = new Float64Array(n * n);
  const vi = new Float64Array(n * n);
  for (let i = 0; i < n; i++) vr[i * n + i] = 1;
  const idx = (i: number, j: number) => i * n + j;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[idx(p, q)] ** 2 + b[idx(p, q)] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq_r = a[idx(p, q)], apq_i = b[idx(p, q)];
        const mag = Math.hypot(apq_r, apq_i);
        if (mag < 1e-300) continue;
        // phase rotation of column q (and its row) makes a_pq real positive
        const cph = apq_r / mag, sph = apq_i / mag; // e^{i phi} = (cph, sph)
        // column q *= e^{-i phi}; row q *= e^{i phi}
        for (let k = 0; k < n; k++) {
          const kr = a[idx(k, q)], ki = b[idx(k, q)];
          a[idx(k, q)] = kr * cph + ki * sph;
          b[idx(k, q)] = ki * cph - kr * sph;
        }
        for (let k = 0; k < n; k++) {
          const kr = a[idx(q, k)], ki = b[idx(q, k)];
          a[idx(q, k)] = kr * cph - ki * sph;
          b[idx(q, k)] = ki * cph + kr * sph;
        }
        for (let k = 0; k < n; k++) {
          const kr = vr[idx(k, q)], ki = vi[idx(k, q)];
          vr[idx(k, q)] = kr * cph + ki * sph;
          vi[idx(k, q)] = ki * cph - kr * sph;
        }
        // real Jacobi rotation in the (p, q) plane
        const app = a[idx(p, p)], aqq = a[idx(q, q)], apq = a[idx(p, q)];
        const theta = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(theta), s = Math.sin(theta);
        for (let k = 0; k < n; k++) {
          // columns
          const pr = a[idx(k, p)], pi = b[idx(k, p)], qr = a[idx(k, q)], qi = b[idx(k, q)];
          a[idx(k, p)] = c * pr - s * qr; b[idx(k, p)] = c * pi - s * qi;
          a[idx(k, q)] = s * pr + c * qr; b[idx(k, q)] = s * pi + c * qi;
        }
        for (let k = 0; k < n; k++) {
          // rows
          const pr = a[idx(p, k)], pi = b[idx(p, k)], qr = a[idx(q, k)], qi = b[idx(q, k)];
          a[idx(p, k)] = c * pr - s * qr; b[idx(p, k)] = c * pi - s * qi;
          a[idx(q, k)] = s * pr + c * qr; b[idx(q, k)] = s * pi + c * qi;
        }
        for (let k = 0; k < n; k++) {
          const pr = vr[idx(k, p)], pi = vi[idx(k, p)], qr = vr[idx(k, q)], qi = vi[idx(k, q)];
          vr[idx(k, p)] = c * pr - s * qr; vi[idx(k, p)] = c * pi - s * qi;
          vr[idx(k, q)] = s * pr + c * qr; vi[idx(k, q)] = s * pi + c * qi;
        }
        a[idx(p, q)] = 0; b[idx(p, q)] = 0; a[idx(q, p)] = 0; b[idx(q, p)] = 0;
      }
    }
  }
  const vals = new Float64Array(n);
  for (let i = 0; i < n; i++) vals[i] = a[idx(i, i)];
  return { vals, vecRe: vr, vecIm: vi };
}
