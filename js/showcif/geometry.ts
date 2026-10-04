/** Cartesian Å geometry. [uvw] uses the direct lattice, including oblique cells. */
export type V3 = [number, number, number];
export const dot = (a: number[], b: number[]) =>
  a.reduce((s, x, i) => s + x * b[i], 0);
export const cross = (a: number[], b: number[]): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export function normalize(v: number[]): V3 {
  const n = Math.hypot(...v);
  if (n < 1e-10) throw Error("Direction must be nonzero.");
  return v.map((x) => x / n) as V3;
}
export function projectionBasis(cell: number[][], uvw: number[]) {
  const beam = normalize(
    [0, 1, 2].map((j) => uvw.reduce((s, u, i) => s + u * cell[i][j], 0)),
  );
  const candidates = [cell[1], cell[2], cell[0]];
  const ref = candidates.find((v) => Math.hypot(...cross(v, beam)) > 1e-6)!;
  const right = normalize(cross(ref, beam));
  return { right, up: normalize(cross(beam, right)), beam };
}
export function cameraBasis(
  basis: ReturnType<typeof projectionBasis>,
  yaw: number,
  pitch: number,
) {
  const { right: r, up: u, beam: b } = basis;
  const right = r.map((v, i) => v * Math.cos(yaw) - b[i] * Math.sin(yaw)) as V3;
  const behind = r.map((v, i) => v * Math.sin(yaw) + b[i] * Math.cos(yaw));
  return {
    right,
    up: u.map(
      (v, i) => v * Math.cos(pitch) - behind[i] * Math.sin(pitch),
    ) as V3,
    beam: u.map(
      (v, i) => v * Math.sin(pitch) + behind[i] * Math.cos(pitch),
    ) as V3,
  };
}
export function cellCorners(cell: number[][]): V3[] {
  return Array.from(
    { length: 8 },
    (_, k) =>
      [0, 1, 2].map((j) =>
        cell.reduce((s, v, i) => s + ((k >> i) & 1) * v[j], 0),
      ) as V3,
  );
}

/** Validate before allocating or drawing an inspection supercell. */
export function repeatCount(atomsPerCell: number, repeats: number[]): number {
  if (
    repeats.length !== 3 ||
    repeats.some((v) => !Number.isSafeInteger(v) || v < 1)
  ) {
    throw Error("Use three positive whole numbers for a, b, and c.");
  }
  const count = atomsPerCell * repeats.reduce((n, v) => n * v, 1);
  if (!Number.isSafeInteger(count) || count > 250000) {
    throw Error("Use a smaller inspection region: at most 250,000 atoms.");
  }
  return count;
}

/** Click-picking reference for the GPU's ASE-order unit-cell instances. */
export function repeatedAtom(
  atoms: Float32Array,
  cell: number[][],
  repeats: number[],
  index: number,
): number[] {
  const n = atoms.length / 4,
    base = (index % n) * 4,
    k = Math.floor(index / n);
  const shift = [
    Math.floor(k / (repeats[1] * repeats[2])),
    Math.floor(k / repeats[2]) % repeats[1],
    k % repeats[2],
  ];
  return [0, 1, 2]
    .map(
      (j) => atoms[base + j] + shift.reduce((s, v, i) => s + v * cell[i][j], 0),
    )
    .concat(atoms[base + 3]);
}

/** Fixed-camera calibration: field width is inversely proportional to magnification. */
export function calibratedFov(
  magnification: number,
  calibration: number[],
): number {
  if (
    calibration.length !== 2 ||
    [...calibration, magnification].some((v) => !Number.isFinite(v) || v <= 0)
  ) {
    throw Error(
      "Supply positive reference magnification, reference FOV, and magnification.",
    );
  }
  const width = calibration[1] * (calibration[0] / magnification);
  if (!Number.isFinite(width) || width <= 0)
    throw Error("Field of view is outside the supported numeric range.");
  return width;
}

/** A readable 1/2/5 scale bar no longer than one quarter of the field width. */
export function scaleBarWidth(span: number): number {
  const power = 10 ** Math.floor(Math.log10(span / 4));
  return [5, 2, 1].find((v) => v * power <= span / 4)! * power;
}

/** Orthogonal side cameras in the beam frame, not crystallographic plane normals. */
export function orthogonalBases(basis: ReturnType<typeof projectionBasis>) {
  return [
    basis,
    { right: basis.right, up: basis.beam, beam: basis.up.map((v) => -v) as V3 },
    { right: basis.up, up: basis.beam, beam: basis.right },
  ];
}

/** Rigid specimen tilt about its center, expressed as inverse camera bases.
 * Positive row/column makes the nominal beam column lean down/right with depth.
 * The rotation vector is row*right + column*up (mrad); not an Euler shear.
 * Dotting original positions with these bases applies the same rigid rotation
 * to atoms, box corners, picking, clipping and the GPU potential geometry.
 */
export function specimenTiltBasis(
  basis: ReturnType<typeof projectionBasis>,
  tiltMrad: number[],
): ReturnType<typeof projectionBasis> {
  if (
    tiltMrad.length !== 2 ||
    tiltMrad.some((v) => !Number.isFinite(v) || Math.abs(v) > 15)
  )
    throw Error("Use row and column specimen tilts between −15 and +15 mrad.");
  const omega = basis.right.map(
    (v, i) => (tiltMrad[0] * v + tiltMrad[1] * basis.up[i]) / 1000,
  );
  const angle = Math.hypot(...omega);
  if (angle === 0) return basis;
  const axis = omega.map((v) => v / angle);
  const rotateInverse = (v: V3): V3 => {
    const c = Math.cos(angle),
      s = Math.sin(angle),
      av = dot(axis, v);
    const axv = cross(axis, v);
    return v.map((x, i) => x * c - axv[i] * s + axis[i] * av * (1 - c)) as V3;
  };
  return {
    right: rotateInverse(basis.right),
    up: rotateInverse(basis.up),
    beam: rotateInverse(basis.beam),
  };
}
