/**
 * Weickenmeier-Kohl electron scattering factors with absorption, for CIFs
 * loaded in the browser, so that their Bloch couplings match the preset
 * structures, which quantem computes in Python.
 *
 * The elastic part is the 8-parameter fit of Weickenmeier & Kohl, Acta Cryst.
 * A 47, 590-597 (1991), https://doi.org/10.1107/S0108767391004774, damped by
 * the Debye-Waller factor; the absorptive part (core loss and phonon / thermal
 * diffuse scattering) is computed analytically from the same fit. This is a
 * line-by-line port of compute_WK_factor in
 * quantem/diffraction/wk_scattering_factors.py (electronmicroscopy/quantem, MIT
 * license), which follows Weickenmeier's original code as adapted by De Graef
 * in EMsoftLib/others.f90; the parameter tables are copied from that module.
 * Function and variable names follow the Fortran original.
 */

// Fit parameters A_i and B_i of Weickenmeier & Kohl (1991), 4 per element, H (Z = 1) to Cf (Z = 98).
const WK_A: number[][] = [
  [0.00427, 0.00957, 0.00802, 0.00209],
  [0.01217, 0.02616, -0.00884, 0.01841],
  [0.00251, 0.03576, 0.00988, 0.0237],
  [0.01596, 0.02959, 0.04024, 0.01001],
  [0.03652, 0.0114, 0.05677, 0.01506],
  [0.04102, 0.04911, 0.05296, 0.00061],
  [0.04123, 0.0574, 0.06529, 0.00373],
  [0.03547, 0.03133, 0.10865, 0.01615],
  [0.03957, 0.07225, 0.09581, 0.00792],
  [0.02597, 0.02197, 0.13762, 0.05394],
  [0.03283, 0.08858, 0.11688, 0.02516],
  [0.03833, 0.17124, 0.03649, 0.04134],
  [0.04388, 0.17743, 0.05047, 0.03957],
  [0.03812, 0.17833, 0.0628, 0.05605],
  [0.04166, 0.17817, 0.09479, 0.04463],
  [0.04003, 0.18346, 0.12218, 0.03753],
  [0.04245, 0.17645, 0.15814, 0.03011],
  [0.05011, 0.16667, 0.17074, 0.04358],
  [0.04058, 0.17582, 0.20943, 0.02922],
  [0.04001, 0.17416, 0.20986, 0.05497],
  [0.09685, 0.14777, 0.20981, 0.04852],
  [0.06667, 0.17356, 0.2271, 0.05957],
  [0.05118, 0.16791, 0.267, 0.06476],
  [0.03204, 0.1846, 0.30764, 0.05052],
  [0.03866, 0.17782, 0.31329, 0.06898],
  [0.05455, 0.1666, 0.33208, 0.06947],
  [0.05942, 0.17472, 0.34423, 0.06828],
  [0.06049, 0.166, 0.37302, 0.07109],
  [0.08034, 0.15838, 0.40116, 0.05467],
  [0.02948, 0.192, 0.42222, 0.0748],
  [0.16157, 0.32976, 0.18964, 0.06148],
  [0.16184, 0.35705, 0.17618, 0.07133],
  [0.0619, 0.18452, 0.416, 0.12793],
  [0.15913, 0.41583, 0.13385, 0.10549],
  [0.16514, 0.41202, 0.129, 0.13209],
  [0.15798, 0.41181, 0.14254, 0.14987],
  [0.16535, 0.44674, 0.24245, 0.03161],
  [0.16039, 0.4447, 0.24661, 0.0584],
  [0.16619, 0.44376, 0.25613, 0.06797],
  [0.16794, 0.44505, 0.27188, 0.07313],
  [0.16552, 0.45008, 0.30474, 0.06161],
  [0.17327, 0.44679, 0.32441, 0.06143],
  [0.16424, 0.45046, 0.33749, 0.07766],
  [0.1875, 0.44919, 0.36323, 0.05388],
  [0.16081, 0.45211, 0.40343, 0.0614],
  [0.16599, 0.43951, 0.41478, 0.08142],
  [0.16547, 0.44658, 0.45401, 0.05959],
  [0.17154, 0.43689, 0.46392, 0.07725],
  [0.15752, 0.44821, 0.48186, 0.08596],
  [0.15732, 0.44563, 0.48507, 0.10948],
  [0.16971, 0.42742, 0.48779, 0.13653],
  [0.14927, 0.43729, 0.49444, 0.1644],
  [0.18053, 0.44724, 0.48163, 0.15995],
  [0.13141, 0.43855, 0.50035, 0.22299],
  [0.31397, 0.55648, 0.39828, 0.04852],
  [0.32756, 0.53927, 0.3983, 0.07607],
  [0.30887, 0.53804, 0.42265, 0.09559],
  [0.28398, 0.53568, 0.46662, 0.10282],
  [0.3516, 0.56889, 0.4201, 0.07246],
  [0.3381, 0.58035, 0.44442, 0.07413],
  [0.35449, 0.59626, 0.43868, 0.07152],
  [0.35559, 0.60598, 0.45165, 0.07168],
  [0.38379, 0.64088, 0.4171, 0.06708],
  [0.40352, 0.64303, 0.40488, 0.08137],
  [0.36838, 0.64761, 0.47222, 0.06854],
  [0.38514, 0.68422, 0.44359, 0.06775],
  [0.3728, 0.67528, 0.47337, 0.0832],
  [0.39335, 0.70093, 0.46774, 0.06658],
  [0.40587, 0.71223, 0.46598, 0.06847],
  [0.39728, 0.73368, 0.47795, 0.06759],
  [0.40697, 0.73576, 0.47481, 0.08291],
  [0.40122, 0.78861, 0.44658, 0.08799],
  [0.41127, 0.76965, 0.46563, 0.1018],
  [0.39978, 0.77171, 0.48541, 0.1154],
  [0.3913, 0.80752, 0.48702, 0.11041],
  [0.40436, 0.80701, 0.48445, 0.12438],
  [0.38816, 0.80163, 0.51922, 0.13514],
  [0.39551, 0.80409, 0.53365, 0.13485],
  [0.4085, 0.83052, 0.53325, 0.11978],
  [0.40092, 0.85415, 0.53346, 0.12747],
  [0.41872, 0.88168, 0.54551, 0.09404],
  [0.43358, 0.88007, 0.52966, 0.12059],
  [0.40858, 0.87837, 0.56392, 0.13698],
  [0.41637, 0.85094, 0.57749, 0.167],
  [0.38951, 0.83297, 0.60557, 0.2077],
  [0.41677, 0.88094, 0.5517, 0.21029],
  [0.50089, 1.0086, 0.5142, 0.05996],
  [0.4747, 0.99363, 0.54721, 0.09206],
  [0.4781, 0.98385, 0.54905, 0.12055],
  [0.47903, 0.97455, 0.55883, 0.14309],
  [0.48351, 0.98292, 0.58877, 0.12425],
  [0.48664, 0.98057, 0.61483, 0.12136],
  [0.46078, 0.97139, 0.66506, 0.13012],
  [0.49148, 0.98583, 0.67674, 0.09725],
  [0.50865, 0.98574, 0.68109, 0.09977],
  [0.46259, 0.97882, 0.73056, 0.12723],
  [0.46221, 0.95749, 0.76259, 0.14086],
  [0.485, 0.95602, 0.77234, 0.13374],
];
const WK_B: number[][] = [
  [4.17218, 16.05892, 26.78365, 69.45643],
  [1.83008, 7.20225, 16.13585, 18.75551],
  [0.0262, 2.00907, 10.80597, 130.49226],
  [0.38968, 1.99268, 46.86913, 108.84167],
  [0.50627, 3.68297, 27.90586, 74.98296],
  [0.41335, 10.98289, 34.80286, 177.19113],
  [0.29792, 7.84094, 22.58809, 72.59254],
  [0.17964, 2.60856, 11.79972, 38.02912],
  [0.16403, 3.96612, 12.43903, 40.05053],
  [0.09101, 0.41253, 5.02463, 17.52954],
  [0.06008, 2.07182, 7.64444, 146.00952],
  [0.07424, 2.87177, 18.06729, 97.00854],
  [0.09086, 2.53252, 30.43883, 98.26737],
  [0.05396, 1.86461, 22.54263, 72.43144],
  [0.05564, 1.625, 24.45354, 64.38264],
  [0.05214, 1.40793, 23.35691, 53.59676],
  [0.04643, 1.15677, 19.34091, 52.88785],
  [0.07991, 1.01436, 15.67109, 39.60819],
  [0.03352, 0.82984, 14.13679, 200.97722],
  [0.02289, 0.71288, 11.18914, 135.0239],
  [0.12527, 1.34248, 12.43524, 131.71112],
  [0.05198, 0.86467, 10.59984, 103.56776],
  [0.03786, 0.5716, 8.30305, 91.78068],
  [0.0024, 0.44931, 7.92251, 86.64058],
  [0.01836, 0.41203, 6.73736, 76.30466],
  [0.03947, 0.43294, 6.26864, 71.2947],
  [0.03962, 0.43253, 6.05175, 68.72437],
  [0.03558, 0.39976, 5.3666, 62.46894],
  [0.05475, 0.45736, 5.38252, 60.43276],
  [0.00137, 0.26535, 4.4804, 54.26088],
  [0.10455, 2.18391, 9.04125, 75.16958],
  [0.0989, 2.06856, 9.89926, 68.13783],
  [0.01642, 0.32542, 3.51888, 44.50604],
  [0.07669, 1.89297, 11.31554, 46.32082],
  [0.08199, 1.76568, 9.87254, 38.1064],
  [0.06939, 1.53446, 8.98025, 33.04365],
  [0.07044, 1.59236, 17.53592, 215.26198],
  [0.06199, 1.41265, 14.33812, 152.80257],
  [0.06364, 1.34205, 13.66551, 125.72522],
  [0.06565, 1.25292, 13.09355, 109.50252],
  [0.05921, 1.15624, 13.24924, 98.69958],
  [0.06162, 1.11236, 12.76149, 90.92026],
  [0.05081, 0.99771, 11.28925, 84.28943],
  [0.0512, 1.08672, 12.23172, 85.27316],
  [0.04662, 0.85252, 10.51121, 74.53949],
  [0.04933, 0.79381, 9.30944, 41.17414],
  [0.04481, 0.75608, 9.34354, 67.91975],
  [0.04867, 0.71518, 8.40595, 64.244],
  [0.03672, 0.64379, 7.83687, 73.37281],
  [0.03308, 0.60931, 7.04977, 64.83582],
  [0.04023, 0.58192, 6.29247, 55.57061],
  [0.02842, 0.50687, 5.60835, 48.28004],
  [0.0383, 0.5834, 6.4755, 47.0882],
  [0.02097, 0.41007, 4.52105, 37.18178],
  [0.07813, 1.45053, 15.05933, 199.4883],
  [0.08444, 1.40227, 13.12939, 160.56676],
  [0.07206, 1.19585, 11.55866, 127.31371],
  [0.05717, 0.98756, 9.95556, 117.31874],
  [0.08249, 1.43427, 12.37363, 150.55968],
  [0.07081, 1.31033, 11.44403, 144.17706],
  [0.07442, 1.3868, 11.54391, 143.72185],
  [0.07155, 1.34703, 11.00432, 140.09138],
  [0.07794, 1.55042, 11.89283, 142.79585],
  [0.08508, 1.60712, 11.45367, 116.64063],
  [0.0652, 1.32571, 10.16884, 134.69034],
  [0.0685, 1.43566, 10.57719, 131.88972],
  [0.06264, 1.26756, 9.46411, 107.50194],
  [0.0675, 1.35829, 9.7648, 127.40374],
  [0.06958, 1.3875, 9.41888, 122.1094],
  [0.06574, 1.31578, 9.13448, 120.98209],
  [0.06517, 1.29452, 8.67569, 100.34878],
  [0.06213, 1.3086, 9.18871, 91.20213],
  [0.06292, 1.23499, 8.42904, 77.59815],
  [0.05693, 1.15762, 7.83077, 67.14066],
  [0.05145, 1.1124, 8.33441, 65.71782],
  [0.05573, 1.11159, 8.00221, 57.35021],
  [0.04855, 0.99356, 7.38693, 51.75829],
  [0.04981, 0.97669, 7.38024, 44.52068],
  [0.05151, 1.00803, 8.03707, 45.01758],
  [0.04693, 0.98398, 7.83562, 46.51474],
  [0.05161, 1.02127, 9.18455, 64.88177],
  [0.05154, 1.03252, 8.49678, 58.79463],
  [0.042, 0.90939, 7.71158, 57.79178],
  [0.04661, 0.87289, 6.84038, 51.36],
  [0.04168, 0.73697, 5.86112, 43.78613],
  [0.04488, 0.83871, 6.4402, 43.5194],
  [0.05786, 1.20028, 13.85073, 172.15909],
  [0.05239, 1.03225, 11.49796, 143.12303],
  [0.05167, 0.98867, 10.52682, 112.18267],
  [0.04931, 0.95698, 9.61135, 95.44649],
  [0.04748, 0.93369, 9.89867, 102.06961],
  [0.0466, 0.89912, 9.69785, 100.23434],
  [0.04323, 0.78798, 8.71624, 92.30811],
  [0.04641, 0.85867, 9.51157, 111.02754],
  [0.04918, 0.87026, 9.41105, 104.98576],
  [0.03904, 0.72797, 8.00506, 86.41747],
  [0.03969, 0.68167, 7.29607, 75.72682],
  [0.04291, 0.69956, 7.38554, 77.18528],
];
// x exp(-x) Ei(x) for 1/x = 0 to 0.1 in steps of 0.005 (Abramowitz and Stegun table 5.1), for RIH2.
const RIH2_TABLE = [1.0, 1.005051, 1.010206, 1.015472, 1.020852, 1.026355, 1.031985, 1.037751, 1.043662, 1.049726, 1.055956, 1.062364, 1.068965, 1.07578, 1.08283, 1.09014, 1.097737, 1.105647, 1.113894, 1.122497, 1.13147];

const EULER_GAMMA = 0.5772156649015329;

/** Exponential integral Ei(x) for x != 0, as scipy.special.expi. */
export function expi(x: number): number {
  if (x < -1) {
    // Ei(x) = -E1(-x), E1 by its continued fraction (modified Lentz)
    const y = -x;
    let b = y + 1;
    let c = 1e300;
    let d = 1 / b;
    let h = d;
    for (let i = 1; i < 300; i++) {
      const an = -i * i;
      b += 2;
      d = 1 / (an * d + b);
      c = b + an / c;
      const del = c * d;
      h *= del;
      if (Math.abs(del - 1) < 1e-16) break;
    }
    return -h * Math.exp(-y);
  }
  // Ei(x) = gamma + ln|x| + sum_k x^k / (k k!)
  let sum = 0;
  let term = 1;
  for (let k = 1; k < 300; k++) {
    term *= x / k;
    const add = term / k;
    sum += add;
    if (Math.abs(add) <= 1e-17 * Math.abs(sum)) break;
  }
  return EULER_GAMMA + Math.log(Math.abs(x)) + sum;
}

/** x exp(-x) Ei(x) for large x, interpolated from RIH2_TABLE. */
function RIH2(x: number): number {
  const idx = Math.floor(200 / x);
  return RIH2_TABLE[idx] + 200 * (RIH2_TABLE[idx + 1] - RIH2_TABLE[idx]) * (1 / x - 0.5e-3 * idx);
}

/** exp(-x1) (Ei(x2) - Ei(x3)). */
function RIH1(x1: number, x2: number, x3: number): number {
  const a = x2 <= 20 ? Math.exp(-x1) * expi(x2) : (Math.exp(x2 - x1) * RIH2(x2)) / x2;
  const b = x3 <= 20 ? Math.exp(-x1) * expi(x3) : (Math.exp(x3 - x1) * RIH2(x3)) / x3;
  return a - b;
}

/** First integral of the absorptive potential. */
function RI1(bi: number, bj: number, G: number): number {
  const G2 = G * G;
  if (Math.max(bi, bj) * G2 <= 0.1) {
    let ri1 = Math.PI * (bi * Math.log((bi + bj) / bi) + bj * Math.log((bi + bj) / bj));
    if (G > 0) {
      let temp = 0.5 * bi * bi * Math.log(bi / (bi + bj)) + 0.5 * bj * bj * Math.log(bj / (bi + bj));
      temp += 0.75 * (bi * bi + bj * bj) - 0.25 * (bi + bj) ** 2;
      temp -= 0.5 * (bi - bj) ** 2;
      ri1 += Math.PI * G2 * temp;
    }
    return ri1;
  }
  let ri1 = 2 * 0.5772157 + Math.log(bi * G2) + Math.log(bj * G2) - 2 * expi((-bi * bj * G2) / (bi + bj));
  ri1 += RIH1(bi * G2, (bi * G2 * bi) / (bi + bj), bi * G2);
  ri1 += RIH1(bj * G2, (bj * G2 * bj) / (bi + bj), bj * G2);
  return (ri1 * Math.PI) / G2;
}

/** Second integral of the absorptive potential, with RMS displacement u. */
function RI2(bi: number, bj: number, G: number, u: number): number {
  const U2 = u * u;
  const U22 = 0.5 * U2;
  const G2 = G * G;
  const BIUH = bi + 0.5 * U2;
  const BJUH = bj + 0.5 * U2;
  const BIU = bi + U2;
  const BJU = bj + U2;
  if (Math.max(bi, bj, U2) * G2 <= 0.1) {
    let ri2 = (bi + U2) * Math.log((bi + bj + U2) / (bi + U2)) + bj * Math.log((bi + bj + U2) / (bj + U2));
    if (U2 > 0) ri2 += U2 * Math.log(U2 / (bj + U2));
    ri2 *= Math.PI;
    let temp = U2 > 0 ? 0.5 * U22 * U22 * Math.log((BIU * BJU) / (U2 * U2)) : 0;
    temp += 0.5 * BIUH * BIUH * Math.log(BIU / (BIUH + BJUH));
    temp += 0.5 * BJUH * BJUH * Math.log(BJU / (BIUH + BJUH));
    temp += 0.25 * BIU * BIU + 0.5 * bi * bi;
    temp += 0.25 * BJU * BJU + 0.5 * bj * bj;
    temp -= 0.25 * (BIUH + BJUH) * (BIUH + BJUH);
    temp -= 0.5 * ((bi * BIU - bj * BJU) / (BIUH + BJUH)) ** 2;
    temp -= U22 * U22;
    return ri2 + Math.PI * G2 * temp;
  }
  let ri2 = expi((-0.5 * U2 * G2 * BIUH) / BIU) + expi((-0.5 * U2 * G2 * BJUH) / BJU);
  ri2 -= expi((-BIUH * BJUH * G2) / (BIUH + BJUH)) + expi(-0.25 * U2 * G2);
  ri2 *= 2;
  ri2 += RIH1(0.5 * U2 * G2, 0.25 * U2 * G2, (0.25 * U2 * U2 * G2) / BIU);
  ri2 += RIH1(0.5 * U2 * G2, 0.25 * U2 * G2, (0.25 * U2 * U2 * G2) / BJU);
  ri2 += RIH1(BIUH * G2, (BIUH * BIUH * G2) / (BIUH + BJUH), (BIUH * BIUH * G2) / BIU);
  ri2 += RIH1(BJUH * G2, (BJUH * BJUH * G2) / (BIUH + BJUH), (BJUH * BJUH * G2) / BJU);
  return (ri2 * Math.PI) / G2;
}

/** Electron wavelength in Angstrom, with the constants of quantem's electron_wavelength_angstrom. */
function wavelength(energyEv: number): number {
  const m = 9.109383e-31;
  const e = 1.602177e-19;
  const c = 299792458;
  const h = 6.62607e-34;
  return (h / Math.sqrt(2 * m * e * energyEv) / Math.sqrt(1 + (e * energyEv) / 2 / m / c ** 2)) * 1e10;
}

/** True for the atomic numbers with tabulated parameters, H (1) to Cf (98). */
export function hasWkFactors(Z: number): boolean {
  return Number.isInteger(Z) && Z >= 1 && Z <= WK_A.length;
}

/**
 * Absorptive, relativistically corrected scattering factor of one atom,
 * [real, imaginary] in Angstrom, at |g| = 1/d (1/Angstrom, no 2 pi).
 *
 * sigma is the RMS thermal displacement (Angstrom) of the Debye-Waller factor
 * and the phonon absorption; 0 means no thermal motion and no phonon term.
 * Both parts include core loss and phonon absorption, as compute_WK_factor
 * with include_core and include_phonon.
 */
export function wkFactor(g: number, Z: number, energyEv: number, sigma: number): [number, number] {
  if (!hasWkFactors(Z)) throw new Error(`no Weickenmeier-Kohl parameters for Z = ${Z}`);
  const A = WK_A[Z - 1];
  const B = WK_B[Z - 1];
  // the Fortran units: G = 2 pi g, S = G / (4 pi) = g / 2
  const G = 2 * Math.PI * g;
  const S = g / 2;
  const kV = energyEv / 1e3;
  const UL = sigma > 0 ? sigma : 0;
  const DWF = UL > 0 ? Math.exp(-0.5 * UL * UL * G * G) : 1;

  // WEKO(A, B, S)
  let WK = 0;
  for (let i = 0; i < 4; i++) {
    const argu = B[i] * S * S;
    if (argu < 0.1) WK += A[i] * B[i] * (1 - 0.5 * argu);
    else if (argu <= 20) WK += (A[i] * (1 - Math.exp(-argu))) / (S * S);
    else WK += A[i] / (S * S);
  }
  const Freal = 4 * Math.PI * DWF * WK;

  // core loss, following FCORE
  const k0 = (2 * Math.PI) / wavelength(energyEv);
  const DE = 6e-3 * Z;
  const thetaE = (DE / (2 * kV)) * ((2 * kV + 1022) / (kV + 1022));
  const R = (0.885 * 0.5289) / Z ** (1 / 3);
  const TA = 1 / (k0 * R);
  const TB = G / (2 * k0);
  const OMEGA = (2 * TB) / TA;
  const KAPPA = thetaE / TA;
  const K2 = KAPPA * KAPPA;
  const O2 = OMEGA * OMEGA;
  const X1 = (OMEGA / ((1 + O2) * Math.sqrt(O2 + 4 * K2))) * Math.log((OMEGA + Math.sqrt(O2 + 4 * K2)) / (2 * KAPPA));
  const root = Math.sqrt((1 + O2) * (1 + O2) + 4 * K2 * O2);
  const X2 = (1 / root) * Math.log((1 + 2 * K2 + O2 + root) / (2 * KAPPA * Math.sqrt(1 + K2)));
  const X3 =
    OMEGA > 1e-2
      ? (1 / (OMEGA * Math.sqrt(O2 + 4 * (1 + K2)))) * Math.log((OMEGA + Math.sqrt(O2 + 4 * (1 + K2))) / (2 * Math.sqrt(1 + K2)))
      : 1 / (4 * (1 + K2));
  const HI = ((2 * Z) / (TA * TA)) * (-X1 + X2 - X3);
  const A0 = 0.5289;
  const Fcore = (4 / (A0 * A0)) * ((2 * Math.PI) / (k0 * k0)) * HI;

  // phonon / thermal diffuse scattering, following FPHON(G, UL, A, B)
  let Fphon = 0;
  if (UL > 0) {
    const A1 = A.map((a) => a * (4 * Math.PI) ** 2);
    const B1 = B.map((b) => b / (4 * Math.PI) ** 2);
    for (let jj = 0; jj < 4; jj++) {
      for (let ii = 0; ii <= jj; ii++) {
        Fphon += (jj !== ii ? 2 : 1) * A1[jj] * A1[ii] * (DWF * RI1(B1[ii], B1[jj], G) - RI2(B1[ii], B1[jj], G, UL));
      }
    }
  }
  const Fimag = Fcore * DWF + Fphon;

  // relativistic correction, then Angstrom without the physicist factors (diffraction.f90)
  const gamma = (kV + 511) / 511;
  const scale = (0.4787801 * 0.664840340614319) / (4 * Math.PI);
  return [Freal * gamma * scale, ((Fimag * gamma * gamma) / k0) * scale];
}
