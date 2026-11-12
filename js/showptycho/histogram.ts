/** Display-only bar heights; contrast coordinates and image values stay unchanged. */
export function histogramBarHeights(
  bins: readonly number[],
  columns: number,
  scale: "linear" | "log",
  finiteCount: number,
): number[] {
  const counts = new Array<number>(columns).fill(0);
  for (let i = 0; i < bins.length; i++) {
    counts[Math.min(columns - 1, Math.floor(i * columns / bins.length))] += bins[i];
  }
  // Cached display bins are peak-normalized. Recover pixel counts without
  // scanning the image again; finiteCount comes from the existing frame stats.
  if (scale === "log") {
    const sum = counts.reduce((total, count) => total + count, 0);
    const factor = sum > 0 ? finiteCount / sum : 0;
    for (let i = 0; i < columns; i++) counts[i] = Math.log1p(counts[i] * factor);
  }
  const peak = Math.max(...counts);
  return peak > 0 ? counts.map(count => count / peak) : counts;
}
