import { formatSavedBytes } from "../shared/exportFormat";

/**
 * One kernel message above this never reaches the browser: the websocket drops
 * it and the widget keeps only its static preview. Same value as
 * BROWSER_MESSAGE_LIMIT_BYTES in src/quantem/widget/state.py.
 */
export const BROWSER_MESSAGE_LIMIT_BYTES = 2 * 1024 ** 3;

/**
 * Status line for a Show3D whose embedded display stack did not arrive.
 *
 * The usual cause is a stack larger than one browser message. expectedBytes is
 * the payload already binned by displayBin, so the native size is
 * expectedBytes * displayBin^2. The display_bin named here is the one the
 * notebook printed: the smallest factor above displayBin whose payload fits,
 * the rule announce_browser_limit uses. WebGPU is not the cause, so the line
 * does not mention it.
 */
export function missingDisplayStackStatus(expectedBytes: number, displayBin = 1): string {
  if (expectedBytes <= BROWSER_MESSAGE_LIMIT_BYTES) {
    return `The ${formatSavedBytes(expectedBytes)} display stack did not reach the browser`;
  }
  const nativeBytes = expectedBytes * displayBin ** 2;
  let factor = displayBin + 1;
  while (nativeBytes / factor ** 2 > BROWSER_MESSAGE_LIMIT_BYTES) factor += 1;
  return `The ${formatSavedBytes(expectedBytes)} display stack did not reach the browser: one browser message holds at most `
    + `${formatSavedBytes(BROWSER_MESSAGE_LIMIT_BYTES)}. Pass display_bin=${factor} `
    + `(${formatSavedBytes(nativeBytes / factor ** 2)}) for an interactive view, as printed in the notebook`;
}
