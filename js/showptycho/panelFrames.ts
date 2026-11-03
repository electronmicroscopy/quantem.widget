import { ValueStore } from "../shared/valueStore";

/** The frame a ShowPtycho panel shows: its scalar data, finite range and histogram bins. */
export type PanelFrame = {
  data: Float32Array | null;
  min: number;
  max: number;
  displayBins: number[] | null;
};

const EMPTY: PanelFrame = { data: null, min: 0, max: 1, displayBins: null };

/**
 * The painted frame of each panel slot (phase, FFT, amplitude), outside React state: a new
 * phase repaints its canvas and re-renders only what reads the frame (the histograms).
 */
export function createPanelFrames(): ValueStore<PanelFrame>[] {
  return [new ValueStore(EMPTY), new ValueStore(EMPTY), new ValueStore(EMPTY)];
}
