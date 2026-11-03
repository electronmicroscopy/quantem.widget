/** Page navigation of a paged gallery (Show2D, Show3D). */
import * as React from "react";

/**
 * The current page, a slider preview page, a requestAnimationFrame-batched page commit for slider
 * drags, page playback state and the label the page bar shows.
 */
export function usePageNavigation(
  nPages: number,
  pageIdx: number,
  setPageIdx: (idx: number) => void,
  pageLabels: string[] | undefined,
  isPaged: boolean,
  defaultPlayFps: number,
) {
  const currentPageIdx = Math.max(0, Math.min((nPages || 1) - 1, Math.round(pageIdx || 0)));
  const [pagePlaying, setPagePlaying] = React.useState(false);
  const [pagePlayFps, setPagePlayFps] = React.useState<number>(defaultPlayFps);
  const [pageSliderPreviewIdx, setPageSliderPreviewIdxState] = React.useState<number | null>(null);
  const pageSliderPreviewIdxRef = React.useRef<number | null>(null);
  const currentPageIdxRef = React.useRef(0);
  const pageCommitPendingRef = React.useRef<number | null>(null);
  const pageCommitRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    currentPageIdxRef.current = currentPageIdx;
  }, [currentPageIdx]);
  const clampPageIdx = React.useCallback((value: number) => (
    Math.max(0, Math.min((nPages || 1) - 1, Math.round(Number(value) || 0)))
  ), [nPages]);
  const setPageSliderPreviewIdx = React.useCallback((value: number | null) => {
    pageSliderPreviewIdxRef.current = value;
    setPageSliderPreviewIdxState(value);
  }, []);
  const commitPageIdx = React.useCallback((value: number, immediate = false) => {
    const next = clampPageIdx(value);
    pageCommitPendingRef.current = next;
    if (immediate) {
      if (pageCommitRafRef.current !== null) {
        window.cancelAnimationFrame(pageCommitRafRef.current);
        pageCommitRafRef.current = null;
      }
      pageCommitPendingRef.current = null;
      if (next !== currentPageIdxRef.current) setPageIdx(next);
      return;
    }
    if (pageCommitRafRef.current !== null) return;
    pageCommitRafRef.current = window.requestAnimationFrame(() => {
      pageCommitRafRef.current = null;
      const pending = pageCommitPendingRef.current;
      pageCommitPendingRef.current = null;
      if (pending !== null && pending !== currentPageIdxRef.current) setPageIdx(pending);
    });
  }, [clampPageIdx, setPageIdx]);
  const stopPagePlayback = React.useCallback(() => {
    setPagePlaying(value => value ? false : value);
  }, []);
  React.useEffect(() => {
    const preview = pageSliderPreviewIdxRef.current;
    if (preview !== null && preview === currentPageIdx) setPageSliderPreviewIdx(null);
  }, [currentPageIdx, setPageSliderPreviewIdx]);
  React.useEffect(() => () => {
    if (pageCommitRafRef.current !== null) {
      window.cancelAnimationFrame(pageCommitRafRef.current);
      pageCommitRafRef.current = null;
    }
  }, []);
  const pageControlIdx = clampPageIdx(pageSliderPreviewIdx ?? currentPageIdx);
  const pageControlLabel = pageLabels?.[pageControlIdx] || `Page ${pageControlIdx + 1}`;
  const pageControlStatus = `${pageControlLabel} ${pageControlIdx + 1}/${nPages || 1}`;
  React.useEffect(() => {
    if (!isPaged || (nPages || 1) <= 1) setPagePlaying(false);
  }, [isPaged, nPages]);
  return {
    currentPageIdx, pagePlaying, setPagePlaying, pagePlayFps, setPagePlayFps, setPageSliderPreviewIdx,
    clampPageIdx, commitPageIdx, stopPagePlayback, pageControlIdx, pageControlLabel, pageControlStatus,
  };
}
