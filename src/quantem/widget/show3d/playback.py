"""Frame transport and per-frame services for Show3D playback.

The browser plays back from one embedded float32 display stack. This module
builds that stack, answers frame requests from the Python side (saved
previews, GIF rendering, ROI plots) and keeps the auto-contrast ranges the
browser uses stable across the whole stack.
"""

import math
import threading

import numpy as np
import torch

# Rec. 709 luminance weights: a true-color stack keeps RGB for the browser and
# one luminance plane for stats, FFT, ROI and scale bars.
_RGB_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _quantize_u8(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map ``[low, high]`` linearly onto 0..255 and truncate to uint8; a flat range maps to 0.

    The browser undoes it with the same ``low`` and ``high``, which travel with
    the bytes, so a uint8 export keeps each panel's own contrast.
    """
    return np.clip((values - low) * (255.0 / (high - low if high > low else 1.0)), 0, 255).astype(np.uint8)


class Show3DPlayback:
    """Display-stack packing, frame access and contrast ranges."""

    def _establish_rgb_luminance(self, data: np.ndarray, *, is_rgb: bool) -> np.ndarray:
        """Store the color stack and return the luminance plane that drives stats.

        True color lives in ``_rgb_data`` for the browser to paint; a Rec. 709
        luminance plane in ``_data`` feeds stats, FFT and ROI. Both the
        constructor and ``set_image`` route through here so a live color update
        cannot leave ``_rgb_data`` stale.
        """
        if is_rgb:
            self._rgb_data = data
            return np.tensordot(data, _RGB_LUMA, axes=([-1], [0])).astype(np.float32)
        self._rgb_data = None
        return data

    def _offline_stack_source(self) -> np.ndarray:
        """The stack the browser indexes per frame: RGB (N, H, W, 3) or the display plane."""
        if self._display_data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        if self.is_rgb and self._rgb_data is not None:
            return np.ascontiguousarray(self._rgb_data, dtype=np.float32)
        return np.ascontiguousarray(self._display_data, dtype=np.float32)

    def _pack_exact_offline_stack(self) -> None:
        """Embed the display stack as exact float32 bytes.

        This is the one transport: every path that replaces the stack must
        repack it, or frame indices past the old end render blank.
        """
        arr = self._offline_stack_source()
        self._offline_min = float(arr.min()) if arr.size else 0.0
        self._offline_max = float(arr.max()) if arr.size else 1.0
        self._offline_mins = []
        self._offline_maxs = []
        self._offline_stack = b""
        self._offline_float_stack = arr.tobytes()

    def _pack_offline_u8_stack(self) -> None:
        """Quantize the display stack to uint8 for compact standalone HTML.

        Each panel is scaled over its own range so a dim panel beside a bright
        one keeps its contrast; the per-panel ranges travel in
        ``_offline_mins`` / ``_offline_maxs``. RGB packs to uint8 color.
        """
        arr = self._offline_stack_source()
        self._offline_float_stack = b""
        self._offline_mins = []
        self._offline_maxs = []
        if self.is_rgb and arr.ndim == 4:
            self._offline_min = 0.0
            self._offline_max = 1.0
            self._offline_stack = np.clip(arr * 255.0, 0, 255).astype(np.uint8).tobytes()
            return
        low = float(arr.min()) if arr.size else 0.0
        high = float(arr.max()) if arr.size else 1.0
        self._offline_min = low
        self._offline_max = high
        panel_count = int(self.n_panels)
        panel_width = int(self.panel_width_px)
        if panel_count > 1 and panel_width > 0 and arr.shape[2] == panel_width * panel_count:
            panels = [arr[:, :, panel * panel_width : (panel + 1) * panel_width] for panel in range(panel_count)]
            panel_lows = [float(panel_values.min()) for panel_values in panels]
            panel_highs = [float(panel_values.max()) for panel_values in panels]
            self._offline_mins = panel_lows
            self._offline_maxs = panel_highs
            quantized = [_quantize_u8(panel_values, panel_low, panel_high)
                         for panel_values, panel_low, panel_high in zip(panels, panel_lows, panel_highs)]
            self._offline_stack = np.concatenate(quantized, axis=2).tobytes()
            return
        self._offline_stack = _quantize_u8(arr, low, high).tobytes()

    def _get_display_frame(self, idx: int | None = None) -> np.ndarray:
        """The display-resolution frame the browser shows, with diff mode applied."""
        idx = self.slice_idx if idx is None else idx
        data = self._display_data
        frame = data[idx]
        if self.diff_mode == "previous":
            return np.zeros_like(frame) if idx == 0 else frame - data[idx - 1]
        if self.diff_mode == "first":
            return frame - data[0]
        return frame

    def _get_display_panel_frame(self, panel: int, idx: int) -> np.ndarray:
        """One panel's slice of the display frame, as the browser crops it from the wide stack."""
        frame = self._get_display_frame(idx)
        panel_width = int(self.panel_width_px) or frame.shape[1] // max(1, int(self.n_panels))
        return frame[:, panel * panel_width : (panel + 1) * panel_width]

    def _get_source_panel_frame(self, panel: int, idx: int) -> np.ndarray:
        """One native-resolution panel frame for GIF rendering."""
        if self._source_panels:
            return np.asarray(self._source_panels[panel][idx])
        frame = np.asarray(self._data[idx])
        if self.shared_panel_source or int(self.n_panels) <= 1:
            return frame
        panel_width = frame.shape[1] // int(self.n_panels)
        return frame[:, panel * panel_width : (panel + 1) * panel_width]

    def _refresh_auto_contrast_ranges(self) -> None:
        """Precompute one percentile range per stack and per independent panel.

        A panel may be a different physical quantity from its neighbor (SSB
        phase beside dark-field counts), so each independent panel gets its own
        range over its whole time series. Ranges come from a 1024-bin
        histogram accumulated frame by frame so the stack is never joined.
        """
        bins = 1024

        def stack_percentiles(frames: list[np.ndarray]) -> tuple[float, float] | None:
            """``(low, high)`` at the percentile sliders from one histogram over every frame; None when non-finite.

            Bin ``b`` of ``bins`` maps back to ``minimum + b / (bins - 1) * (maximum - minimum)``.
            """
            minimum = min(float(np.min(frame)) for frame in frames)
            maximum = max(float(np.max(frame)) for frame in frames)
            total = sum(int(frame.size) for frame in frames)
            if total <= 0 or not math.isfinite(minimum) or not math.isfinite(maximum):
                return None
            if minimum == maximum:
                return minimum, maximum
            counts = np.zeros(bins, dtype=np.int64)
            for frame in frames:
                counts += np.histogram(frame, bins=bins, range=(minimum, maximum))[0]
            cumulative = np.cumsum(counts)
            low_bin = int(np.searchsorted(cumulative, int(total * self.percentile_low / 100.0), side="left"))
            high_bin = int(np.searchsorted(cumulative, int(np.ceil(total * self.percentile_high / 100.0)), side="left"))
            last_bin = bins - 1
            low_bin = max(0, min(last_bin, low_bin))
            high_bin = max(0, min(last_bin, high_bin))
            return (minimum + (low_bin / last_bin) * (maximum - minimum),
                    minimum + (high_bin / last_bin) * (maximum - minimum))

        stack_range = None
        if self.n_slices > 0 and self.auto_contrast:
            stack_range = stack_percentiles([np.asarray(self._get_display_frame(i)) for i in range(int(self.n_slices))])
        if stack_range is None:
            self.auto_vmins = []
            self.auto_vmaxs = []
            self.auto_vmins_per_panel = []
            self.auto_vmaxs_per_panel = []
            return
        panel_vmins: list[float] = []
        panel_vmaxs: list[float] = []
        if int(self.n_panels) > 1 and not self.shared_panel_source:
            for panel in range(int(self.n_panels)):
                panel_range = stack_percentiles(
                    [np.asarray(self._get_display_panel_frame(panel, i)) for i in range(int(self.n_slices))]
                )
                if panel_range is not None:
                    panel_vmins.append(panel_range[0])
                    panel_vmaxs.append(panel_range[1])
        with self.hold_sync():
            self.auto_vmins = [stack_range[0]] * int(self.n_slices)
            self.auto_vmaxs = [stack_range[1]] * int(self.n_slices)
            self.auto_vmins_per_panel = panel_vmins
            self.auto_vmaxs_per_panel = panel_vmaxs

    def _on_diff_mode_change(self, change: dict | None = None) -> None:
        """Recompute a zero-centered data range so diff colormaps pin black at 0."""
        data = self._display_data
        if self.diff_mode == "off":
            # Restore the constructor's full-resolution range so toggling
            # Off > Previous > Off is idempotent.
            self.data_min = float(self._data_min_off)
            self.data_max = float(self._data_max_off)
        elif self.n_slices < 2:
            self.data_min = 0.0
            self.data_max = 0.0
        else:
            diffs = data[1:] - data[:-1] if self.diff_mode == "previous" else data[1:] - data[0:1]
            self.data_min = min(0.0, float(diffs.min()))
            self.data_max = max(0.0, float(diffs.max()))
        self._refresh_auto_contrast_ranges()

    def _on_roi_change(self, change: dict | None = None) -> None:
        """Debounce the all-frame ROI plot so a drag does not freeze the kernel."""
        if int(self.n_panels) > 1 or not self.roi_active:
            self.roi_plot_data = b""
            return
        if self.roi_list and self.roi_selected_idx < 0:
            self.roi_selected_idx = 0
        if self._roi_plot_timer is not None:
            self._roi_plot_timer.cancel()
        self._roi_plot_timer = threading.Timer(0.5, self._compute_roi_plot)
        self._roi_plot_timer.start()

    def _compute_roi_plot(self) -> None:
        """Mean of the selected ROI on every display frame, as float32 bytes.

        The mask is built at display resolution so stats match what is drawn.
        """
        idx = self.roi_selected_idx
        if idx < 0 or idx >= len(self.roi_list):
            self.roi_plot_data = b""
            return
        # Square and rectangle edges use strict inequalities so the mask matches
        # the browser's strokeRect outline exactly.
        roi = self.roi_list[idx]
        rows, cols = np.ogrid[0 : self.height, 0 : self.width]
        row, col = float(roi.get("row", 0)), float(roi.get("col", 0))
        radius = max(1.0, float(roi.get("radius", 10)))
        shape = roi.get("shape", "circle")
        dist2 = (cols - col) ** 2 + (rows - row) ** 2
        if shape == "square":
            mask = (np.abs(cols - col) < radius) & (np.abs(rows - row) < radius)
        elif shape == "rectangle":
            half_width = max(1.0, float(roi.get("width", 20)) / 2.0)
            half_height = max(1.0, float(roi.get("height", 20)) / 2.0)
            mask = (np.abs(cols - col) < half_width) & (np.abs(rows - row) < half_height)
        elif shape == "annular":
            mask = (dist2 >= max(0.0, float(roi.get("radius_inner", 5))) ** 2) & (dist2 <= radius**2)
        else:
            mask = dist2 <= radius**2
        if not mask.any():
            self.roi_plot_data = b""
            return
        data_t = torch.from_numpy(np.ascontiguousarray(self._display_data))
        if self.diff_mode == "previous":
            data_t = torch.cat([torch.zeros_like(data_t[:1]), data_t[1:] - data_t[:-1]])
        elif self.diff_mode == "first":
            data_t = data_t - data_t[0:1]
        means_t = data_t[:, torch.from_numpy(mask)].mean(dim=1)
        self.roi_plot_data = means_t.numpy().tobytes()
