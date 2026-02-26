"""
Align2DBulk: GPU-accelerated bulk alignment of N frames to a reference.

Aligns a stack of images via FFT phase correlation on GPU (MPS/CUDA),
crops to the common overlap region, and optionally bins the output.
Provides an interactive widget for visual verification.

When torch is available the entire pipeline runs on GPU — batched FFT,
DFT sub-pixel refinement, grid_sample shifts, avg_pool2d binning — with
a single transfer back to CPU at the end.
"""
import json
import pathlib

import anywidget
import numpy as np
import traitlets

from quantem.widget.align2d import _cross_correlate_fft, _compute_ncc
from quantem.widget.array_utils import to_numpy, bin2d, apply_shift
from quantem.widget.io import IOResult
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _select_device(device):
    """Pick the best available torch device."""
    if not _HAS_TORCH:
        return None
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── GPU pipeline ─────────────────────────────────────────────────────────

def _tukey_2d_torch(h, w, alpha, device):
    """2D Tukey window on a torch device."""
    def _t1d(n):
        if n <= 1:
            return torch.ones(n, device=device)
        x = torch.linspace(0, 1, n, device=device)
        win = torch.ones(n, device=device)
        left = x < alpha / 2
        right = x > 1 - alpha / 2
        win[left] = 0.5 * (1 + torch.cos(2 * torch.pi / alpha * (x[left] - alpha / 2)))
        win[right] = 0.5 * (1 + torch.cos(2 * torch.pi / alpha * (x[right] - 1 + alpha / 2)))
        return win
    return _t1d(h).unsqueeze(1) * _t1d(w).unsqueeze(0)


def _dft_upsample_torch(fa_conj_fb, peak_y, peak_x, device, upsample_factor=100, region=1.5):
    """Matrix DFT sub-pixel refinement on GPU (Guizar-Sicairos et al. 2008)."""
    h, w = fa_conj_fb.shape
    size = int(np.ceil(region * upsample_factor))
    offs = (torch.arange(size, device=device, dtype=torch.float32) - size // 2) / upsample_factor
    ups_y = peak_y + offs
    ups_x = peak_x + offs
    freq_y = torch.fft.fftfreq(h, device=device) * h
    freq_x = torch.fft.fftfreq(w, device=device) * w
    row_kernel = torch.exp(2j * torch.pi * ups_y[:, None] * freq_y[None, :] / h)
    col_kernel = torch.exp(2j * torch.pi * ups_x[:, None] * freq_x[None, :] / w)
    upsampled = (row_kernel @ fa_conj_fb @ col_kernel.T).real
    peak = torch.argmax(upsampled)
    return float(ups_y[int(peak // size)]), float(ups_x[int(peak % size)])


def _align_stack_torch(frames_np, reference, max_shift, bin_factor, device):
    """
    Full GPU alignment pipeline — upload once, download once.

    1. Batched FFT + phase correlation (all frames at once)
    2. Per-frame DFT sub-pixel refinement (on GPU)
    3. Batched grid_sample for shift application
    4. Crop to common overlap on GPU
    5. Batched NCC on cropped aligned data (vectorized)
    6. avg_pool2d for binning on GPU
    7. Single .cpu().numpy() at the end
    """
    n, h, w = frames_np.shape
    stack = torch.as_tensor(frames_np, dtype=torch.float32, device=device)
    limit_x = max_shift if max_shift > 0 else int(w * 0.2)
    limit_y = max_shift if max_shift > 0 else int(h * 0.2)
    # Tukey window — computed once, reused for all frames
    win = _tukey_2d_torch(h, w, 0.2, device)
    # Search mask — computed once
    search_mask = None
    if limit_x > 0 or limit_y > 0:
        msy = min(limit_y, h // 2) if limit_y > 0 else h // 2
        msx = min(limit_x, w // 2) if limit_x > 0 else w // 2
        vy = torch.zeros(h, dtype=torch.bool, device=device)
        vy[:msy + 1] = True
        vy[max(h - msy, 0):] = True
        vx = torch.zeros(w, dtype=torch.bool, device=device)
        vx[:msx + 1] = True
        vx[max(w - msx, 0):] = True
        search_mask = vy.unsqueeze(1) & vx.unsqueeze(0)
    # ── Batched FFT (single call for all frames) ──────────────────
    stack_win = (stack - stack.mean(dim=(-2, -1), keepdim=True)) * win
    fa_all = torch.fft.fft2(stack_win)
    del stack_win
    fa_ref = fa_all[reference]
    # ── Per-frame cross-correlation + DFT refinement ─────────────
    # Process one frame at a time to avoid N×H×W complex intermediates
    offsets = [(0.0, 0.0)] * n
    for i in range(n):
        if i == reference:
            continue
        cp = fa_ref * fa_all[i].conj()
        cp_norm = cp / (cp.abs() + 1e-10)
        xcorr = torch.fft.ifft2(cp_norm).real
        if search_mask is not None:
            xcorr = xcorr.masked_fill(~search_mask, -torch.inf)
        peak_idx = xcorr.view(-1).argmax()
        peak_y = int(peak_idx // w)
        peak_x = int(peak_idx % w)
        sub_y, sub_x = _dft_upsample_torch(cp, peak_y, peak_x, device)
        dy = float(sub_y if sub_y <= h / 2 else sub_y - h)
        dx = float(sub_x if sub_x <= w / 2 else sub_x - w)
        offsets[i] = (
            max(-limit_x, min(limit_x, dx)),
            max(-limit_y, min(limit_y, dy)),
        )
    del fa_all
    # ── Batched grid_sample ──────────────────────────────────────────
    base_y = torch.linspace(-1, 1, h, device=device)
    base_x = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(base_y, base_x, indexing="ij")
    base_grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)
    # Build per-frame shift vectors (vectorized, no Python loop)
    shift_vals = torch.tensor(offsets, dtype=torch.float32, device=device)  # (N, 2)
    dx_norm = shift_vals[:, 0] * 2.0 / w  # (N,)
    dy_norm = shift_vals[:, 1] * 2.0 / h  # (N,)
    grids = base_grid.unsqueeze(0).expand(n, -1, -1, -1).clone()
    grids[:, :, :, 0] -= dx_norm.view(n, 1, 1)
    grids[:, :, :, 1] -= dy_norm.view(n, 1, 1)
    aligned = F.grid_sample(
        stack.unsqueeze(1), grids, mode="bilinear", padding_mode="zeros", align_corners=True,
    ).squeeze(1)
    # ── Crop to common overlap on GPU ────────────────────────────────
    all_dx = [o[0] for o in offsets]
    all_dy = [o[1] for o in offsets]
    y0 = int(np.ceil(max(0, -min(all_dy))))
    y1 = int(np.floor(h - max(0, max(all_dy))))
    x0 = int(np.ceil(max(0, -min(all_dx))))
    x1 = int(np.floor(w - max(0, max(all_dx))))
    crop_box = None
    if y1 > y0 and x1 > x0:
        aligned = aligned[:, y0:y1, x0:x1]
        crop_box = (y0, y1, x0, x1)
    # ── Batched NCC on aligned data (vectorized, no interpolation) ──
    means = aligned.mean(dim=(-2, -1), keepdim=True)
    centered = aligned - means
    sq_sums = (centered ** 2).sum(dim=(-2, -1))
    ref_c = centered[reference]
    cross = (ref_c.unsqueeze(0) * centered).sum(dim=(-2, -1))
    denoms = torch.sqrt(sq_sums[reference] * sq_sums)
    ncc_tensor = cross / (denoms + 1e-10)
    nccs = ncc_tensor.tolist()
    # ── Bin on GPU ───────────────────────────────────────────────────
    if bin_factor > 1:
        _, ch, cw = aligned.shape
        oh, ow = ch // bin_factor, cw // bin_factor
        aligned = aligned[:, :oh * bin_factor, :ow * bin_factor]
        aligned = F.avg_pool2d(aligned.unsqueeze(1), kernel_size=bin_factor).squeeze(1)
    # ── Single transfer to CPU ───────────────────────────────────────
    return aligned.cpu().numpy(), offsets, nccs, crop_box


# ── CPU fallback (when torch is unavailable) ─────────────────────────────

def _align_stack_numpy(frames, reference, max_shift, bin_factor):
    """CPU fallback using numpy FFT and bilinear shifts."""
    n, h, w = frames.shape
    ref = frames[reference]
    limit_x = max_shift if max_shift > 0 else int(w * 0.2)
    limit_y = max_shift if max_shift > 0 else int(h * 0.2)
    offsets = [(0.0, 0.0)] * n
    nccs = [0.0] * n
    nccs[reference] = 1.0
    for i in range(n):
        if i == reference:
            continue
        dx, dy = _cross_correlate_fft(ref, frames[i], limit_x, limit_y)
        dx = max(-limit_x, min(limit_x, dx))
        dy = max(-limit_y, min(limit_y, dy))
        offsets[i] = (dx, dy)
        nccs[i] = _compute_ncc(ref, frames[i], dx, dy)
    aligned = np.empty_like(frames)
    for i in range(n):
        if i == reference:
            aligned[i] = frames[i]
        else:
            dx, dy = offsets[i]
            aligned[i] = apply_shift(frames[i], dy, dx)
    all_dx = [o[0] for o in offsets]
    all_dy = [o[1] for o in offsets]
    y0 = int(np.ceil(max(0, -min(all_dy))))
    y1 = int(np.floor(h - max(0, max(all_dy))))
    x0 = int(np.ceil(max(0, -min(all_dx))))
    x1 = int(np.floor(w - max(0, max(all_dx))))
    crop_box = None
    if y1 > y0 and x1 > x0:
        aligned = aligned[:, y0:y1, x0:x1]
        crop_box = (y0, y1, x0, x1)
    if bin_factor > 1:
        aligned = bin2d(aligned, bin_factor)
    return aligned.astype(np.float32), offsets, nccs, crop_box


# ── Dispatcher ───────────────────────────────────────────────────────────

def _align_stack(frames, reference, max_shift, bin_factor, device_str):
    """Route to GPU or CPU pipeline."""
    device = _select_device(device_str)
    if device is not None:
        return _align_stack_torch(frames, reference, max_shift, bin_factor, device)
    return _align_stack_numpy(frames, reference, max_shift, bin_factor)


# ── Widget ───────────────────────────────────────────────────────────────

class Align2DBulk(anywidget.AnyWidget):
    """
    GPU-accelerated bulk alignment of N frames to a reference.

    Parameters
    ----------
    images : array-like or IOResult
        3D stack ``(N, H, W)``, list of 2D arrays, or an ``IOResult``.
    reference : int, default 0
        Index of the reference frame.
    max_shift : int, default 0
        Maximum shift in pixels. When 0, defaults to 20% of image size.
    bin : int, default 1
        Spatial binning factor.
    roi : dict or tuple, optional
        Region of interest to crop before alignment. Accepts a
        ``Show3D.roi`` dict or a ``(row, col, height, width)`` tuple.
    cmap : str, default "gray"
        Colormap name.
    title : str, default ""
        Widget title.
    device : str or None, default None
        Torch device (``"cpu"``, ``"mps"``, ``"cuda"``). Auto-detected if None.

    Examples
    --------
    >>> viewer = Show3D(raw_stack)
    >>> viewer.add_roi(256, 256, shape="rectangle")
    >>> bulk = Align2DBulk(raw_stack, roi=viewer.roi, max_shift=20)
    >>> bulk.stack.shape
    (50, ...)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "align2d_bulk.js"

    # Metadata
    n_images = traitlets.Int(1).tag(sync=True)
    current_idx = traitlets.Int(0).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    raw_height = traitlets.Int(1).tag(sync=True)
    raw_width = traitlets.Int(1).tag(sync=True)

    # Image bytes: "after" panel (aligned ref + frame)
    ref_bytes = traitlets.Bytes(b"").tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Image bytes: "before" panel (raw frame with moving average)
    raw_frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Alignment results (JSON)
    offsets_json = traitlets.Unicode("[]").tag(sync=True)
    crop_box_json = traitlets.Unicode("").tag(sync=True)

    # Display
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("gray").tag(sync=True)
    opacity = traitlets.Float(0.5).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    avg_window = traitlets.Int(1).tag(sync=True)

    # Alignment parameters (visible in UI)
    bin_factor = traitlets.Int(1).tag(sync=True)
    max_shift_px = traitlets.Int(0).tag(sync=True)
    reference_idx = traitlets.Int(0).tag(sync=True)

    # Realign trigger
    _realign_requested = traitlets.Int(0).tag(sync=True)

    # Tool visibility / locking
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> list[str]:
        return normalize_tool_groups("Align2DBulk", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_navigation: bool = False,
        disable_stats: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
    ) -> list[str]:
        return build_tool_groups(
            "Align2DBulk",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "navigation": disable_navigation,
                "stats": disable_stats,
                "view": disable_view,
                "export": disable_export,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_navigation: bool = False,
        hide_stats: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
    ) -> list[str]:
        return build_tool_groups(
            "Align2DBulk",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "navigation": hide_navigation,
                "stats": hide_stats,
                "view": hide_view,
                "export": hide_export,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    def __init__(
        self,
        images,
        reference: int = 0,
        max_shift: int = 0,
        bin: int = 1,
        roi=None,
        cmap: str = "gray",
        title: str = "",
        device=None,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_navigation: bool = False,
        disable_stats: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_navigation: bool = False,
        hide_stats: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._device_str = device
        # Extract from IOResult
        if isinstance(images, IOResult):
            if not title and images.title:
                title = images.title
            images = images.data
        # List of 2D → 3D stack
        if isinstance(images, (list, tuple)):
            images = np.stack([to_numpy(img).astype(np.float32) for img in images])
        frames = to_numpy(images).astype(np.float32)
        # Apply ROI crop (from Show3D roi_list dict or (row, col, height, width) tuple)
        if roi is not None:
            if isinstance(roi, dict):
                r, c = roi["row"], roi["col"]
                h, w = roi["height"], roi["width"]
            else:
                r, c, h, w = roi
            r0 = r - h // 2
            c0 = c - w // 2
            frames = frames[:, r0:r0 + h, c0:c0 + w].copy()
        self._raw_frames = frames
        # Pre-cache raw frame bytes (avoids .tobytes() on every slider tick)
        self._raw_bytes_cache = [f.tobytes() for f in frames]
        # Run alignment (always crop to common overlap)
        self._aligned, self._offsets, self._nccs, self._crop_box = _align_stack(
            self._raw_frames, reference, max_shift, bin, device,
        )
        # Pre-cache aligned frame bytes
        self._aligned_bytes_cache = [f.tobytes() for f in self._aligned]
        # Set traits
        n, ah, aw = self._aligned.shape
        _, rh, rw = self._raw_frames.shape
        self.n_images = n
        self.height = ah
        self.width = aw
        self.raw_height = rh
        self.raw_width = rw
        self.title = title
        self.cmap = cmap
        self.bin_factor = bin
        self.max_shift_px = max_shift
        self.reference_idx = reference
        if self._crop_box:
            y0, y1, x0, x1 = self._crop_box
            self.crop_box_json = json.dumps({"y0": y0, "y1": y1, "x0": x0, "x1": x1})
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_navigation=disable_navigation,
            disable_stats=disable_stats,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_navigation=hide_navigation,
            hide_stats=hide_stats,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_all=hide_all,
        )
        self._update_offsets_json()
        self._send_ref()
        self._send_frame(0)
        self.observe(self._on_idx_change, names=["current_idx"])
        self.observe(self._on_realign, names=["_realign_requested"])
        self.observe(self._on_avg_window_change, names=["avg_window"])

    def _update_offsets_json(self):
        entries = []
        for i, ((dx, dy), ncc) in enumerate(zip(self._offsets, self._nccs)):
            entries.append({"idx": i, "dx": round(dx, 2), "dy": round(dy, 2), "ncc": round(ncc, 4)})
        self.offsets_json = json.dumps(entries)

    def _send_ref(self):
        self.ref_bytes = self._aligned_bytes_cache[self.reference_idx]

    def _send_frame(self, idx):
        with self.hold_sync():
            self.frame_bytes = self._aligned_bytes_cache[idx]
            self._send_raw_frame(idx)

    def _send_raw_frame(self, idx):
        w = min(self.avg_window, self.n_images)
        if w > 1:
            half = w // 2
            start = max(0, idx - half)
            end = min(self.n_images, start + w)
            start = max(0, end - w)
            self.raw_frame_bytes = self._raw_frames[start:end].mean(axis=0).astype(np.float32).tobytes()
        else:
            self.raw_frame_bytes = self._raw_bytes_cache[idx]

    def _on_idx_change(self, change):
        self._send_frame(change["new"])

    def _on_avg_window_change(self, change):
        self._send_raw_frame(self.current_idx)

    def _on_realign(self, change):
        self._aligned, self._offsets, self._nccs, self._crop_box = _align_stack(
            self._raw_frames,
            self.reference_idx,
            self.max_shift_px,
            self.bin_factor,
            self._device_str,
        )
        self._aligned_bytes_cache = [f.tobytes() for f in self._aligned]
        n, ah, aw = self._aligned.shape
        self.n_images = n
        self.height = ah
        self.width = aw
        self._update_offsets_json()
        self._send_ref()
        self._send_frame(self.current_idx)

    @property
    def stack(self) -> np.ndarray:
        """Aligned (+ cropped + binned) 3D array ``(N, H', W')``, float32."""
        return self._aligned

    @property
    def offsets(self) -> list[tuple[float, float]]:
        """List of ``(dx, dy)`` offsets per frame."""
        return list(self._offsets)

    @property
    def ncc(self) -> list[float]:
        """List of NCC values per frame."""
        return list(self._nccs)

    @property
    def crop_box(self):
        """``(y0, y1, x0, x1)`` crop region or None."""
        return self._crop_box

    def __repr__(self) -> str:
        n, h, w = self._aligned.shape
        dx = [o[0] for o in self._offsets]
        dy = [o[1] for o in self._offsets]
        ncc_min = min(self._nccs) if self._nccs else 0
        ncc_mean = sum(self._nccs) / len(self._nccs) if self._nccs else 0
        drift_max = max(max(abs(v) for v in dx), max(abs(v) for v in dy)) if dx else 0
        lines = [
            f"Align2DBulk({n} frames, {h}×{w}, ref={self.reference_idx})",
            f"  drift: dx=[{min(dx):.1f}, {max(dx):.1f}] dy=[{min(dy):.1f}, {max(dy):.1f}] max={drift_max:.1f} px",
            f"  NCC: min={ncc_min:.4f} mean={ncc_mean:.4f}",
        ]
        if self._crop_box:
            y0, y1, x0, x1 = self._crop_box
            lines.append(f"  crop: [{y0}:{y1}, {x0}:{x1}]")
        if self.bin_factor > 1:
            lines.append(f"  bin: {self.bin_factor}×")
        return "\n".join(lines)

    @property
    def mean(self) -> np.ndarray:
        """Mean of all aligned frames — SNR improves by sqrt(N)."""
        return self._aligned.mean(axis=0)


bind_tool_runtime_api(Align2DBulk, "Align2DBulk")
