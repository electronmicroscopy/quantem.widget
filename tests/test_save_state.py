"""Regression shield for the ``save_state`` contract (Show2D / Show3D / Show4DSTEM).

Background: an anywidget syncs its pixel buffers as ``sync=True`` traits. On
notebook save, ipywidgets serializes those buffers into ``metadata.widgets`` -
a 5-panel 4k Show2D baked ~1 GB into a single .ipynb. The fix: ``save_state``
(default False) drops the bulk buffers from the FULL-state snapshot (the save
path, ``get_state(key=None)``) and instead attaches a static PNG so a cold
reopen still shows the render. ``save_state=True`` embeds everything for
kernel-less restore.

The danger in that fix is trimming too much: if the trim also hit the TARGETED
``send_state`` path (``get_state(key=<name/set>)``), the frontend would never
receive the buffer and the widget would render blank - invisible to unit tests
that only check output size. These tests lock both halves of the contract so a
future edit can't silently reintroduce either the bloat or the blank render.
"""
import pathlib

import numpy as np
import pytest

from quantem.widget import Show2D, Show3D, Show4DSTEM, ShowEDS


def _make(widget, *, save_state):
    """Construct a small instance of each widget plus the trait key that carries
    its live-render pixels (the one that must survive the targeted send path)."""
    if widget is Show2D:
        data = [np.random.rand(128, 128).astype("float32") for _ in range(2)]
        return Show2D(data, save_state=save_state), "frame_bytes"
    if widget is Show3D:
        return Show3D(np.random.rand(4, 128, 128).astype("float32"),
                      save_state=save_state), "frame_bytes"
    if widget is ShowEDS:
        return ShowEDS(np.random.rand(16, 16, 32).astype("float32"),
                       np.linspace(0.1, 10.0, 32), save_state=save_state), "cube_bytes"
    return Show4DSTEM(np.random.rand(8, 8, 16, 16).astype("float32"),
                      save_state=save_state), "virtual_image_bytes"


WIDGETS = [Show2D, Show3D, Show4DSTEM, ShowEDS]


@pytest.mark.parametrize("widget", WIDGETS)
def test_targeted_send_state_never_trimmed(widget):
    """The render path must stay intact. ``send_state`` / ``hold_sync`` call
    ``get_state`` with a specific key (or set), never ``None``; the trim must
    only fire on the full ``key=None`` snapshot. If a targeted lookup loses the
    pixel key, the frontend renders blank."""
    w, render_key = _make(widget, save_state=False)
    assert render_key in w.get_state(render_key), (
        f"{widget.__name__}: targeted get_state({render_key!r}) dropped the key "
        f"- live render would go blank")
    assert render_key in w.get_state({render_key, "widget_version"}), (
        f"{widget.__name__}: hold_sync batch lost {render_key!r}")


@pytest.mark.parametrize("widget", WIDGETS)
def test_full_snapshot_trims_bulk_buffers(widget):
    """save_state=False: no bulk pixel buffer may appear in the saved-notebook
    snapshot. This is the anti-1GB guard."""
    w, _ = _make(widget, save_state=False)
    full = w.get_state()
    leaked = [k for k in w._UNSAVED_HEAVY_KEYS if k in full]
    assert not leaked, (
        f"{widget.__name__}: bulk buffers {leaked} leaked into saved state "
        f"- notebook will bloat")


@pytest.mark.parametrize("widget", WIDGETS)
def test_static_png_fallback_present(widget):
    """save_state=False: a static image/png must be attached so a kernel-less
    reopen (GitHub, nbviewer, cold Lab) still shows the render."""
    w, _ = _make(widget, save_state=False)
    bundle = w._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    assert "image/png" in (data or {}), (
        f"{widget.__name__}: no static PNG fallback for a cold reopen")


@pytest.mark.parametrize("widget", WIDGETS)
def test_save_state_true_does_not_force_png(widget):
    """save_state=True embeds full interactive state, so it must NOT inject the
    static PNG (the live widget restores from the embedded buffers instead)."""
    w, _ = _make(widget, save_state=True)
    bundle = w._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    assert "image/png" not in (data or {}), (
        f"{widget.__name__}: save_state=True should not attach the static PNG")


@pytest.mark.parametrize("widget", WIDGETS)
def test_default_is_save_state_false(widget):
    """The whole point: persistence is opt-in. A plain construction must behave
    as save_state=False (static PNG present)."""
    w, _ = _make(widget, save_state=False)
    assert w._save_state is False


def test_show2d_static_png_preserves_sparse_large_content():
    """The fallback PNG must show real image content, not a blank dark tile.

    A 4k-style sparse image is exactly where stride sampling fails: if the
    bright feature lands between sampled rows/columns, the static fallback looks
    nearly black even though the live widget rendered fine. The PNG path should
    area-downsample instead, so thin/sparse features survive notebook save.
    """
    import base64
    import io

    from PIL import Image

    image = np.zeros((1024, 1024), dtype=np.float32)
    image[101:109, 611:619] = 1000
    widget = Show2D(image, labels=["Sparse feature"], save_state=False, verbose=False)

    bundle = widget._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    png = data["image/png"]
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB"))

    nonwhite = ~np.all(decoded > 245, axis=-1)
    warm_signal = (
        (decoded[..., 0] > 140)
        & (decoded[..., 1] > 80)
        & (decoded[..., 2] < 230)
        & nonwhite
    )
    assert int(warm_signal.sum()) > 20


def test_show2d_first_render_clears_heavy_frame_buffer():
    """After JS paints, later notebook saves should not keep pixel buffers."""
    image = np.random.default_rng(0).random((128, 128), dtype=np.float32)
    widget = Show2D(image, save_state=False, verbose=False)
    assert widget.frame_bytes

    widget._on_first_render({"new": True})

    assert widget.frame_bytes == b""
    assert "frame_bytes" not in widget.get_state()
    assert not widget._get_embed_state().get("buffers")


def test_show3d_first_render_clears_heavy_transfer_buffers():
    """After JS paints, later notebook saves should not keep transfer buffers."""
    stack = np.random.default_rng(1).random((6, 128, 128), dtype=np.float32)
    widget = Show3D(stack, save_state=False)
    widget.frame_bytes = b"frame-transfer"
    widget._buffer_bytes = b"chunk-transfer"

    widget._on_first_render({"new": True})

    assert widget.frame_bytes == b""
    assert widget._buffer_bytes == b""
    full = widget.get_state()
    assert "frame_bytes" not in full
    assert "_buffer_bytes" not in full
    buffers = widget._get_embed_state().get("buffers", [])
    assert not [b for b in buffers if b.get("path") in (["frame_bytes"], ["_buffer_bytes"])]
    assert not [b for b in buffers if b.get("data")]


def _assert_export_state_keeps_buffer(widget, key: str) -> None:
    """HTML export clones must opt into the full embedded state."""
    state = widget.get_state()
    assert widget._save_state is True
    assert key in state
    value = state[key]
    if isinstance(value, (bytes, bytearray)):
        assert len(value) > 0


def test_html_export_clones_keep_bulk_buffers():
    """Standalone HTML export must embed the data it needs to render offline.

    ``export_html`` builds an export-only clone and then asks ipywidgets for a
    full dependency state. That call hits ``get_state(key=None)``. If the clone
    stays on the notebook-save default (``_save_state=False``), the heavy pixel
    payload is trimmed and the exported HTML renders blank.
    """
    rng = np.random.default_rng(10)
    show2d = Show2D(rng.random((64, 64), dtype=np.float32), save_state=False, verbose=False)
    show2d_clone = show2d._clone_for_html_export(quantized=False)
    try:
        _assert_export_state_keeps_buffer(show2d_clone, "frame_bytes")
    finally:
        show2d_clone.close()
        show2d.close()

    show3d = Show3D(rng.random((3, 32, 32), dtype=np.float32), save_state=False)
    show3d_clone = show3d._clone_for_html_export(quantized=False)
    try:
        _assert_export_state_keeps_buffer(show3d_clone, "_offline_float_stack")
    finally:
        show3d_clone.close()
        show3d.close()

    stem = Show4DSTEM(
        rng.integers(0, 100, (4, 4, 8, 8), dtype=np.uint16),
        save_state=False,
        verbose=False,
    )
    stem_clone = stem._clone_for_html_export(dtype="uint16", det_bin=1)
    try:
        _assert_export_state_keeps_buffer(stem_clone, "_offline_stack")
    finally:
        stem_clone.close()
        stem.close()


def test_show4dstem_bslz4_html_export_embeds_self_with_bulk_state(monkeypatch, tmp_path):
    """Show4DSTEM's bslz4 export branch embeds ``self``, not an export clone."""
    widget = Show4DSTEM(
        np.random.default_rng(11).integers(0, 100, (4, 4, 8, 8), dtype=np.uint16),
        save_state=False,
        verbose=False,
    )
    widget._offline_bslz4 = "{}"
    widget._offline_stack = b"offline-stack"
    captured = {}

    def fake_dependency_state(views, drop_defaults=False):
        view = views[0]
        state = view.get_state()
        captured["save_state"] = view._save_state
        captured["has_stack"] = "_offline_stack" in state
        captured["stack_size"] = len(state.get("_offline_stack", b""))
        return {}

    def fake_embed_minimal_html(filename, *, views, title, drop_defaults, state):
        pathlib.Path(filename).write_text("<html><head></head><body></body></html>")

    monkeypatch.setattr("ipywidgets.embed.dependency_state", fake_dependency_state)
    monkeypatch.setattr("ipywidgets.embed.embed_minimal_html", fake_embed_minimal_html)

    try:
        widget._write_html_export(tmp_path / "show4dstem.html", dtype="uint16", det_bin=1)
    finally:
        widget.close()

    assert captured == {"save_state": True, "has_stack": True, "stack_size": len(b"offline-stack")}
    assert widget._save_state is False


def test_export_html_size_scales_with_embedded_data(tmp_path):
    """Public ``export_html`` must not collapse to a tiny empty widget shell.

    Regression for the save_state opt-in bug: export clones used the notebook
    default ``_save_state=False``, so ipywidgets stripped the heavy buffers and
    wrote a small HTML file with no offline image stack.
    """
    rng = np.random.default_rng(12)
    cases = []

    data2d = rng.random((3, 512, 512), dtype=np.float32)
    show2d = Show2D(data2d, save_state=False, verbose=False, title="size-show2d")
    cases.append((show2d, data2d.nbytes, "frame_bytes", tmp_path / "show2d.html"))

    data3d = rng.random((8, 256, 256), dtype=np.float32)
    show3d = Show3D(data3d, save_state=False, title="size-show3d")
    cases.append((show3d, data3d.nbytes, "_offline_float_stack", tmp_path / "show3d.html"))

    stem_data = rng.integers(0, 1000, (32, 32, 32, 32), dtype=np.uint16)
    stem = Show4DSTEM(stem_data, save_state=False, verbose=False, title="size-show4dstem")
    cases.append((stem, stem_data.nbytes, "_offline_stack", tmp_path / "show4dstem.html"))

    try:
        for widget, raw_bytes, marker, path in cases:
            exported = widget.export_html(path, encoding="full")
            html = exported.read_text(errors="ignore")
            assert exported.stat().st_size > raw_bytes / 2
            assert marker in html
    finally:
        for widget, _, _, _ in cases:
            widget.close()


# ---------------------------------------------------------------------------
# Show2D static-PNG fidelity: the fallback must map pixels to colors exactly
# like the live widget (colormap, contrast window, log scale, linked/per-panel
# ranges, diff panels), not merely "show something".
# ---------------------------------------------------------------------------

def _decode_png(widget) -> np.ndarray:
    """Decode the widget's static fallback PNG to an (H, W, 3) uint8 array."""
    import base64
    import io

    from PIL import Image

    png_b64 = widget._static_png_b64()
    assert png_b64
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB"))


def test_show2d_format_stat_matches_widget_stats_row():
    """Stats line must format like JS formatNumber: 5.85e+3, 0.42, 0."""
    fmt = Show2D._format_stat
    assert fmt(0) == "0"
    assert fmt(5852.3) == "5.85e+3"
    assert fmt(0.0042) == "4.20e-3"
    assert fmt(0.42) == "0.42"
    assert fmt(-12345.0) == "-1.23e+4"


def test_show2d_static_panel_rgb_bit_matches_independent_cmap():
    """The panel colormap path must equal an independently computed
    cmap(clip((frame - vmin) / (vmax - vmin))) at bit level. This is the
    parity anchor: the PNG renderer feeds imshow these exact RGB bytes."""
    from matplotlib import colormaps

    rng = np.random.default_rng(0)
    frame = rng.random((64, 64)).astype(np.float32) * 100
    for cmap_name in ("gray", "inferno", "viridis"):
        widget = Show2D(frame, cmap=cmap_name, verbose=False)
        (vmin, vmax), = widget._resolve_panel_display_ranges([frame])
        got = widget._static_panel_rgb(frame, vmin, vmax, cmap_name)
        expected_norm = np.clip((frame - vmin) / (vmax - vmin), 0.0, 1.0)
        expected = (colormaps.get_cmap(cmap_name)(expected_norm)[..., :3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(got, expected)


def test_show2d_png_gray_cmap_is_channel_equal_inferno_is_not():
    """Wrong-colormap regression: gray must produce (almost) only R==G==B
    pixels; inferno must produce a large channel-diverse fraction."""
    rng = np.random.default_rng(1)
    frame = rng.random((256, 256)).astype(np.float32)

    gray_png = _decode_png(Show2D(frame, cmap="gray", verbose=False))
    channel_spread = gray_png.max(axis=-1).astype(int) - gray_png.min(axis=-1).astype(int)
    # Tolerance 3 for PNG quantization; the histogram contrast markers are the
    # only intentionally colored pixels and they are a sliver of the figure.
    assert (channel_spread <= 3).mean() > 0.99

    inferno_png = _decode_png(Show2D(frame, cmap="inferno", verbose=False))
    inferno_spread = inferno_png.max(axis=-1).astype(int) - inferno_png.min(axis=-1).astype(int)
    assert (inferno_spread > 30).mean() > 0.05


def test_show2d_png_log_scale_brightens_skewed_data():
    """log1p on heavily skewed data lifts mid-tones: with a gray colormap the
    panel's mean luminance must increase. Locks that log_scale actually feeds
    the PNG's pixel mapping."""
    rng = np.random.default_rng(2)
    frame = (rng.random((256, 256)).astype(np.float32) ** 4) * 1000

    def panel_mean(widget) -> float:
        decoded = _decode_png(widget).astype(float)
        luminance = decoded.mean(axis=-1)
        panel = luminance[luminance < 240]  # exclude white figure background
        return float(panel.mean())

    linear_mean = panel_mean(Show2D(frame, cmap="gray", log_scale=False, verbose=False))
    log_mean = panel_mean(Show2D(frame, cmap="gray", log_scale=True, verbose=False))
    assert log_mean > linear_mean + 20


def test_show2d_png_vmin_vmax_clipping_saturates():
    """An explicit narrow [vmin, vmax] window must show up as large saturated
    regions at the colormap endpoints; the unclipped render must not."""
    from matplotlib import colormaps

    gradient = np.tile(np.linspace(0, 1, 256, dtype=np.float32), (256, 1))
    cmap = colormaps.get_cmap("inferno")
    lo_color = np.array(cmap(0.0)[:3]) * 255
    hi_color = np.array(cmap(1.0)[:3]) * 255

    def saturated_fraction(widget) -> tuple[float, float]:
        decoded = _decode_png(widget).astype(float)
        near_lo = (np.abs(decoded - lo_color).max(axis=-1) < 10).mean()
        near_hi = (np.abs(decoded - hi_color).max(axis=-1) < 10).mean()
        return float(near_lo), float(near_hi)

    clipped_lo, clipped_hi = saturated_fraction(
        Show2D(gradient, cmap="inferno", vmin=0.4, vmax=0.6, verbose=False))
    open_lo, open_hi = saturated_fraction(Show2D(gradient, cmap="inferno", verbose=False))
    assert clipped_lo > 0.05 and clipped_hi > 0.05
    assert clipped_lo > 5 * max(open_lo, 1e-4)
    assert clipped_hi > 5 * max(open_hi, 1e-4)


def test_show2d_auto_contrast_uses_full_res_percentiles():
    """auto_contrast must resolve to the 2/98 percentiles of the FULL frame
    (the same cut the widget computes), not of the binned PNG pixels."""
    rng = np.random.default_rng(3)
    frame = rng.normal(100, 25, (1024, 1024)).astype(np.float32)
    widget = Show2D(frame, auto_contrast=True, verbose=False)
    (vmin, vmax), = widget._resolve_panel_display_ranges([frame])
    expected_lo, expected_hi = np.percentile(frame, (2, 98))
    np.testing.assert_allclose([vmin, vmax], [expected_lo, expected_hi], rtol=1e-6)


def test_show2d_linked_contrast_shares_one_range():
    """Gallery + link_contrast=True (widget default): all panels share the
    merged range. link_contrast=False: each panel uses its own extrema."""
    dim_frame = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    bright_frame = dim_frame * 100
    linked = Show2D([dim_frame, bright_frame], link_contrast=True, verbose=False)
    ranges = linked._resolve_panel_display_ranges([dim_frame, bright_frame])
    assert ranges[0] == ranges[1] == (0.0, 100.0)
    unlinked = Show2D([dim_frame, bright_frame], link_contrast=False, verbose=False)
    ranges = unlinked._resolve_panel_display_ranges([dim_frame, bright_frame])
    assert ranges[0] == (0.0, 1.0)
    assert ranges[1] == (0.0, 100.0)


def test_show2d_per_image_vmin_vmax_lists():
    """List vmin/vmax must resolve per panel and beat linked contrast."""
    frame_a = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    frame_b = frame_a * 10
    widget = Show2D([frame_a, frame_b], vmin=[0.1, 1.0], vmax=[0.9, 9.0], verbose=False)
    ranges = widget._resolve_panel_display_ranges([frame_a, frame_b])
    assert ranges[0] == (pytest.approx(0.1), pytest.approx(0.9))
    assert ranges[1] == (pytest.approx(1.0), pytest.approx(9.0))


def test_show2d_diff_mode_adds_signed_diff_panel():
    """diff_mode with 2 images: 3 panels; the diff panel is ref - other with a
    symmetric window and a diverging colormap, never log-scaled."""
    rng = np.random.default_rng(4)
    frame_a = rng.random((64, 64)).astype(np.float32)
    frame_b = rng.random((64, 64)).astype(np.float32)
    widget = Show2D([frame_a, frame_b], diff_mode=True, log_scale=True, verbose=False)
    specs = widget._static_panel_specs()
    assert len(specs) == 3
    diff_spec = specs[2]
    assert diff_spec["label"] == "Diff (A − B)"
    assert diff_spec["cmap"] == "RdBu"
    assert diff_spec["apply_log"] is False
    np.testing.assert_array_equal(diff_spec["frame"], frame_a - frame_b)
    assert diff_spec["vmin"] == -diff_spec["vmax"]
    png = _decode_png(widget)
    assert png.size > 0


@pytest.mark.parametrize("cmap", ["gray", "inferno", "viridis"])
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize("auto_contrast", [False, True])
def test_show2d_png_settings_sweep(cmap, log_scale, auto_contrast):
    """Every contrast/colormap combination must render a decodable PNG with
    the gallery + labels layout (3 panels, 3 labels in the specs)."""
    rng = np.random.default_rng(5)
    frames = [rng.random((96, 96)).astype(np.float32) * scale for scale in (1, 10, 100)]
    widget = Show2D(frames, labels=["a", "b", "c"], cmap=cmap, log_scale=log_scale,
                    auto_contrast=auto_contrast, verbose=False)
    specs = widget._static_panel_specs()
    assert [spec["label"] for spec in specs] == ["a", "b", "c"]
    assert all(spec["vmax"] > spec["vmin"] for spec in specs)
    assert all("Mean" in spec["stats"] for spec in specs)
    decoded = _decode_png(widget)
    assert decoded.shape[0] > 100 and decoded.shape[1] > 300  # 3-across gallery


def test_show2d_static_scale_bar_label_matches_widget_format():
    """The PNG's scale bar text must be character-identical to the live
    widget's canvas label (js/figure.ts formatScaleLabel + drawScaleBarHiDPI):
    60 CSS px target bar, nice 1/2/5 rounding, length units re-laddered to a
    clean integer (10 A -> "1 nm"), uncalibrated data labeled in "px"."""
    from quantem.widget.show2d import _format_scale_label

    frame = np.zeros((512, 512), dtype=np.float32)
    calibrated = Show2D(frame, sampling=0.23, units="A", labels=["cal"], verbose=False)
    # single image -> live canvas is SINGLE_IMAGE_TARGET = 500 CSS px wide
    effective_zoom = 500 / 512
    (label, zoom_text, bar_text, bar_px), = calibrated._static_overlay_texts()
    assert label == "cal"
    assert zoom_text == "1.0×"
    # 60 css px / effectiveZoom * 0.23 A = 14.1 A -> nice 10 A -> integer nm
    assert bar_text == "1 nm"
    assert bar_px == pytest.approx(10 / 0.23 * effective_zoom)

    uncalibrated = Show2D(frame, verbose=False)
    (_, _, bar_text, bar_px), = uncalibrated._static_overlay_texts()
    assert bar_text == "50 px"  # 61.4 px -> nice 50, unit "px" when pixel_size == 0
    assert bar_px == pytest.approx(50 * effective_zoom)

    assert _format_scale_label(0.5, "nm") == "5 Å"     # sub-1 re-ladders down
    assert _format_scale_label(13.8, "A") == "1 nm"    # 10 A reads as 1 nm
    assert _format_scale_label(20, "mrad") == "20 mrad"  # non-length keeps unit


def test_show2d_static_zoom_badge_and_center_crop():
    """zoom=1.8 must produce the widget's badge text (JS zoom.toFixed(1) + x),
    shorten the bar to the zoomed field of view, and crop the central 1/zoom
    window exactly like the live canvas transform."""
    frame = np.random.default_rng(7).random((512, 512)).astype(np.float32)
    widget = Show2D(frame, zoom=1.8, verbose=False)
    (_, zoom_text, bar_text, bar_px), = widget._static_overlay_texts()
    assert zoom_text == "1.8×"
    effective_zoom = 1.8 * 500 / 512
    assert bar_text == "20 px"  # 60 / 1.76 = 34.1 -> nice 20
    assert bar_px == pytest.approx(20 * effective_zoom)
    rows, cols = Show2D._center_crop_slices(512, 512, 1.8)
    assert cols.stop - cols.start == round(512 / 1.8) == 284
    assert rows.start == (512 - 284) // 2
    assert _decode_png(widget).size > 0  # zoomed render still produces a PNG


def test_show2d_png_render_perf_two_4k_frames():
    """The PNG path (area-bin first, then normalize) must stay cheap even on
    2x 4096x4096 float32: under 1.5 s per display."""
    import time

    rng = np.random.default_rng(6)
    frames = rng.random((2, 4096, 4096), dtype=np.float32)
    widget = Show2D(frames, verbose=False)
    start = time.perf_counter()
    png_b64 = widget._static_png_b64()
    elapsed = time.perf_counter() - start
    assert png_b64
    assert elapsed < 1.5, f"static PNG took {elapsed:.2f}s"


def test_show2d_display_defers_static_png_render(monkeypatch):
    """Displaying the widget must NOT render the PNG synchronously (matplotlib
    would block every cell that shows a widget); the deferred post_execute
    fill must then update the placeholder sibling with the real image/png."""
    import IPython

    frame = np.random.default_rng(8).random((64, 64)).astype(np.float32)
    widget = Show2D(frame, verbose=False)
    png_calls = []
    original_png = Show2D._static_png_b64
    monkeypatch.setattr(Show2D, "_static_png_b64",
                        lambda self, **kw: (png_calls.append(1), original_png(self, **kw))[1])
    displayed, updated, hooks = [], [], {}

    class FakeHandle:
        def update(self, data, raw=False, metadata=None):
            updated.append((data, metadata))

    class FakeEvents:
        def register(self, name, fn):
            hooks[name] = fn

        def unregister(self, name, fn):
            hooks.pop(name, None)

    class ZMQInteractiveShell:  # name is what the kernel check looks at
        events = FakeEvents()

    shell = ZMQInteractiveShell()

    def fake_display(data, raw=False, metadata=None, display_id=None, **kw):
        displayed.append((data, metadata, display_id))
        return FakeHandle() if display_id else None

    monkeypatch.setattr(IPython, "get_ipython", lambda: shell)
    monkeypatch.setattr("IPython.display.display", fake_display)

    widget._ipython_display_()
    assert not png_calls, "PNG was rendered synchronously at display time"
    # widget bundle + empty placeholder sibling with the hide marker
    assert len(displayed) == 2
    placeholder_data, placeholder_meta, display_id = displayed[1]
    assert display_id is True
    assert "image/jpeg" not in placeholder_data
    assert "quantem-static-fallback" in placeholder_data["text/html"]
    assert placeholder_meta == {"quantem.widget": {"static_fallback": True}}
    # cell finishes -> post_execute hook fills the placeholder with the PNG
    assert "post_execute" in hooks
    hooks["post_execute"]()
    assert png_calls, "deferred fill never rendered the PNG"
    assert "post_execute" not in hooks, "one-shot hook did not unregister"
    fill_data, fill_meta = updated[-1]
    assert isinstance(fill_data["image/jpeg"], bytes) and len(fill_data["image/jpeg"]) > 1000
    assert "quantem-static-fallback" in fill_data["text/html"]
    assert fill_meta == {"quantem.widget": {"static_fallback": True}}


def test_show2d_sibling_static_output_via_nbconvert(tmp_path):
    """Executing a notebook must leave a saved preview on the Show2D cell.

    Depending on the frontend that calls ``display()``, the fallback may be a
    separate sibling output or an image fallback inside the widget mime bundle.
    Both are acceptable as long as the output can render statically and widget
    metadata stays tiny.
    """
    import json
    import subprocess
    import sys

    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(
        "import numpy as np\n"
        "from IPython.display import display\n"
        "from quantem.widget import Show2D\n"
        "display(Show2D(np.random.rand(48, 48).astype('float32'), verbose=False))\n"
    )]
    nb_path = tmp_path / "show2d_sibling.ipynb"
    nbformat.write(nb, nb_path)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", str(nb_path)],
        check=True, capture_output=True, timeout=180)
    executed = nbformat.read(nb_path, as_version=4)
    outputs = executed.cells[0].outputs
    display_outputs = [o for o in outputs if o.output_type == "display_data"]
    assert display_outputs, f"expected display output, got {outputs}"
    widget_outputs = [
        output
        for output in display_outputs
        if "application/vnd.jupyter.widget-view+json" in output.data
    ]
    assert widget_outputs, f"expected widget-view bundle, got {outputs}"
    fallback_outputs = [
        output
        for output in display_outputs
        if "image/jpeg" in output.data or "image/png" in output.data
    ]
    assert fallback_outputs, f"expected static image fallback, got {outputs}"
    fallback = fallback_outputs[-1]
    image_key = "image/jpeg" if "image/jpeg" in fallback.data else "image/png"
    assert len(fallback.data[image_key]) > 1000
    if "text/html" in fallback.data:
        assert "quantem-static-fallback" in fallback.data["text/html"]
        assert fallback.metadata.get("quantem.widget", {}).get("static_fallback") is True
    # metadata.widgets guard. nbclient reconstructs widget state by replaying
    # comm traffic, so two known constants of headless execution appear here no
    # matter what get_state() trims: the anywidget JS bundle (_esm) and the
    # initial 48x48 frame upload (~12 KB, needed for live render; a real Lab
    # save goes through the trimmed get_state(), covered by
    # test_full_snapshot_trims_bulk_buffers). Everything else must stay tiny -
    # this fails if e.g. the static PNG or a duplicate pixel buffer starts
    # leaking into the synced state.
    widget_states = executed.metadata["widgets"]["application/vnd.jupyter.widget-state+json"]["state"]
    anymodel = next(s for s in widget_states.values() if s.get("model_name") == "AnyModel")
    body = dict(anymodel["state"])
    body.pop("_esm", None)
    body.pop("_static_fallback_jpeg", None)
    buffer_bytes = sum(len(b["data"]) for b in anymodel.get("buffers", []))
    assert buffer_bytes < 20_000
    assert len(json.dumps(body)) < 20_000


def test_show2d_cmd_s_snapshot_keeps_static_preview_without_heavy_pixels():
    """JupyterLab Cmd+S uses the full widget-state snapshot.

    A lightweight Show2D save must not embed ``frame_bytes``/detail/export
    buffers, but it still needs a compact preview in the model state. Otherwise
    a saved notebook can rehydrate a live widget with no pixels and show a
    blank output after reopen.
    """
    widget = Show2D(
        np.random.default_rng(14).random((3, 256, 256), dtype=np.float32),
        display_bin=2,
        save_state=False,
        verbose=False,
    )
    state = widget.get_state()

    assert "_static_fallback_jpeg" in state
    assert len(state["_static_fallback_jpeg"]) > 1000
    assert "frame_bytes" not in state
    assert "_detail_bytes" not in state
    assert "export_payload" not in state


def test_show3d_cmd_s_snapshot_keeps_static_preview_without_heavy_pixels():
    """Show3D lightweight notebook saves must reopen with a compact preview."""
    widget = Show3D(
        np.random.default_rng(15).random((5, 64, 64), dtype=np.float32),
        save_state=False,
        title="save-state show3d",
    )
    state = widget.get_state()

    assert "_static_fallback_jpeg" in state
    assert len(state["_static_fallback_jpeg"]) > 1000
    assert "frame_bytes" not in state
    assert "_buffer_bytes" not in state
    assert "_offline_stack" not in state
    assert "_offline_float_stack" not in state
    assert "export_payload" not in state


def test_show4dstem_cmd_s_snapshot_keeps_two_panel_static_preview():
    """Show4DSTEM saved preview should show virtual image and diffraction."""
    import base64
    import io

    from PIL import Image

    rng = np.random.default_rng(19)
    data = rng.integers(0, 1000, (10, 12, 24, 24), dtype=np.uint16)
    widget = Show4DSTEM(
        data,
        sampling=(0.2, 0.2, 0.8, 0.8),
        units=["nm", "nm", "mrad", "mrad"],
        panel_width_px=128,
        save_state=False,
        verbose=False,
        title="save-state 4dstem",
    )

    state = widget.get_state()
    png_b64 = widget._static_png_b64(max_px=128, dpi=160)
    assert png_b64
    decoded = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")

    assert decoded.width > decoded.height
    assert decoded.width >= (decoded.height - 24) * 2 - 8
    assert "_static_fallback_jpeg" in state
    assert len(state["_static_fallback_jpeg"]) > 1000
    assert "_offline_stack" not in state
    assert "export_payload" not in state
    assert "_gif_data" not in state
    widget.close()


def test_show3d_static_overlay_matches_show2d_style_metadata():
    """Show3D saved previews should carry the same context as Show2D."""
    widget = Show3D(
        np.random.default_rng(16).random((3, 128, 128), dtype=np.float32),
        labels=["zero", "one", "two"],
        panel_titles=["ADF"],
        sampling=(0.23, 0.23),
        units=("A", "A"),
        save_state=False,
    )
    widget.slice_idx = 1

    (label, zoom_text, bar_text, bar_px), = widget._static_overlay_texts([0], 1)

    assert label == "ADF · one 2/3"
    assert zoom_text == "1.0×"
    assert bar_text == "5 Å"
    assert bar_px == pytest.approx(5 / 0.23 * 500 / 128)
    assert widget._static_png_b64()


def test_show3d_static_png_pixel_matches_show2d_current_frame_gallery():
    """Show3D saved previews must be pixel-identical to Show2D galleries."""
    import base64
    import io

    from PIL import Image

    rng = np.random.default_rng(17)
    panels = []
    for panel in range(6):
        frames = rng.random((5, 192, 192), dtype=np.float32)
        panels.append(frames + panel * 0.2)
    widget = Show3D(
        *panels,
        panel_titles=[f"P{panel + 1:02d}" for panel in range(len(panels))],
        panel_frame_labels=[
            [f"defocus {frame - 2:+.1f} nm" for frame in range(5)]
            for _ in panels
        ],
        max_cols=3,
        panel_gap=3,
        size=180,
        sampling=0.05,
        units="nm",
        auto_contrast=True,
        save_state=False,
    )
    widget.slice_idx = 3

    show3d_png = widget._static_png_b64(max_px=256, dpi=160)
    reference = Show2D(
        [widget._get_display_panel_frame(panel, widget.slice_idx) for panel in range(6)],
        labels=[widget._static_panel_title(panel, widget.slice_idx) for panel in range(6)],
        ncols=3,
        gallery_gap_px=3,
        size=180,
        sampling=0.05,
        units="nm",
        cmap=widget.cmap,
        auto_contrast=widget.auto_contrast,
        link_contrast=widget.link_contrast,
        show_stats=False,
        show_controls=False,
        verbose=False,
        save_state=False,
    )
    show2d_png = reference._static_png_b64(max_px=256, dpi=160)

    show3d_rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(show3d_png))).convert("RGB"))
    show2d_rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(show2d_png))).convert("RGB"))
    np.testing.assert_array_equal(show3d_rgb, show2d_rgb)


@pytest.mark.parametrize(
    ("n_panels", "max_cols", "size", "panel_gap", "hidden", "scale_bar"),
    [
        (1, 1, 0, 0, [], True),
        (2, 2, 0, 0, [], True),
        (4, 2, 180, 3, [], True),
        (6, 3, 160, 4, [1, 4], True),
        (9, 3, 0, 2, [], False),
    ],
)
def test_show3d_static_png_pixel_matches_show2d_layout_matrix(
    n_panels,
    max_cols,
    size,
    panel_gap,
    hidden,
    scale_bar,
):
    """Show3D static fallback must stay pixel-perfect across panel layouts."""
    import base64
    import io

    from PIL import Image

    rng = np.random.default_rng(18 + n_panels)
    stacks = []
    for panel in range(n_panels):
        frame_stack = rng.random((4, 96, 112), dtype=np.float32)
        stacks.append(frame_stack + panel * 0.1)
    widget = Show3D(
        *stacks,
        panel_titles=[f"P{panel + 1:02d}" for panel in range(n_panels)],
        panel_frame_labels=[
            [f"frame-label-{frame + 1}" for frame in range(4)]
            for _ in range(n_panels)
        ],
        max_cols=max_cols,
        panel_gap=panel_gap,
        size=size,
        sampling=0.12,
        units="nm",
        auto_contrast=True,
        show_scale_bar=scale_bar,
        hidden_panels=hidden,
        save_state=False,
    )
    widget.slice_idx = 2
    visible = [panel for panel in range(n_panels) if panel not in set(hidden)] or [0]
    reference = Show2D(
        [widget._get_display_panel_frame(panel, widget.slice_idx) for panel in visible],
        labels=[widget._static_panel_title(panel, widget.slice_idx) for panel in visible],
        ncols=max(1, min(max_cols, len(visible))),
        gallery_gap_px=panel_gap,
        size=size,
        sampling=0.12,
        units="nm",
        scale_bar_visible=scale_bar,
        cmap=widget.cmap,
        auto_contrast=widget.auto_contrast,
        link_contrast=widget.link_contrast,
        show_stats=False,
        show_controls=False,
        verbose=False,
        save_state=False,
    )

    show3d_png = widget._static_png_b64(max_px=220, dpi=160)
    show2d_png = reference._static_png_b64(max_px=220, dpi=160)
    show3d_rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(show3d_png))).convert("RGB"))
    show2d_rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(show2d_png))).convert("RGB"))

    np.testing.assert_array_equal(show3d_rgb, show2d_rgb)


# ---------------------------------------------------------------------------
# Static-fallback sibling contract, shared by all four widgets via
# StaticFallbackMixin: display publishes the widget bundle plus an EMPTY
# placeholder sibling; the deferred post_execute fill swaps in the PNG with
# the quantem-static-fallback marker; and the full get_state snapshot never
# carries the heavy buffers. A regression in any widget reopens BLACK in
# JupyterLab (the ShowEDS bug this suite was extended for).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widget", WIDGETS)
def test_sibling_static_fallback_contract(widget, monkeypatch):
    import IPython

    w, _ = _make(widget, save_state=False)
    png_calls = []
    # patch on the instance's real class: the public Show4DSTEM name is a
    # backend dispatcher function, not the widget class itself
    widget_cls = type(w)
    original_png = widget_cls._static_png_b64
    monkeypatch.setattr(widget_cls, "_static_png_b64",
                        lambda self, **kw: (png_calls.append(1), original_png(self, **kw))[1])
    displayed, updated, hooks = [], [], {}

    class FakeHandle:
        def update(self, data, raw=False, metadata=None):
            updated.append((data, metadata))

    class FakeEvents:
        def register(self, name, fn):
            hooks[name] = fn

        def unregister(self, name, fn):
            hooks.pop(name, None)

    class ZMQInteractiveShell:  # name is what the kernel check looks at
        events = FakeEvents()

    shell = ZMQInteractiveShell()

    def fake_display(data, raw=False, metadata=None, display_id=None, **kw):
        displayed.append((data, metadata, display_id))
        return FakeHandle() if display_id else None

    monkeypatch.setattr(IPython, "get_ipython", lambda: shell)
    monkeypatch.setattr("IPython.display.display", fake_display)

    w._ipython_display_()
    assert not png_calls, f"{widget.__name__}: PNG rendered synchronously at display time"
    # widget bundle + empty placeholder sibling with the hide marker
    assert len(displayed) == 2, f"{widget.__name__}: expected widget + placeholder sibling"
    placeholder_data, placeholder_meta, display_id = displayed[1]
    assert display_id is True
    assert "image/jpeg" not in placeholder_data
    assert "quantem-static-fallback" in placeholder_data["text/html"]
    assert placeholder_meta == {"quantem.widget": {"static_fallback": True}}
    # cell finishes -> post_execute hook fills the placeholder with the PNG
    assert "post_execute" in hooks
    hooks["post_execute"]()
    assert png_calls, f"{widget.__name__}: deferred fill never rendered the PNG"
    assert "post_execute" not in hooks, "one-shot hook did not unregister"
    fill_data, fill_meta = updated[-1]
    assert isinstance(fill_data["image/jpeg"], bytes) and len(fill_data["image/jpeg"]) > 1000
    assert "quantem-static-fallback" in fill_data["text/html"]
    assert fill_meta == {"quantem.widget": {"static_fallback": True}}
    # the saved-notebook snapshot must stay free of the bulk buffers
    full = w.get_state()
    leaked = [k for k in w._UNSAVED_HEAVY_KEYS if k in full]
    assert not leaked, f"{widget.__name__}: heavy keys {leaked} leaked into full get_state"


@pytest.mark.parametrize("widget", WIDGETS)
def test_sibling_not_emitted_with_save_state_true(widget, monkeypatch):
    """save_state=True embeds full interactive state; no sibling placeholder."""
    import IPython

    w, _ = _make(widget, save_state=True)
    displayed = []

    class ZMQInteractiveShell:
        events = None

    monkeypatch.setattr(IPython, "get_ipython", lambda: ZMQInteractiveShell())
    monkeypatch.setattr("IPython.display.display",
                        lambda data, **kw: displayed.append(data))

    w._ipython_display_()
    assert len(displayed) == 1, f"{widget.__name__}: sibling emitted despite save_state=True"
