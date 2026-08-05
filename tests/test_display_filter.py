"""Display-filter contract: view transforms only, raw counts stay intact.

The helper backs the Show2D/Show3D display-filter knobs, so these tests are
the workflows a microscopist actually runs: leave the default alone and see
raw counts, turn on bin2_anscombe for a sparse EDS map and see speckle drop,
blend chemistry on HAADF without white atom cores.
"""

import numpy as np
import pytest

from quantem.widget.utils.display_filter import (
    apply_display_filter,
    blend_map_on_haadf,
    format_display_filter_banner,
    magenta_cmap,
)


def _sparse_eds_map(seed: int = 7, shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Synthetic sparse EDS map: Poisson counts on a faint lattice of dots."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    lattice = 0.25 * (1 + np.cos(2 * np.pi * rows / 16) * np.cos(2 * np.pi * cols / 16))
    return rng.poisson(lattice).astype(np.float32)


def test_default_filter_is_lossless_identity():
    """The default view shows exactly the stored counts (house rule 2)."""
    counts = _sparse_eds_map()
    for mode in ("none", "off", "raw"):
        view = apply_display_filter(counts, mode=mode)
        np.testing.assert_allclose(view, counts)
        assert view.dtype == np.float32
    assert counts.base is None and counts.dtype == np.float32  # input untouched


def test_bin2_anscombe_suppresses_speckle_keeps_shape():
    """bin2_anscombe on a sparse Poisson map cuts high-frequency speckle
    while the display array keeps the raw (n_rows, n_cols) shape."""
    from scipy import ndimage

    counts = _sparse_eds_map()
    view = apply_display_filter(counts, mode="bin2_anscombe", sigma=8)
    assert view.shape == counts.shape

    def high_freq_energy(a):
        return float(np.var(a - ndimage.gaussian_filter(a, 4.0)))

    assert high_freq_energy(view) < 0.2 * high_freq_energy(counts)


def test_bin2_zoom_back_preserves_odd_shapes():
    """Odd-sized survey crops keep their shape through the bin2 round trip."""
    counts = _sparse_eds_map(shape=(257, 255))
    view = apply_display_filter(counts, mode="bin2", sigma=4)
    assert view.shape == (257, 255)


def test_blend_never_white_at_bright_columns():
    """Chemistry on HAADF: a saturated map pixel on a bright column renders
    magenta, not white (the whole point of the fixed blend)."""
    map_01 = np.ones((32, 32), dtype=np.float32)
    haadf_01 = np.ones((32, 32), dtype=np.float32)
    rgb = blend_map_on_haadf(map_01, haadf_01, alpha=0.95, haadf_gain=0.35)
    assert rgb.shape == (32, 32, 3)
    red, green, blue = rgb[16, 16]
    assert green < 0.75, f"white-ish blend: rgb={rgb[16, 16]}"
    assert red > green and blue > green  # magenta hue survives full brightness
    top = magenta_cmap()(1.0)[:3]
    assert not np.allclose(top, (1.0, 1.0, 1.0), atol=0.05)


def test_banner_announces_active_reduction_only():
    """The one-line notice appears when a reduction is active and tells the
    user how to get native counts back; the lossless default stays silent."""
    banner = format_display_filter_banner("bin2_anscombe", 8)
    assert banner == "denoise: bin2_anscombe σ=8 (set denoise='none' for raw counts)"
    assert format_display_filter_banner("none", 4) == ""
    assert "bin2" in format_display_filter_banner("none", 0, spatial_bin=2)


def test_show2d_default_view_bit_identical_and_raw_untouched(capsys):
    """A plain Show2D(map) shows exactly the stored counts: no banner, and
    the wire bytes match a raw float32 pack of the data."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    widget = Show2D(counts, verbose=False)
    assert widget.denoise == "none"
    assert widget.denoise_banner == ""
    assert "display:" not in capsys.readouterr().out
    # frame_bytes is the raw float32 pack (zero-padded to a multiple of 3)
    sent = np.frombuffer(widget.frame_bytes, dtype=np.float32, count=counts.size)
    np.testing.assert_array_equal(sent.reshape(counts.shape), counts)


def test_show2d_filter_knobs_rerender_live(capsys):
    """Turning on bin2_anscombe re-filters the view in place (no reload),
    announces the reduction once, and going back to none restores raw bytes
    while the stored array never changes."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    kept = counts.copy()
    widget = Show2D(counts, verbose=False)
    # These assertions watch the pixels Python puts on the wire, so pin the
    # widget to the Python filter path. By default the browser owns the
    # filter and Python ships raw (see
    # test_browser_owns_display_filter_by_default).
    widget._webgpu_filter_ok = False
    raw_bytes = widget.frame_bytes
    widget.denoise = "bin2_anscombe"  # compound alias -> (anscombe, bin 2)
    assert widget.frame_bytes != raw_bytes
    out = capsys.readouterr().out
    assert out.count("denoise: anscombe σ=4 bin2") == 1
    assert "denoise='none'" in out
    sigma_bytes = widget.frame_bytes
    widget.denoise_sigma = 10.0
    assert widget.frame_bytes != sigma_bytes  # sigma change re-filters too
    # The compound alias raised the bin knob; raw needs both knobs reset.
    widget.denoise = "none"
    widget.denoise_bin = 1
    assert widget.frame_bytes == raw_bytes
    np.testing.assert_array_equal(widget._data[0], kept)  # raw counts intact


def test_show2d_gallery_filters_scalar_panels_and_persists_state():
    """A raw-vs-filtered A/B gallery: per-panel filtering applies to every
    scalar panel, and the three knobs round-trip through saved state."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    widget = Show2D(
        [counts, counts],
        denoise="bin2_anscombe",
        denoise_sigma=8,
        verbose=False,
    )
    state = widget.state_dict()
    assert state["denoise"] == "anscombe"  # compound kwarg normalized
    assert state["denoise_bin"] == 2
    assert state["denoise_sigma"] == 8.0
    restored = Show2D([counts, counts], verbose=False)
    restored.load_state_dict(state)
    assert restored.denoise == "anscombe"
    assert restored.frame_bytes == widget.frame_bytes


def test_show3d_filter_knobs_rerender_and_never_mutate(capsys):
    """A scrubbable EDS stack: default playback buffer is raw, turning the
    filter on re-sends a filtered buffer (with the banner), sigma re-filters,
    and none restores the identical raw buffer while .array stays intact."""
    from quantem.widget import Show3D

    stack = np.stack([_sparse_eds_map(seed=s, shape=(64, 64)) for s in range(3)])
    kept = stack.copy()
    widget = Show3D(stack, verbose=False, offline=False)
    # These assertions watch the pixels Python puts on the wire, so pin the
    # widget to the Python filter path. By default the browser owns the
    # filter and Python ships raw (see
    # test_browser_owns_display_filter_by_default).
    widget._webgpu_filter_ok = False
    assert widget.denoise == "none"
    widget._send_buffer(0)  # what the browser's first prefetch triggers
    raw_buffer = widget._buffer_bytes
    assert raw_buffer
    widget.denoise = "bin2_anscombe"  # compound alias -> (anscombe, bin 2)
    assert widget._buffer_bytes != raw_buffer
    assert "denoise: anscombe" in capsys.readouterr().out
    sigma_buffer = widget._buffer_bytes
    widget.denoise_sigma = 12.0
    assert widget._buffer_bytes != sigma_buffer
    # The compound alias raised the bin knob; raw needs both knobs reset.
    widget.denoise = "none"
    widget.denoise_bin = 1
    assert widget._buffer_bytes == raw_buffer
    np.testing.assert_array_equal(widget._data, kept)
    state = widget.state_dict()
    assert state["denoise"] == "none" and state["denoise_sigma"] == 12.0


def test_show2d_per_panel_filter_lists():
    """A raw-vs-filtered A/B gallery from ONE constructor call: per-panel
    filter/sigma lists, panel 0 stays bit-identical raw while panel 1 is
    denoised, and UI edits scope to the selected panel."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    widget = Show2D(
        [counts, counts],
        denoise=["none", "bin2_anscombe"],
        denoise_sigma=[0.0, 8.0],
        verbose=False,
    )
    # These assertions watch the pixels Python puts on the wire, so pin the
    # widget to the Python filter path. By default the browser owns the
    # filter and Python ships raw (see
    # test_browser_owns_display_filter_by_default).
    widget._webgpu_filter_ok = False
    assert widget.denoise_scope == "panel"  # sequence => per-panel scope
    n = counts.size
    sent = np.frombuffer(widget.frame_bytes[: 2 * n * 4], dtype=np.float32).reshape(2, 64, 64)
    np.testing.assert_array_equal(sent[0], counts)  # panel 0: raw, bit-identical
    assert not np.array_equal(sent[1], counts)  # panel 1: filtered view
    assert "p1:anscombe" in widget.denoise_banner and "bin2" in widget.denoise_banner
    # Panel scope: editing the knob touches only the selected panel
    widget.selected_idx = 0
    assert widget.denoise == "none"  # editor mirrors the selected panel
    widget.denoise = "gaussian"
    assert widget.denoise_modes == ["gaussian", "anscombe"]
    widget.selected_idx = 1
    assert widget.denoise == "anscombe"


def test_show2d_eight_panel_eds_gallery_scoped_and_linked_edits():
    """An 8-panel EDS gallery (4 sparse maps denoised, 4 raw references):
    each panel packs with its own knobs, clicking a panel and turning a knob
    edits only that panel while unlinked, and relinking the Denoise switch
    (denoise_scope="all") broadcasts the next edit to every panel."""
    from quantem.widget import Show2D

    maps = [_sparse_eds_map(seed=s, shape=(64, 64)) for s in range(8)]
    widget = Show2D(
        maps,
        denoise=["anscombe"] * 4 + ["none"] * 4,
        denoise_sigma=[6.0, 8.0, 10.0, 12.0] + [4.0] * 4,
        verbose=False,
    )
    # These assertions watch the pixels Python puts on the wire, so pin the
    # widget to the Python filter path. By default the browser owns the
    # filter and Python ships raw (see
    # test_browser_owns_display_filter_by_default).
    widget._webgpu_filter_ok = False
    assert widget.denoise_scope == "panel"  # per-panel lists => unlinked
    n = 64 * 64
    sent = np.frombuffer(widget.frame_bytes[: 8 * n * 4], dtype=np.float32).reshape(8, 64, 64)
    for panel in range(4):  # denoised maps: filtered view, raw data intact
        assert not np.array_equal(sent[panel], maps[panel])
        np.testing.assert_array_equal(widget._data[panel], maps[panel])
    for panel in range(4, 8):  # reference panels: bit-identical raw
        np.testing.assert_array_equal(sent[panel], maps[panel])
    # Click panel 5 (a raw reference), turn on gaussian: only panel 5 changes.
    widget.selected_idx = 5
    assert widget.denoise == "none"  # editor knobs mirror the clicked panel
    widget.denoise = "gaussian"
    assert widget.denoise_modes == ["anscombe"] * 4 + ["none", "gaussian", "none", "none"]
    # Relink through the Denoise switch in the Link group: the next edit
    # broadcasts to every panel.
    widget.denoise_scope = "all"
    widget.denoise = "anscombe"
    assert widget.denoise_modes == ["anscombe"] * 8


def test_show2d_underlay_composes_chemistry_on_haadf():
    """underlay=True on (haadf, map) adds the blend as a third RGB panel:
    haadf gray | map | map-on-HAADF, with bright columns colored (not white),
    and the alpha slider re-blends live without touching the sources."""
    from quantem.widget import Show2D

    rng = np.random.default_rng(3)
    haadf = rng.random((64, 64)).astype(np.float32)
    eds_map = _sparse_eds_map(shape=(64, 64))
    widget = Show2D(
        [haadf, eds_map],
        underlay=True,
        denoise="bin2_anscombe",
        denoise_sigma=8,
        cmap="magenta",
        verbose=False,
    )
    assert widget.n_images == 3
    assert widget.is_rgb == [False, False, True]
    assert widget.labels[-1] == "map on HAADF"
    blend = widget._rgb_frames[-1]
    assert blend.shape == (64, 64, 3)
    bright = blend[blend.max(axis=-1) > 0.5]
    assert bright.size == 0 or not np.any(np.all(bright > 0.9, axis=-1)), "white cores in blend"
    before = blend.copy()
    widget.underlay_alpha = 0.5
    assert not np.array_equal(widget._rgb_frames[-1], before)  # live re-blend
    np.testing.assert_array_equal(widget._data[0], haadf)  # sources untouched
    np.testing.assert_array_equal(widget._data[1], eds_map)


def test_underlay_slider_repacks_the_synced_rgb_block_the_browser_reads():
    """Moving a blend slider must repaint the browser, not just the Python-side
    ``_rgb_frames``. The canvas decodes the synced ``frame_bytes`` trait, so the
    regression this guards is a blend that re-blends internally yet ships the
    stale RGB block: turning the alpha or HAADF-gain knob has to change both the
    ``frame_bytes`` buffer and the last-panel RGB block decoded out of it.
    """
    from quantem.widget import Show2D

    rng = np.random.default_rng(11)
    haadf = rng.random((128, 128)).astype(np.float32)
    eds_map = _sparse_eds_map(shape=(128, 128))
    widget = Show2D(
        [haadf, eds_map],
        underlay=True,
        denoise="anscombe",
        denoise_bin=2,
        denoise_sigma=8,
        cmap="magenta",
        verbose=False,
    )
    per = 128 * 128
    rgb_start = 2 * per  # panels 0,1 are one grayscale block each; blend is last
    rgb_floats = 3 * per

    def rgb_block(widget):
        floats = np.frombuffer(
            widget.frame_bytes[: (rgb_start + rgb_floats) * 4], dtype=np.float32
        )
        return floats[rgb_start : rgb_start + rgb_floats].reshape(128, 128, 3).copy()

    before_bytes = bytes(widget.frame_bytes)
    before_block = rgb_block(widget)
    before_blend = widget._compute_underlay_blend().copy()

    widget.underlay_alpha = 0.2
    assert bytes(widget.frame_bytes) != before_bytes  # synced buffer changed
    assert not np.array_equal(rgb_block(widget), before_block)  # decoded block changed
    assert not np.array_equal(widget._compute_underlay_blend(), before_blend)

    # The HAADF-gain knob is the second blend control and must repaint too.
    gain_bytes = bytes(widget.frame_bytes)
    gain_block = rgb_block(widget)
    widget.underlay_haadf_gain = 0.9
    assert bytes(widget.frame_bytes) != gain_bytes
    assert not np.array_equal(rgb_block(widget), gain_block)

    np.testing.assert_array_equal(widget._data[0], haadf)  # sources untouched
    np.testing.assert_array_equal(widget._data[1], eds_map)


def test_underlay_fig4_knobs_restretch_the_synced_blend():
    """The Fig4 parity knobs (display_gamma, stretch_percentiles) re-blend live
    just like the alpha/gain sliders: each turns the synced frame_bytes buffer
    and the computed blend, and the two raw sources stay untouched."""
    from quantem.widget import Show2D

    rng = np.random.default_rng(21)
    haadf = rng.random((96, 96)).astype(np.float32)
    eds_map = _sparse_eds_map(shape=(96, 96))
    widget = Show2D(
        [haadf, eds_map], underlay=True, cmap="magenta", verbose=False
    )
    gamma_bytes = bytes(widget.frame_bytes)
    gamma_blend = widget._compute_underlay_blend().copy()
    widget.display_gamma = 1.3
    assert bytes(widget.frame_bytes) != gamma_bytes
    assert not np.array_equal(widget._compute_underlay_blend(), gamma_blend)

    stretch_bytes = bytes(widget.frame_bytes)
    widget.stretch_percentiles = [10.0, 95.0]
    assert bytes(widget.frame_bytes) != stretch_bytes

    np.testing.assert_array_equal(widget._data[0], haadf)  # sources untouched
    np.testing.assert_array_equal(widget._data[1], eds_map)


def test_show2d_dual_composite_is_magenta_plus_green_and_gain_scrubs():
    """underlay_mode='dual' on two maps composes a magenta+green RGB panel:
    map A -> magenta (R, B), map B -> green (G). The per-channel dual_gain
    sliders re-blend live and never touch the stored maps."""
    from quantem.widget import Show2D

    map_a = np.zeros((32, 32), np.float32)
    map_a[8, 8] = 12.0  # A-only site -> should read magenta
    map_b = np.zeros((32, 32), np.float32)
    map_b[24, 24] = 12.0  # B-only site -> should read green
    widget = Show2D(
        [map_a, map_b], underlay=True, underlay_mode="dual", verbose=False
    )
    assert widget.n_images == 3
    assert widget.is_rgb == [False, False, True]
    assert widget.labels[-1] == "dual composite"
    blend = widget._rgb_frames[-1]
    assert blend.shape == (32, 32, 3)
    a_site = blend[8, 8]
    assert a_site[0] > a_site[1] and a_site[2] > a_site[1], "A site should read magenta"
    b_site = blend[24, 24]
    assert b_site[1] > b_site[0] and b_site[1] > b_site[2], "B site should read green"

    before_bytes = bytes(widget.frame_bytes)
    widget.dual_gain = [2.0, 0.5]
    assert bytes(widget.frame_bytes) != before_bytes
    np.testing.assert_array_equal(widget._data[0], map_a)  # sources untouched
    np.testing.assert_array_equal(widget._data[1], map_b)


def test_show2d_webgpu_negotiation_ships_raw_frames():
    """The browser reports a real WebGPU adapter (_webgpu_filter_ok=True): the
    widget ships raw counts for the WGSL port to filter client-side, keeps
    announcing the reduction, and repacks the Python-filtered view when the
    flag drops again (browser without WebGPU)."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    widget = Show2D(counts, denoise="bin2_anscombe", denoise_sigma=8, verbose=False)
    n_bytes = counts.size * 4
    # Default: the browser owns the filter, so RAW counts ship without any
    # negotiation round trip. Regression guard - this used to default to the
    # Python scipy path, which re-sent the whole frame over comm on every knob
    # edit and made live sigma drags unusable.
    sent = np.frombuffer(widget.frame_bytes[:n_bytes], dtype=np.float32).reshape(64, 64)
    np.testing.assert_array_equal(sent, counts)
    assert "anscombe" in widget.denoise_banner  # reduction still announced
    widget._webgpu_filter_ok = False  # software adapter: Python fallback
    python_view = np.frombuffer(widget.frame_bytes[:n_bytes], dtype=np.float32).reshape(64, 64).copy()
    assert not np.array_equal(python_view, counts)  # scipy path filtered it
    widget._webgpu_filter_ok = True  # back on a real adapter
    again = np.frombuffer(widget.frame_bytes[:n_bytes], dtype=np.float32).reshape(64, 64)
    np.testing.assert_array_equal(again, counts)


def test_deprecated_display_filter_kwargs_warn_and_still_apply():
    """The display_filter-era kwargs still work for one release but warn: a
    microscopist reopening an old notebook sees the map denoised and a clear
    pointer to the new denoise= name."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    with pytest.warns(DeprecationWarning, match="display_filter is deprecated"):
        widget = Show2D(counts, display_filter="anscombe", display_sigma=8, verbose=False)
    assert widget.denoise == "anscombe"  # deprecated alias filled the new kwarg
    assert widget.denoise_sigma == 8.0


def test_new_denoise_kwarg_wins_over_deprecated_alias_and_still_warns():
    """Passing both the new denoise= and its deprecated display_filter= alias:
    the new kwarg wins the value (even at its "none" default) while the
    deprecated alias still raises its warning."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    with pytest.warns(DeprecationWarning, match="display_filter is deprecated"):
        widget = Show2D(counts, denoise="none", display_filter="anscombe", verbose=False)
    assert widget.denoise == "none"  # explicit new kwarg wins even at its default


def test_show2d_html_export_ships_raw_frames_and_knobs():
    """An exported HTML clone carries raw counts plus the filter knobs, so the
    kernel-less page scrubs filter/sigma live through the browser port."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    widget = Show2D(counts, denoise="anscombe", denoise_sigma=6, verbose=False)
    clone = widget._clone_for_html_export(quantized=False)
    try:
        assert clone._webgpu_filter_ok is True
        sent = np.frombuffer(clone.frame_bytes[: counts.size * 4], dtype=np.float32).reshape(64, 64)
        np.testing.assert_array_equal(sent, counts)
        assert list(clone.denoise_modes) == ["anscombe"]
        assert list(clone.denoise_sigmas) == [6.0]
    finally:
        clone.close()


def test_show3d_html_export_enables_browser_filter_for_raw_stack():
    """A full-precision Show3D export keeps raw frames and tells the offline
    browser to apply its restored denoise settings before painting the canvas."""
    from quantem.widget import Show3D

    counts = np.stack([_sparse_eds_map(shape=(32, 32)) for _ in range(3)])
    widget = Show3D(counts, denoise="gaussian", denoise_sigma=6, offline=False)
    clone = widget._clone_for_html_export(quantized=False)
    try:
        # C1: full export is packed after construction, expect browser ownership.
        assert clone._webgpu_filter_ok is True
        sent = np.frombuffer(clone._offline_float_stack, dtype=np.float32).reshape(counts.shape)
        np.testing.assert_array_equal(sent, counts)
        assert clone.denoise == "gaussian"
        assert clone.denoise_sigma == 6.0
        assert clone.denoise_enabled is True
    finally:
        clone.close()
        widget.close()


def test_gallery_pad_ratio_announces_a_real_border():
    """A gallery can use display padding without claiming a phantom border."""
    from quantem.widget import Show2D

    a = _sparse_eds_map(shape=(64, 64))
    b = _sparse_eds_map(seed=9, shape=(64, 64))
    gallery = Show2D([a, b], pad_ratio=0.1, verbose=False)
    assert "pad 10%" in gallery.view_banner
    assert gallery.height > 64
    assert gallery.width > 64


def test_diff_reference_survives_state_round_trip():
    """A drift A/B gallery in diff mode: point the diff at panel 1, save, then
    reload into a fresh widget -> the saved reference comes back instead of
    silently reverting to panel 0."""
    from quantem.widget import Show2D

    a = _sparse_eds_map(shape=(64, 64))
    b = _sparse_eds_map(seed=9, shape=(64, 64))
    widget = Show2D([a, b], diff_mode=True, verbose=False)
    widget.diff_reference = 1
    restored = Show2D([a, b], verbose=False)
    restored.load_state_dict(widget.state_dict())
    assert restored.diff_reference == 1


def test_set_denoise_applies_to_all_panels_by_default():
    """set_denoise is the imperative twin of denoise=: on a 2-panel gallery a
    bare call denoises both panels, folds the bin knob, and chains."""
    from quantem.widget import Show2D

    a = _sparse_eds_map(shape=(64, 64))
    b = _sparse_eds_map(seed=9, shape=(64, 64))
    widget = Show2D([a, b], verbose=False)
    returned = widget.set_denoise("anscombe", sigma=8, bin=2)
    assert returned is widget  # chainable, like crop_to_view / set_roi
    assert widget.denoise_modes == ["anscombe", "anscombe"]
    assert widget.denoise_bins == [2, 2]
    assert widget.denoise_sigmas == [8.0, 8.0]


def test_set_denoise_scoped_to_one_panel_leaves_others_raw():
    """Passing panels=[1] builds a raw-vs-denoised A/B from one imperative
    call: only panel 1 changes, denoise_scope switches to per-panel, and panel
    0 stays bit-identical raw while the stored arrays never change."""
    from quantem.widget import Show2D

    a = _sparse_eds_map(shape=(64, 64))
    b = _sparse_eds_map(seed=9, shape=(64, 64))
    widget = Show2D([a, b], verbose=False)
    # These assertions watch the pixels Python puts on the wire, so pin the
    # widget to the Python filter path. By default the browser owns the
    # filter and Python ships raw (see
    # test_browser_owns_display_filter_by_default).
    widget._webgpu_filter_ok = False
    widget.set_denoise("anscombe", sigma=8, panels=[1])
    assert widget.denoise_modes == ["none", "anscombe"]
    assert widget.denoise_scope == "panel"
    n = a.size
    sent = np.frombuffer(widget.frame_bytes[: 2 * n * 4], dtype=np.float32).reshape(2, 64, 64)
    np.testing.assert_array_equal(sent[0], a)  # panel 0: raw, bit-identical
    assert not np.array_equal(sent[1], b)  # panel 1: filtered view
    np.testing.assert_array_equal(widget._data[1], b)  # stored counts intact


def test_explicit_all_scope_with_per_panel_sequence_is_rejected():
    """denoise_scope='all' broadcasts one setting, so pairing it with a
    per-panel sequence is a contradiction: the widget raises instead of
    silently dropping the explicit scope."""
    from quantem.widget import Show2D

    a = _sparse_eds_map(shape=(64, 64))
    with pytest.raises(ValueError, match="broadcasts one setting"):
        Show2D([a, a], denoise=["none", "anscombe"], denoise_scope="all", verbose=False)


def test_browser_owns_display_filter_by_default():
    """Python must never run the scipy display filter unless the frontend says
    it has to. js/displayFilter.ts carries WGSL and CPU paths that match NumPy,
    CPU port, so every viewer can filter client-side; leaving it to Python cost
    a full frame re-send per knob edit (~0.5 s at 2048^2) and made the denoise
    sliders feel broken. Guards the default on every widget that filters."""
    import numpy as np

    from quantem.widget import Show2D, Show3D

    counts = _sparse_eds_map(shape=(32, 32))
    two_d = Show2D(counts, denoise="gaussian", denoise_sigma=4, verbose=False)
    assert two_d._webgpu_filter_ok is True

    stack = np.stack([counts, counts])
    three_d = Show3D(stack, denoise="gaussian", denoise_sigma=4, verbose=False)
    assert three_d._webgpu_filter_ok is True

    # With the default flag the wire frame is the raw frame: no scipy call.
    frame = np.asarray(counts, dtype=np.float32)
    np.testing.assert_array_equal(three_d._wire_frame(frame), frame)
