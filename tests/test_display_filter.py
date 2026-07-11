"""Display-filter contract: view transforms only, raw counts stay intact.

The helper backs the Show2D/Show3D display-filter knobs, so these tests are
the workflows a microscopist actually runs: leave the default alone and see
raw counts, turn on bin2_anscombe for a sparse EDS map and see speckle drop,
blend chemistry on HAADF without white atom cores.
"""

import numpy as np

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
        view = apply_display_filter(counts, filter=mode)
        np.testing.assert_allclose(view, counts)
        assert view.dtype == np.float32
    assert counts.base is None and counts.dtype == np.float32  # input untouched


def test_bin2_anscombe_suppresses_speckle_keeps_shape():
    """bin2_anscombe on a sparse Poisson map cuts high-frequency speckle
    while the display array keeps the raw (n_rows, n_cols) shape."""
    from scipy import ndimage

    counts = _sparse_eds_map()
    view = apply_display_filter(counts, filter="bin2_anscombe", sigma=8)
    assert view.shape == counts.shape

    def high_freq_energy(a):
        return float(np.var(a - ndimage.gaussian_filter(a, 4.0)))

    assert high_freq_energy(view) < 0.2 * high_freq_energy(counts)


def test_bin2_zoom_back_preserves_odd_shapes():
    """Odd-sized survey crops keep their shape through the bin2 round trip."""
    counts = _sparse_eds_map(shape=(257, 255))
    view = apply_display_filter(counts, filter="bin2", sigma=4)
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
    assert banner == "display: bin2_anscombe σ=8 (set display_filter='none' for raw counts)"
    assert format_display_filter_banner("none", 4) == ""
    assert "bin2" in format_display_filter_banner("none", 0, spatial_bin=2)


def test_show2d_default_view_bit_identical_and_raw_untouched(capsys):
    """A plain Show2D(map) shows exactly the stored counts: no banner, and
    the wire bytes match a raw float32 pack of the data."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    widget = Show2D(counts, verbose=False)
    assert widget.display_filter == "none"
    assert widget.display_filter_banner == ""
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
    raw_bytes = widget.frame_bytes
    widget.display_filter = "bin2_anscombe"
    assert widget.frame_bytes != raw_bytes
    out = capsys.readouterr().out
    assert out.count("display: bin2_anscombe") == 1
    assert "display_filter='none'" in out
    sigma_bytes = widget.frame_bytes
    widget.display_sigma = 10.0
    assert widget.frame_bytes != sigma_bytes  # sigma change re-filters too
    widget.display_filter = "none"
    assert widget.frame_bytes == raw_bytes
    np.testing.assert_array_equal(widget._data[0], kept)  # raw counts intact


def test_show2d_gallery_filters_scalar_panels_and_persists_state():
    """A raw-vs-filtered A/B gallery: per-panel filtering applies to every
    scalar panel, and the three knobs round-trip through saved state."""
    from quantem.widget import Show2D

    counts = _sparse_eds_map(shape=(64, 64))
    widget = Show2D(
        [counts, counts],
        display_filter="bin2_anscombe",
        display_sigma=8,
        verbose=False,
    )
    state = widget.state_dict()
    assert state["display_filter"] == "bin2_anscombe"
    assert state["display_sigma"] == 8.0
    assert state["spatial_bin"] == 1
    restored = Show2D([counts, counts], verbose=False)
    restored.load_state_dict(state)
    assert restored.display_filter == "bin2_anscombe"
    assert restored.frame_bytes == widget.frame_bytes


def test_show3d_filter_knobs_rerender_and_never_mutate(capsys):
    """A scrubbable EDS stack: default playback buffer is raw, turning the
    filter on re-sends a filtered buffer (with the banner), sigma re-filters,
    and none restores the identical raw buffer while .array stays intact."""
    from quantem.widget import Show3D

    stack = np.stack([_sparse_eds_map(seed=s, shape=(64, 64)) for s in range(3)])
    kept = stack.copy()
    widget = Show3D(stack, verbose=False, offline=False)
    assert widget.display_filter == "none"
    widget._send_buffer(0)  # what the browser's first prefetch triggers
    raw_buffer = widget._buffer_bytes
    assert raw_buffer
    widget.display_filter = "bin2_anscombe"
    assert widget._buffer_bytes != raw_buffer
    assert "display: bin2_anscombe" in capsys.readouterr().out
    sigma_buffer = widget._buffer_bytes
    widget.display_sigma = 12.0
    assert widget._buffer_bytes != sigma_buffer
    widget.display_filter = "none"
    assert widget._buffer_bytes == raw_buffer
    np.testing.assert_array_equal(widget._data, kept)
    state = widget.state_dict()
    assert state["display_filter"] == "none" and state["display_sigma"] == 12.0


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
        display_filter="bin2_anscombe",
        display_sigma=8,
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
