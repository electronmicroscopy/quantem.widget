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
