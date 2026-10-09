"""Display-filter contract: view transforms only, raw counts stay intact.

The helper backs the ShowDiffraction denoise views and is the reference for the
Show2D/Show3D browser filter, so these tests are the workflows a microscopist
actually runs: leave the default alone and see raw counts, turn on a binned
Anscombe smooth for a sparse map and see speckle drop.
"""


import numpy as np


from quantem.widget.utils.display_filter import apply_display_filter, format_display_filter_banner


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


def test_binned_anscombe_suppresses_speckle_keeps_shape():
    """A 2x-binned Anscombe smooth of a sparse Poisson map cuts high-frequency
    speckle while the display array keeps the raw (n_rows, n_cols) shape."""
    from scipy import ndimage

    counts = _sparse_eds_map()
    view = apply_display_filter(counts, mode="anscombe", sigma=8, spatial_bin=2)
    assert view.shape == counts.shape

    def high_freq_energy(a):
        return float(np.var(a - ndimage.gaussian_filter(a, 4.0)))

    assert high_freq_energy(view) < 0.2 * high_freq_energy(counts)


def test_bin2_zoom_back_preserves_odd_shapes():
    """Odd-sized survey crops keep their shape through the bin2 round trip."""
    counts = _sparse_eds_map(shape=(257, 255))
    view = apply_display_filter(counts, mode="gaussian", sigma=4, spatial_bin=2)
    assert view.shape == (257, 255)


def test_banner_announces_active_reduction_only():
    """The one-line notice appears when a reduction is active and tells the
    user how to get native counts back; the lossless default stays silent."""
    banner = format_display_filter_banner("anscombe", 8, spatial_bin=2)
    assert banner == "denoise: anscombe σ=8 bin2 (set denoise='none' for raw counts)"
    assert format_display_filter_banner("none", 4) == ""
    assert "bin2" in format_display_filter_banner("none", 0, spatial_bin=2)
