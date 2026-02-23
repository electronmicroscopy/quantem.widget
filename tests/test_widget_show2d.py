import numpy as np
import pytest
import torch

from quantem.widget import Show2D

try:
    import h5py  # type: ignore

    _HAS_H5PY = True
except Exception:
    h5py = None
    _HAS_H5PY = False

def test_show2d_single_numpy():
    """Single image from numpy array."""
    data = np.random.rand(32, 32).astype(np.float32)
    widget = Show2D(data)
    assert widget.n_images == 1
    assert widget.height == 32
    assert widget.width == 32
    assert len(widget.frame_bytes) > 0

def test_show2d_single_torch():
    """Single image from PyTorch tensor."""
    data = torch.rand(32, 32)
    widget = Show2D(data)
    assert widget.n_images == 1
    assert widget.height == 32
    assert widget.width == 32

def test_show2d_multiple_numpy():
    """Gallery mode from list of numpy arrays."""
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Show2D(images, labels=["A", "B", "C"])
    assert widget.n_images == 3
    assert widget.labels == ["A", "B", "C"]

def test_show2d_multiple_torch():
    """Gallery mode from list of PyTorch tensors."""
    images = [torch.rand(16, 16) for _ in range(3)]
    widget = Show2D(images)
    assert widget.n_images == 3

def test_show2d_3d_array():
    """3D array treated as multiple images."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.n_images == 5

def test_show2d_stats():
    """Statistics are computed correctly."""
    data = np.ones((16, 16), dtype=np.float32) * 42.0
    widget = Show2D(data)
    assert widget.stats_mean[0] == pytest.approx(42.0)
    assert widget.stats_min[0] == pytest.approx(42.0)
    assert widget.stats_max[0] == pytest.approx(42.0)
    assert widget.stats_std[0] == pytest.approx(0.0)

def test_show2d_colormap():
    """Colormap option is applied."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, cmap="viridis")
    assert widget.cmap == "viridis"

def test_show2d_different_sizes():
    """Gallery with different sized images resizes to largest."""
    images = [
        np.random.rand(16, 16).astype(np.float32),
        np.random.rand(32, 32).astype(np.float32),
    ]
    widget = Show2D(images)
    assert widget.height == 32
    assert widget.width == 32

def test_show2d_display_options():
    """Log scale and auto contrast options are accepted."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, log_scale=True, auto_contrast=True)
    assert widget.log_scale is True
    assert widget.auto_contrast is True

def test_show2d_title():
    """Title parameter is stored."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, title="Test Title")
    assert widget.title == "Test Title"

def test_show2d_default_labels():
    """Labels default to 'Image 1', 'Image 2', etc."""
    data = np.random.rand(3, 16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.labels == ["Image 1", "Image 2", "Image 3"]

def test_show2d_scale_bar():
    """Scale bar parameters are stored."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, pixel_size=1.5, scale_bar_visible=False)
    assert widget.pixel_size == pytest.approx(1.5)
    assert widget.scale_bar_visible is False

def test_show2d_ncols():
    """ncols parameter is stored."""
    data = np.random.rand(6, 16, 16).astype(np.float32)
    widget = Show2D(data, ncols=2)
    assert widget.ncols == 2

def test_show2d_canvas_size():
    """canvas_size parameter is stored."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, canvas_size=500)
    assert widget.canvas_size == 500

def test_show2d_show_controls():
    """show_controls can be toggled."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, show_controls=False)
    assert widget.show_controls is False

def test_show2d_show_stats():
    """show_stats can be toggled."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, show_stats=False)
    assert widget.show_stats is False

def test_show2d_disabled_tools_default():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.disabled_tools == []

def test_show2d_disabled_tools_custom():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, disabled_tools=["display", "ROI", "profile"])
    assert widget.disabled_tools == ["display", "roi", "profile"]

def test_show2d_disabled_tools_flags():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, disable_display=True, disable_navigation=True, disable_view=True)
    assert widget.disabled_tools == ["display", "navigation", "view"]

def test_show2d_disabled_tools_disable_all():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, disable_all=True, disable_display=True)
    assert widget.disabled_tools == ["all"]

def test_show2d_disabled_tools_unknown_raises():
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show2D(data, disabled_tools=["not_real"])

def test_show2d_disabled_tools_trait_assignment_normalizes():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    widget.disabled_tools = ["DISPLAY", "display", "roi"]
    assert widget.disabled_tools == ["display", "roi"]

def test_show2d_hidden_tools_default():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.hidden_tools == []

def test_show2d_hidden_tools_custom():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, hidden_tools=["display", "ROI", "profile"])
    assert widget.hidden_tools == ["display", "roi", "profile"]

def test_show2d_hidden_tools_flags():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, hide_display=True, hide_navigation=True, hide_view=True)
    assert widget.hidden_tools == ["display", "navigation", "view"]

def test_show2d_hidden_tools_hide_all():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, hide_all=True, hide_display=True)
    assert widget.hidden_tools == ["all"]

def test_show2d_hidden_tools_unknown_raises():
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show2D(data, hidden_tools=["not_real"])

def test_show2d_hidden_tools_trait_assignment_normalizes():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    widget.hidden_tools = ["DISPLAY", "display", "roi"]
    assert widget.hidden_tools == ["display", "roi"]

def test_show2d_constant_image_stats():
    """Constant image doesn't crash stats computation."""
    data = np.zeros((16, 16), dtype=np.float32)
    widget = Show2D(data)
    assert widget.stats_mean[0] == pytest.approx(0.0)
    assert widget.stats_std[0] == pytest.approx(0.0)

def test_show2d_single_image_is_3d_internally():
    """2D input is wrapped to 3D (1, H, W) internally."""
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    assert widget.n_images == 1
    # frame_bytes contains exactly 1 * 16 * 16 float32 values
    assert len(widget.frame_bytes) == 1 * 16 * 16 * 4

def test_show2d_large_gallery():
    """Large gallery (20 images) works."""
    data = np.random.rand(20, 8, 8).astype(np.float32)
    widget = Show2D(data)
    assert widget.n_images == 20
    assert len(widget.stats_mean) == 20
    assert len(widget.stats_min) == 20
    assert len(widget.stats_max) == 20
    assert len(widget.stats_std) == 20

def test_show2d_gallery_stats_per_image():
    """Stats are computed per image in gallery."""
    img1 = np.ones((8, 8), dtype=np.float32) * 10.0
    img2 = np.ones((8, 8), dtype=np.float32) * 20.0
    widget = Show2D([img1, img2])
    assert widget.stats_mean[0] == pytest.approx(10.0)
    assert widget.stats_mean[1] == pytest.approx(20.0)


def test_show2d_add_roi():
    """add_roi() creates an ROI and activates ROI mode."""
    data = np.ones((32, 32), dtype=np.float32) * 5.0
    w = Show2D(data)
    assert w.roi_active is False
    assert len(w.roi_list) == 0
    w.add_roi(row=16, col=16, shape="square")
    assert w.roi_active is True
    assert len(w.roi_list) == 1
    assert w.roi_list[0]["shape"] == "square"
    assert w.roi_list[0]["row"] == 16
    assert w.roi_list[0]["col"] == 16
    assert w.roi_selected_idx == 0

def test_show2d_add_roi_defaults():
    """add_roi() defaults to center of image when no position given."""
    data = np.ones((64, 128), dtype=np.float32)
    w = Show2D(data)
    w.add_roi()
    assert w.roi_list[0]["row"] == 32
    assert w.roi_list[0]["col"] == 64
    assert w.roi_list[0]["shape"] == "square"

def test_show2d_clear_rois():
    """clear_rois() removes all ROIs and deactivates ROI mode."""
    data = np.ones((32, 32), dtype=np.float32)
    w = Show2D(data)
    w.add_roi(row=10, col=10)
    w.add_roi(row=20, col=20)
    assert len(w.roi_list) == 2
    w.clear_rois()
    assert len(w.roi_list) == 0
    assert w.roi_selected_idx == -1
    assert w.roi_active is False

def test_show2d_roi_shape_methods():
    """roi_circle/square/rectangle/annular change ROI shape."""
    data = np.ones((32, 32), dtype=np.float32)
    w = Show2D(data)
    w.add_roi(row=16, col=16)
    w.roi_circle(radius=8)
    assert w.roi_list[0]["shape"] == "circle"
    assert w.roi_list[0]["radius"] == 8
    w.roi_square(half_size=6)
    assert w.roi_list[0]["shape"] == "square"
    assert w.roi_list[0]["radius"] == 6
    w.roi_rectangle(width=12, height=8)
    assert w.roi_list[0]["shape"] == "rectangle"
    assert w.roi_list[0]["width"] == 12
    assert w.roi_list[0]["height"] == 8
    w.roi_annular(inner=3, outer=7)
    assert w.roi_list[0]["shape"] == "annular"
    assert w.roi_list[0]["radius_inner"] == 3
    assert w.roi_list[0]["radius"] == 7

def test_show2d_delete_selected_roi():
    """delete_selected_roi() removes the selected ROI."""
    data = np.ones((32, 32), dtype=np.float32)
    w = Show2D(data)
    w.add_roi(row=10, col=10)
    w.add_roi(row=20, col=20)
    assert len(w.roi_list) == 2
    assert w.roi_selected_idx == 1
    w.delete_selected_roi()
    assert len(w.roi_list) == 1
    assert w.roi_list[0]["row"] == 10

def test_show2d_add_roi_chaining():
    """ROI methods return Self for chaining."""
    data = np.ones((32, 32), dtype=np.float32)
    w = Show2D(data)
    result = w.add_roi(row=16, col=16).roi_square(half_size=5)
    assert result is w

# ── FFT Hann Window Ground Truth ──────────────────────────────────────────

def _js_hann_1d(n):
    """Port of 1D Hann vector from applyHannWindow2D (js/webgpu-fft.ts line 178-180).
    Extracts the exact loop: hannW[i] = 0.5 * (1 - cos(2*pi*i / wDenom))."""
    denom = n - 1 if n > 1 else 1
    return np.array([0.5 * (1 - np.cos(2 * np.pi * i / denom)) for i in range(n)])

def _js_apply_hann_window_2d(data, width, height):
    """Exact Python port of applyHannWindow2D from js/webgpu-fft.ts lines 175-186.
    Applies in-place and returns the modified array for convenience."""
    hann_w = _js_hann_1d(width)
    hann_h = _js_hann_1d(height)
    # Port of: data[offset + c] *= hr * hannW[c]  (outer product multiplication)
    data *= np.outer(hann_h, hann_w)
    return data

def _next_pow2(n):
    """Port of nextPow2 from js/webgpu-fft.ts."""
    p = 1
    while p < n:
        p *= 2
    return p

def test_show2d_hann_js_port_matches_numpy():
    """Ported JS 1D Hann window must match np.hanning for all practical sizes."""
    for n in [8, 15, 20, 32, 64]:
        js_1d = _js_hann_1d(n)
        np_hann = np.hanning(n)
        np.testing.assert_allclose(js_1d, np_hann, atol=1e-10,
                                   err_msg=f"JS port != np.hanning at N={n}")
        assert js_1d[0] == 0.0, f"First element should be 0 for N={n}"
        assert js_1d[-1] == 0.0, f"Last element should be 0 for N={n}"

def test_show2d_hann_js_port_2d_separable():
    """Ported JS 2D Hann = outer product of np.hanning (all edges zero)."""
    np.random.seed(42)
    crop = np.random.rand(15, 20).astype(np.float32)
    result = _js_apply_hann_window_2d(crop.copy(), 20, 15)
    expected = crop * np.outer(np.hanning(15), np.hanning(20)).astype(np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert np.all(result[0, :] == 0), "Top edge should be zero"
    assert np.all(result[-1, :] == 0), "Bottom edge should be zero"
    assert np.all(result[:, 0] == 0), "Left edge should be zero"
    assert np.all(result[:, -1] == 0), "Right edge should be zero"

def test_show2d_hann_then_pad_fft_pipeline():
    """Full pipeline using JS port: window(crop) → pad → FFT matches NumPy equivalent."""
    np.random.seed(123)
    crop_h, crop_w = 15, 20
    pad_h, pad_w = _next_pow2(crop_h), _next_pow2(crop_w)
    crop_data = np.random.rand(crop_h, crop_w).astype(np.float32)
    # Correct pipeline (what our JS now does): window at crop dims, then pad
    windowed = _js_apply_hann_window_2d(crop_data.copy(), crop_w, crop_h)
    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:crop_h, :crop_w] = windowed
    fft_correct = np.abs(np.fft.fftshift(np.fft.fft2(padded)))
    # Verify seamless transition: windowed edges are zero → pad region is zero
    assert windowed[-1, crop_w // 2] == 0.0, "Bottom edge of windowed crop should be 0"
    assert padded[crop_h, crop_w // 2] == 0.0, "Pad region should be 0"
    # Wrong pipeline (what our JS used to do): pad first, then window at padded dims
    padded_wrong = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded_wrong[:crop_h, :crop_w] = crop_data
    _js_apply_hann_window_2d(padded_wrong, pad_w, pad_h)
    fft_wrong = np.abs(np.fft.fftshift(np.fft.fft2(padded_wrong)))
    # Wrong pipeline has discontinuity at crop boundary
    assert padded_wrong[crop_h - 1, crop_w // 2] != 0.0, "Wrong: crop boundary not zero"
    # Correct pipeline should have less total energy (no leakage)
    energy_correct = np.sum(fft_correct ** 2)
    energy_wrong = np.sum(fft_wrong ** 2)
    assert energy_correct < energy_wrong, "Correct windowing should have less spectral energy"

def test_show2d_fft_window_trait():
    """fft_window trait defaults True and roundtrips via state_dict."""
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data)
    assert w.fft_window is True
    w2 = Show2D(data, fft_window=False)
    assert w2.fft_window is False
    sd = w2.state_dict()
    assert sd["fft_window"] is False
    w3 = Show2D(data, state=sd)
    assert w3.fft_window is False

# ── State Protocol ────────────────────────────────────────────────────────

def test_show2d_state_dict_roundtrip():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, cmap="viridis", log_scale=True, auto_contrast=True,
               title="Test", pixel_size=2.5, show_fft=True, fft_window=False,
               disabled_tools=["display", "view"], hidden_tools=["stats"])
    w.roi_active = True
    w.roi_list = [{"shape": "circle", "row": 10, "col": 15, "radius": 5}]
    w.roi_selected_idx = 0
    sd = w.state_dict()
    assert sd["fft_window"] is False
    w2 = Show2D(data, state=sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.auto_contrast is True
    assert w2.title == "Test"
    assert w2.pixel_size == pytest.approx(2.5)
    assert w2.show_fft is True
    assert w2.fft_window is False
    assert w2.disabled_tools == ["display", "view"]
    assert w2.hidden_tools == ["stats"]
    assert w2.roi_active is True
    assert len(w2.roi_list) == 1
    assert w2.roi_list[0]["row"] == 10
    assert w2.roi_list[0]["col"] == 15

def test_show2d_state_dict_keys():
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data)
    keys = set(w.state_dict().keys())
    assert "disabled_tools" in keys
    assert "hidden_tools" in keys
    assert "show_stats" in keys
    assert "show_fft" in keys

def test_show2d_save_load_file(tmp_path):
    import json
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data, cmap="magma", title="Saved")
    path = tmp_path / "show2d_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show2D"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["cmap"] == "magma"
    assert saved["state"]["title"] == "Saved"
    w2 = Show2D(data, state=str(path))
    assert w2.cmap == "magma"
    assert w2.title == "Saved"

def test_show2d_rejects_legacy_flat_state_file(tmp_path):
    import json

    data = np.random.rand(16, 16).astype(np.float32)
    path = tmp_path / "legacy_show2d_state.json"
    path.write_text(json.dumps({"cmap": "magma", "title": "Legacy"}, indent=2))

    with pytest.raises(ValueError, match="versioned envelope"):
        Show2D(data, state=str(path))

def test_show2d_summary(capsys):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, title="My Image", cmap="viridis")
    w.summary()
    out = capsys.readouterr().out
    assert "My Image" in out
    assert "32×32" in out
    assert "viridis" in out

def test_show2d_repr():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, cmap="inferno")
    r = repr(w)
    assert "Show2D" in r
    assert "32×32" in r
    assert "inferno" in r

def test_show2d_repr_gallery():
    data = np.random.rand(3, 16, 16).astype(np.float32)
    w = Show2D(data)
    r = repr(w)
    assert "3×16×16" in r
    assert "idx=0" in r

def test_show2d_set_image():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data, cmap="viridis", log_scale=True)
    assert widget.height == 16
    assert widget.width == 16

    new_data = np.random.rand(32, 24).astype(np.float32)
    widget.set_image(new_data)
    assert widget.height == 32
    assert widget.width == 24
    assert widget.n_images == 1
    assert widget.cmap == "viridis"
    assert widget.log_scale is True
    assert len(widget.frame_bytes) == 32 * 24 * 4

def test_show2d_set_image_gallery():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Show2D(data)
    new_data = [np.random.rand(20, 20).astype(np.float32) for _ in range(3)]
    widget.set_image(new_data, labels=["A", "B", "C"])
    assert widget.n_images == 3
    assert widget.height == 20
    assert widget.width == 20
    assert widget.labels == ["A", "B", "C"]
    assert widget.selected_idx == 0

def test_show2d_from_png_file(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.fromarray((np.ones((8, 6), dtype=np.uint8) * 33)).save(path)

    widget = Show2D.from_png(path)
    assert widget.n_images == 1
    assert widget.height == 8
    assert widget.width == 6
    assert widget.title == "img"

def test_show2d_from_png_folder_gallery(tmp_path):
    from PIL import Image

    folder = tmp_path / "png_stack"
    folder.mkdir()
    for i in range(3):
        Image.fromarray((np.ones((8, 6), dtype=np.uint8) * (10 + i))).save(folder / f"slice_{i:02d}.png")

    widget = Show2D.from_folder(folder, file_type="png")
    assert widget.n_images == 3
    assert widget.labels[0] == "slice_00.png"

    widget2 = Show2D.from_folder(folder, file_type="png", mode="mean")
    assert widget2.n_images == 1

def test_show2d_from_path_folder_requires_file_type(tmp_path):
    from PIL import Image

    folder = tmp_path / "png_stack"
    folder.mkdir()
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(folder / "a.png")

    with pytest.raises(ValueError, match="file_type is required"):
        Show2D.from_path(folder)

def test_show2d_from_tiff_file_gallery_and_reduce(tmp_path):
    from PIL import Image

    tiff_path = tmp_path / "stack.tiff"
    frames = [
        Image.fromarray((np.ones((7, 5), dtype=np.uint8) * 20)),
        Image.fromarray((np.ones((7, 5), dtype=np.uint8) * 22)),
    ]
    frames[0].save(tiff_path, save_all=True, append_images=frames[1:])

    gallery = Show2D.from_tiff(tiff_path)
    assert gallery.n_images == 2
    assert gallery.labels[0].startswith("stack[0]")

    reduced = Show2D.from_tiff(tiff_path, mode="mean")
    assert reduced.n_images == 1
    assert reduced.stats_mean[0] == pytest.approx(21.0)

def test_show2d_from_mixed_folder_explicit_type(tmp_path):
    from PIL import Image

    folder = tmp_path / "mixed_stack"
    folder.mkdir()
    Image.fromarray((np.ones((6, 4), dtype=np.uint8) * 7)).save(folder / "a.png")
    frames = [
        Image.fromarray((np.ones((6, 4), dtype=np.uint8) * 21)),
        Image.fromarray((np.ones((6, 4), dtype=np.uint8) * 22)),
    ]
    frames[0].save(folder / "b.tiff", save_all=True, append_images=frames[1:])

    png_widget = Show2D.from_folder(folder, file_type="png")
    assert png_widget.n_images == 1
    assert png_widget.labels == ["a.png"]

    tiff_widget = Show2D.from_folder(folder, file_type="tiff")
    assert tiff_widget.n_images == 2
    assert tiff_widget.labels[0].startswith("b[0]")

def test_show2d_rejects_dataset_path_for_non_emd(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.fromarray((np.ones((8, 6), dtype=np.uint8) * 33)).save(path)

    with pytest.raises(ValueError, match="dataset_path is only supported"):
        Show2D.from_path(path, dataset_path="/data/signal")

def test_show2d_from_path_rejects_file_type_for_file(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.fromarray((np.ones((8, 6), dtype=np.uint8) * 33)).save(path)

    with pytest.raises(ValueError, match="file_type is only used for folder"):
        Show2D.from_path(path, file_type="png")

def test_show2d_invalid_reduce_mode(tmp_path):
    from PIL import Image

    folder = tmp_path / "png_stack"
    folder.mkdir()
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(folder / "a.png")

    with pytest.raises(ValueError, match="Unknown reduce mode"):
        Show2D.from_folder(folder, file_type="png", mode="median")

@pytest.mark.skipif(not _HAS_H5PY or h5py is None, reason="h5py not available")
def test_show2d_from_emd_with_dataset_path_and_reduce(tmp_path):
    emd_path = tmp_path / "stack.emd"
    with h5py.File(emd_path, "w") as h5f:  # type: ignore[arg-type]
        h5f.create_dataset("preview/thumb", data=np.ones((5, 5), dtype=np.float32) * 2.0)
        h5f.create_dataset("data/signal", data=np.ones((3, 7, 5), dtype=np.float32) * 13.0)

    gallery = Show2D.from_emd(emd_path, dataset_path="/data/signal")
    assert gallery.n_images == 3
    assert gallery.height == 7
    assert gallery.width == 5
    assert gallery.labels[0].startswith("stack[0]")

    reduced = Show2D.from_emd(emd_path, dataset_path="/data/signal", mode="max")
    assert reduced.n_images == 1
    assert reduced.stats_mean[0] == pytest.approx(13.0)

@pytest.mark.skipif(not _HAS_H5PY or h5py is None, reason="h5py not available")
def test_show2d_from_emd_highdim_default_and_reduction(tmp_path):
    emd_path = tmp_path / "highdim.emd"
    arr = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    with h5py.File(emd_path, "w") as h5f:  # type: ignore[arg-type]
        h5f.create_dataset("data/signal", data=arr)

    gallery = Show2D.from_emd(emd_path, dataset_path="/data/signal")
    assert gallery.n_images == 6
    assert gallery.height == 4
    assert gallery.width == 4

    reduced = Show2D.from_emd(emd_path, dataset_path="/data/signal", mode="mean")
    assert reduced.n_images == 1
    assert reduced.height == 4
    assert reduced.width == 4

# ── save_image ───────────────────────────────────────────────────────────

def test_show2d_save_image_png(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, cmap="viridis")
    out = w.save_image(tmp_path / "out.png")
    assert out.exists()
    assert out.stat().st_size > 0
    from PIL import Image
    img = Image.open(out)
    assert img.size == (32, 32)

def test_show2d_save_image_pdf(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, cmap="inferno")
    out = w.save_image(tmp_path / "out.pdf")
    assert out.exists()
    assert out.stat().st_size > 0

def test_show2d_save_image_gallery_idx(tmp_path):
    imgs = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Show2D(imgs)
    out0 = w.save_image(tmp_path / "img0.png", idx=0)
    out2 = w.save_image(tmp_path / "img2.png", idx=2)
    assert out0.exists()
    assert out2.exists()

def test_show2d_save_image_log_auto(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Show2D(data, log_scale=True, auto_contrast=True)
    out = w.save_image(tmp_path / "out.png")
    assert out.exists()

def test_show2d_save_image_bad_format(tmp_path):
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data)
    with pytest.raises(ValueError, match="Unsupported format"):
        w.save_image(tmp_path / "out.bmp")

def test_show2d_save_image_bad_idx(tmp_path):
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data)
    with pytest.raises(IndexError):
        w.save_image(tmp_path / "out.png", idx=5)

def test_show2d_widget_version_is_set():
    data = np.random.rand(16, 16).astype(np.float32)
    w = Show2D(data)
    assert w.widget_version != "unknown"
