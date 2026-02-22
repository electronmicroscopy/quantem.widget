import numpy as np
import pytest
import torch

from quantem.widget import Edit2D

def test_edit2d_single_numpy():
    data = np.random.rand(32, 32).astype(np.float32)
    widget = Edit2D(data)
    assert widget.n_images == 1
    assert widget.height == 32
    assert widget.width == 32
    assert len(widget.frame_bytes) > 0

def test_edit2d_single_torch():
    data = torch.rand(32, 32)
    widget = Edit2D(data)
    assert widget.n_images == 1
    assert widget.height == 32
    assert widget.width == 32

def test_edit2d_default_bounds():
    data = np.random.rand(64, 48).astype(np.float32)
    widget = Edit2D(data)
    assert widget.crop_top == 0
    assert widget.crop_left == 0
    assert widget.crop_bottom == 64
    assert widget.crop_right == 48

def test_edit2d_custom_bounds():
    data = np.random.rand(64, 48).astype(np.float32)
    widget = Edit2D(data, bounds=(10, 5, 50, 40))
    assert widget.crop_top == 10
    assert widget.crop_left == 5
    assert widget.crop_bottom == 50
    assert widget.crop_right == 40

def test_edit2d_result_crop():
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    widget = Edit2D(data, bounds=(2, 3, 7, 8))
    result = widget.result
    assert result.shape == (5, 5)
    np.testing.assert_array_equal(result, data[2:7, 3:8])

def test_edit2d_result_full_image():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data)
    result = widget.result
    assert result.shape == (16, 16)
    np.testing.assert_array_equal(result, data)

def test_edit2d_result_pad():
    data = np.ones((10, 10), dtype=np.float32) * 5.0
    widget = Edit2D(data, bounds=(-2, -3, 12, 13), fill_value=0.0)
    result = widget.result
    assert result.shape == (14, 16)
    # Corners should be fill value
    assert result[0, 0] == 0.0
    assert result[0, 15] == 0.0
    assert result[13, 0] == 0.0
    # Interior should be original data
    assert result[2, 3] == 5.0
    assert result[11, 12] == 5.0

def test_edit2d_result_crop_and_pad():
    data = np.ones((10, 10), dtype=np.float32) * 3.0
    widget = Edit2D(data, bounds=(-2, 2, 8, 12), fill_value=-1.0)
    result = widget.result
    assert result.shape == (10, 10)
    # Top-left should be padded (row < 0 in image space)
    assert result[0, 0] == -1.0
    # Within image overlap
    assert result[2, 0] == 3.0
    # Right side padding (col >= 10 in image space)
    assert result[2, 8] == -1.0

def test_edit2d_result_pure_pad():
    data = np.ones((10, 10), dtype=np.float32)
    widget = Edit2D(data, bounds=(-20, -20, -10, -10), fill_value=7.0)
    result = widget.result
    assert result.shape == (10, 10)
    assert np.all(result == 7.0)

def test_edit2d_fill_value():
    data = np.zeros((10, 10), dtype=np.float32)
    widget = Edit2D(data, bounds=(-1, -1, 11, 11), fill_value=42.0)
    result = widget.result
    assert result[0, 0] == 42.0
    assert result[1, 1] == 0.0

def test_edit2d_crop_bounds_property():
    data = np.random.rand(32, 32).astype(np.float32)
    widget = Edit2D(data, bounds=(5, 10, 25, 30))
    assert widget.crop_bounds == (5, 10, 25, 30)

def test_edit2d_crop_bounds_setter():
    data = np.random.rand(32, 32).astype(np.float32)
    widget = Edit2D(data)
    widget.crop_bounds = (1, 2, 3, 4)
    assert widget.crop_top == 1
    assert widget.crop_left == 2
    assert widget.crop_bottom == 3
    assert widget.crop_right == 4

def test_edit2d_crop_size():
    data = np.random.rand(32, 32).astype(np.float32)
    widget = Edit2D(data, bounds=(5, 10, 25, 30))
    assert widget.crop_size == (20, 20)

def test_edit2d_multi_image_numpy():
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    widget = Edit2D(images, labels=["A", "B", "C"])
    assert widget.n_images == 3
    assert widget.labels == ["A", "B", "C"]

def test_edit2d_multi_image_result():
    images = [np.arange(100, dtype=np.float32).reshape(10, 10) for _ in range(3)]
    widget = Edit2D(images, bounds=(2, 3, 7, 8))
    result = widget.result
    assert isinstance(result, list)
    assert len(result) == 3
    for r in result:
        assert r.shape == (5, 5)

def test_edit2d_multi_image_default_labels():
    images = [np.random.rand(8, 8).astype(np.float32) for _ in range(2)]
    widget = Edit2D(images)
    assert widget.labels == ["Image 1", "Image 2"]

def test_edit2d_colormap():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, cmap="viridis")
    assert widget.cmap == "viridis"

def test_edit2d_log_scale():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, log_scale=True)
    assert widget.log_scale is True

def test_edit2d_auto_contrast_default():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data)
    assert widget.auto_contrast is True

def test_edit2d_title():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, title="My Crop")
    assert widget.title == "My Crop"

def test_edit2d_show_stats():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, show_stats=False)
    assert widget.show_stats is False

def test_edit2d_control_group_visibility_defaults():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data)
    assert widget.show_display_controls is True
    assert widget.show_edit_controls is True
    assert widget.show_histogram is True

def test_edit2d_control_group_visibility_custom():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(
        data,
        show_display_controls=False,
        show_edit_controls=False,
        show_histogram=False,
    )
    assert widget.show_display_controls is False
    assert widget.show_edit_controls is False
    assert widget.show_histogram is False

def test_edit2d_disabled_tools_default():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data)
    assert widget.disabled_tools == []

def test_edit2d_disabled_tools_custom():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, disabled_tools=["display", "Edit", "histogram"])
    assert widget.disabled_tools == ["display", "edit", "histogram"]

def test_edit2d_disabled_tools_flags():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, disable_edit=True, disable_display=True, disable_navigation=True)
    assert widget.disabled_tools == ["edit", "display", "navigation"]

def test_edit2d_disabled_tools_disable_all_flag():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, disable_all=True, disable_edit=True)
    assert widget.disabled_tools == ["all"]

def test_edit2d_disabled_tools_unknown_value_raises():
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Edit2D(data, disabled_tools=["not_real"])

def test_edit2d_disabled_tools_trait_assignment_normalizes():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data)
    widget.disabled_tools = ["DISPLAY", "display", "edit"]
    assert widget.disabled_tools == ["display", "edit"]

def test_edit2d_hidden_tools_default():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data)
    assert widget.hidden_tools == []

def test_edit2d_hidden_tools_custom():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, hidden_tools=["display", "Edit", "histogram"])
    assert widget.hidden_tools == ["display", "edit", "histogram"]

def test_edit2d_hidden_tools_flags():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, hide_edit=True, hide_display=True, hide_navigation=True)
    assert widget.hidden_tools == ["edit", "display", "navigation"]

def test_edit2d_hidden_tools_hide_all_flag():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, hide_all=True, hide_edit=True)
    assert widget.hidden_tools == ["all"]

def test_edit2d_hidden_tools_unknown_value_raises():
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Edit2D(data, hidden_tools=["not_real"])

def test_edit2d_hidden_tools_trait_assignment_normalizes():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data)
    widget.hidden_tools = ["DISPLAY", "display", "edit"]
    assert widget.hidden_tools == ["display", "edit"]

def test_edit2d_pixel_size():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, pixel_size=1.5)
    assert widget.pixel_size == 1.5

def test_edit2d_stats():
    data = np.ones((16, 16), dtype=np.float32) * 42.0
    widget = Edit2D(data)
    assert widget.stats_mean == pytest.approx(42.0)
    assert widget.stats_min == pytest.approx(42.0)
    assert widget.stats_max == pytest.approx(42.0)
    assert widget.stats_std == pytest.approx(0.0)

def test_edit2d_negative_bounds():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, bounds=(-5, -5, 10, 10))
    assert widget.crop_top == -5
    assert widget.crop_left == -5
    assert widget.crop_size == (15, 15)

def test_edit2d_zero_size_crop():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data, bounds=(5, 5, 5, 5))
    result = widget.result
    assert result.shape == (0, 0)

def test_edit2d_frame_bytes_size():
    data = np.random.rand(32, 24).astype(np.float32)
    widget = Edit2D(data)
    assert len(widget.frame_bytes) == 32 * 24 * 4  # float32 = 4 bytes

def test_edit2d_3d_array():
    data = np.random.rand(3, 16, 16).astype(np.float32)
    widget = Edit2D(data)
    assert widget.n_images == 3

def test_edit2d_repr():
    data = np.random.rand(32, 32).astype(np.float32)
    widget = Edit2D(data, bounds=(5, 10, 25, 30))
    r = repr(widget)
    assert "Edit2D" in r
    assert "32x32" in r
    assert "20x20" in r

def test_edit2d_set_image():
    data = np.random.rand(16, 16).astype(np.float32)
    widget = Edit2D(data)
    assert widget.height == 16

    new_data = np.random.rand(32, 24).astype(np.float32)
    widget.set_image(new_data)
    assert widget.height == 32
    assert widget.width == 24
    assert widget.crop_top == 0
    assert widget.crop_bottom == 32
    assert widget.crop_right == 24

# ── State Protocol ────────────────────────────────────────────────────────

def test_edit2d_state_dict_roundtrip():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Edit2D(data, cmap="viridis", title="My Crop",
               bounds=(5, 10, 25, 30), fill_value=7.0, log_scale=True,
               show_display_controls=False, show_edit_controls=True,
               show_histogram=False, disabled_tools=["edit", "view"],
               hidden_tools=["stats"])
    sd = w.state_dict()
    w2 = Edit2D(data, state=sd)
    assert w2.cmap == "viridis"
    assert w2.title == "My Crop"
    assert w2.log_scale is True
    assert w2.fill_value == 7.0
    assert w2.crop_top == 5
    assert w2.crop_left == 10
    assert w2.crop_bottom == 25
    assert w2.crop_right == 30
    assert w2.show_display_controls is False
    assert w2.show_edit_controls is True
    assert w2.show_histogram is False
    assert w2.disabled_tools == ["edit", "view"]
    assert w2.hidden_tools == ["stats"]

def test_edit2d_state_dict_keys():
    data = np.random.rand(16, 16).astype(np.float32)
    w = Edit2D(data)
    sd = w.state_dict()
    expected = {
        "title", "cmap", "mode", "log_scale", "auto_contrast",
        "show_controls", "show_stats",
        "show_display_controls", "show_edit_controls", "show_histogram",
        "disabled_tools", "hidden_tools",
        "pixel_size", "fill_value",
        "crop_top", "crop_left", "crop_bottom", "crop_right",
        "shared",
    }
    assert set(sd.keys()) == expected

def test_edit2d_save_load_file(tmp_path):
    import json
    data = np.random.rand(32, 32).astype(np.float32)
    w = Edit2D(data, cmap="plasma", title="Saved Crop")
    path = tmp_path / "edit2d_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Edit2D"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["cmap"] == "plasma"
    w2 = Edit2D(data, state=str(path))
    assert w2.cmap == "plasma"
    assert w2.title == "Saved Crop"

def test_edit2d_summary(capsys):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Edit2D(data, title="My Crop", cmap="inferno", bounds=(5, 10, 25, 30))
    w.summary()
    out = capsys.readouterr().out
    assert "My Crop" in out
    assert "32×32" in out
    assert "20×20" in out  # crop size

def test_edit2d_widget_version_is_set():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Edit2D(data)
    assert w.widget_version != "unknown"

def test_edit2d_show_controls_default():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Edit2D(data)
    assert w.show_controls is True

# ── save_image ───────────────────────────────────────────────────────────

def test_edit2d_save_image_png(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Edit2D(data, cmap="viridis")
    out = w.save_image(tmp_path / "out.png")
    assert out.exists()
    assert out.stat().st_size > 0
    from PIL import Image
    img = Image.open(out)
    assert img.size == (32, 32)

def test_edit2d_save_image_pdf(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Edit2D(data, cmap="inferno")
    out = w.save_image(tmp_path / "out.pdf")
    assert out.exists()
    assert out.stat().st_size > 0

def test_edit2d_save_image_crop_mode(tmp_path):
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    w = Edit2D(data, bounds=(2, 3, 7, 8))
    out = w.save_image(tmp_path / "cropped.png")
    assert out.exists()
    from PIL import Image
    img = Image.open(out)
    assert img.size == (5, 5)

def test_edit2d_save_image_bad_format(tmp_path):
    data = np.random.rand(16, 16).astype(np.float32)
    w = Edit2D(data)
    with pytest.raises(ValueError, match="Unsupported format"):
        w.save_image(tmp_path / "out.bmp")


# ── Independent mode (shared=False) ──────────────────────────────────

def test_edit2d_shared_default():
    data = np.random.rand(16, 16).astype(np.float32)
    w = Edit2D(data)
    assert w.shared is True


def test_edit2d_independent_crop_result():
    import json
    images = [np.arange(100, dtype=np.float32).reshape(10, 10) for _ in range(3)]
    w = Edit2D(images, shared=False)
    # Set per-image crops via JSON
    crops = [
        {"top": 0, "left": 0, "bottom": 5, "right": 5},
        {"top": 2, "left": 3, "bottom": 8, "right": 9},
        {"top": 1, "left": 1, "bottom": 9, "right": 9},
    ]
    w.per_image_crops_json = json.dumps(crops)
    result = w.result
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0].shape == (5, 5)
    assert result[1].shape == (6, 6)
    assert result[2].shape == (8, 8)
    # Verify content: first image crop [0:5, 0:5]
    np.testing.assert_array_equal(result[0], images[0][:5, :5])


def test_edit2d_independent_mask_result():
    images = [np.ones((8, 8), dtype=np.float32) * (i + 1) for i in range(2)]
    w = Edit2D(images, shared=False, mode="mask")
    # Create per-image masks: mask top-left of image 0, bottom-right of image 1
    mask0 = np.zeros((8, 8), dtype=np.uint8)
    mask0[:4, :4] = 255
    mask1 = np.zeros((8, 8), dtype=np.uint8)
    mask1[4:, 4:] = 255
    w.per_image_masks_bytes = (mask0.tobytes() + mask1.tobytes())
    result = w.result
    assert isinstance(result, list)
    assert len(result) == 2
    # Image 0: top-left masked (fill_value=0)
    assert result[0][0, 0] == 0.0
    assert result[0][4, 4] == 1.0
    # Image 1: bottom-right masked
    assert result[1][0, 0] == 2.0
    assert result[1][7, 7] == 0.0


def test_edit2d_independent_state_dict_roundtrip():
    import json
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Edit2D(images, shared=False)
    crops = [
        {"top": 0, "left": 0, "bottom": 8, "right": 8},
        {"top": 4, "left": 4, "bottom": 12, "right": 12},
        {"top": 2, "left": 2, "bottom": 14, "right": 14},
    ]
    w.per_image_crops_json = json.dumps(crops)
    sd = w.state_dict()
    assert sd["shared"] is False
    assert "per_image_crops" in sd
    assert len(sd["per_image_crops"]) == 3
    assert sd["per_image_crops"][0]["bottom"] == 8

    # Restore via state param
    w2 = Edit2D(images, state=sd)
    assert w2.shared is False
    restored = w2._get_per_image_crops()
    assert restored[1]["top"] == 4
    assert restored[1]["right"] == 12


def test_edit2d_independent_save_load_file(tmp_path):
    import json as json_mod
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(2)]
    w = Edit2D(images, shared=False, cmap="viridis")
    crops = [
        {"top": 0, "left": 0, "bottom": 10, "right": 10},
        {"top": 3, "left": 3, "bottom": 13, "right": 13},
    ]
    w.per_image_crops_json = json_mod.dumps(crops)
    path = tmp_path / "edit2d_indep.json"
    w.save(str(path))
    assert path.exists()
    saved = json_mod.loads(path.read_text())
    assert saved["state"]["shared"] is False
    assert len(saved["state"]["per_image_crops"]) == 2

    w2 = Edit2D(images, state=str(path))
    assert w2.shared is False
    assert w2.cmap == "viridis"


def test_edit2d_independent_summary(capsys):
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Edit2D(images, shared=False)
    w.summary()
    out = capsys.readouterr().out
    assert "independent" in out
    assert "3 images" in out


def test_edit2d_independent_repr():
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Edit2D(images, shared=False)
    r = repr(w)
    assert "independent" in r
    assert "3 images" in r


def test_edit2d_set_image_resets_independent_state():
    import json
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(2)]
    w = Edit2D(images, shared=False)
    crops = [
        {"top": 2, "left": 2, "bottom": 10, "right": 10},
        {"top": 4, "left": 4, "bottom": 12, "right": 12},
    ]
    w.per_image_crops_json = json.dumps(crops)
    # set_image resets independent state
    new_data = np.random.rand(24, 24).astype(np.float32)
    w.set_image(new_data)
    assert w.per_image_crops_json == "[]"
    assert w.per_image_masks_bytes == b""


def test_edit2d_load_state_dict_clears_stale_independent_state():
    import json
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Edit2D(images, shared=False)
    # Set per-image crops
    crops = [
        {"top": 2, "left": 2, "bottom": 10, "right": 10},
        {"top": 4, "left": 4, "bottom": 12, "right": 12},
        {"top": 0, "left": 0, "bottom": 16, "right": 16},
    ]
    w.per_image_crops_json = json.dumps(crops)
    assert w.shared is False
    assert w.per_image_crops_json != "[]"
    # Restore shared-mode state — stale per-image data should be cleared
    w.load_state_dict({"shared": True, "cmap": "viridis"})
    assert w.shared is True
    assert w.per_image_crops_json == "[]"
    assert w.per_image_masks_bytes == b""


def test_edit2d_independent_initial_bounds():
    import json
    images = [np.arange(100, dtype=np.float32).reshape(10, 10) for _ in range(3)]
    w = Edit2D(images, shared=False, bounds=(2, 3, 8, 9))
    # bounds should populate per_image_crops_json on init
    crops = json.loads(w.per_image_crops_json)
    assert len(crops) == 3
    for c in crops:
        assert c == {"top": 2, "left": 3, "bottom": 8, "right": 9}
    # crop_bounds should return the initial bounds
    assert w.crop_bounds == (2, 3, 8, 9)
    # result should use these bounds
    result = w.result
    assert isinstance(result, list)
    for r in result:
        assert r.shape == (6, 6)


def test_edit2d_independent_crop_bounds_setter():
    images = [np.arange(100, dtype=np.float32).reshape(10, 10) for _ in range(3)]
    w = Edit2D(images, shared=False)
    # Set crop for image 0 via setter
    w.crop_bounds = (1, 2, 8, 9)
    assert w.crop_bounds == (1, 2, 8, 9)
    # Result reflects the new bounds
    r = w.result[0]
    assert r.shape == (7, 7)
    # Image 1 still has default full-image bounds
    w.selected_idx = 1
    assert w.crop_bounds == (0, 0, 10, 10)
    # Set a different crop for image 1
    w.crop_bounds = (3, 3, 7, 7)
    assert w.crop_bounds == (3, 3, 7, 7)
    # Verify both images have independent bounds
    w.selected_idx = 0
    assert w.crop_bounds == (1, 2, 8, 9)


def test_edit2d_independent_crop_bounds_property():
    import json
    images = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]
    w = Edit2D(images, shared=False)
    crops = [
        {"top": 0, "left": 0, "bottom": 8, "right": 8},
        {"top": 4, "left": 4, "bottom": 12, "right": 12},
        {"top": 2, "left": 2, "bottom": 14, "right": 14},
    ]
    w.per_image_crops_json = json.dumps(crops)
    # Default selected_idx=0
    assert w.crop_bounds == (0, 0, 8, 8)
    assert w.crop_size == (8, 8)
    # Switch to image 1
    w.selected_idx = 1
    assert w.crop_bounds == (4, 4, 12, 12)
    assert w.crop_size == (8, 8)
    # Switch to image 2
    w.selected_idx = 2
    assert w.crop_bounds == (2, 2, 14, 14)
    assert w.crop_size == (12, 12)
