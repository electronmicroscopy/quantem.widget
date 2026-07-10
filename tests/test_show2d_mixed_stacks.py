from __future__ import annotations

import numpy as np
import pytest

from quantem.widget import Show2D


def _mixed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    static_a = np.arange(30, dtype=np.float32).reshape(5, 6)
    stack = np.stack(
        [np.full((4, 5), frame, dtype=np.float32) for frame in range(4)],
        axis=0,
    )
    static_b = np.full((3, 4), 9.0, dtype=np.float32)
    return static_a, stack, static_b


def test_mixed_static_and_stack_panels_normalize_independently() -> None:
    static_a, stack, static_b = _mixed_data()
    widget = Show2D(
        [static_a, stack, static_b],
        labels=["overview", "HAADF", "map"],
        panel_frame_indices=[0, -1, 0],
        verbose=False,
    )

    assert widget.n_images == 3
    assert widget.panel_frame_counts == [1, 4, 1]
    assert widget.panel_frame_indices == [0, 3, 0]
    assert widget.panel_stack_offsets == [-1, 0, -1]
    assert widget._data.shape == (3, 5, 6)
    assert widget._panel_stacks[1].shape == (4, 5, 6)
    assert widget.stats_mean[1] == pytest.approx(3.0)

    expected_floats = 4 * 5 * 6
    packed = np.frombuffer(
        widget.panel_stack_bytes[: expected_floats * 4],
        dtype=np.float32,
    ).reshape(4, 5, 6)
    np.testing.assert_array_equal(packed, widget._display_panel_stacks[1])


def test_bare_3d_input_remains_a_static_gallery() -> None:
    data = np.zeros((4, 8, 9), dtype=np.float32)
    widget = Show2D(data, verbose=False)

    assert widget.n_images == 4
    assert widget.panel_frame_counts == [1, 1, 1, 1]
    assert widget.panel_stack_bytes == b""


def test_panel_frame_change_updates_python_analysis_without_repacking() -> None:
    widget = Show2D(
        list(_mixed_data()),
        labels=["overview", "HAADF", "map"],
        verbose=False,
    )
    packed_before = widget.panel_stack_bytes

    assert widget.set_panel_frame("HAADF", -1) is widget
    assert widget.panel_frame_indices == [0, 3, 0]
    assert widget.stats_mean[1] == pytest.approx(3.0)
    assert widget.panel_stack_bytes is packed_before

    with pytest.raises(IndexError, match="out of range"):
        widget.set_panel_frame("HAADF", 4)
    with pytest.raises(IndexError, match="one frame"):
        widget.set_panel_frame("overview", 1)


def test_panel_frame_state_survives_hide_reorder_and_state_roundtrip() -> None:
    data = list(_mixed_data())
    widget = Show2D(
        data,
        labels=["overview", "HAADF", "map"],
        panel_frame_indices=[0, 2, 0],
        hidden_panels=["map"],
        panel_order=["HAADF", "map", "overview"],
        verbose=False,
    )
    state = widget.state_dict()

    assert state["panel_frame_indices"] == [0, 2, 0]
    assert widget.visible_panels == [1, 0]

    restored = Show2D(data, labels=["overview", "HAADF", "map"], verbose=False)
    restored.load_state_dict(state)
    assert restored.panel_frame_indices == [0, 2, 0]
    assert restored.visible_panels == [1, 0]
    assert restored.stats_mean[1] == pytest.approx(2.0)


def test_set_image_accepts_mixed_static_and_stack_panels() -> None:
    widget = Show2D(np.zeros((2, 8, 8), dtype=np.float32), verbose=False)
    widget.set_image(
        list(_mixed_data()),
        labels=["overview", "HAADF", "map"],
        panel_frame_indices=[0, 1, 0],
    )

    assert widget.labels == ["overview", "HAADF", "map"]
    assert widget.panel_frame_counts == [1, 4, 1]
    assert widget.panel_frame_indices == [0, 1, 0]
    assert widget.stats_mean[1] == pytest.approx(1.0)


def test_rotation_applies_to_every_frame_in_a_local_stack() -> None:
    static_a, stack, _ = _mixed_data()
    widget = Show2D([static_a, stack], panel_frame_indices=[0, 2], verbose=False)
    original_frame = widget._panel_stacks[1][2].copy()

    widget.rotate(1, 90)

    rotated = np.rot90(original_frame)
    panel = widget._panel_stacks[1][2]
    row0 = (panel.shape[0] - rotated.shape[0]) // 2
    col0 = (panel.shape[1] - rotated.shape[1]) // 2
    np.testing.assert_array_equal(
        panel[row0:row0 + rotated.shape[0], col0:col0 + rotated.shape[1]],
        rotated,
    )
    assert widget.panel_frame_indices == [0, 2]


@pytest.mark.parametrize("quantized", [False, True])
def test_html_export_clone_keeps_all_local_frames(quantized: bool) -> None:
    widget = Show2D(
        list(_mixed_data()),
        labels=["overview", "HAADF", "map"],
        panel_frame_indices=[0, 3, 0],
        verbose=False,
    )

    clone = widget._clone_for_html_export(quantized=quantized)
    try:
        assert clone.panel_frame_counts == [1, 4, 1]
        assert clone.panel_frame_indices == [0, 3, 0]
        assert len(clone.panel_stack_bytes) >= 4 * 5 * 6 * (1 if quantized else 4)
        assert clone._save_state is True
        assert clone.offline is quantized
    finally:
        clone.close()


def test_notebook_state_drops_or_keeps_stack_payload_by_save_state() -> None:
    transient = Show2D(list(_mixed_data()), save_state=False, verbose=False)
    persisted = Show2D(list(_mixed_data()), save_state=True, verbose=False)

    assert "panel_stack_bytes" not in transient.get_state()
    assert persisted.get_state()["panel_stack_bytes"]


def test_standalone_html_embeds_local_stack_protocol(tmp_path) -> None:
    widget = Show2D(
        list(_mixed_data()),
        labels=["overview", "HAADF", "map"],
        panel_frame_indices=[0, 2, 0],
        verbose=False,
    )

    output = widget.export_html(tmp_path / "mixed-stack.html", encoding="uint8")
    html = output.read_text()

    assert "panel_frame_counts" in html
    assert "panel_frame_indices" in html
    assert "panel_stack_bytes" in html
    assert "HAADF" in html


def test_mixed_rgb_and_local_stack_has_corrective_error() -> None:
    rgb = np.zeros((8, 9, 3), dtype=np.float32)
    stack = np.zeros((5, 8, 9), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="RGB panels with local grayscale"):
        Show2D([rgb, stack], verbose=False)
