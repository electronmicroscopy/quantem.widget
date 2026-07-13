from __future__ import annotations

import numpy as np
import pytest
import traitlets

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


def test_multiple_local_stack_panels_keep_independent_frames() -> None:
    frame_counts = [2, 3, 4, 5]
    offsets = [0.0, 10.0, 20.0, 30.0]
    stacks = [
        np.stack(
            [
                np.full((5, 6), offset + frame, dtype=np.float32)
                for frame in range(count)
            ],
            axis=0,
        )
        for count, offset in zip(frame_counts, offsets)
    ]
    widget = Show2D(
        stacks,
        labels=["baseline", "coarse z", "fine z", "alternative"],
        panel_frame_indices=[0, 1, 2, -1],
        verbose=False,
    )

    assert widget.panel_frame_counts == frame_counts
    assert widget.panel_frame_indices == [0, 1, 2, 4]
    np.testing.assert_allclose(widget.stats_mean, [0.0, 11.0, 22.0, 34.0])

    widget.set_panel_frame("fine z", 3)
    assert widget.panel_frame_indices == [0, 1, 3, 4]
    np.testing.assert_allclose(widget.stats_mean, [0.0, 11.0, 23.0, 34.0])

    widget.set_panel_frame("baseline", -1)
    assert widget.panel_frame_indices == [1, 1, 3, 4]
    np.testing.assert_allclose(widget.stats_mean, [1.0, 11.0, 23.0, 34.0])


def test_panel_playback_speed_is_configurable_without_extra_panel_state() -> None:
    default_widget = Show2D(list(_mixed_data()), verbose=False)
    configured = Show2D(
        list(_mixed_data()),
        panel_playback_fps=4.5,
        verbose=False,
    )

    assert default_widget.panel_playback_fps == pytest.approx(10.0)
    assert configured.panel_playback_fps == pytest.approx(4.5)
    configured.set_image(list(_mixed_data()))
    assert configured.panel_playback_fps == pytest.approx(4.5)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_panel_playback_speed_rejects_invalid_values(value: float) -> None:
    with pytest.raises(traitlets.TraitError, match="panel_playback_fps"):
        Show2D(list(_mixed_data()), panel_playback_fps=value, verbose=False)


def test_panel_playback_speed_caps_at_browser_budget() -> None:
    widget = Show2D(
        list(_mixed_data()),
        panel_playback_fps=120,
        verbose=False,
    )

    assert widget.panel_playback_fps == pytest.approx(30.0)


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
        panel_playback_fps=4,
        hidden_panels=["map"],
        panel_order=["HAADF", "map", "overview"],
        verbose=False,
    )
    state = widget.state_dict()

    assert state["panel_frame_indices"] == [0, 2, 0]
    assert state["panel_playback_fps"] == pytest.approx(4.0)
    assert widget.visible_panels == [1, 0]

    restored = Show2D(data, labels=["overview", "HAADF", "map"], verbose=False)
    restored.load_state_dict(state)
    assert restored.panel_frame_indices == [0, 2, 0]
    assert restored.panel_playback_fps == pytest.approx(4.0)
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
        panel_playback_fps=4,
        verbose=False,
    )

    clone = widget._clone_for_html_export(quantized=quantized)
    try:
        assert clone.panel_frame_counts == [1, 4, 1]
        assert clone.panel_frame_indices == [0, 3, 0]
        assert clone.panel_playback_fps == pytest.approx(4.0)
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
        panel_playback_fps=4,
        verbose=False,
    )

    output = widget.export_html(tmp_path / "mixed-stack.html", encoding="uint8")
    html = output.read_text()

    assert "panel_frame_counts" in html
    assert "panel_frame_indices" in html
    assert "panel_playback_fps" in html
    assert "panel_stack_bytes" in html
    assert "HAADF" in html


def test_mixed_rgb_and_local_stack_has_corrective_error() -> None:
    rgb = np.zeros((8, 9, 3), dtype=np.float32)
    stack = np.zeros((5, 8, 9), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="RGB panels with local grayscale"):
        Show2D([rgb, stack], verbose=False)
