from __future__ import annotations

import pathlib

import numpy as np
import pytest

from quantem.widget import Show3D


def _panels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.full((3, 4, 5), fill_value=i, dtype=np.float32)
        for i in range(3)
    )


def test_show3d_hides_panels_by_title_or_index() -> None:
    widget = Show3D(
        *_panels(),
        panel_titles=["SSB", "Mean DP", "Probe"],
        hidden_panels=["Mean DP"],
        show_controls=False,
    )

    assert widget.hidden_panels == [1]
    assert widget.visible_panels == [0, 2]

    widget.hide_panel(2)
    assert widget.hidden_panels == [1, 2]
    assert widget.visible_panels == [0]

    widget.show_panel("Mean DP")
    assert widget.hidden_panels == [2]

    widget.show_all_panels()
    assert widget.hidden_panels == []
    assert widget.visible_panels == [0, 1, 2]


def test_show3d_panel_title_must_be_unique_for_visibility_lookup() -> None:
    with pytest.raises(ValueError, match="not unique"):
        Show3D(
            *_panels()[:2],
            panel_titles=["duplicate", "duplicate"],
            hidden_panels=["duplicate"],
            show_controls=False,
        )


def test_show3d_refuses_to_hide_every_panel() -> None:
    widget = Show3D(
        *_panels()[:2],
        panel_titles=["SSB", "Mean DP"],
        show_controls=False,
    )

    with pytest.raises(ValueError, match="hide every panel"):
        widget.set_hidden_panels([0, 1])

    with pytest.raises(ValueError, match="hide every panel"):
        widget.hide_panel("SSB", "Mean DP")


def test_show3d_statistics_are_opt_in() -> None:
    widget = Show3D(np.zeros((3, 4, 5), dtype=np.float32), show_controls=False)

    assert widget.show_stats is False
    assert widget.state_dict()["show_stats"] is False

    explicit = Show3D(
        np.zeros((3, 4, 5), dtype=np.float32),
        show_stats=True,
        show_controls=False,
    )
    assert explicit.show_stats is True


def test_show3d_set_image_replaces_stack_and_triggers_new_frame_transfer() -> None:
    widget = Show3D(
        *_panels()[:2],
        panel_titles=["SSB", "Mean DP"],
        hidden_panels=["Mean DP"],
        panel_order=["Mean DP", "SSB"],
        offline=False,
        show_controls=False,
        verbose=False,
    )
    widget.slice_idx = 2
    widget.playing = True
    old_seq = int(widget.frame_seq)

    data = np.stack(
        [
            np.full((6, 7), 3.0, dtype=np.float32),
            np.full((6, 7), 9.0, dtype=np.float32),
        ]
    )
    widget.set_image(data, labels=["fresh 0", "fresh 1"])

    assert widget.n_panels == 1
    assert widget.n_slices == 2
    assert widget.height == 6
    assert widget.width == 7
    assert widget.labels == ["fresh 0", "fresh 1"]
    assert widget.slice_idx == 1
    assert widget.playing is False
    assert widget.hidden_panels == []
    assert widget.panel_order == []
    assert widget.panel_titles == []
    assert widget.starred == [-1]
    assert widget.frame_seq > old_seq
    assert bytes(widget.frame_bytes) == data[1].tobytes()
    assert widget._buffer_bytes == b""


def test_show3d_controls_collapsed_roundtrips_state_and_html(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        *_panels()[:2],
        panel_titles=["SSB", "Mean DP"],
        show_controls=True,
        controls_collapsed=True,
        verbose=False,
    )

    assert widget.controls_collapsed is True
    assert widget.expand_controls() is widget
    assert widget.controls_collapsed is False
    assert widget.collapse_controls() is widget
    assert widget.controls_collapsed is True
    assert widget.toggle_controls() is widget
    assert widget.controls_collapsed is False

    widget.collapse_controls()
    state = widget.state_dict()
    assert state["show_controls"] is True
    assert state["controls_collapsed"] is True

    restored = Show3D(*_panels()[:2], panel_titles=["SSB", "Mean DP"], verbose=False)
    restored.load_state_dict(state)
    assert restored.controls_collapsed is True

    out = widget.export_html(tmp_path / "show3d_controls_collapsed.html", encoding="full")
    html = out.read_text()
    assert "controls_collapsed" in html


def test_show3d_ui_mode_presets_and_overrides() -> None:
    presentation = Show3D(*_panels()[:2], ui_mode="presentation", verbose=False)
    assert presentation.show_title is True
    assert presentation.show_controls is True
    assert presentation.controls_collapsed is True
    assert presentation.show_stats is False
    assert presentation.show_panel_titles is True
    assert presentation.show_resize_handles is False
    assert presentation.show_zoom_indicator is False
    assert presentation.scale_bar_visible is True

    report = Show3D(*_panels()[:2], ui_mode="report", verbose=False)
    assert report.show_title is True
    assert report.show_controls is False
    assert report.controls_collapsed is False
    assert report.show_stats is False
    assert report.show_panel_titles is True
    assert report.show_resize_handles is False
    assert report.show_zoom_indicator is False
    assert report.scale_bar_visible is True

    minimal = Show3D(*_panels()[:2], ui_mode="minimal", verbose=False)
    assert minimal.show_title is False
    assert minimal.show_controls is False
    assert minimal.controls_collapsed is False
    assert minimal.show_panel_titles is False
    assert minimal.show_resize_handles is False
    assert minimal.show_zoom_indicator is False
    assert minimal.scale_bar_visible is False

    override = Show3D(
        *_panels()[:2],
        ui_mode="minimal",
        show_title=True,
        show_controls=True,
        controls_collapsed=True,
        show_panel_titles=True,
        show_resize_handles=True,
        show_zoom_indicator=True,
        show_scale_bar=True,
        verbose=False,
    )
    assert override.show_title is True
    assert override.show_controls is True
    assert override.controls_collapsed is True
    assert override.show_panel_titles is True
    assert override.show_resize_handles is True
    assert override.show_zoom_indicator is True
    assert override.scale_bar_visible is True

    with pytest.raises(ValueError, match="show_scale_bar"):
        Show3D(_panels()[0], show_scale_bar=True, scale_bar_visible=False)


def test_show3d_fft_layout_validates_and_roundtrips(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        *_panels()[:2],
        panel_titles=["SSB", "Mean DP"],
        show_fft=True,
        fft_layout="overlay",
        fft_overlay_position="bottom-left",
        fft_overlay_size=0.5,
        fft_overlay_zoom=2.0,
        show_zoom_indicator=False,
        show_controls=False,
    )

    assert widget.fft_layout == "overlay"
    assert widget.fft_overlay_position == "bottom-left"
    assert widget.fft_overlay_size == 0.5
    assert widget.fft_overlay_zoom == 2.0
    # C1: hidden zoom chrome with an initialized FFT view, expect both values
    # to survive state and standalone HTML round trips independently.
    assert widget.show_zoom_indicator is False
    state = widget.state_dict()
    assert state["fft_layout"] == "overlay"
    assert state["fft_overlay_position"] == "bottom-left"
    assert state["fft_overlay_size"] == 0.5
    assert state["fft_overlay_zoom"] == 2.0
    assert state["show_zoom_indicator"] is False

    restored = Show3D(*_panels()[:2], panel_titles=["SSB", "Mean DP"], show_controls=False)
    restored.load_state_dict(state)
    assert restored.fft_layout == "overlay"
    assert restored.fft_overlay_position == "bottom-left"
    assert restored.fft_overlay_size == 0.5
    assert restored.fft_overlay_zoom == 2.0
    assert restored.show_zoom_indicator is False

    out = widget.export_html(tmp_path / "show3d_fft_layout.html", encoding="full")
    exported = out.read_text()
    assert "fft_layout" in exported
    assert "fft_overlay_position" in exported
    assert "fft_overlay_size" in exported
    assert "fft_overlay_zoom" in exported
    assert "show_zoom_indicator" in exported

    with pytest.raises(ValueError, match="fft_layout"):
        Show3D(np.zeros((3, 4, 5), dtype=np.float32), fft_layout="floating")
    with pytest.raises(ValueError, match="fft_overlay_position"):
        Show3D(np.zeros((3, 4, 5), dtype=np.float32), fft_overlay_position="center")
    with pytest.raises(ValueError, match="fft_overlay_size"):
        Show3D(np.zeros((3, 4, 5), dtype=np.float32), fft_overlay_size=0.9)
    with pytest.raises(ValueError, match="fft_overlay_zoom"):
        Show3D(np.zeros((3, 4, 5), dtype=np.float32), fft_overlay_zoom=0.5)


def test_show3d_hidden_panels_roundtrip_in_state_and_html(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        *_panels()[:2],
        panel_titles=["SSB", "Mean DP"],
        hidden_panels=["Mean DP"],
        show_controls=False,
    )

    state = widget.state_dict()
    assert state["hidden_panels"] == [1]

    restored = Show3D(*_panels()[:2], panel_titles=["SSB", "Mean DP"], show_controls=False)
    restored.load_state_dict(state)
    assert restored.hidden_panels == [1]
    assert restored.visible_panels == [0]

    out = widget.export_html(tmp_path / "show3d_hidden_panel.html", encoding="full")
    html = out.read_text()
    assert "hidden_panels" in html
    assert "Mean DP" in html


def test_show3d_panel_order_controls_visible_order_and_handoff() -> None:
    widget = Show3D(
        *_panels(),
        panel_titles=["SSB", "Mean DP", "Probe"],
        panel_order=["Probe", "SSB", "Mean DP"],
        hidden_panels=["Mean DP"],
        show_controls=False,
    )

    assert widget.panel_order == [2, 0, 1]
    assert widget.ordered_panels == [2, 0, 1]
    assert widget.visible_panels == [2, 0]

    widget.move_panel("SSB", 2)
    assert widget.panel_order == [2, 1, 0]
    assert widget.visible_panels == [2, 0]

    out = widget.to_show2d(frame=1)
    assert out.labels == ["Probe 2/3", "SSB 2/3"]
    np.testing.assert_allclose(out._data[0], _panels()[2][1])
    np.testing.assert_allclose(out._data[1], _panels()[0][1])

    widget.reset_panel_order()
    assert widget.panel_order == []
    assert widget.visible_panels == [0, 2]


def test_show3d_panel_order_roundtrips_state_and_validates(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        *_panels(),
        panel_titles=["SSB", "Mean DP", "Probe"],
        show_controls=False,
    )
    widget.set_panel_order([2, 0, 1])
    state = widget.state_dict()

    assert state["panel_order"] == [2, 0, 1]

    restored = Show3D(*_panels(), panel_titles=["SSB", "Mean DP", "Probe"], show_controls=False)
    restored.load_state_dict(state)
    assert restored.panel_order == [2, 0, 1]
    assert restored.visible_panels == [2, 0, 1]

    with pytest.raises(ValueError, match="every panel"):
        widget.set_panel_order([0, 1])
    with pytest.raises(Exception, match="panel_order"):
        widget.panel_order = [0, 0, 1]

    smaller = Show3D(*_panels()[:2], panel_titles=["SSB", "Mean DP"], show_controls=False)
    smaller.load_state_dict(state)
    assert smaller.panel_order == []
    assert smaller.visible_panels == [0, 1]

    out = widget.export_html(tmp_path / "show3d_panel_order.html", encoding="full")
    html = out.read_text()
    assert "panel_order" in html


def test_show3d_panel_frame_labels_roundtrip_in_state_and_html(tmp_path: pathlib.Path) -> None:
    frame_labels = [
        ["SSB iter 1", "SSB iter 2", "SSB iter 3"],
        ["Mean DP iter 1", "Mean DP iter 2", "Mean DP iter 3"],
    ]
    widget = Show3D(
        *_panels()[:2],
        panel_titles=["SSB", "Mean DP"],
        panel_frame_labels=frame_labels,
        show_controls=False,
    )

    state = widget.state_dict()
    assert state["panel_frame_labels"] == frame_labels

    restored = Show3D(*_panels()[:2], panel_titles=["SSB", "Mean DP"], show_controls=False)
    restored.load_state_dict(state)
    assert restored.panel_frame_labels == frame_labels

    out = widget.export_html(tmp_path / "show3d_panel_frame_labels.html", encoding="full")
    html = out.read_text()
    assert "panel_frame_labels" in html
    assert "SSB iter 2" in html
    assert "Mean DP iter 3" in html


def test_show3d_panel_frame_labels_validate_shape() -> None:
    with pytest.raises(Exception, match="panel_frame_labels"):
        Show3D(
            *_panels()[:2],
            panel_titles=["SSB", "Mean DP"],
            panel_frame_labels=[["only one panel"]],
            show_controls=False,
        )


def test_show3d_frame_metadata_generates_labels_and_roundtrips_html(tmp_path: pathlib.Path) -> None:
    frame_metadata = [
        {"iteration": np.int64(1), "defocus_nm": np.float32(-12.5), "loss": 0.031},
        {"iteration": np.int64(2), "defocus_nm": np.float32(-10.0), "loss": 0.024},
        {"iteration": np.int64(3), "defocus_nm": np.float32(-8.5), "loss": 0.019},
    ]
    panel_frame_metadata = [
        [
            {"iteration": 1, "defocus_nm": -12.5, "loss": 0.031},
            {"iteration": 2, "defocus_nm": -10.0, "loss": 0.024},
            {"iteration": 3, "defocus_nm": -8.5, "loss": 0.019},
        ],
        [
            {"iteration": 1, "defocus_nm": -6.0, "loss": 0.041},
            {"iteration": 2, "defocus_nm": -5.5, "loss": 0.035},
            {"iteration": 3, "defocus_nm": -5.0, "loss": 0.028},
        ],
    ]
    widget = Show3D(
        *_panels()[:2],
        panel_titles=["SSB phase", "SSB amplitude"],
        frame_metadata=frame_metadata,
        panel_frame_metadata=panel_frame_metadata,
        frame_label_format="iter {iteration} · df={defocus_nm:.1f} nm · loss={loss:.3f}",
        show_controls=False,
    )

    assert widget.labels[1] == "iter 2 · df=-10.0 nm · loss=0.024"
    assert widget.panel_frame_labels[1][2] == "iter 3 · df=-5.0 nm · loss=0.028"
    assert widget.frame_metadata[0]["iteration"] == 1
    assert isinstance(widget.frame_metadata[0]["iteration"], int)

    state = widget.state_dict()
    assert state["frame_label_format"] == "iter {iteration} · df={defocus_nm:.1f} nm · loss={loss:.3f}"
    assert state["panel_frame_metadata"][1][0]["defocus_nm"] == -6.0

    restored = Show3D(*_panels()[:2], panel_titles=["SSB phase", "SSB amplitude"], show_controls=False)
    restored.load_state_dict(state)
    assert restored.panel_frame_metadata == state["panel_frame_metadata"]
    assert restored.panel_frame_labels == widget.panel_frame_labels

    out = widget.export_html(tmp_path / "show3d_frame_metadata.html", encoding="full")
    html = out.read_text()
    assert "panel_frame_metadata" in html
    assert "iter 3" in html
    assert "df=-5.0 nm" in html


def test_show3d_frame_metadata_callable_formatter() -> None:
    widget = Show3D(
        np.zeros((3, 4, 5), dtype=np.float32),
        frame_metadata=[{"iteration": i + 1, "loss": 1 / (i + 1)} for i in range(3)],
        frame_label_format=lambda meta, frame_idx, panel_idx: (
            f"step {meta['iteration']} frame {frame_idx + 1} loss {meta['loss']:.2f}"
        ),
        show_controls=False,
    )

    assert widget.labels == [
        "step 1 frame 1 loss 1.00",
        "step 2 frame 2 loss 0.50",
        "step 3 frame 3 loss 0.33",
    ]
    assert widget.frame_label_format == ""


def test_show3d_frame_metadata_format_missing_key_is_helpful() -> None:
    with pytest.raises(KeyError, match="missing metadata key"):
        Show3D(
            np.zeros((1, 4, 5), dtype=np.float32),
            frame_metadata=[{"iteration": 1}],
            frame_label_format="{iteration} {defocus_nm}",
            show_controls=False,
        )

    with pytest.raises(Exception, match="panel_frame_labels"):
        Show3D(
            *_panels()[:2],
            panel_titles=["SSB", "Mean DP"],
            panel_frame_labels=[["too short"], ["Mean DP iter 1", "Mean DP iter 2", "Mean DP iter 3"]],
            show_controls=False,
        )
