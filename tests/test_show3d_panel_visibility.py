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


def test_show3d_fft_layout_validates_and_roundtrips(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        *_panels()[:2],
        panel_titles=["SSB", "Mean DP"],
        show_fft=True,
        fft_layout="right",
        show_controls=False,
    )

    assert widget.fft_layout == "right"
    state = widget.state_dict()
    assert state["fft_layout"] == "right"

    restored = Show3D(*_panels()[:2], panel_titles=["SSB", "Mean DP"], show_controls=False)
    restored.load_state_dict(state)
    assert restored.fft_layout == "right"

    out = widget.export_html(tmp_path / "show3d_fft_layout.html", encoding="full")
    assert "fft_layout" in out.read_text()

    with pytest.raises(ValueError, match="fft_layout"):
        Show3D(np.zeros((3, 4, 5), dtype=np.float32), fft_layout="floating")


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
