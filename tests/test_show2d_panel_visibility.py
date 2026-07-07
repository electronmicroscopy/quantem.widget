from __future__ import annotations

import pathlib

import numpy as np
import pytest

from quantem.widget import Show2D


def _images() -> np.ndarray:
    return np.stack(
        [np.full((4, 5), fill_value=i, dtype=np.float32) for i in range(3)]
    )


def test_show2d_hides_panels_by_label_or_index() -> None:
    widget = Show2D(
        _images(),
        labels=["Raw", "Filtered", "Residual"],
        hidden_panels=["Filtered"],
        show_controls=False,
        verbose=False,
    )

    assert widget.hidden_panels == [1]
    assert widget.visible_panels == [0, 2]

    widget.hide_panel(2)
    assert widget.hidden_panels == [1, 2]
    assert widget.visible_panels == [0]

    widget.show_panel("Filtered")
    assert widget.hidden_panels == [2]

    widget.show_all_panels()
    assert widget.hidden_panels == []
    assert widget.visible_panels == [0, 1, 2]


def test_show2d_refuses_to_hide_every_panel() -> None:
    widget = Show2D(
        _images()[:2],
        labels=["Raw", "Filtered"],
        show_controls=False,
        verbose=False,
    )

    with pytest.raises(ValueError, match="hide every panel"):
        widget.set_hidden_panels([0, 1])

    with pytest.raises(ValueError, match="hide every panel"):
        widget.hide_panel("Raw", "Filtered")


def test_show2d_panel_label_must_be_unique_for_visibility_lookup() -> None:
    with pytest.raises(ValueError, match="not unique"):
        Show2D(
            _images()[:2],
            labels=["duplicate", "duplicate"],
            hidden_panels=["duplicate"],
            show_controls=False,
            verbose=False,
        )


def test_show2d_starred_panels_roundtrip() -> None:
    widget = Show2D(
        _images(),
        labels=["Raw", "Filtered", "Residual"],
        starred=["Filtered"],
        show_controls=False,
        verbose=False,
    )

    assert widget.starred == [0, 1, 0]
    assert widget.starred_panels == [1]

    widget.star_panel("Residual")
    assert widget.starred_panels == [1, 2]

    widget.unstar_panel(1)
    assert widget.starred_panels == [2]


def test_show2d_panel_order_controls_visible_order_and_handoff() -> None:
    widget = Show2D(
        _images(),
        labels=["Raw", "Filtered", "Residual"],
        panel_order=["Residual", "Raw", "Filtered"],
        hidden_panels=["Filtered"],
        show_controls=False,
        verbose=False,
    )

    assert widget.panel_order == [2, 0, 1]
    assert widget.ordered_panels == [2, 0, 1]
    assert widget.visible_panels == [2, 0]

    widget.move_panel("Raw", 2)
    assert widget.panel_order == [2, 1, 0]
    assert widget.visible_panels == [2, 0]

    show3d = widget.to_show3d()
    assert show3d.labels == ["Residual", "Raw"]

    widget.reset_panel_order()
    assert widget.panel_order == []
    assert widget.visible_panels == [0, 2]


def test_show2d_panel_order_roundtrips_state_and_validates() -> None:
    widget = Show2D(
        _images(),
        labels=["Raw", "Filtered", "Residual"],
        show_controls=False,
        verbose=False,
    )
    widget.set_panel_order([2, 0, 1])
    state = widget.state_dict()
    assert state["panel_order"] == [2, 0, 1]

    restored = Show2D(_images(), labels=["Raw", "Filtered", "Residual"], show_controls=False, verbose=False)
    restored.load_state_dict(state)
    assert restored.panel_order == [2, 0, 1]
    assert restored.visible_panels == [2, 0, 1]

    with pytest.raises(ValueError, match="every panel exactly once"):
        widget.set_panel_order([0, 1])

    with pytest.raises(Exception, match="panel_order"):
        widget.panel_order = [0, 0, 1]


def test_show2d_hidden_and_starred_roundtrip_in_state_and_html(tmp_path: pathlib.Path) -> None:
    widget = Show2D(
        _images()[:2],
        labels=["Raw", "Filtered"],
        panel_order=["Filtered", "Raw"],
        hidden_panels=["Filtered"],
        starred=["Raw"],
        show_controls=False,
        verbose=False,
    )

    state = widget.state_dict()
    assert state["labels"] == ["Raw", "Filtered"]
    assert state["panel_order"] == [1, 0]
    assert state["hidden_panels"] == [1]
    assert state["starred"] == [1, 0]

    restored = Show2D(_images()[:2], labels=["Raw", "Filtered"], show_controls=False, verbose=False)
    restored.load_state_dict(state)
    assert restored.panel_order == [1, 0]
    assert restored.hidden_panels == [1]
    assert restored.visible_panels == [0]
    assert restored.starred_panels == [0]

    out = widget.export_html(tmp_path / "show2d_hidden_starred.html", encoding="full")
    html = out.read_text()
    assert "hidden_panels" in html
    assert "panel_order" in html
    assert "starred" in html
    assert "Filtered" in html


def test_show2d_controls_collapsed_roundtrips_state_and_html(tmp_path: pathlib.Path) -> None:
    widget = Show2D(
        _images()[:2],
        labels=["Raw", "Filtered"],
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

    restored = Show2D(_images()[:2], labels=["Raw", "Filtered"], verbose=False)
    restored.load_state_dict(state)
    assert restored.controls_collapsed is True

    out = widget.export_html(tmp_path / "show2d_controls_collapsed.html", encoding="full")
    html = out.read_text()
    assert "controls_collapsed" in html


def test_show2d_ui_mode_presets_and_overrides() -> None:
    presentation = Show2D(_images()[:2], ui_mode="presentation", verbose=False)
    assert presentation.show_title is True
    assert presentation.show_controls is True
    assert presentation.controls_collapsed is True
    assert presentation.show_stats is False
    assert presentation.show_panel_titles is True
    assert presentation.scale_bar_visible is True

    report = Show2D(_images()[:2], ui_mode="report", verbose=False)
    assert report.show_title is True
    assert report.show_controls is False
    assert report.controls_collapsed is False
    assert report.show_stats is False
    assert report.show_panel_titles is True
    assert report.scale_bar_visible is True

    minimal = Show2D(_images()[:2], ui_mode="minimal", verbose=False)
    assert minimal.show_title is False
    assert minimal.show_controls is False
    assert minimal.controls_collapsed is False
    assert minimal.show_stats is False
    assert minimal.show_panel_titles is False
    assert minimal.scale_bar_visible is False

    override = Show2D(
        _images()[:2],
        ui_mode="minimal",
        show_title=True,
        show_controls=True,
        controls_collapsed=True,
        show_panel_titles=True,
        show_scale_bar=True,
        verbose=False,
    )
    assert override.show_title is True
    assert override.show_controls is True
    assert override.controls_collapsed is True
    assert override.show_panel_titles is True
    assert override.scale_bar_visible is True

    alias = Show2D(_images()[:1], show_scale_bar=False, verbose=False)
    assert alias.scale_bar_visible is False

    with pytest.raises(ValueError, match="show_scale_bar"):
        Show2D(_images()[:1], show_scale_bar=True, scale_bar_visible=False, verbose=False)
