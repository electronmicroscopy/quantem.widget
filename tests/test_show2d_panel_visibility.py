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


def test_show2d_hidden_and_starred_roundtrip_in_state_and_html(tmp_path: pathlib.Path) -> None:
    widget = Show2D(
        _images()[:2],
        labels=["Raw", "Filtered"],
        hidden_panels=["Filtered"],
        starred=["Raw"],
        show_controls=False,
        verbose=False,
    )

    state = widget.state_dict()
    assert state["labels"] == ["Raw", "Filtered"]
    assert state["hidden_panels"] == [1]
    assert state["starred"] == [1, 0]

    restored = Show2D(_images()[:2], labels=["Raw", "Filtered"], show_controls=False, verbose=False)
    restored.load_state_dict(state)
    assert restored.hidden_panels == [1]
    assert restored.visible_panels == [0]
    assert restored.starred_panels == [0]

    out = widget.export_html(tmp_path / "show2d_hidden_starred.html", encoding="full")
    html = out.read_text()
    assert "hidden_panels" in html
    assert "starred" in html
    assert "Filtered" in html
