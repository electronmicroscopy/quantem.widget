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
