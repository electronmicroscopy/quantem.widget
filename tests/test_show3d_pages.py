from __future__ import annotations

import pathlib

import numpy as np

from quantem.widget import Show3D


def _paged_stack() -> np.ndarray:
    pages = []
    for page in range(3):
        panels = []
        for panel in range(4):
            frames = []
            for frame in range(5):
                frames.append(
                    np.full((6, 7), page * 100 + panel * 10 + frame, dtype=np.float32)
                )
            panels.append(frames)
        pages.append(panels)
    return np.asarray(pages, dtype=np.float32)


def test_show3d_accepts_5d_paged_panels() -> None:
    widget = Show3D(
        _paged_stack(),
        panel_titles=["raw", "filtered", "residual", "probe"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        show_controls=False,
        verbose=False,
    )

    assert widget.n_pages == 3
    assert widget.panels_per_page == 4
    assert widget.n_panels == 12
    assert widget.n_slices == 5
    assert widget.page_labels == ["lambda 0.01", "lambda 0.03", "lambda 0.10"]
    assert widget.panel_titles[:4] == ["raw", "filtered", "residual", "probe"]
    assert widget.visible_panels == [0, 1, 2, 3]

    widget.page_idx = 2
    assert widget.visible_panels == [8, 9, 10, 11]
    assert np.all(widget._get_display_panel_frame(9, 4) == 214)


def test_show3d_accepts_dict_pages_and_page_stars() -> None:
    base = _paged_stack()
    widget = Show3D(
        [
            {"title": "iteration 10", "stacks": base[0], "panel_titles": ["a", "b", "c", "d"]},
            {"title": "iteration 20", "stacks": base[1], "panel_titles": ["a", "b", "c", "d"]},
        ],
        show_controls=False,
        verbose=False,
    )

    assert widget.n_pages == 2
    assert widget.panels_per_page == 4
    assert widget.page_labels == ["iteration 10", "iteration 20"]
    assert widget.panel_titles == ["a", "b", "c", "d", "a", "b", "c", "d"]

    assert widget.star_page(1) is widget
    assert widget.starred_pages == [1]
    assert widget.unstar_page(1) is widget
    assert widget.starred_pages == []


def test_show3d_paged_state_and_html_roundtrip(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        _paged_stack(),
        panel_titles=["raw", "filtered", "residual", "probe"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        show_controls=True,
        verbose=False,
    )
    widget.page_idx = 1
    widget.hide_panel(5)
    widget.star_page(1)
    widget.star_panel(6, frame=3)

    restored = Show3D(
        _paged_stack(),
        panel_titles=["raw", "filtered", "residual", "probe"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        verbose=False,
    )
    restored.load_state_dict(widget.state_dict())

    assert restored.page_idx == 1
    assert restored.page_starred == [0, 1, 0]
    assert restored.hidden_panels == [5]
    assert restored.visible_panels == [4, 6, 7]
    assert restored.starred[6] == 3

    out = widget.export_html(tmp_path / "show3d_pages.html", encoding="uint8")
    html = out.read_text()
    assert "page_idx" in html
    assert "page_starred" in html
    assert "lambda 0.03" in html


def test_show3d_paged_to_show2d_uses_current_visible_page() -> None:
    widget = Show3D(
        _paged_stack(),
        panel_titles=["raw", "filtered", "residual", "probe"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        show_controls=False,
        verbose=False,
    )
    widget.page_idx = 2
    widget.hide_panel(9)
    show2d = widget.to_show2d(frame=4)

    assert show2d.n_images == 3
    assert show2d.labels == ["raw 5/5", "residual 5/5", "probe 5/5"]
    assert np.all(show2d._data[0] == 204)
    assert np.all(show2d._data[1] == 224)
    assert np.all(show2d._data[2] == 234)
