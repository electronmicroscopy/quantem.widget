import pathlib

import numpy as np

from quantem.widget import Show2D


def _paged_stack() -> np.ndarray:
    rows, cols = np.mgrid[:16, :16]
    pages = []
    for page in range(3):
        panels = []
        for panel in range(4):
            panels.append(
                np.exp(-((rows - 5 - page) ** 2 + (cols - 5 - panel) ** 2) / 18)
                + 0.1 * page
                + 0.03 * panel
            )
        pages.append(panels)
    return np.asarray(pages, dtype=np.float32)


def test_show2d_accepts_4d_pages() -> None:
    data = _paged_stack()
    widget = Show2D(
        data,
        labels=["raw", "filtered", "residual", "score"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        ncols=2,
        verbose=False,
    )

    assert widget.n_pages == 3
    assert widget.panels_per_page == 4
    assert widget.n_images == 12
    assert widget.labels[:4] == ["raw", "filtered", "residual", "score"]
    assert widget.labels[4:8] == ["raw", "filtered", "residual", "score"]
    assert widget.visible_panels == [0, 1, 2, 3]

    widget.page_idx = 2
    assert widget.visible_panels == [8, 9, 10, 11]


def test_show2d_accepts_dict_pages_and_page_stars() -> None:
    data = _paged_stack()
    widget = Show2D(
        [
            {"title": "iteration 10", "images": data[0], "labels": ["a", "b", "c", "d"]},
            {"title": "iteration 20", "images": data[1], "labels": ["a", "b", "c", "d"]},
        ],
        verbose=False,
    )

    assert widget.page_labels == ["iteration 10", "iteration 20"]
    assert widget.labels == ["a", "b", "c", "d", "a", "b", "c", "d"]
    widget.star_page(1)
    assert widget.starred_pages == [1]
    widget.unstar_page(1)
    assert widget.starred_pages == []


def test_show2d_dict_pages_can_share_global_panel_labels() -> None:
    data = _paged_stack()
    widget = Show2D(
        [
            {"title": "iteration 10", "images": data[0]},
            {"title": "iteration 20", "images": data[1]},
        ],
        labels=["raw", "filtered", "residual", "score"],
        verbose=False,
    )

    assert widget.labels == ["raw", "filtered", "residual", "score"] * 2


def test_show2d_page_state_and_html_roundtrip(tmp_path: pathlib.Path) -> None:
    widget = Show2D(
        _paged_stack(),
        labels=["raw", "filtered", "residual", "score"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        verbose=False,
    )
    widget.page_idx = 1
    widget.star_page(1)
    state = widget.state_dict()

    restored = Show2D(
        _paged_stack(),
        labels=["raw", "filtered", "residual", "score"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        verbose=False,
    )
    restored.load_state_dict(state)
    assert restored.page_idx == 1
    assert restored.page_starred == [0, 1, 0]
    assert restored.visible_panels == [4, 5, 6, 7]

    out = widget.export_html(tmp_path / "show2d_pages.html", encoding="uint8")
    html = out.read_text()
    assert "page_idx" in html
    assert "page_starred" in html
    assert "lambda 0.03" in html


def test_show2d_hidden_page_slots_follow_paged_layout(tmp_path: pathlib.Path) -> None:
    widget = Show2D(
        _paged_stack(),
        labels=["raw", "filtered", "residual", "score"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        verbose=False,
    )

    widget.hide_panel(1)
    assert widget.hidden_page_slots == [1]
    assert widget.visible_panels == [0, 2, 3]

    widget.page_idx = 2
    assert widget.visible_panels == [8, 10, 11]

    restored = Show2D(
        _paged_stack(),
        labels=["raw", "filtered", "residual", "score"],
        page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
        verbose=False,
    )
    restored.load_state_dict(widget.state_dict())
    restored.page_idx = 1
    assert restored.hidden_page_slots == [1]
    assert restored.visible_panels == [4, 6, 7]

    out = widget.export_html(tmp_path / "show2d_hidden_page_slots.html", encoding="uint8")
    assert "hidden_page_slots" in out.read_text()


def test_show2d_folder_item_pages_restore_and_export_partial_page(
    tmp_path: pathlib.Path,
) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    arrays = []
    for index in range(45):
        array = np.full((8, 9), index, dtype=np.float32)
        arrays.append(array)
        np.save(folder / f"frame_{index:03d}.npy", array)

    widget = Show2D.from_folder(folder, watch=False, page_size=20)
    restored = Show2D(np.stack(arrays), verbose=False)
    try:
        widget.page_idx = 2
        widget.hide_panel(40)
        state = widget.state_dict()
        assert state["folder_page_size"] == 20

        # C1: page metadata installs before page_idx and hidden-state
        # validation, expect the saved final page to survive an unpaged target.
        restored.load_state_dict(state)
        assert restored.page_kind == "items"
        assert (restored.n_pages, restored.panels_per_page) == (3, 20)
        assert restored.page_idx == 2
        assert restored.visible_panels == [41, 42, 43, 44]

        # C2: a standalone folder-page export retains item semantics and the
        # real partial-page label without padded panels.
        out = widget.export_html(
            tmp_path / "show2d_folder_item_pages.html",
            encoding="uint8",
        )
        html = out.read_text()
        assert "page_kind" in html
        assert r"Images 41\u201345" in html
        assert "frame_044" in html
    finally:
        widget.close()
        restored.close()
