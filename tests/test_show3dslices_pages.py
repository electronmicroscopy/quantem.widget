from __future__ import annotations

from pathlib import Path

import numpy as np

from quantem.widget import Show3DSlices


ROOT = Path(__file__).resolve().parents[1]


def _volume(offset: float = 0.0) -> np.ndarray:
    return np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) + offset


def test_show3dslices_4d_page_shorthand_preserves_geometry() -> None:
    data = np.stack([_volume(), _volume(100)], axis=0)
    widget = Show3DSlices(
        data,
        page_labels=["raw", "corrected"],
        panel_titles=["object"],
        show_controls=False,
    )

    assert widget.n_pages == 2
    assert widget.panels_per_page == 1
    assert widget.panel_count == 2
    assert widget.page_labels == ["raw", "corrected"]
    assert widget.panel_titles == ["object", "object"]
    assert widget.page_idx == 0
    assert widget.active_panel == 0

    widget.slice_z = 1
    widget.slice_y = 2
    widget.slice_x = 3
    widget.set_page(1)

    assert widget.page_idx == 1
    assert widget.active_panel == 1
    assert (widget.slice_z, widget.slice_y, widget.slice_x) == (1, 2, 3)
    np.testing.assert_array_equal(widget._active_data(), data[1])


def test_show3dslices_5d_pages_preserve_active_panel_slot() -> None:
    data = np.stack(
        [
            np.stack([_volume(), _volume(10)], axis=0),
            np.stack([_volume(100), _volume(110)], axis=0),
        ],
        axis=0,
    )
    widget = Show3DSlices(
        data,
        page_labels=["single slice", "multislice"],
        panel_titles=["object", "error"],
        show_controls=False,
    )

    assert widget.n_pages == 2
    assert widget.panels_per_page == 2
    assert widget.panel_count == 4
    assert widget.panel_titles == ["object", "error", "object", "error"]

    widget.active_panel = 1
    widget.next_page()
    assert widget.page_idx == 1
    assert widget.active_panel == 3
    np.testing.assert_array_equal(widget._active_data(), data[1, 1])

    widget.active_panel = 0
    assert widget.page_idx == 0


def test_show3dslices_accepts_explicit_page_descriptors() -> None:
    widget = Show3DSlices(
        [
            {"title": "before", "volume": _volume(), "panel_titles": ["object"]},
            {"title": "after", "volume": _volume(50), "panel_titles": ["object"]},
        ],
        show_controls=False,
    )

    assert widget.n_pages == 2
    assert widget.panels_per_page == 1
    assert widget.page_labels == ["before", "after"]
    assert widget.panel_titles == ["object", "object"]


def test_show3dslices_page_state_and_html_roundtrip(tmp_path: Path) -> None:
    data = np.stack([_volume(), _volume(100)], axis=0)
    widget = Show3DSlices(
        data,
        title="depth comparison",
        page_labels=["raw", "corrected"],
        show_controls=True,
    )
    widget.page_idx = 1

    restored = Show3DSlices(
        data,
        page_labels=["raw", "corrected"],
        show_controls=False,
    )
    restored.load_state_dict(widget.state_dict())
    assert restored.page_idx == 1
    assert restored.active_panel == 1
    assert restored.page_labels == ["raw", "corrected"]

    out = widget.export_html(tmp_path / "show3dslices_pages.html", encoding="uint8")
    html = out.read_text(encoding="utf-8")
    assert "page_idx" in html
    assert "page_labels" in html
    assert "corrected" in html


def test_show3dslices_frontend_uses_page_controls() -> None:
    source = (ROOT / "js" / "show3dslices" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert 'useModelState<number>("n_pages")' in source
    assert 'useModelState<number>("page_idx")' in source
    assert 'useModelState<string[]>("page_labels")' in source
    assert 'aria-label="Show3DSlices page"' in source
    assert 'title={pagePlaying ? "Pause page playback" : "Play pages"}' in source
    assert "Page switches comparable volumes while preserving" in source
    assert 'flexDirection: { xs: "column", md: "row" }' in source
    assert 'flexDirection: { xs: "column", sm: "row" }' in source
