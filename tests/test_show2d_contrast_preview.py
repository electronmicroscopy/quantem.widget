"""Frontend contracts for independent Show2D contrast previews."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unlinked_contrast_preview_is_panel_local() -> None:
    """Dragging one histogram must not repaint the rest of the gallery."""
    source = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")

    assert "const previewIndices = linkedContrast ? visibleIndices : [idx];" in source
    assert "engine.renderSlotsToImageBitmap(previewIndices, bitmapRanges, ls)" in source
    assert "const panel = previewIndices[k];" in source


def test_auto_contrast_ranges_seed_the_fast_preview_state() -> None:
    """The first manual drag must start from every panel's visible auto range."""
    source = (ROOT / "js" / "show2d" / "index.tsx").read_text(encoding="utf-8")

    assert "const nextPerImage = new Map(contrastRef.current.perImage);" in source
    assert "contrastRef.current.perImage = nextPerImage;" in source
    assert "contrastRef.current.linked = state;" in source
    assert "setContrastState(i, { vminPct: min, vmaxPct: max }, leavingAuto)" in source
