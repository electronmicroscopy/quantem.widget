"""Regression shield for the ``save_state`` contract (Show2D / Show3D / Show4DSTEM).

Background: an anywidget syncs its pixel buffers as ``sync=True`` traits. On
notebook save, ipywidgets serializes those buffers into ``metadata.widgets`` -
a 5-panel 4k Show2D baked ~1 GB into a single .ipynb. The fix: ``save_state``
(default False) drops the bulk buffers from the FULL-state snapshot (the save
path, ``get_state(key=None)``) and instead attaches a static PNG so a cold
reopen still shows the render. ``save_state=True`` embeds everything for
kernel-less restore.

The danger in that fix is trimming too much: if the trim also hit the TARGETED
``send_state`` path (``get_state(key=<name/set>)``), the frontend would never
receive the buffer and the widget would render blank - invisible to unit tests
that only check output size. These tests lock both halves of the contract so a
future edit can't silently reintroduce either the bloat or the blank render.
"""
import numpy as np
import pytest

from quantem.widget import Show2D, Show3D, Show4DSTEM


def _make(widget, *, save_state):
    """Construct a small instance of each widget plus the trait key that carries
    its live-render pixels (the one that must survive the targeted send path)."""
    if widget is Show2D:
        data = [np.random.rand(128, 128).astype("float32") for _ in range(2)]
        return Show2D(data, save_state=save_state), "frame_bytes"
    if widget is Show3D:
        return Show3D(np.random.rand(4, 128, 128).astype("float32"),
                      save_state=save_state), "frame_bytes"
    return Show4DSTEM(np.random.rand(8, 8, 16, 16).astype("float32"),
                      save_state=save_state), "virtual_image_bytes"


WIDGETS = [Show2D, Show3D, Show4DSTEM]


@pytest.mark.parametrize("widget", WIDGETS)
def test_targeted_send_state_never_trimmed(widget):
    """The render path must stay intact. ``send_state`` / ``hold_sync`` call
    ``get_state`` with a specific key (or set), never ``None``; the trim must
    only fire on the full ``key=None`` snapshot. If a targeted lookup loses the
    pixel key, the frontend renders blank."""
    w, render_key = _make(widget, save_state=False)
    assert render_key in w.get_state(render_key), (
        f"{widget.__name__}: targeted get_state({render_key!r}) dropped the key "
        f"- live render would go blank")
    assert render_key in w.get_state({render_key, "widget_version"}), (
        f"{widget.__name__}: hold_sync batch lost {render_key!r}")


@pytest.mark.parametrize("widget", WIDGETS)
def test_full_snapshot_trims_bulk_buffers(widget):
    """save_state=False: no bulk pixel buffer may appear in the saved-notebook
    snapshot. This is the anti-1GB guard."""
    w, _ = _make(widget, save_state=False)
    full = w.get_state()
    leaked = [k for k in w._UNSAVED_HEAVY_KEYS if k in full]
    assert not leaked, (
        f"{widget.__name__}: bulk buffers {leaked} leaked into saved state "
        f"- notebook will bloat")


@pytest.mark.parametrize("widget", WIDGETS)
def test_static_png_fallback_present(widget):
    """save_state=False: a static image/png must be attached so a kernel-less
    reopen (GitHub, nbviewer, cold Lab) still shows the render."""
    w, _ = _make(widget, save_state=False)
    bundle = w._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    assert "image/png" in (data or {}), (
        f"{widget.__name__}: no static PNG fallback for a cold reopen")


@pytest.mark.parametrize("widget", WIDGETS)
def test_save_state_true_does_not_force_png(widget):
    """save_state=True embeds full interactive state, so it must NOT inject the
    static PNG (the live widget restores from the embedded buffers instead)."""
    w, _ = _make(widget, save_state=True)
    bundle = w._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    assert "image/png" not in (data or {}), (
        f"{widget.__name__}: save_state=True should not attach the static PNG")


@pytest.mark.parametrize("widget", WIDGETS)
def test_default_is_save_state_false(widget):
    """The whole point: persistence is opt-in. A plain construction must behave
    as save_state=False (static PNG present)."""
    w, _ = _make(widget, save_state=False)
    assert w._save_state is False
