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
import pathlib

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


def test_show2d_static_png_preserves_sparse_large_content():
    """The fallback PNG must show real image content, not a blank dark tile.

    A 4k-style sparse image is exactly where stride sampling fails: if the
    bright feature lands between sampled rows/columns, the static fallback looks
    nearly black even though the live widget rendered fine. The PNG path should
    area-downsample instead, so thin/sparse features survive notebook save.
    """
    import base64
    import io

    from PIL import Image

    image = np.zeros((1024, 1024), dtype=np.float32)
    image[101:109, 611:619] = 1000
    widget = Show2D(image, labels=["Sparse feature"], save_state=False, verbose=False)

    bundle = widget._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    png = data["image/png"]
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB"))

    nonwhite = ~np.all(decoded > 245, axis=-1)
    warm_signal = (
        (decoded[..., 0] > 140)
        & (decoded[..., 1] > 80)
        & (decoded[..., 2] < 230)
        & nonwhite
    )
    assert int(warm_signal.sum()) > 20


def test_show2d_first_render_clears_heavy_frame_buffer():
    """After JS paints, later notebook saves should not keep pixel buffers."""
    image = np.random.default_rng(0).random((128, 128), dtype=np.float32)
    widget = Show2D(image, save_state=False, verbose=False)
    assert widget.frame_bytes

    widget._on_first_render({"new": True})

    assert widget.frame_bytes == b""
    assert "frame_bytes" not in widget.get_state()
    assert not widget._get_embed_state().get("buffers")


def _assert_export_state_keeps_buffer(widget, key: str) -> None:
    """HTML export clones must opt into the full embedded state."""
    state = widget.get_state()
    assert widget._save_state is True
    assert key in state
    value = state[key]
    if isinstance(value, (bytes, bytearray)):
        assert len(value) > 0


def test_html_export_clones_keep_bulk_buffers():
    """Standalone HTML export must embed the data it needs to render offline.

    ``export_html`` builds an export-only clone and then asks ipywidgets for a
    full dependency state. That call hits ``get_state(key=None)``. If the clone
    stays on the notebook-save default (``_save_state=False``), the heavy pixel
    payload is trimmed and the exported HTML renders blank.
    """
    rng = np.random.default_rng(10)
    show2d = Show2D(rng.random((64, 64), dtype=np.float32), save_state=False, verbose=False)
    show2d_clone = show2d._clone_for_html_export(quantized=False)
    try:
        _assert_export_state_keeps_buffer(show2d_clone, "frame_bytes")
    finally:
        show2d_clone.close()
        show2d.close()

    show3d = Show3D(rng.random((3, 32, 32), dtype=np.float32), save_state=False)
    show3d_clone = show3d._clone_for_html_export(quantized=False)
    try:
        _assert_export_state_keeps_buffer(show3d_clone, "_offline_float_stack")
    finally:
        show3d_clone.close()
        show3d.close()

    stem = Show4DSTEM(
        rng.integers(0, 100, (4, 4, 8, 8), dtype=np.uint16),
        save_state=False,
        verbose=False,
    )
    stem_clone = stem._clone_for_html_export(dtype="uint16", det_bin=1)
    try:
        _assert_export_state_keeps_buffer(stem_clone, "_offline_stack")
    finally:
        stem_clone.close()
        stem.close()


def test_show4dstem_bslz4_html_export_embeds_self_with_bulk_state(monkeypatch, tmp_path):
    """Show4DSTEM's bslz4 export branch embeds ``self``, not an export clone."""
    widget = Show4DSTEM(
        np.random.default_rng(11).integers(0, 100, (4, 4, 8, 8), dtype=np.uint16),
        save_state=False,
        verbose=False,
    )
    widget._offline_bslz4 = "{}"
    widget._offline_stack = b"offline-stack"
    captured = {}

    def fake_dependency_state(views, drop_defaults=False):
        view = views[0]
        state = view.get_state()
        captured["save_state"] = view._save_state
        captured["has_stack"] = "_offline_stack" in state
        captured["stack_size"] = len(state.get("_offline_stack", b""))
        return {}

    def fake_embed_minimal_html(filename, *, views, title, drop_defaults, state):
        pathlib.Path(filename).write_text("<html><head></head><body></body></html>")

    monkeypatch.setattr("ipywidgets.embed.dependency_state", fake_dependency_state)
    monkeypatch.setattr("ipywidgets.embed.embed_minimal_html", fake_embed_minimal_html)

    try:
        widget._write_html_export(tmp_path / "show4dstem.html", dtype="uint16", det_bin=1)
    finally:
        widget.close()

    assert captured == {"save_state": True, "has_stack": True, "stack_size": len(b"offline-stack")}
    assert widget._save_state is False


def test_export_html_size_scales_with_embedded_data(tmp_path):
    """Public ``export_html`` must not collapse to a tiny empty widget shell.

    Regression for the save_state opt-in bug: export clones used the notebook
    default ``_save_state=False``, so ipywidgets stripped the heavy buffers and
    wrote a small HTML file with no offline image stack.
    """
    rng = np.random.default_rng(12)
    cases = []

    data2d = rng.random((3, 512, 512), dtype=np.float32)
    show2d = Show2D(data2d, save_state=False, verbose=False, title="size-show2d")
    cases.append((show2d, data2d.nbytes, "frame_bytes", tmp_path / "show2d.html"))

    data3d = rng.random((8, 256, 256), dtype=np.float32)
    show3d = Show3D(data3d, save_state=False, title="size-show3d")
    cases.append((show3d, data3d.nbytes, "_offline_float_stack", tmp_path / "show3d.html"))

    stem_data = rng.integers(0, 1000, (32, 32, 32, 32), dtype=np.uint16)
    stem = Show4DSTEM(stem_data, save_state=False, verbose=False, title="size-show4dstem")
    cases.append((stem, stem_data.nbytes, "_offline_stack", tmp_path / "show4dstem.html"))

    try:
        for widget, raw_bytes, marker, path in cases:
            exported = widget.export_html(path, encoding="full")
            html = exported.read_text(errors="ignore")
            assert exported.stat().st_size > raw_bytes / 2
            assert marker in html
    finally:
        for widget, _, _, _ in cases:
            widget.close()
