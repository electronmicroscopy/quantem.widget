"""Regression shield for the ``save_state`` contract (Plot2D / Show1D / Show2D / Show3D / Show4DSTEM).

Background: an anywidget syncs its pixel buffers as ``sync=True`` traits. On
notebook save, ipywidgets serializes those buffers into ``metadata.widgets`` -
a 5-panel 4k Show2D baked ~1 GB into a single .ipynb. The fix: ``save_state``
(default False) drops the bulk buffers from the FULL-state snapshot (the save
path, ``get_state(key=None)``) and instead attaches a static preview image so a
cold reopen still shows the render. ``save_state=True`` embeds everything for
kernel-less restore.

The danger in that fix is trimming too much: if the trim also hit the TARGETED
``send_state`` path (``get_state(key=<name/set>)``), the frontend would never
receive the buffer and the widget would render blank - invisible to unit tests
that only check output size. These tests lock both halves of the contract so a
future edit can't silently reintroduce either the bloat or the blank render.
"""


import base64


import io




import numpy as np


import pytest


from PIL import Image


from quantem.widget import Plot2D, Show1D, Show2D, Show3D, Show4DSTEM


from quantem.widget.show4dstem.export import export_clone


IMAGE_MIME_KEYS = ("image/jpeg", "image/webp", "image/png")


def _mos2_like_stack(frames: int, rows: int, cols: int) -> np.ndarray:
    """Tiny deterministic 1H-MoS2-like HAADF lattice for preview-format tests."""
    y, x = np.mgrid[:rows, :cols].astype(np.float32)
    spacing = max(7.0, min(rows, cols) / 8.5)
    sigma_mo = max(0.65, spacing * 0.115)
    sigma_s2 = max(0.72, spacing * 0.135)
    angle = np.deg2rad(8.0)
    a1 = spacing * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    a2 = spacing * np.array([np.cos(angle + np.pi / 3.0), np.sin(angle + np.pi / 3.0)], dtype=np.float32)
    basis = [
        (np.array([0.0, 0.0], dtype=np.float32), 1.00, sigma_mo),
        ((a1 + a2) / 3.0, 0.42, sigma_s2),
    ]
    out: list[np.ndarray] = []
    for idx in range(frames):
        frame = np.full((rows, cols), 0.035, dtype=np.float32)
        origin = np.array(
            [cols * 0.08 + 0.16 * idx, rows * 0.10 - 0.11 * idx],
            dtype=np.float32,
        )
        for lattice_row in range(-2, int(rows / spacing) + 4):
            for lattice_col in range(-2, int(cols / spacing) + 4):
                cell = origin + lattice_col * a1 + lattice_row * a2
                for offset, amp, sigma in basis:
                    cx, cy = cell + offset
                    if (
                        -3 * sigma <= cx < cols + 3 * sigma
                        and -3 * sigma <= cy < rows + 3 * sigma
                    ):
                        r2 = (x - cx) ** 2 + (y - cy) ** 2
                        frame += amp * np.exp(-r2 / (2.0 * sigma**2))
        frame *= 1.0 + 0.035 * np.sin((y + 0.7 * idx) / max(rows, 1) * 2.0 * np.pi)
        frame += 0.014 * np.sin((x + 2.0 * y + idx) / 5.0)
        frame -= float(frame.min())
        frame /= float(frame.max()) + 1e-6
        out.append(frame.astype(np.float32))
    return np.stack(out, axis=0)


def _make(widget, *, save_state):
    """Construct a small instance of each widget plus the trait key that carries
    its live-render pixels (the one that must survive the targeted send path)."""
    if widget is Plot2D:
        data = np.arange(24, dtype=float).reshape(4, 6)
        return Plot2D(data, x=np.arange(6), y=np.arange(4),
                      save_state=save_state), "data_bytes"
    if widget is Show1D:
        # snapshot_bytes is empty on a plain trace widget but the KEY must
        # still survive the targeted path (it streams when a monitor attaches).
        return Show1D(np.random.rand(64).astype("float32"),
                      save_state=save_state), "snapshot_bytes"
    if widget is Show2D:
        data = [np.random.rand(128, 128).astype("float32") for _ in range(2)]
        return Show2D(data, save_state=save_state), "frame_bytes"
    if widget is Show3D:
        return Show3D(np.random.rand(4, 128, 128).astype("float32"),
                      save_state=save_state), "_offline_float_stack"
    return Show4DSTEM(np.random.rand(8, 8, 16, 16).astype("float32"),
                      save_state=save_state), "virtual_image_bytes"


WIDGETS = [Plot2D, Show1D, Show2D, Show3D, Show4DSTEM]


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
    assert render_key in w.get_state({render_key, "layout"}), (
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
def test_static_fallback_present(widget):
    """save_state=False: a static image fallback must be attached so a kernel-less
    reopen (GitHub, nbviewer, cold Lab) still shows the render."""
    w, _ = _make(widget, save_state=False)
    bundle = w._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    image_keys = [key for key in IMAGE_MIME_KEYS if key in (data or {})]
    assert image_keys == ["image/png" if widget is Plot2D else "image/jpeg"], (
        f"{widget.__name__}: missing expected static fallback for a cold reopen")


@pytest.mark.parametrize("widget", WIDGETS)
def test_save_state_true_does_not_force_png(widget):
    """save_state=True embeds full interactive state, so it must NOT inject a
    static preview (the live widget restores from the embedded buffers instead)."""
    w, _ = _make(widget, save_state=True)
    bundle = w._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    image_keys = [key for key in IMAGE_MIME_KEYS if key in (data or {})]
    assert not image_keys, (
        f"{widget.__name__}: save_state=True should not attach the static preview")


@pytest.mark.parametrize("widget", WIDGETS)
def test_default_is_save_state_false(widget):
    """The whole point: persistence is opt-in. A plain construction must behave
    as save_state=False (static preview present)."""
    w, _ = _make(widget, save_state=False)
    assert w._save_state is False


@pytest.mark.parametrize(
    ("format_name", "mime", "pil_format"),
    [
        ("jpeg", "image/jpeg", "JPEG"),
        ("webp", "image/webp", "WEBP"),
        ("png", "image/png", "PNG"),
    ],
)
@pytest.mark.parametrize("widget_cls", [Show3D])
def test_notebook_preview_format_controls_saved_fallback(widget_cls, format_name, mime, pil_format):
    """Show2D/Show3D should expose the saved-notebook preview format directly.

    The compatibility trait name stays ``_static_fallback_jpeg`` for old
    frontends, but the MIME trait tells the frontend how to interpret the bytes.
    """
    lattice = _mos2_like_stack(3, 64, 64)
    if widget_cls is Show2D:
        widget = Show2D(
            lattice[0],
            notebook_preview_format=format_name,
            notebook_preview_quality=85,
            save_state=False,
            verbose=False,
        )
    else:
        widget = Show3D(
            lattice,
            notebook_preview_format=format_name,
            notebook_preview_quality=85,
            save_state=False,
        )

    bundle = widget._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    image_keys = [key for key in IMAGE_MIME_KEYS if key in (data or {})]
    assert image_keys == [mime]
    with Image.open(io.BytesIO(base64.b64decode(data[mime]))) as img:
        assert img.format == pil_format
        assert min(img.size) > 10
        assert np.asarray(img.convert("RGB")).std() > 0

    state = widget.get_state()
    assert state["_static_fallback_mime"] == mime
    assert "_static_fallback_jpeg" in state
    with Image.open(io.BytesIO(base64.b64decode(state["_static_fallback_jpeg"]))) as img:
        assert img.format == pil_format


def test_show4dstem_notebook_preview_format_none_disables_static_preview():
    """A live-only Show4DSTEM notebook can suppress the static sibling."""
    widget = Show4DSTEM(
        np.ones((4, 4, 8, 8), dtype=np.float32),
        notebook_preview_format=None,
        precompute_virtual_images=False,
        save_state=False,
        verbose=False,
    )
    try:
        bundle = widget._repr_mimebundle_()
        data = bundle[0] if isinstance(bundle, tuple) else bundle

        assert not [key for key in IMAGE_MIME_KEYS if key in (data or {})]
        state = widget.get_state()
        assert "_static_fallback_jpeg" not in state
        assert "_static_fallback_mime" not in state
    finally:
        widget.close()


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
    stem_clone = export_clone(stem, "uint16", 1, 1)
    try:
        _assert_export_state_keeps_buffer(stem_clone, "_offline_stack")
    finally:
        stem_clone.close()
        stem.close()


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
    cases.append((show2d, data2d.nbytes, "frame_bytes", tmp_path / "show2d.html", {"encoding": "full"}))

    data3d = rng.random((8, 256, 256), dtype=np.float32)
    show3d = Show3D(data3d, save_state=False, title="size-show3d")
    cases.append((show3d, data3d.nbytes, "_offline_float_stack", tmp_path / "show3d.html", {"encoding": "full"}))

    stem_data = rng.integers(0, 1000, (32, 32, 32, 32), dtype=np.uint16)
    stem = Show4DSTEM(stem_data, save_state=False, verbose=False, title="size-show4dstem")
    cases.append((stem, stem_data.nbytes, "_offline_stack", tmp_path / "show4dstem.html", {"dtype": "uint16"}))

    try:
        for widget, raw_bytes, marker, path, kwargs in cases:
            exported = widget.export_html(path, **kwargs)
            html = exported.read_text(errors="ignore")
            assert exported.stat().st_size > raw_bytes / 2
            assert marker in html
    finally:
        for widget, _, _, _, _ in cases:
            widget.close()


def _decode_png(widget) -> np.ndarray:
    """Decode the widget's static fallback PNG to an (H, W, 3) uint8 array."""
    import base64
    import io

    from PIL import Image

    png_b64 = widget._static_png_b64()
    assert png_b64
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB"))


@pytest.mark.parametrize("cmap", ["gray", "inferno", "viridis"])
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize("auto_contrast", [False, True])
def test_show2d_png_settings_sweep(cmap, log_scale, auto_contrast):
    """Every contrast/colormap combination must render a decodable PNG with
    the gallery + labels layout (3 panels, 3 labels in the specs)."""
    rng = np.random.default_rng(5)
    frames = [rng.random((96, 96)).astype(np.float32) * scale for scale in (1, 10, 100)]
    widget = Show2D(frames, labels=["a", "b", "c"], cmap=cmap, log_scale=log_scale,
                    auto_contrast=auto_contrast, verbose=False)
    specs = widget._static_panel_specs()
    assert [spec["label"] for spec in specs] == ["a", "b", "c"]
    assert all(spec["vmax"] > spec["vmin"] for spec in specs)
    assert all("Mean" in spec["stats"] for spec in specs)
    decoded = _decode_png(widget)
    assert decoded.shape[0] > 100 and decoded.shape[1] > 300  # 3-across gallery


def test_show2d_png_render_perf_two_4k_frames():
    """The PNG path (area-bin first, then normalize) must stay cheap even on
    2x 4096x4096 float32: under 1.5 s per display."""
    import time

    rng = np.random.default_rng(6)
    frames = rng.random((2, 4096, 4096), dtype=np.float32)
    widget = Show2D(frames, verbose=False)
    start = time.perf_counter()
    png_b64 = widget._static_png_b64()
    elapsed = time.perf_counter() - start
    assert png_b64
    assert elapsed < 1.5, f"static PNG took {elapsed:.2f}s"


def test_show2d_display_defers_static_png_render(monkeypatch):
    """Displaying the widget must NOT render the PNG synchronously (matplotlib
    would block every cell that shows a widget); the deferred post_execute
    fill must then update the placeholder sibling with the real image/png."""
    import IPython

    frame = np.random.default_rng(8).random((64, 64)).astype(np.float32)
    widget = Show2D(frame, verbose=False)
    png_calls = []
    original_png = Show2D._static_png_b64
    monkeypatch.setattr(Show2D, "_static_png_b64",
                        lambda self, **kw: (png_calls.append(1), original_png(self, **kw))[1])
    displayed, updated, hooks = [], [], {}

    class FakeHandle:
        def update(self, data, raw=False, metadata=None):
            updated.append((data, metadata))

    class FakeEvents:
        def register(self, name, fn):
            hooks[name] = fn

        def unregister(self, name, fn):
            hooks.pop(name, None)

    class ZMQInteractiveShell:  # name is what the kernel check looks at
        events = FakeEvents()

    shell = ZMQInteractiveShell()

    def fake_display(data, raw=False, metadata=None, display_id=None, **kw):
        displayed.append((data, metadata, display_id))
        return FakeHandle() if display_id else None

    monkeypatch.setattr(IPython, "get_ipython", lambda: shell)
    monkeypatch.setattr("IPython.display.display", fake_display)

    widget._ipython_display_()
    assert not png_calls, "PNG was rendered synchronously at display time"
    # widget bundle + empty placeholder sibling with the hide marker
    assert len(displayed) == 2
    placeholder_data, placeholder_meta, display_id = displayed[1]
    assert display_id is True
    assert "image/jpeg" not in placeholder_data
    assert "quantem-static-fallback" in placeholder_data["text/html"]
    assert placeholder_meta == {"quantem.widget": {"static_fallback": True}}
    # cell finishes -> post_execute hook fills the placeholder with the PNG
    assert "post_execute" in hooks
    hooks["post_execute"]()
    assert png_calls, "deferred fill never rendered the PNG"
    assert "post_execute" not in hooks, "one-shot hook did not unregister"
    fill_data, fill_meta = updated[-1]
    assert isinstance(fill_data["image/jpeg"], bytes) and len(fill_data["image/jpeg"]) > 1000
    assert "quantem-static-fallback" in fill_data["text/html"]
    assert fill_meta == {"quantem.widget": {"static_fallback": True}}


def test_static_fallback_env_kill_switch(monkeypatch):
    """QUANTEM_WIDGET_STATIC_FALLBACK=0 (docs/CI builds) must emit ONLY the
    interactive widget output: no in-bundle preview image and no static
    sibling display, so built docs pages show a single widget, not a
    duplicate image under it."""
    import IPython

    monkeypatch.setenv("QUANTEM_WIDGET_STATIC_FALLBACK", "0")
    frame = np.random.default_rng(9).random((32, 32)).astype(np.float32)
    widget = Show2D(frame, verbose=False)

    bundle = widget._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    assert "image/jpeg" not in data
    assert "image/webp" not in data

    displayed = []

    class ZMQInteractiveShell:  # name is what the kernel check looks at
        pass

    def fake_display(data, raw=False, metadata=None, display_id=None, **kw):
        displayed.append((data, metadata, display_id))

    monkeypatch.setattr(IPython, "get_ipython", lambda: ZMQInteractiveShell())
    monkeypatch.setattr("IPython.display.display", fake_display)
    widget._ipython_display_()
    assert len(displayed) == 1, "static sibling was emitted despite kill switch"


def test_show2d_sibling_static_output_via_nbconvert(tmp_path):
    """Executing a notebook must leave a saved preview on the Show2D cell.

    Depending on the frontend that calls ``display()``, the fallback may be a
    separate sibling output or an image fallback inside the widget mime bundle.
    Both are acceptable as long as the output can render statically and widget
    metadata stays tiny.
    """
    import json
    import subprocess
    import sys

    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(
        "import numpy as np\n"
        "from IPython.display import display\n"
        "from quantem.widget import Show2D\n"
        "display(Show2D(np.random.rand(48, 48).astype('float32'), verbose=False))\n"
    )]
    nb_path = tmp_path / "show2d_sibling.ipynb"
    nbformat.write(nb, nb_path)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", str(nb_path)],
        check=True, capture_output=True, timeout=180)
    executed = nbformat.read(nb_path, as_version=4)
    outputs = executed.cells[0].outputs
    display_outputs = [o for o in outputs if o.output_type == "display_data"]
    assert display_outputs, f"expected display output, got {outputs}"
    widget_outputs = [
        output
        for output in display_outputs
        if "application/vnd.jupyter.widget-view+json" in output.data
    ]
    assert widget_outputs, f"expected widget-view bundle, got {outputs}"
    fallback_outputs = [
        output
        for output in display_outputs
        if "image/jpeg" in output.data or "image/png" in output.data
    ]
    assert fallback_outputs, f"expected static image fallback, got {outputs}"
    fallback = fallback_outputs[-1]
    image_key = "image/jpeg" if "image/jpeg" in fallback.data else "image/png"
    assert len(fallback.data[image_key]) > 1000
    if "text/html" in fallback.data:
        assert "quantem-static-fallback" in fallback.data["text/html"]
        assert fallback.metadata.get("quantem.widget", {}).get("static_fallback") is True
    # metadata.widgets guard. nbclient reconstructs widget state by replaying
    # comm traffic, so two known constants of headless execution appear here no
    # matter what get_state() trims: the anywidget JS bundle (_esm) and the
    # initial 48x48 frame upload (~12 KB, needed for live render; a real Lab
    # save goes through the trimmed get_state(), covered by
    # test_full_snapshot_trims_bulk_buffers). Everything else must stay tiny -
    # this fails if e.g. the static PNG or a duplicate pixel buffer starts
    # leaking into the synced state.
    widget_states = executed.metadata["widgets"]["application/vnd.jupyter.widget-state+json"]["state"]
    anymodel = next(s for s in widget_states.values() if s.get("model_name") == "AnyModel")
    body = dict(anymodel["state"])
    body.pop("_esm", None)
    body.pop("_static_fallback_jpeg", None)
    body.pop("_static_fallback_mime", None)
    buffer_bytes = sum(len(b["data"]) for b in anymodel.get("buffers", []))
    assert buffer_bytes < 20_000
    assert len(json.dumps(body)) < 20_000


def test_show2d_cmd_s_snapshot_keeps_static_preview_without_heavy_pixels():
    """JupyterLab Cmd+S uses the full widget-state snapshot.

    A lightweight Show2D save must not embed ``frame_bytes``/detail/export
    buffers, but it still needs a compact preview in the model state. Otherwise
    a saved notebook can rehydrate a live widget with no pixels and show a
    blank output after reopen.
    """
    widget = Show2D(
        np.random.default_rng(14).random((3, 256, 256), dtype=np.float32),
        display_bin=2,
        save_state=False,
        verbose=False,
    )
    state = widget.get_state()

    assert "_static_fallback_jpeg" in state
    assert state["_static_fallback_mime"] == "image/jpeg"
    assert len(state["_static_fallback_jpeg"]) > 1000
    assert "frame_bytes" not in state
    assert "_detail_bytes" not in state
    assert "export_payload" not in state


def test_show4dstem_cmd_s_snapshot_keeps_two_panel_static_preview():
    """Show4DSTEM saved preview should show virtual image and diffraction."""
    import base64
    import io

    from PIL import Image

    rng = np.random.default_rng(19)
    data = rng.integers(0, 1000, (10, 12, 24, 24), dtype=np.uint16)
    widget = Show4DSTEM(
        data,
        sampling=(0.2, 0.2, 0.8, 0.8),
        units=["nm", "nm", "mrad", "mrad"],
        panel_width_px=128,
        save_state=False,
        verbose=False,
        title="save-state 4dstem",
    )

    state = widget.get_state()
    png_b64 = widget._static_png_b64(max_px=128, dpi=160)
    assert png_b64
    decoded = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")

    assert decoded.width > decoded.height
    assert decoded.width >= (decoded.height - 24) * 2 - 8
    assert "_static_fallback_jpeg" in state
    assert len(state["_static_fallback_jpeg"]) > 1000
    assert "_offline_stack" not in state
    assert "export_payload" not in state
    assert "_gif_data" not in state
    widget.close()


def test_show2d_static_gallery_avoids_pyplot_figure_manager(monkeypatch):
    """A saved gallery must not join Jupyter's inline figure lifecycle.

    The fallback is rendered from a ``post_execute`` callback. Registering
    its figure through pyplot lets matplotlib-inline's neighboring callback
    flush or clear the gallery before ``savefig``, producing a correctly sized
    but all-white JPEG after notebook save/reopen.
    """
    import matplotlib.pyplot as plt

    def fail_pyplot_figure(*args, **kwargs):
        raise AssertionError("saved Show2D previews must use an unmanaged Figure")

    monkeypatch.setattr(plt, "figure", fail_pyplot_figure)
    rng = np.random.default_rng(181)
    widget = Show2D(
        [rng.random((96, 96), dtype=np.float32) for _ in range(4)],
        labels=["Ba", "Ti", "O", "Sr"],
        ncols=2,
        cmap="inferno",
        save_state=False,
        verbose=False,
    )

    png = widget._static_png_b64(max_px=192)
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB"))

    assert image.shape[0] > 100 and image.shape[1] > 100
    assert image.std() > 5


@pytest.mark.parametrize("widget", WIDGETS)
def test_sibling_static_fallback_contract(widget, monkeypatch):
    import IPython

    w, _ = _make(widget, save_state=False)
    png_calls = []
    widget_cls = type(w)
    original_png = widget_cls._static_png_b64
    monkeypatch.setattr(widget_cls, "_static_png_b64",
                        lambda self, **kw: (png_calls.append(1), original_png(self, **kw))[1])
    displayed, updated, hooks = [], [], {}

    class FakeHandle:
        def update(self, data, raw=False, metadata=None):
            updated.append((data, metadata))

    class FakeEvents:
        def register(self, name, fn):
            hooks[name] = fn

        def unregister(self, name, fn):
            hooks.pop(name, None)

    class ZMQInteractiveShell:  # name is what the kernel check looks at
        events = FakeEvents()

    shell = ZMQInteractiveShell()

    def fake_display(data, raw=False, metadata=None, display_id=None, **kw):
        displayed.append((data, metadata, display_id))
        return FakeHandle() if display_id else None

    monkeypatch.setattr(IPython, "get_ipython", lambda: shell)
    monkeypatch.setattr("IPython.display.display", fake_display)

    w._ipython_display_()
    assert not png_calls, f"{widget.__name__}: PNG rendered synchronously at display time"
    # widget bundle + empty placeholder sibling with the hide marker
    assert len(displayed) == 2, f"{widget.__name__}: expected widget + placeholder sibling"
    placeholder_data, placeholder_meta, display_id = displayed[1]
    assert display_id is True
    assert "image/jpeg" not in placeholder_data
    assert "quantem-static-fallback" in placeholder_data["text/html"]
    assert placeholder_meta == {"quantem.widget": {"static_fallback": True}}
    # cell finishes -> post_execute hook fills the placeholder with the PNG
    assert "post_execute" in hooks
    hooks["post_execute"]()
    assert png_calls, f"{widget.__name__}: deferred fill never rendered the PNG"
    assert "post_execute" not in hooks, "one-shot hook did not unregister"
    fill_data, fill_meta = updated[-1]
    preview_mime = "image/png" if widget is Plot2D else "image/jpeg"
    assert isinstance(fill_data[preview_mime], bytes)
    assert len(fill_data[preview_mime]) > 1000
    assert "quantem-static-fallback" in fill_data["text/html"]
    assert fill_meta == {"quantem.widget": {"static_fallback": True}}
    # the saved-notebook snapshot must stay free of the bulk buffers
    full = w.get_state()
    leaked = [k for k in w._UNSAVED_HEAVY_KEYS if k in full]
    assert not leaked, f"{widget.__name__}: heavy keys {leaked} leaked into full get_state"


def test_deferred_static_fallback_failure_does_not_emit_traceback(monkeypatch):
    """A failed saved-notebook preview must not pollute notebook exports."""
    import IPython

    w, _ = _make(Show2D, save_state=False)
    displayed, updated, hooks = [], [], {}

    class FakeHandle:
        def update(self, data, raw=False, metadata=None):
            updated.append((data, metadata))

    class FakeEvents:
        def register(self, name, fn):
            hooks[name] = fn

        def unregister(self, name, fn):
            hooks.pop(name, None)

    class ZMQInteractiveShell:
        events = FakeEvents()

    def fake_display(data, raw=False, metadata=None, display_id=None, **kw):
        displayed.append((data, metadata, display_id))
        return FakeHandle() if display_id else None

    def fail_preview(*args, **kwargs):
        raise RuntimeError("preview renderer failed")

    monkeypatch.setattr(IPython, "get_ipython", lambda: ZMQInteractiveShell())
    monkeypatch.setattr("IPython.display.display", fake_display)
    monkeypatch.setattr(type(w), "_static_png_b64", fail_preview)

    w._ipython_display_()
    assert "post_execute" in hooks
    hooks["post_execute"]()

    assert "post_execute" not in hooks
    assert len(displayed) == 2
    assert updated == []


@pytest.mark.parametrize("widget", WIDGETS)
def test_sibling_not_emitted_with_save_state_true(widget, monkeypatch):
    """save_state=True embeds full interactive state; no sibling placeholder."""
    import IPython

    w, _ = _make(widget, save_state=True)
    displayed = []

    class ZMQInteractiveShell:
        events = None

    monkeypatch.setattr(IPython, "get_ipython", lambda: ZMQInteractiveShell())
    monkeypatch.setattr("IPython.display.display",
                        lambda data, **kw: displayed.append(data))

    w._ipython_display_()
    assert len(displayed) == 1, f"{widget.__name__}: sibling emitted despite save_state=True"


def test_show1d_save_state_true_keeps_buffers():
    from quantem.widget import Show1D

    w = Show1D(np.random.rand(64).astype("float32"), save_state=True)
    full = w.get_state()
    assert "snapshot_bytes" in full
    assert "export_payload" in full
