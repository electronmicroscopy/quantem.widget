"""Static-PNG fallback machinery shared by the display widgets.

JupyterLab never rehydrates a widget from saved state on a cold reopen: the
widget-view output renders "Error displaying widget: model not found" and the
cell goes black. Each widget therefore publishes a SEPARATE display_data
sibling carrying a static PNG render (its ``_static_png_b64``); the live
frontend hides the sibling while the interactive view is mounted, and a cold
reopen (no widget JS) shows the PNG instead of nothing. ``save_state=True``
embeds the full interactive state, so no sibling is emitted.

Mixed into Show2D / Show3D / Show4DSTEM / ShowEDS ahead of AnyWidget. Each
class supplies ``_static_png_b64()`` (its own widget-format render) and
``_save_state``; everything here is render-agnostic plumbing.
"""

import base64


class StaticFallbackMixin:
    """Publish-and-hide plumbing for the per-widget static PNG fallback."""

    def _repr_mimebundle_(self, **kwargs):
        """Display bundle: interactive widget live, static PNG for cold reopen.

        When ``save_state`` is False we add an ``image/png`` fallback to the
        widget-view bundle. Live Jupyter renders the interactive widget (richest
        mime); a kernel-less reopen falls back to the PNG. When ``save_state`` is
        True the full state is embedded, so no static fallback is needed.
        """
        png = None
        if not getattr(self, "_save_state", False):
            # Build the preview before delegating to anywidget. In the common
            # "last expression" path, Jupyter captures widget state while the
            # widget-view bundle is being produced; storing after super() is too
            # late and a saved lightweight model reopens blank.
            png = self._static_png_b64()
            if png:
                store = getattr(self, "_store_static_fallback_preview", None)
                if callable(store):
                    store(png)
        bundle = super()._repr_mimebundle_(**kwargs)
        if getattr(self, "_save_state", False) or bundle is None:
            return bundle
        if png:
            data = bundle[0] if isinstance(bundle, tuple) else bundle
            data["image/png"] = png
        return bundle

    def _ipython_display_(self):
        """Publish the widget view plus a SEPARATE static-PNG sibling output.

        JupyterLab's widget renderer outranks ``image/png`` inside a single
        bundle: on a cold reopen with ``save_state=False`` it shows "Error
        displaying widget: model not found" and never falls back to the PNG
        living in the same output. Publishing the render as its own
        display_data output survives, because Lab only swallows the
        widget-view output. ``save_state=True`` embeds full interactive
        state, so no sibling is emitted. The sibling carries this widget's
        model id so the live frontend can hide it while the interactive view
        is mounted.

        The sibling starts as an EMPTY placeholder and is filled with the
        PNG only after the cell finishes: rendering at display time would
        block the cell output on matplotlib for every widget shown, and the
        PNG is only ever consumed by a notebook save that happens later.
        """
        from IPython.display import display
        # super() bundle only: the in-bundle PNG _repr_mimebundle_ adds for
        # direct consumers (nbsphinx, repr displays) would render here
        # synchronously; the sibling fill below covers the notebook-save path
        bundle = super()._repr_mimebundle_()
        if bundle is None:
            return
        data, metadata = bundle if isinstance(bundle, tuple) else (bundle, {})
        display(data, raw=True, metadata=metadata or None)
        if not getattr(self, "_save_state", False):
            self._display_static_sibling_deferred()

    def _static_fallback_marker(self, image_b64: str | None = None,
                                mime: str = "image/jpeg") -> str:
        """The sibling's HTML: an ``img.quantem-static-fallback`` tagged with
        this widget's model id, which is exactly what the live frontend's
        hide effect queries. The empty placeholder keeps the tag (hidden) so
        the hide keeps working after ``update_display_data`` swaps in the
        render."""
        note = f"{type(self).__name__} static render (for saved-notebook viewing)"
        src = f' src="data:{mime};base64,{image_b64}"' if image_b64 else ' style="display:none"'
        return (f'<img class="quantem-static-fallback" '
                f'data-quantem-model-id="{self.model_id}"{src} alt="{note}">')

    @staticmethod
    def _png_to_jpeg_b64(png_b64: str, quality: int = 88) -> str:
        """Re-encode the fallback render as JPEG for the saved notebook.

        The sibling stores the image twice (``image/jpeg`` for GitHub/nbviewer
        plus the html data-URL JupyterLab actually renders), and noisy STEM
        content compresses poorly as PNG (~2.7 bytes/px): a survey notebook
        hit 20 MB of PNG siblings. JPEG q88 is ~10x smaller and visually
        equivalent for a reopen preview; the render is photographic, not
        line art.
        """
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _display_static_sibling_deferred(self):
        """Reserve the sibling output now, render the PNG after the cell ends.

        The placeholder is published with a ``display_id``; a one-shot
        ``post_execute`` hook renders the PNG once the current cell finishes
        and rewrites the placeholder via ``update_display_data``. The widget
        therefore appears instantly, the notebook file saved any time after
        the cell carries the render, and the live view never shows a
        duplicate (the placeholder is empty, then hidden by the frontend).
        ``post_execute`` fires inside the same execute request, so headless
        runners (nbconvert/nbclient) capture the update deterministically.
        Terminal IPython has no notebook file to save, so outside a kernel
        (no ZMQInteractiveShell) no sibling is emitted at all.
        """
        from IPython import get_ipython
        from IPython.display import display
        shell = get_ipython()
        if shell is None or type(shell).__name__ != "ZMQInteractiveShell":
            return
        handle = display(
            {"text/html": self._static_fallback_marker(), "text/plain": ""},
            raw=True, display_id=True,
            metadata={"quantem.widget": {"static_fallback": True}},
        )

        def fill():
            png_b64 = self._static_png_b64()
            if not png_b64 or handle is None:
                return
            store = getattr(self, "_store_static_fallback_preview", None)
            if callable(store):
                store(png_b64)
            jpeg_b64 = self._png_to_jpeg_b64(png_b64)
            note = f"{type(self).__name__} static render (for saved-notebook viewing)"
            handle.update(
                {
                    "image/jpeg": base64.b64decode(jpeg_b64),
                    "text/html": self._static_fallback_marker(jpeg_b64),
                    "text/plain": note,
                },
                raw=True,
                metadata={"quantem.widget": {"static_fallback": True}},
            )

        def fill_once():
            try:
                shell.events.unregister("post_execute", fill_once)
            except ValueError:
                pass
            fill()

        shell.events.register("post_execute", fill_once)
        self._static_fallback_fill = fill  # test hook: flush the deferred render
