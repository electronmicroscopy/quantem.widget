"""Static-preview fallback machinery shared by the display widgets.

JupyterLab never rehydrates a widget from saved state on a cold reopen: the
widget-view output renders "Error displaying widget: model not found" and the
cell goes black. Each widget therefore publishes a SEPARATE display_data
sibling carrying an encoded static render (generated from its
``_static_png_b64``); the live frontend hides the sibling while the interactive
view is mounted, and a cold reopen (no widget JS) shows the image instead of
nothing. ``save_state=True`` embeds the full interactive state, so no sibling
is emitted.

Mixed into Show2D / Show3D / Show4DSTEM / ShowEDS ahead of AnyWidget. Each
class supplies ``_static_png_b64()`` (its own widget-format render) and
``_save_state``; everything here is render-agnostic plumbing.
"""

import base64
import os


class StaticFallbackMixin:
    """Publish-and-hide plumbing for the per-widget static fallback."""

    _NOTEBOOK_PREVIEW_FORMATS = {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "webp": "image/webp",
        "png": "image/png",
    }

    def _configure_static_fallback(
        self,
        *,
        notebook_preview_format: str | None = "jpeg",
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
    ) -> None:
        """Store Python-only settings for the saved-notebook preview.

        The preview is a human-facing fallback, not analysis data.  ``jpeg`` is
        the default because it is widely supported by notebook frontends and is
        compact for noisy microscopy images.  ``webp`` is smaller but less
        universal; ``png`` is lossless but can make notebooks much larger.
        """
        fmt = self._normalize_notebook_preview_format(notebook_preview_format)
        quality = int(notebook_preview_quality)
        if not 1 <= quality <= 100:
            raise ValueError(
                "notebook_preview_quality must be between 1 and 100; "
                f"got {notebook_preview_quality!r}"
            )
        max_px = int(notebook_preview_max_px)
        if max_px <= 0:
            raise ValueError(
                "notebook_preview_max_px must be positive; "
                f"got {notebook_preview_max_px!r}"
            )
        self._notebook_preview_format = fmt
        self._notebook_preview_quality = quality
        self._notebook_preview_max_px = max_px
        self._notebook_preview_mime = (
            "" if fmt is None else self._NOTEBOOK_PREVIEW_FORMATS[fmt]
        )

    @classmethod
    def _normalize_notebook_preview_format(cls, value: str | None) -> str | None:
        if value is None:
            return None
        fmt = str(value).strip().lower()
        if fmt in {"", "none", "off", "false"}:
            return None
        if fmt == "jpg":
            fmt = "jpeg"
        if fmt not in cls._NOTEBOOK_PREVIEW_FORMATS:
            raise ValueError(
                "notebook_preview_format must be 'jpeg', 'webp', 'png', or None; "
                f"got {value!r}"
            )
        return fmt

    def _static_fallback_enabled(self) -> bool:
        # Docs/CI builds bake full interactive widget state into the published
        # HTML, so the saved-notebook preview would only render as a duplicate
        # image under the live widget. QUANTEM_WIDGET_STATIC_FALLBACK=0 lets
        # those builds emit the interactive output alone.
        env = os.environ.get("QUANTEM_WIDGET_STATIC_FALLBACK", "").strip().lower()
        if env in {"0", "off", "false", "none"}:
            return False
        return getattr(self, "_notebook_preview_format", "jpeg") is not None

    def _static_fallback_mime_type(self) -> str:
        fmt = getattr(self, "_notebook_preview_format", "jpeg")
        if fmt is None:
            return ""
        return self._NOTEBOOK_PREVIEW_FORMATS[fmt]

    def _static_fallback_png_b64(self) -> str | None:
        max_px = int(getattr(self, "_notebook_preview_max_px", 512))
        try:
            return self._static_png_b64(max_px=max_px)
        except TypeError:
            return self._static_png_b64()

    def _repr_mimebundle_(self, **kwargs):
        """Display bundle: interactive widget live, static image for cold reopen.

        When ``save_state`` is False we add an image fallback to the
        widget-view bundle. Live Jupyter renders the interactive widget (richest
        mime); a kernel-less reopen falls back to the static image. When
        ``save_state`` is True the full state is embedded, so no static fallback
        is needed.
        """
        encoded = None
        if not getattr(self, "_save_state", False) and self._static_fallback_enabled():
            # Build the preview before delegating to anywidget. In the common
            # "last expression" path, Jupyter captures widget state while the
            # widget-view bundle is being produced; storing after super() is too
            # late and a saved lightweight model reopens blank.
            png = self._static_fallback_png_b64()
            if png:
                store = getattr(self, "_store_static_fallback_preview", None)
                if callable(store):
                    store(png)
                encoded = self._encode_static_fallback_b64(png)
        bundle = super()._repr_mimebundle_(**kwargs)
        if getattr(self, "_save_state", False) or bundle is None:
            return bundle
        if encoded:
            mime, image_b64 = encoded
            data = bundle[0] if isinstance(bundle, tuple) else bundle
            data[mime] = image_b64
        return bundle

    def _ipython_display_(self):
        """Publish the widget view plus a SEPARATE static-image sibling output.

        JupyterLab's widget renderer outranks image MIME data inside a single
        bundle: on a cold reopen with ``save_state=False`` it shows "Error
        displaying widget: model not found" and never falls back to the image
        living in the same output. Publishing the render as its own
        display_data output survives, because Lab only swallows the
        widget-view output. ``save_state=True`` embeds full interactive
        state, so no sibling is emitted. The sibling carries this widget's
        model id so the live frontend can hide it while the interactive view
        is mounted.

        The sibling starts as an EMPTY placeholder and is filled with the image
        only after the cell finishes: rendering at display time would
        block the cell output on matplotlib for every widget shown, and the
        image is only ever consumed by a notebook save that happens later.
        """
        from IPython.display import display
        # super() bundle only: the in-bundle image _repr_mimebundle_ adds for
        # direct consumers (nbsphinx, repr displays) would render here
        # synchronously; the sibling fill below covers the notebook-save path
        bundle = super()._repr_mimebundle_()
        if bundle is None:
            return
        data, metadata = bundle if isinstance(bundle, tuple) else (bundle, {})
        display(data, raw=True, metadata=metadata or None)
        if not getattr(self, "_save_state", False) and self._static_fallback_enabled():
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

    def _encode_static_fallback_b64(self, png_b64: str) -> tuple[str, str] | None:
        """Encode the PNG render into the configured notebook preview format."""
        fmt = getattr(self, "_notebook_preview_format", "jpeg")
        if fmt is None:
            return None
        mime = self._NOTEBOOK_PREVIEW_FORMATS[fmt]
        if fmt == "png":
            return mime, png_b64
        if fmt in {"jpeg", "webp"}:
            return mime, self._png_to_format_b64(
                png_b64,
                fmt=fmt,
                quality=int(getattr(self, "_notebook_preview_quality", 88)),
            )
        raise AssertionError(f"unhandled notebook preview format: {fmt!r}")

    @staticmethod
    def _png_to_format_b64(png_b64: str, *, fmt: str, quality: int) -> str:
        """Re-encode the fallback render for the saved notebook.

        The sibling stores the image both as an image MIME bundle and as the
        html data-URL JupyterLab actually renders. Noisy STEM content
        compresses poorly as PNG, while JPEG/WebP previews are visually
        equivalent for reopen review; the render is photographic, not line art.
        """
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
        buf = io.BytesIO()
        if fmt == "jpeg":
            img.save(buf, format="JPEG", quality=quality, optimize=True)
        elif fmt == "webp":
            try:
                img.save(buf, format="WEBP", quality=quality, method=4)
            except Exception as exc:
                raise RuntimeError(
                    "Pillow could not encode WebP notebook preview. Use "
                    "notebook_preview_format='jpeg' or install a Pillow build "
                    "with WebP support."
                ) from exc
        else:
            raise ValueError(f"Unsupported fallback format: {fmt!r}")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _png_to_jpeg_b64(png_b64: str, quality: int = 88) -> str:
        """Backward-compatible helper for existing widget fallback code."""
        return StaticFallbackMixin._png_to_format_b64(
            png_b64,
            fmt="jpeg",
            quality=quality,
        )

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
            png_b64 = self._static_fallback_png_b64()
            if not png_b64 or handle is None:
                return
            store = getattr(self, "_store_static_fallback_preview", None)
            if callable(store):
                store(png_b64)
            encoded = self._encode_static_fallback_b64(png_b64)
            if not encoded:
                return
            mime, image_b64 = encoded
            note = f"{type(self).__name__} static render (for saved-notebook viewing)"
            handle.update(
                {
                    mime: base64.b64decode(image_b64),
                    "text/html": self._static_fallback_marker(image_b64, mime=mime),
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
            try:
                fill()
            except Exception:
                # The sibling preview is a saved-notebook convenience. The live
                # widget has already displayed, so a renderer edge case should
                # not pollute collaborator exports with callback tracebacks.
                return

        shell.events.register("post_execute", fill_once)
        self._static_fallback_fill = fill  # test hook: flush the deferred render
