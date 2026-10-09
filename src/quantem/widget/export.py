"""Standalone HTML export shared by every widget with an export button.

``HtmlExportMixin`` owns the toolbar handshake (the ``export_*`` traits and
the request observer), the default file name, the in-memory bytes path and the
standalone write. A widget supplies ``_clone_for_html_export`` (an export-only
copy of itself with the requested packing) and, when it has options, the
``_html_export_options`` parser plus the matching ``export_html`` signature.
``write_widget_html`` embeds any widget as a one-file page and applies the
shell fixes below so the page opens on phones and on a cold CDN cache.
"""

import json
import pathlib
import tempfile

import traitlets

_MOBILE_VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'
_ANYWIDGET_REQUIREJS_CONFIG = """<script id="quantem-widget-anywidget-requirejs">
if (window.require && window.require.config) {
  window.require.config({
    // waitSeconds: requirejs defaults to 7s per module. A first-time viewer on a
    // cold cache fetches several MB from the CDN; when the anywidget module
    // misses that window, html-manager falls back to a SECOND copy of the
    // module, the model's binding registers in one copy and the view looks it
    // up in the other, and the widget dies with "WidgetBinding not found".
    // 0 disables the timeout entirely - slow is recoverable, split-brain is not.
    waitSeconds: 0,
    paths: {
      anywidget: "https://cdn.jsdelivr.net/npm/anywidget@~0.11.*/dist/index"
    }
  });
}
</script>"""
_STANDALONE_EXPORT_STYLE = """<style id="quantem-widget-export-layout">
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
}
body {
  box-sizing: border-box;
}
*, *::before, *::after {
  box-sizing: inherit;
}
</style>"""


def ensure_mobile_viewport(path: str | pathlib.Path) -> pathlib.Path:
    """Add standalone HTML shell tags and widget-manager module paths if needed."""

    html_path = pathlib.Path(path)
    html = html_path.read_text(encoding="utf-8")
    changed = False
    needs_anywidget = any(
        f'"{key}": "anywidget"' in html for key in ("model_module", "_model_module", "view_module", "_view_module")
    )
    if needs_anywidget and 'id="quantem-widget-anywidget-requirejs"' not in html:
        marker = '<script src="https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager'
        marker_index = html.find(marker)
        if marker_index >= 0:
            html = f"{html[:marker_index]}{_ANYWIDGET_REQUIREJS_CONFIG}\n{html[marker_index:]}"
        elif "</head>" in html:
            html = html.replace("</head>", f"    {_ANYWIDGET_REQUIREJS_CONFIG}\n</head>", 1)
        else:
            html = f"{_ANYWIDGET_REQUIREJS_CONFIG}\n{html}"
        changed = True
    needs_viewport = 'name="viewport"' not in html and "name='viewport'" not in html
    needs_layout = 'id="quantem-widget-export-layout"' not in html
    if "<head>" in html:
        if needs_viewport:
            html = html.replace("<head>", f"<head>\n    {_MOBILE_VIEWPORT_META}", 1)
        if needs_layout:
            html = html.replace("</head>", f"    {_STANDALONE_EXPORT_STYLE}\n</head>", 1)
    else:
        viewport = f"{_MOBILE_VIEWPORT_META}\n" if needs_viewport else ""
        layout = f"{_STANDALONE_EXPORT_STYLE}\n" if needs_layout else ""
        html = f"{viewport}{layout}{html}"
    if changed or needs_viewport or needs_layout:
        html_path.write_text(html, encoding="utf-8")
    return html_path


def export_slug(title: str, fallback: str) -> str:
    """File-name stem from a widget title: lowercase, one underscore between words."""
    label = (title or "").strip() or fallback
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or fallback


def write_widget_html(path: str | pathlib.Path, widget, title: str) -> pathlib.Path:
    """Embed one widget with its full state as a standalone HTML page."""
    from ipywidgets.embed import dependency_state, embed_minimal_html

    export_path = pathlib.Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    embed_minimal_html(
        str(export_path),
        views=[widget],
        title=title,
        drop_defaults=False,
        state=dependency_state([widget], drop_defaults=False),
    )
    ensure_mobile_viewport(export_path)
    return export_path


class HtmlExportMixin(traitlets.HasTraits):
    """Toolbar export handshake and standalone HTML writing.

    The frontend writes a JSON request into ``export_request``; Python answers
    through ``export_status`` and, for browser downloads, ``export_payload``
    with the matching ``export_payload_id`` / ``export_filename``. Widgets
    register ``_on_export_request_change`` on ``export_request`` in their
    ``__init__`` so a subclass override still receives the request.
    """

    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)

    def _on_export_request_change(self, change: dict) -> None:
        """React to a toolbar request on ``export_request``: clear the download payload or serve the export."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", ""))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            self._export_request(payload, mode)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, NotImplementedError, MemoryError) as exc:
            # the toolbar shows the failure; raising would only reach the kernel log
            self.export_status = f"Export failed: {exc}"

    def _export_request(self, payload: dict, mode: str) -> None:
        """Serve one toolbar request: a browser download or a file in the kernel cwd."""
        options = self._html_export_options(payload, mode)
        if payload.get("download"):
            filename = str(payload.get("filename") or self._default_html_export_path(**options).name)
            request_id = str(payload.get("id") or "")
            self.export_status = f"Preparing {filename}..."
            html = self._html_export_bytes(**options)
            self.export_filename = filename
            self.export_payload = html
            self.export_payload_id = request_id
            size_mb = len(html) / (1024 * 1024)
            self.export_status = f"Ready {filename} ({size_mb:.1f} MB, {self._export_mode_label(**options)})"
        else:
            self.export_status = f"Exporting {mode} HTML..."
            self.export_html(**options)

    def _html_export_options(self, payload: dict, mode: str) -> dict:
        """Keyword options for ``export_html`` parsed from the toolbar request; none by default."""
        return {}

    def _export_mode_label(self, quantized: bool = False, downsample: int = 1) -> str:
        """Packing named in the toolbar status, so the user sees whether the page holds float32 or uint8 pixels."""
        if not quantized:
            return "full float32"
        if int(downsample) > 1:
            return f"uint8, {int(downsample)}x downsample"
        return "uint8"

    def _html_export_bytes(self, **options) -> bytes:
        """The standalone page as bytes, for the browser download path."""
        with tempfile.TemporaryDirectory(prefix=f"{type(self).__name__.lower()}-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path(**options).name
            self._write_html_export(path, **options)
            return path.read_bytes()

    def _write_html_export(self, path: str | pathlib.Path, *, title: str | None = None, **options) -> pathlib.Path:
        """Write the standalone page from an export clone, then release the clone."""
        export_widget = self._clone_for_html_export(**options)
        try:
            return write_widget_html(path, export_widget, title or self.title or type(self).__name__)
        finally:
            self._release_export_clone(export_widget)
            # Widget.close leaves the child Layout model open, one more per export
            export_widget.layout.close()

    def _release_export_clone(self, clone) -> None:
        """Close the export clone so its comm and pixel buffers do not outlive the export; widgets override it."""
        clone.close()
