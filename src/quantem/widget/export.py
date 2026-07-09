"""Structural protocol for standalone HTML exports.

The viewer widgets do not inherit from a shared export base class.  Keep it that
way: each widget owns its data packing and export modes.  This module is the
small, discoverable contract for tools, docs, and future widgets.
"""

from __future__ import annotations

import pathlib
from typing import Any, Protocol, TypeGuard, runtime_checkable

HtmlExportPath = str | pathlib.Path | None

HTML_EXPORT_TRAITS = (
    "export_request",
    "export_status",
    "export_enabled",
    "export_payload",
    "export_payload_id",
    "export_filename",
)

_MOBILE_VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'
_ANYWIDGET_REQUIREJS_CONFIG = """<script id="quantem-widget-anywidget-requirejs">
if (window.require && window.require.config) {
  window.require.config({
    paths: {
      anywidget: "https://cdn.jsdelivr.net/npm/anywidget@0.11.0/dist/index.min"
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
    needs_anywidget = (
        '"model_module": "anywidget"' in html
        or '"_model_module": "anywidget"' in html
        or '"view_module": "anywidget"' in html
        or '"_view_module": "anywidget"' in html
    )
    if needs_anywidget and 'id="quantem-widget-anywidget-requirejs"' not in html:
        marker = '<script src="https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager'
        marker_idx = html.find(marker)
        if marker_idx >= 0:
            html = f"{html[:marker_idx]}{_ANYWIDGET_REQUIREJS_CONFIG}\n{html[marker_idx:]}"
        elif "</head>" in html:
            html = html.replace("</head>", f"    {_ANYWIDGET_REQUIREJS_CONFIG}\n</head>", 1)
        else:
            html = f"{_ANYWIDGET_REQUIREJS_CONFIG}\n{html}"
        changed = True
    if "<head>" in html:
        if 'name="viewport"' not in html and "name='viewport'" not in html:
            html = html.replace("<head>", f"<head>\n    {_MOBILE_VIEWPORT_META}", 1)
            changed = True
        if 'id="quantem-widget-export-layout"' not in html:
            html = html.replace("</head>", f"    {_STANDALONE_EXPORT_STYLE}\n</head>", 1)
            changed = True
    else:
        prefix = ""
        if 'name="viewport"' not in html and "name='viewport'" not in html:
            prefix += f"{_MOBILE_VIEWPORT_META}\n"
        if 'id="quantem-widget-export-layout"' not in html:
            prefix += f"{_STANDALONE_EXPORT_STYLE}\n"
        if prefix:
            html = f"{prefix}{html}"
            changed = True
    if changed:
        html_path.write_text(html, encoding="utf-8")
    return html_path


@runtime_checkable
class SupportsHtmlExport(Protocol):
    """Widget object with a Python API for standalone HTML export.

    Implementations should write an HTML file that can hydrate the widget with
    the ipywidgets HTML manager and run without a live Python kernel.  The
    preferred public options are ``mode`` for file layout, ``encoding`` for data
    storage, and ``downsample`` for shape reduction. Existing widget-specific
    names such as ``quantized``, ``dtype``, ``det_bin``, and ``binning`` remain
    compatibility aliases.
    """

    def export_html(
        self,
        path: HtmlExportPath = None,
        *,
        title: str | None = None,
        **options: Any,
    ) -> pathlib.Path:
        """Write a standalone HTML artifact and return the written path."""


@runtime_checkable
class SupportsFrontendHtmlExport(SupportsHtmlExport, Protocol):
    """Widget object with the standard in-widget HTML export bridge."""

    export_request: str
    export_status: str
    export_enabled: bool
    export_payload: bytes
    export_payload_id: str
    export_filename: str


def supports_html_export(obj: object) -> TypeGuard[SupportsHtmlExport]:
    """Return whether ``obj`` exposes the structural HTML export API."""

    return callable(getattr(obj, "export_html", None))


__all__ = [
    "HTML_EXPORT_TRAITS",
    "HtmlExportPath",
    "SupportsFrontendHtmlExport",
    "SupportsHtmlExport",
    "ensure_mobile_viewport",
    "supports_html_export",
]
