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
    "supports_html_export",
]
