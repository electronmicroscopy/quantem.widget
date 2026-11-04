"""The diffraction simulator in a Dash app.

:func:`register` serves ``diffraction-sim.js`` from the app's Flask server
and adds one clientside callback, matched on the component ids, that draws a
simulator whenever its settings change. :func:`layout` returns the
simulator's place in a layout: a ``dcc.Store`` with the settings and the
``html.Div`` the simulator draws into. A server callback that writes
:func:`quantem.diffsim.settings` to ``settings_id(name)`` redraws it, for
example when the page shows another structure. No calculation runs on the
server.
"""

from __future__ import annotations

import json
from typing import Any

from dash import MATCH, Input, Output, dcc, html

from quantem.diffsim import __version__, bundle_path, settings

SETTINGS_TYPE = "quantem-diffsim-settings"
MOUNT_TYPE = "quantem-diffsim-mount"
BUNDLE_ROUTE = f"_quantem_diffsim/diffraction-sim-{__version__}.js"

# Draws the simulator for one settings store. Dash gives a pattern-matching component the DOM id
# of its id dict with sorted keys; the module import is cached by the browser after the first draw.
_MOUNT_JS = """
function (settings) {
  const dc = window.dash_clientside;
  if (!settings) return dc.no_update;
  const store = dc.callback_context.inputs_list[0].id;
  const id = Object.assign({}, store, { type: %(mount_type)s });
  const key = JSON.stringify(id, Object.keys(id).sort());
  const draw = (attempt) => {
    const el = document.getElementById(key);
    if (!el) {
      if (attempt < 120) requestAnimationFrame(() => draw(attempt + 1));
      return;
    }
    import(%(url)s).then((module) => {
      if (el.quantemDiffsimCleanup) el.quantemDiffsimCleanup();
      el.replaceChildren();
      el.quantemDiffsimCleanup = module.default.render({ model: { get: (name) => settings[name] }, el });
    });
  };
  draw(0);
  return dc.no_update;
}
"""


def settings_id(name: str) -> dict[str, str]:
    """Id of the ``dcc.Store`` that holds the settings of simulator ``name``, for callback outputs."""
    return {"type": SETTINGS_TYPE, "index": name}


def layout(name: str, structure: Any = None, **values: Any) -> html.Div:
    """The simulator's place in a Dash layout.

    Parameters
    ----------
    name : str
        Unique name of this simulator in the app.
    structure : pymatgen Structure, ase.Atoms, str or pathlib.Path, optional
        Initial crystal, as for :func:`quantem.diffsim.settings`. Leave it out
        when a callback provides the settings.
    **values
        Initial settings, as for :func:`quantem.diffsim.settings`.

    Returns
    -------
    dash.html.Div
        A ``dcc.Store`` (id ``settings_id(name)``) and the ``html.Div`` the
        simulator draws into.
    """
    data = settings(structure, **values) if structure is not None or values else None
    return html.Div(
        [dcc.Store(id=settings_id(name), data=data), html.Div(id={"type": MOUNT_TYPE, "index": name})],
    )


def register(app) -> str:
    """Serve the bundle and add the drawing callback; once per app, before it runs.

    Repeated calls (one per component that uses the simulator) do nothing.

    Parameters
    ----------
    app : dash.Dash
        The app.

    Returns
    -------
    str
        URL of the bundle, relative to the page and versioned so browsers can
        cache it.
    """
    url = getattr(app, "_quantem_diffsim_url", None)
    if url is not None:
        return url
    from flask import send_file

    bundle = bundle_path()
    app.server.add_url_rule(
        app.config.routes_pathname_prefix + BUNDLE_ROUTE,
        endpoint="quantem_diffsim_bundle",
        view_func=lambda: send_file(bundle, mimetype="text/javascript", max_age=31536000),
    )
    url = app.get_relative_path("/" + BUNDLE_ROUTE)
    app.clientside_callback(
        _MOUNT_JS % {"mount_type": json.dumps(MOUNT_TYPE), "url": json.dumps(url)},
        Output({"type": MOUNT_TYPE, "index": MATCH}, "className"),
        Input({"type": SETTINGS_TYPE, "index": MATCH}, "data"),
    )
    app._quantem_diffsim_url = url
    return url


__all__ = ["BUNDLE_ROUTE", "MOUNT_TYPE", "SETTINGS_TYPE", "layout", "register", "settings_id"]
