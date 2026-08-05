"""Internal ShowPtycho collection manifest and catalog helpers."""

from __future__ import annotations

import html
import json
import pathlib

from quantem.widget.command_launcher import write_command_launcher


SHOWPTYCHO_COLLECTION_FORMAT = "quantem.showptycho.collection.v1"


def showptycho_collection_folder(path: pathlib.Path) -> pathlib.Path:
    """Return a validated top-level ShowPtycho collection folder."""

    folder = path.parent if path.is_file() and path.name == "index.html" else path
    manifest = folder / "manifest.json"
    index = folder / "index.html"
    if not folder.is_dir() or not manifest.is_file() or not index.is_file():
        raise ValueError(f"not a ShowPtycho collection: {path}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ShowPtycho collection manifest is invalid: {manifest}"
        ) from exc
    if payload.get("format") != SHOWPTYCHO_COLLECTION_FORMAT:
        raise ValueError(f"not a ShowPtycho collection: {path}")
    return folder


def is_showptycho_collection(path: pathlib.Path) -> bool:
    """Return whether ``path`` is a top-level ShowPtycho collection."""

    try:
        showptycho_collection_folder(path)
    except (OSError, ValueError):
        return False
    return True


def write_showptycho_collection(
    folder: pathlib.Path,
    datasets: list[dict[str, object]],
    *,
    title: str,
) -> pathlib.Path:
    """Write the portable catalog that opens every dataset-specific viewer."""

    payload = {
        "schema_version": 1,
        "format": SHOWPTYCHO_COLLECTION_FORMAT,
        "title": title,
        "datasets": datasets,
    }
    (folder / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    cards = []
    for dataset in datasets:
        label = html.escape(str(dataset["label"]))
        viewer = html.escape(str(dataset["viewer"]), quote=True)
        raw_viewer = html.escape(str(dataset["show4dstem"]), quote=True)
        details = []
        if dataset.get("backend"):
            details.append(str(dataset["backend"]).upper())
        if dataset.get("num_bf") is not None:
            details.append(f"{int(dataset['num_bf']):,} bright-field pixels")
        if dataset.get("loss") is not None:
            details.append(f"loss {float(dataset['loss']):.6g}")
        summary = html.escape(" · ".join(details) or "Ready to open")
        cards.append(
            f'<article><h2>{label}</h2><p>{summary}</p>'
            f'<nav><a href="{viewer}">Open ShowPtycho</a>'
            f'<a class="secondary" href="{raw_viewer}">Open Show4DSTEM</a>'
            f'</nav></article>'
        )
    plural = "s" if len(datasets) != 1 else ""
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 72rem; padding: 2rem; }}
header {{ margin-bottom: 2rem; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(18rem,1fr)); gap: 1rem; }}
article {{ border: 1px solid color-mix(in srgb,currentColor 22%,transparent); border-radius: .8rem; padding: 1.2rem; }}
h1,h2 {{ margin-top: 0; }} p {{ opacity: .72; }}
nav {{ display: flex; flex-wrap: wrap; gap: .55rem; }}
a {{ display: inline-block; padding: .55rem .8rem; border-radius: .45rem; background: #5b5bd6; color: white; text-decoration: none; }}
a.secondary {{ background: transparent; color: inherit; border: 1px solid currentColor; }}
</style></head><body><header><h1>{html.escape(title)}</h1>
<p>{len(datasets)} dataset{plural}. Open either browser viewer directly. ShowPtycho owns its exact bright-field cache; Show4DSTEM reads the original compressed detector data through read-only links.</p>
</header><main>{''.join(cards)}</main></body></html>"""
    (folder / "index.html").write_text(document, encoding="utf-8")
    write_command_launcher(folder, "ShowPtycho")
    return folder
