"""Rewrite widget ``save()`` files of an earlier quantem.widget release into the current layout, once.

quantem.widget reads one state layout (``quantem.widget-state/3``) and converts nothing inside the package. A view
someone arranged by hand (contrast, zoom, ROIs, stars) is worth carrying over, so this script moves
``quantem.widget-state/2`` envelopes to /3:

- the grid of panels is ``columns`` in every widget: Show2D ``ncols``, Show3D ``max_cols`` and Show4DSTEM
  ``compare_cols`` (now ``compare_columns``) are renamed;
- keys of removed features are dropped: folder watching (Show2D ``folder_page_size`` and ``page_kind``) and Show1D's
  trial review
  (``show_review``, ``trial_notes``, ``trial_tags``, ``show_trial_notes``, ``show_starred_only``, ``review_mode``,
  ``trial_sort_key``, ``trial_sort_descending``, ``trial_filter_text``, ``top_trial_count`` and
  ``starred_snapshot_image_labels``, which only the review read).

No number changes.

    python scripts/rewrite_widget_state.py --dry-run path/to/view.json
    python scripts/rewrite_widget_state.py path/to/view.json path/to/other_view.json

``--dry-run`` prints what would change and writes nothing. Without it each file is replaced and the earlier one is
kept beside it as ``<name>.before``. A file already at /3, or not a widget state envelope, is left alone.
"""

import argparse
import json
import pathlib

SCHEMA = "quantem.widget-state/3"
EARLIER = "quantem.widget-state/2"
# widget_name -> {earlier key: current key}
RENAMED = {
    "Show2D": {"ncols": "columns"},
    "Show3D": {"max_cols": "columns"},
    "Show4DSTEM": {"compare_cols": "compare_columns"},
}
# widget_name -> keys of removed features
DROPPED = {
    "Show1D": ("show_review", "trial_notes", "trial_tags", "show_trial_notes", "show_starred_only", "review_mode",
               "trial_sort_key", "trial_sort_descending", "trial_filter_text", "top_trial_count",
               "starred_snapshot_image_labels"),
    "Show2D": ("folder_page_size", "page_kind"),
}


def current(envelope: dict) -> tuple[dict, list[str]]:
    """The /3 envelope of a /2 one, and one line per changed key."""
    widget = envelope.get("widget_name")
    renamed, dropped = RENAMED.get(widget, {}), DROPPED.get(widget, ())
    state, changes = {}, []
    for key, value in envelope["state"].items():
        if key in dropped:
            changes.append(f"drop {key}")
        elif key in renamed:
            state[renamed[key]] = value
            changes.append(f"{key} -> {renamed[key]}")
        else:
            state[key] = value
    return {**envelope, "schema": SCHEMA, "state": state}, changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true", help="print the changes, write nothing")
    args = parser.parse_args(argv)
    for path in args.paths:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict) or envelope.get("schema") != EARLIER or not isinstance(envelope.get("state"), dict):
            print(f"{path}: left alone (schema {envelope.get('schema') if isinstance(envelope, dict) else None!r})")
            continue
        rewritten, changes = current(envelope)
        print(f"{path}: {EARLIER} -> {SCHEMA}; " + ("; ".join(changes) if changes else "no key changes"))
        if args.dry_run:
            continue
        before = path.with_name(path.name + ".before")
        if not before.exists():
            before.write_bytes(path.read_bytes())
        path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
