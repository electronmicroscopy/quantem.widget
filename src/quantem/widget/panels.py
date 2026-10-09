"""Multi-panel reference, order and visibility helpers shared by the galleries.

Panels are addressed by zero-based index or by their exact title. The plain
functions hold the resolution and ordering rules once; ``PanelsMixin`` wires
them to the standard ``hidden_panels`` / ``panel_order`` / ``starred`` traits
of Show2D and Show3D, and Show4DSTEM's compare grid calls the functions with
its ``compare_*`` traits.
"""

from collections.abc import Callable, Sequence

import traitlets


def resolve_panel_ref(panel: int | str, count: int, title_for_index: Callable[[int], str]) -> int:
    """Resolve a panel index or exact title into a zero-based panel index.

    Titles must match exactly and uniquely; an ambiguous title is an error
    rather than a silent first match, because a report that hides or stars
    the wrong panel is worse than one that fails loudly.
    """
    if isinstance(panel, bool):
        raise TypeError("panel must be an integer index or exact label, not bool")
    if isinstance(panel, int):
        index = int(panel)
        if 0 <= index < count:
            return index
        raise ValueError(f"panel index {index} out of range [0, {count})")
    if isinstance(panel, str):
        titles = [title_for_index(i) for i in range(count)]
        matches = [i for i, title in enumerate(titles) if title == panel]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"panel label {panel!r} is not unique; use a zero-based panel index instead"
            )
        available = ", ".join(repr(title) for title in titles)
        raise ValueError(f"unknown panel label {panel!r}; available labels: {available}")
    raise TypeError(
        f"panel must be an integer index or exact label, got {type(panel).__name__}"
    )


def normalize_panel_refs(
    panels: Sequence[int | str] | int | str,
    count: int,
    title_for_index: Callable[[int], str],
    *,
    allow_empty: bool = False,
) -> list[int]:
    """Resolve and de-duplicate panel references, keeping the caller's order."""
    if isinstance(panels, (str, int)) and not isinstance(panels, bool):
        values: Sequence[int | str] = [panels]
    else:
        values = panels
    resolved = list(dict.fromkeys(resolve_panel_ref(panel, count, title_for_index) for panel in values))
    if not resolved and not allow_empty:
        raise ValueError("at least one panel index or label is required")
    return resolved


def validate_panel_order(values: Sequence[object], count: int) -> list[int]:
    """Trait validator body: an empty order or a full permutation of the panels."""
    values = list(values or [])
    if not values:
        return []
    clean: list[int] = []
    try:
        for value in values:
            if isinstance(value, bool):
                raise TypeError
            clean.append(int(value))
    except (TypeError, ValueError) as exc:
        raise traitlets.TraitError("panel_order must contain integer panel indices") from exc
    expected = list(range(count))
    if len(clean) != count or sorted(clean) != expected:
        raise traitlets.TraitError(
            "panel_order must include every panel index exactly once "
            f"(expected a permutation of {expected!r})"
        )
    return clean


class PanelsMixin(traitlets.HasTraits):
    """Panel reference resolution and the ``panel_order`` / ``selected_panels`` validators.

    The widget supplies ``_panel_count`` and ``_panel_title_for_index``.
    """

    def _panel_count(self) -> int:
        """Number of panels; the widget supplies it."""
        raise NotImplementedError

    def _panel_title_for_index(self, panel: int) -> str:
        """Title of one panel, which ``resolve_panel_ref`` matches labels against; the widget supplies it."""
        raise NotImplementedError

    def _resolve_panel_ref(self, panel: int | str) -> int:
        """``resolve_panel_ref`` over this widget's panels."""
        return resolve_panel_ref(panel, self._panel_count(), self._panel_title_for_index)

    def _normalize_panel_refs(
        self,
        panels: Sequence[int | str] | int | str,
        *,
        allow_empty: bool = False,
    ) -> list[int]:
        """``normalize_panel_refs`` over this widget's panels."""
        return normalize_panel_refs(panels, self._panel_count(), self._panel_title_for_index, allow_empty=allow_empty)

    @traitlets.validate("selected_panels")
    def _validate_selected_panels(self, proposal: dict) -> list[int]:
        """Normalize the UI multi-panel selection to existing panel indices."""
        count = self._panel_count()
        clean: list[int] = []
        seen: set[int] = set()
        for value in proposal["value"]:
            if isinstance(value, bool):
                continue
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < count and index not in seen:
                clean.append(index)
                seen.add(index)
        return clean

    @traitlets.validate("panel_order")
    def _validate_panel_order(self, proposal: dict) -> list[int]:
        """Normalize optional display order over source panel indices."""
        return validate_panel_order(proposal["value"], self._panel_count())
