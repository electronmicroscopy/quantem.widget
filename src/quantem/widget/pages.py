"""Paged-gallery state shared by Show2D and Show3D.

A paged widget flattens ``pages x panels`` into one panel stack and syncs the
page metadata (``n_pages``, ``panels_per_page``, ``page_idx``, ``page_labels``,
``page_starred``, ``hidden_page_slots``) separately; the browser renders only
the active page. The validators and star helpers over those traits are the
same for every paged widget and live here once.
"""

from collections.abc import Sequence
from typing import Self

import traitlets


def resolve_page_labels(page_labels: Sequence[str | None] | None, inferred: Sequence[str], n_pages: int) -> list[str]:
    """Explicit page labels win over the inferred ones; the count must match the pages."""
    resolved = ["" if label is None else str(label) for label in (page_labels if page_labels is not None else inferred)]
    if len(resolved) != n_pages:
        raise ValueError(f"page_labels length ({len(resolved)}) must match n_pages ({n_pages})")
    return resolved


class PagesMixin(traitlets.HasTraits):
    """Validators and star helpers over the synced page traits.

    A widget whose pages are folder items (Show2D ``page_kind == "items"``)
    hides absolute panels rather than reusable slots and says so through
    ``_uses_item_pages``.
    """

    def _uses_item_pages(self) -> bool:
        """Whether pages are folder items, whose hidden panels are absolute indices rather than per-page slots."""
        return False

    @traitlets.validate("page_idx")
    def _validate_page_idx(self, proposal: dict) -> int:
        """Clamp the active page index to the available page range."""
        n_pages = max(1, int(self.n_pages))
        return int(max(0, min(int(proposal["value"]), n_pages - 1)))

    @traitlets.validate("page_starred")
    def _validate_page_starred(self, proposal: dict) -> list[int]:
        """Normalize per-page star flags."""
        flags = list(proposal["value"])
        n_pages = max(1, int(self.n_pages))
        if not flags:
            return [0] * n_pages
        if len(flags) != n_pages:
            raise traitlets.TraitError(
                f"page_starred length ({len(flags)}) must equal n_pages ({n_pages})"
            )
        return [1 if int(flag) else 0 for flag in flags]

    def _normalize_hidden_page_slots(
        self,
        values: Sequence[object],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Normalize reusable hidden panel slots for paged galleries."""
        if int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0 or self._uses_item_pages():
            return []
        n_slots = int(self.panels_per_page)
        clean_set: set[int] = set()
        for value in values or []:
            if isinstance(value, bool):
                continue
            try:
                slot = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= slot < n_slots:
                clean_set.add(slot)
        clean = sorted(clean_set)
        if len(clean) >= n_slots:
            if drop_if_full:
                clean = clean[:-1]
            else:
                raise traitlets.TraitError(
                    "hidden_page_slots cannot hide every page slot; at least one panel must remain visible"
                )
        return clean

    def _hidden_page_slots_from_panels(
        self,
        panels: Sequence[int],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Map absolute hidden panel indices to reusable page slots."""
        if int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0 or self._uses_item_pages():
            return []
        n_panels = self._panel_count()
        per_page = int(self.panels_per_page)
        slots = [
            int(panel) % per_page
            for panel in panels
            if 0 <= int(panel) < n_panels
        ]
        return self._normalize_hidden_page_slots(slots, drop_if_full=drop_if_full)

    @traitlets.validate("hidden_page_slots")
    def _validate_hidden_page_slots(self, proposal: dict) -> list[int]:
        """Normalize hidden page slots for paged galleries."""
        return self._normalize_hidden_page_slots(proposal["value"])

    def star_page(self, page: int) -> Self:
        """Mark a page with a star."""
        index = int(page)
        if index < 0 or index >= int(self.n_pages):
            raise ValueError(f"page index {index} out of range [0, {self.n_pages})")
        starred = list(self.page_starred)
        if len(starred) != int(self.n_pages):
            starred = [0] * int(self.n_pages)
        starred[index] = 1
        self.page_starred = starred
        return self
