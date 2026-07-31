# Docs Design Rules

Non-negotiable layout rules for this docs site. They exist because each one
was broken at least once and reached the published site. Fix forward from the
existing design — do not reinvent it during theme bumps or reworks.

## Widgets get the horizontal pixels

The pages exist to showcase wide interactive widgets. Every layout decision
gives spare width to the article column, not to chrome.

- **No right-hand "Contents" (page-toc) sidebar. Ever.**
  `docs/_static/custom.css` hides `.bd-sidebar-secondary`, and
  `docs/_config.yml` sets `secondary_sidebar_items: []`. A "hideable
  contents" rework reintroduced it once (2026-07-30) and was reverted the
  same day. Do not add a page-toc rail, a toc toggle button, or a
  `toc-toggle.js` back.
- **Left navigation is fixed at 16rem on desktop — never proportional.**
  The theme's default sidebar width is a percentage of the viewport, so big
  monitors silently spend hundreds of pixels on nav. `custom.css` pins
  `width / max-width / flex-basis` to `16rem` at `min-width: 960px`.

## Width and collapse offset move in lockstep

If the sidebar's laid-out width and the theme's collapsed offset disagree,
the whole article is dragged off-screen by the difference (shipped broken
twice: −25px on phones, −116px on desktops). `custom.css` therefore overrides
`.pst-sidebar-hidden { margin-left: -16rem; }` right next to the 16rem width.
Change one, change both. Below 960px the theme's off-canvas drawer is left
completely alone.

## The mobile hamburger must survive theme drift

Both sphinx themes wire the drawer to `document.querySelector('.primary-toggle')`
— the first match — and newer pydata themes render an extra hidden button
first, killing the visible hamburger. Two defenses, keep both:

- `docs/_static/nav-toggle-fix.js` forwards clicks from unwired toggle
  buttons to the wired one (idempotent, safe to double-load).
- `scripts/check_docs_nav_toggle.py` runs in the docs workflow after every
  build and fails the deploy when a page has multiple toggles without the
  shim.

## Verify locally before any push

CI is never the first build. Before pushing docs changes:

1. `scripts/docs_preview.sh` (full build + no-store server on port 8767);
   for CSS-only tweaks, `--no-build` and copy the asset into
   `docs/_build/html/_static/`.
2. Drive the built pages at phone (375/420), laptop (~1500), and large
   desktop (~1900) widths — with the left sidebar **both expanded and
   collapsed**, and the phone drawer opened and closed.
3. `python scripts/check_docs_nav_toggle.py docs/_build/html` must pass, and
   built pages must contain zero `img.quantem-static-fallback` elements.
